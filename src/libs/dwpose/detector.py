import cv2
import numpy as np
import torch


def _nms(boxes, scores, threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


def _multiclass_nms(boxes, scores, nms_thresh, score_thresh):
    results = []
    num_classes = scores.shape[1]

    for cls in range(num_classes):
        cls_scores = scores[:, cls]
        mask = cls_scores > score_thresh
        if mask.sum() == 0:
            continue

        valid_scores = cls_scores[mask]
        valid_boxes = boxes[mask]
        keep = _nms(valid_boxes, valid_scores, nms_thresh)

        if len(keep) > 0:
            cls_ids = np.ones((len(keep), 1)) * cls
            dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_ids], 1)
            results.append(dets)

    return np.concatenate(results, 0) if results else None


def _decode_output(output, img_size):
    strides = [8, 16, 32]
    grids, exp_strides = [], []

    for stride in strides:
        h, w = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        exp_strides.append(np.full((*grid.shape[:2], 1), stride))

    grids = np.concatenate(grids, 1)
    exp_strides = np.concatenate(exp_strides, 1)
    output[..., :2] = (output[..., :2] + grids) * exp_strides
    output[..., 2:4] = np.exp(output[..., 2:4]) * exp_strides

    return output


def _prepare_input(img, size, swap=(2, 0, 1)):
    padded = np.ones((size[0], size[1], 3), dtype=np.uint8) * 114 if len(img.shape) == 3 else np.ones(size, dtype=np.uint8) * 114
    ratio = min(size[0] / img.shape[0], size[1] / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded[:int(img.shape[0] * ratio), :int(img.shape[1] * ratio)] = resized
    padded = padded.transpose(swap)
    return np.ascontiguousarray(padded, dtype=np.float32), ratio


def run_detection(model, image, target_classes=[0]):
    input_size = (640, 640)
    inp, ratio = _prepare_input(image, input_size)

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    tensor = torch.from_numpy(inp[None, :, :, :]).to(device, dtype)

    out = model(tensor).float().cpu().detach().numpy()
    preds = _decode_output(out[0], input_size)

    boxes = preds[:, :4]
    scores = preds[:, 4:5] * preds[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = _multiclass_nms(boxes_xyxy, scores, nms_thresh=0.45, score_thresh=0.1)
    if dets is None:
        return None

    final_boxes, final_scores, final_classes = dets[:, :4], dets[:, 4], dets[:, 5]
    valid = (final_scores > 0.3) & np.isin(final_classes, target_classes)
    return final_boxes[valid]
