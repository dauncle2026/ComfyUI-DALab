from typing import Tuple

import cv2
import numpy as np
import torch


def _bbox_to_center_scale(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center, scale = center[0], scale[0]
    return center, scale


def _adjust_aspect(scale: np.ndarray, ratio: float) -> np.ndarray:
    w, h = np.hsplit(scale, [1])
    return np.where(w > h * ratio, np.hstack([w, w / ratio]), np.hstack([h * ratio, h]))


def _rotate_pt(pt: np.ndarray, angle: float) -> np.ndarray:
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s], [s, c]]) @ pt


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return b + np.r_[-d[1], d[0]]


def _warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, out_size: Tuple[int, int], inv: bool = False) -> np.ndarray:
    src_w, dst_w, dst_h = scale[0], out_size[0], out_size[1]
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_pt(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = _third_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _third_point(dst[0, :], dst[1, :])

    return cv2.getAffineTransform(np.float32(dst), np.float32(src)) if inv else cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _affine_transform(img: np.ndarray, input_size: Tuple[int, int], scale: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w, h = input_size
    scale = _adjust_aspect(scale, w / h)
    mat = _warp_matrix(center, scale, 0, (w, h))
    return cv2.warpAffine(img, mat, (int(w), int(h)), flags=cv2.INTER_LINEAR), scale


def _decode_simcc(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, K, Wx = x.shape
    x_flat, y_flat = x.reshape(N * K, -1), y.reshape(N * K, -1)

    x_locs = np.argmax(x_flat, axis=1)
    y_locs = np.argmax(y_flat, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    max_x = np.amax(x_flat, axis=1)
    max_y = np.amax(y_flat, axis=1)
    mask = max_x > max_y
    max_x[mask] = max_y[mask]
    vals = max_x
    locs[vals <= 0.] = -1

    return locs.reshape(N, K, 2), vals.reshape(N, K)


def _prepare_batch(img: np.ndarray, bboxes, input_size: Tuple[int, int]):
    h, w = img.shape[:2]
    images, centers, scales = [], [], []

    if len(bboxes) == 0:
        bboxes = [[0, 0, w, h]]

    for box in bboxes:
        bbox = np.array([box[0], box[1], box[2], box[3]])
        center, scale = _bbox_to_center_scale(bbox, padding=1.25)
        warped, scale = _affine_transform(img, input_size, scale, center)

        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        warped = (warped - mean) / std

        images.append(warped)
        centers.append(center)
        scales.append(scale)

    return images, centers, scales


def _run_model(model, images, batch_size=5):
    orig_count = len(images)
    while len(images) % batch_size != 0:
        images.append(np.zeros_like(images[0]))

    inp = np.stack(images, axis=0).transpose(0, 3, 1, 2)
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    tensor = torch.from_numpy(inp).to(device, dtype)

    out1, out2 = [], []
    for i in range(tensor.shape[0] // batch_size):
        batch_out = model(tensor[i * batch_size:(i + 1) * batch_size])
        out1.append(batch_out[0].float())
        out2.append(batch_out[1].float())

    out1 = torch.cat(out1, dim=0)[:orig_count].cpu().detach().numpy()
    out2 = torch.cat(out2, dim=0)[:orig_count].cpu().detach().numpy()

    return [[out1[i:i+1, ...], out2[i:i+1, ...]] for i in range(orig_count)]


def _process_outputs(outputs, input_size, centers, scales, split_ratio=2.0):
    all_kpts, all_scores = [], []

    for i, (simcc_x, simcc_y) in enumerate(outputs):
        locs, scores = _decode_simcc(simcc_x, simcc_y)
        kpts = locs / split_ratio
        kpts = kpts / input_size * scales[i] + centers[i] - scales[i] / 2
        all_kpts.append(kpts[0])
        all_scores.append(scores[0])

    return np.array(all_kpts), np.array(all_scores)


def run_estimation(model, bboxes, image, input_size=(288, 384)):
    images, centers, scales = _prepare_batch(image, bboxes, input_size)
    outputs = _run_model(model, images)
    return _process_outputs(outputs, input_size, centers, scales)
