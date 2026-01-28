import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from .types import PoseResult
from .core import PoseEngine
from .draw import render_body, render_hand, render_face


def _to_hwc3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    H, W, C = x.shape
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        rgb = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        return (rgb * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(np.uint8)


def _safe_copy(x):
    return np.ascontiguousarray(x.copy()).copy()


def _pad_to_64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def _resize_with_pad(img, resolution, method="INTER_CUBIC"):
    img = _to_hwc3(img)
    H, W, _ = img.shape
    if resolution == 0:
        return img, lambda x: x

    k = float(resolution) / float(min(H, W))
    Ht, Wt = int(np.round(H * k)), int(np.round(W * k))
    interp = getattr(cv2, method) if k > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (Wt, Ht), interpolation=interp)
    Hp, Wp = _pad_to_64(Ht), _pad_to_64(Wt)
    padded = np.pad(img, [[0, Hp], [0, Wp], [0, 0]], mode='edge')

    def unpad(x):
        return _safe_copy(x[:Ht, :Wt, ...])

    return _safe_copy(padded), unpad


def _validate_input(img, out_type, **kw):
    if img is None:
        raise ValueError("Input image required.")
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)
        out_type = out_type or "pil"
    else:
        out_type = out_type or "np"
    return img, out_type


def render_poses(poses: List[PoseResult], H, W, body=True, hand=True, face=True, scale_sticks=False):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for pose in poses:
        if body:
            canvas = render_body(canvas, pose.body.keypoints, scale_sticks)
        if hand:
            canvas = render_hand(canvas, pose.left_hand)
            canvas = render_hand(canvas, pose.right_hand)
        if face:
            canvas = render_face(canvas, pose.face)
    return canvas


def poses_to_dict(poses: List[PoseResult], height: int, width: int) -> dict:
    def compress(kps):
        if not kps:
            return None
        return [v for k in kps for v in ([float(k.x), float(k.y), 1.0] if k else [0.0, 0.0, 0.0])]

    return {
        'people': [{
            'pose_keypoints_2d': compress(p.body.keypoints),
            'face_keypoints_2d': compress(p.face),
            'hand_left_keypoints_2d': compress(p.left_hand),
            'hand_right_keypoints_2d': compress(p.right_hand),
        } for p in poses],
        'canvas_height': height,
        'canvas_width': width,
    }


_cached_engine = PoseEngine()


class DwposeDetector:
    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def release(cls):
        """Release models from GPU memory."""
        global _cached_engine
        if _cached_engine is not None:
            _cached_engine.release()
            _cached_engine = PoseEngine()

    @classmethod
    def from_local(cls, det_model_path: str, pose_model_path: str, torchscript_device="cuda"):
        global _cached_engine

        det_name = os.path.basename(det_model_path)
        pose_name = os.path.basename(pose_model_path)

        if _cached_engine.det_model is None or _cached_engine.det_name != det_name:
            t = PoseEngine(det_model_path, None, device=torchscript_device)
            t.pose_model = _cached_engine.pose_model
            t.pose_name = _cached_engine.pose_name
            t.pose_input_size = getattr(_cached_engine, 'pose_input_size', (288, 384))
            _cached_engine = t

        if _cached_engine.pose_model is None or _cached_engine.pose_name != pose_name:
            t = PoseEngine(None, pose_model_path, device=torchscript_device)
            t.det_model = _cached_engine.det_model
            t.det_name = _cached_engine.det_name
            _cached_engine = t

        return cls(_cached_engine)

    def detect(self, image) -> List[PoseResult]:
        with torch.no_grad():
            info = self.engine(image.copy())
            return PoseEngine.to_pose_results(info)

    def __call__(self, image, detect_resolution=512, include_body=True, include_hand=False,
                 include_face=False, output_type="pil", image_and_json=False,
                 upscale_method="INTER_CUBIC", scale_sticks=False, **kw):

        image, output_type = _validate_input(image, output_type, **kw)
        image, _ = _resize_with_pad(image, 0, upscale_method)
        poses = self.detect(image)

        canvas = render_poses(poses, image.shape[0], image.shape[1],
                              body=include_body, hand=include_hand,
                              face=include_face, scale_sticks=scale_sticks)
        canvas, unpad = _resize_with_pad(canvas, detect_resolution, upscale_method)
        result = _to_hwc3(unpad(canvas))

        if output_type == "pil":
            result = Image.fromarray(result)

        if image_and_json:
            return result, poses_to_dict(poses, image.shape[0], image.shape[1])
        return result
