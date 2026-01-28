from typing import List, Optional
from timeit import default_timer

import numpy as np
import torch

from .detector import run_detection
from .estimator import run_estimation
from .types import PoseResult, BodyResult, Keypoint


def _get_input_size(filename):
    size = (192, 256)
    if "384" in filename:
        size = (288, 384)
    elif "256" in filename:
        size = (256, 256)
    return size


class PoseEngine:
    def __init__(self, det_path: Optional[str] = None, pose_path: Optional[str] = None, device="cuda"):
        import os
        self.det_name = det_path and os.path.basename(det_path)
        self.pose_name = pose_path and os.path.basename(pose_path)
        self.det_model, self.pose_model = None, None

        if det_path is not None:
            self.det_model = torch.jit.load(det_path)
            self.det_model.to(device)

        if pose_path is not None:
            self.pose_model = torch.jit.load(pose_path)
            self.pose_model.to(device)

        if self.pose_name:
            self.pose_input_size = _get_input_size(self.pose_name)

    def __call__(self, image) -> Optional[np.ndarray]:
        boxes = None

        if self.det_model is None:
            print("DWPose: No detector, using full image.")
            boxes = []
        else:
            t0 = default_timer()
            boxes = run_detection(self.det_model, image, target_classes=[0])
            if boxes is None or boxes.shape[0] == 0:
                return None

        t0 = default_timer()
        kpts, scores = run_estimation(self.pose_model, boxes, image, self.pose_input_size)

        count = 'full image'
        if hasattr(boxes, 'shape') and boxes.shape[0] > 0:
            count = f"{boxes.shape[0]} people"

        info = np.concatenate((kpts, scores[..., None]), axis=-1)
        neck = np.mean(info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(info[:, 5, 2:4] > 0.3, info[:, 6, 2:4] > 0.3).astype(int)
        info = np.insert(info, 17, neck, axis=1)

        mm_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        op_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        info[:, op_idx] = info[:, mm_idx]

        return info

    def release(self):
        """Release models from GPU memory."""
        if self.det_model is not None:
            del self.det_model
            self.det_model = None
        if self.pose_model is not None:
            del self.pose_model
            self.pose_model = None
        torch.cuda.empty_cache()

    @staticmethod
    def to_pose_results(keypoints_info: Optional[np.ndarray]) -> List[PoseResult]:
        def parse_part(part: np.ndarray) -> Optional[List[Optional[Keypoint]]]:
            kps = [Keypoint(x, y, s, i) if s >= 0.3 else None for i, (x, y, s) in enumerate(part)]
            return None if all(k is None for k in kps) else kps

        def sum_scores(kps: Optional[List[Optional[Keypoint]]]) -> float:
            return sum(k.score for k in kps if k is not None) if kps else 0.0

        results = []
        if keypoints_info is None:
            return results

        for inst in keypoints_info:
            body_kps = parse_part(inst[:18]) or ([None] * 18)
            left_hand = parse_part(inst[92:113])
            right_hand = parse_part(inst[113:134])
            face = parse_part(inst[24:92])

            if face is not None:
                face.append(body_kps[14])
                face.append(body_kps[15])

            body = BodyResult(body_kps, sum_scores(body_kps), len(body_kps))
            results.append(PoseResult(body, left_hand, right_hand, face))

        return results
