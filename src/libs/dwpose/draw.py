import math
from typing import List, Optional, Union

import cv2
import matplotlib
import numpy as np

from .types import Keypoint

EPS = 0.01


def check_normalized(keypoints: List[Optional[Keypoint]]) -> bool:
    valid = [0 <= abs(k.x) <= 1 and 0 <= abs(k.y) <= 1 for k in keypoints if k is not None]
    return all(valid) if valid else False


def render_body(canvas: np.ndarray, keypoints: List[Keypoint], scale_sticks: bool = False) -> np.ndarray:
    if not check_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    CH, CW, _ = canvas.shape
    stick_width = 4
    max_dim = max(CW, CH)
    scale = 1 if not scale_sticks or max_dim < 500 else min(2 + (max_dim // 1000), 7)

    limbs = [
        [2, 3], [2, 6], [3, 4], [4, 5],
        [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16],
        [16, 18],
    ]

    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
        [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
        [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
        [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
        [255, 0, 170], [255, 0, 85]
    ]

    for (i1, i2), color in zip(limbs, colors):
        p1, p2 = keypoints[i1 - 1], keypoints[i2 - 1]
        if p1 is None or p2 is None:
            continue

        Y = np.array([p1.x, p2.x]) * float(W)
        X = np.array([p1.y, p2.y]) * float(H)
        mx, my = np.mean(X), np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        poly = cv2.ellipse2Poly((int(my), int(mx)), (int(length / 2), stick_width * scale), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(float(c) * 0.6) for c in color])

    for kp, color in zip(keypoints, colors):
        if kp is None:
            continue
        x, y = int(kp.x * W), int(kp.y * H)
        cv2.circle(canvas, (x, y), 4, color, thickness=-1)

    return canvas


def render_hand(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    if not keypoints:
        return canvas

    if not check_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
        [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]
    ]

    for idx, (e1, e2) in enumerate(edges):
        k1, k2 = keypoints[e1], keypoints[e2]
        if k1 is None or k2 is None:
            continue

        x1, y1 = int(k1.x * W), int(k1.y * H)
        x2, y2 = int(k2.x * W), int(k2.y * H)
        if x1 > EPS and y1 > EPS and x2 > EPS and y2 > EPS:
            hsv_color = matplotlib.colors.hsv_to_rgb([idx / float(len(edges)), 1.0, 1.0]) * 255
            cv2.line(canvas, (x1, y1), (x2, y2), hsv_color, thickness=2)

    for kp in keypoints:
        if kp is None:
            continue
        x, y = int(kp.x * W), int(kp.y * H)
        if x > EPS and y > EPS:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    return canvas


def render_face(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    if not keypoints:
        return canvas

    if not check_normalized(keypoints):
        H, W = 1.0, 1.0
    else:
        H, W, _ = canvas.shape

    for kp in keypoints:
        if kp is None:
            continue
        x, y = int(kp.x * W), int(kp.y * H)
        if x > EPS and y > EPS:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)

    return canvas
