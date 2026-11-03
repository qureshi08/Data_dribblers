"""Pitch template in metres and the camera to pitch homography.

The 32 keypoints follow the order of Roboflow's football-pitch-detection model
(sports/configs/soccer.py), but use real FIFA line dimensions on a 105 x 68 m pitch,
because the model finds real line intersections.
"""
import cv2
import numpy as np

LENGTH, WIDTH = 105.0, 68.0
PBL, PBW = 16.5, 40.32      # penalty box length, width
GBL, GBW = 5.5, 18.32       # goal box length, width
CCR, PSD = 9.15, 11.0       # centre circle radius, penalty spot distance
GOAL_W = 7.32

L, W = LENGTH, WIDTH
PITCH_VERTICES = np.array([
    (0, 0), (0, (W - PBW) / 2), (0, (W - GBW) / 2), (0, (W + GBW) / 2), (0, (W + PBW) / 2), (0, W),
    (GBL, (W - GBW) / 2), (GBL, (W + GBW) / 2), (PSD, W / 2),
    (PBL, (W - PBW) / 2), (PBL, (W - GBW) / 2), (PBL, (W + GBW) / 2), (PBL, (W + PBW) / 2),
    (L / 2, 0), (L / 2, W / 2 - CCR), (L / 2, W / 2 + CCR), (L / 2, W),
    (L - PBL, (W - PBW) / 2), (L - PBL, (W - GBW) / 2), (L - PBL, (W + GBW) / 2), (L - PBL, (W + PBW) / 2),
    (L - PSD, W / 2), (L - GBL, (W - GBW) / 2), (L - GBL, (W + GBW) / 2),
    (L, 0), (L, (W - PBW) / 2), (L, (W - GBW) / 2), (L, (W + GBW) / 2), (L, (W + PBW) / 2), (L, W),
    (L / 2 - CCR, W / 2), (L / 2 + CCR, W / 2),
], dtype=np.float32)


def fit_homography(keypoints_xy, keypoints_conf, min_conf=0.5):
    """Frame pixels -> pitch metres. Returns None when fewer than 4 good, spread out points."""
    mask = keypoints_conf > min_conf
    if mask.sum() < 4:
        return None
    src = keypoints_xy[mask].astype(np.float32)
    dst = PITCH_VERTICES[mask]
    # four points on one line give a degenerate homography
    if np.linalg.matrix_rank(np.c_[dst, np.ones(len(dst))]) < 3:
        return None
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def to_pitch(H, points_xy):
    if H is None or len(points_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def on_pitch(xy, margin=3.0):
    return (xy[:, 0] > -margin) & (xy[:, 0] < LENGTH + margin) & (xy[:, 1] > -margin) & (xy[:, 1] < WIDTH + margin)
