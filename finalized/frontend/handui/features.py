"""Geometry helpers that turn 21 hand landmarks into gesture-friendly measurements.

Landmarks are (21, 3) arrays in pixel-like units: x and y in image pixels and
z scaled to the same units as x, so distances and angles are isotropic.
"""
from __future__ import annotations

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (THUMB, INDEX, MIDDLE, RING, PINKY)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
TIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (0, 5, 9, 13, 17)

FEATURE_SIZE = 21 * 2 + 5 + 4


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def palm_size(lm: np.ndarray) -> float:
    """Average side of the wrist / index-knuckle / pinky-knuckle triangle."""
    w, i, p = lm[WRIST], lm[5], lm[17]
    return float(np.linalg.norm(i - w) + np.linalg.norm(p - w) + np.linalg.norm(i - p)) / 3 + 1e-6


def palm_center(lm: np.ndarray) -> np.ndarray:
    """Wrist and knuckles barely move when the fingers curl, so this stays put during a grab."""
    return lm[list(PALM_POINTS)].mean(axis=0)


def finger_extension(lm: np.ndarray) -> np.ndarray:
    """Per-finger extension in [0, 1] (thumb, index, middle, ring, pinky); 1 = straight."""
    ext = np.empty(5, dtype=np.float32)
    size = palm_size(lm)

    # Thumb: an extended thumb sits far from the pinky knuckle; a tucked one crosses the palm.
    d = np.linalg.norm(lm[4] - lm[17]) / size
    ext[0] = np.clip((d - 0.85) / 0.55, 0.0, 1.0)

    # Other fingers: total bend over the knuckle and both finger joints.
    for i, (mcp, pip, dip, tip) in enumerate(FINGERS[1:], start=1):
        bend = (
            _angle(lm[mcp] - lm[WRIST], lm[pip] - lm[mcp])
            + _angle(lm[pip] - lm[mcp], lm[dip] - lm[pip])
            + _angle(lm[dip] - lm[pip], lm[tip] - lm[dip])
        )
        ext[i] = np.clip(1.0 - (bend - 0.7) / 2.0, 0.0, 1.0)
    return ext


def grab_strength(extension: np.ndarray) -> float:
    """0 = open hand, 1 = tight fist (thumb ignored).

    Averages the mean and the max finger extension, so a single straight
    finger (pointing) keeps the score low instead of reading as a fist.
    """
    fingers = extension[1:]
    return float(1.0 - (fingers.mean() + fingers.max()) / 2)


def pinch_distance(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[4] - lm[8]) / palm_size(lm))


def gesture_features(lm: np.ndarray, handedness: str) -> np.ndarray:
    """Position-, scale- and handedness-invariant feature vector for the custom classifier.

    Rotation is deliberately kept so pose pairs like thumbs-up and thumbs-down stay distinct.
    """
    size = palm_size(lm)
    pts = (lm - lm[WRIST]) / size
    if handedness == "Left":
        pts[:, 0] *= -1  # mirror left hands so one model covers both
    thumb_to_tips = [np.linalg.norm(lm[4] - lm[t]) / size for t in TIPS[1:]]
    return np.concatenate([pts[:, :2].ravel(), finger_extension(lm), thumb_to_tips]).astype(np.float32)


def rotate_landmarks(lm: np.ndarray, angle: float) -> np.ndarray:
    """Rotate in the image plane around the wrist (used for training augmentation)."""
    c, s = np.cos(angle), np.sin(angle)
    out = lm.copy()
    rel = lm[:, :2] - lm[WRIST, :2]
    out[:, 0] = lm[WRIST, 0] + rel[:, 0] * c - rel[:, 1] * s
    out[:, 1] = lm[WRIST, 1] + rel[:, 0] * s + rel[:, 1] * c
    return out
