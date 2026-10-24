"""Synthetic hand landmarks for testing gestures without a camera."""
from __future__ import annotations

import numpy as np

from handui import features as F
from handui.tracker import Hand

IMAGE_SIZE = (1280, 720)

# Hand-local layout, palm facing the camera, image y pointing down; units ~ palm size.
_KNUCKLES = {"index": (-0.38, -1.0), "middle": (-0.12, -1.08), "ring": (0.13, -1.02), "pinky": (0.36, -0.88)}
_LENGTHS = {"index": (0.45, 0.27, 0.22), "middle": (0.5, 0.3, 0.23),
            "ring": (0.46, 0.28, 0.22), "pinky": (0.36, 0.22, 0.2)}
_FINGER_JOINT_BEND = (1.5, 1.7, 1.2)   # radians per joint at curl = 1
_THUMB_CMC = (-0.32, -0.28)
_THUMB_DIR = np.array([-0.756, -0.655, 0.0])
_THUMB_LENGTHS = (0.33, 0.3, 0.25)
_THUMB_JOINT_BEND = (0.6, 0.9, 0.9)

POSES = {  # curl per finger: thumb, index, middle, ring, pinky
    "open": (0, 0, 0, 0, 0),
    "fist": (1, 1, 1, 1, 1),
    "point": (1, 0, 1, 1, 1),
    "peace": (1, 0, 0, 1, 1),
}


def _chain(base, direction, lengths, bends, curl, bend_toward):
    pts, p, angle = [], np.array(base, dtype=float), 0.0
    for length, bend in zip(lengths, bends):
        angle += bend * curl
        d = np.cos(angle) * direction + np.sin(angle) * bend_toward
        p = p + length * d
        pts.append(p)
    return pts


def local_hand(curls=(0, 0, 0, 0, 0), pinch: bool = False) -> np.ndarray:
    lm = np.zeros((21, 3))
    cmc = np.array([*_THUMB_CMC, 0.0])
    lm[1] = cmc
    toward_palm = np.array([0.655, -0.756, -0.3])
    toward_palm /= np.linalg.norm(toward_palm)
    lm[2:5] = _chain(cmc, _THUMB_DIR, _THUMB_LENGTHS, _THUMB_JOINT_BEND, curls[0], toward_palm)

    for name, curl, idx in zip(_KNUCKLES, curls[1:], (5, 9, 13, 17)):
        mcp = np.array([*_KNUCKLES[name], 0.0])
        direction = mcp / np.linalg.norm(mcp)
        lm[idx] = mcp
        lm[idx + 1:idx + 4] = _chain(mcp, direction, _LENGTHS[name], _FINGER_JOINT_BEND, curl,
                                     np.array([0.0, 0.0, -1.0]))
    if pinch:
        lm[4] = lm[8] + (0.03, 0.0, 0.0)
        lm[3] = (lm[2] + lm[4]) / 2
    return lm


def make_hand(curls=(0, 0, 0, 0, 0), pos=(640, 360), scale=110.0, rotation=0.0, pinch=False,
              handedness="Right", noise=0.0, rng: np.random.Generator | None = None) -> Hand:
    lm = local_hand(curls, pinch)
    if noise:
        lm = lm + (rng or np.random.default_rng()).normal(0, noise, lm.shape)
    if handedness == "Left":
        lm[:, 0] *= -1
    c, s = np.cos(rotation), np.sin(rotation)
    xy = lm[:, :2] @ np.array([[c, s], [-s, c]])
    lm = np.column_stack([xy, lm[:, 2]]) * scale
    lm[:, :2] += np.asarray(pos) - F.palm_center(lm)[:2]
    return Hand(lm.astype(np.float32), handedness, 0.99, IMAGE_SIZE)


def pose(name: str, **kwargs) -> Hand:
    return make_hand(POSES[name], **kwargs)
