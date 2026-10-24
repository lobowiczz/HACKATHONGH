"""Webcam hand tracking built on the MediaPipe Tasks HandLandmarker."""
from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .config import LANDMARKER_PATH

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class Hand:
    landmarks: np.ndarray        # (21, 3): x, y in pixels; z in the same units as x
    handedness: str              # "Left" or "Right" (as the user sees it in the mirrored view)
    score: float
    image_size: tuple[int, int]  # (width, height) of the frame the landmarks belong to

    @property
    def normalized(self) -> np.ndarray:
        w, h = self.image_size
        return self.landmarks[:, :2] / (w, h)


def ensure_model(path: Path = LANDMARKER_PATH) -> Path:
    """Download the MediaPipe hand landmark model (~7.5 MB) on first use."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmark model to {path} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(path)
    return path


def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. On macOS, camera access belongs to the app that "
            "launched Python (Terminal, iTerm, VS Code, Claude, ...). Allow it in System Settings > "
            "Privacy & Security > Camera, then quit and reopen that app. The first run can fail "
            "while the permission prompt is showing; just run again."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class HandTracker:
    """Feed mirrored BGR frames in, get Hand objects out.

    MediaPipe assumes selfie (mirrored) input when labelling handedness, so
    flip camera frames horizontally before calling process().
    """

    def __init__(self, max_hands: int = 1, min_confidence: float = 0.6, model_path: Path | None = None):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path or ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray) -> list[Hand]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # VIDEO mode needs increasing timestamps
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            arr = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
            hands.append(Hand(arr, handed[0].category_name, float(handed[0].score), (w, h)))
        hands.sort(key=lambda hand: hand.score, reverse=True)
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
