"""Tunable thresholds, file locations and gesture-to-action bindings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
CLASSIFIER_PATH = MODELS_DIR / "gesture_classifier.joblib"
SAMPLES_PATH = DATA_DIR / "gesture_samples.csv"


@dataclass
class GestureConfig:
    # Grab: fires on a closed fist (grab strength) or a thumb-index pinch.
    # The *_on / *_off pairs give hysteresis so a grab doesn't flicker.
    grab_on: float = 0.7           # features.grab_strength, 0 = open .. 1 = fist
    grab_off: float = 0.4
    pinch_grab: bool = True
    pinch_on: float = 0.25         # thumb-tip to index-tip distance / palm size
    pinch_off: float = 0.45
    grab_confirm_frames: int = 2

    # Swipe: a fast, mostly straight move of an open hand.
    swipe_window: float = 0.35     # seconds of motion considered
    swipe_distance: float = 0.22   # fraction of frame width (horizontal)
    swipe_distance_y: float = 0.28  # fraction of frame height (vertical)
    swipe_axis_ratio: float = 1.8  # dominant axis must beat the other by this much
    swipe_min_open: float = 0.5    # mean finger extension needed to count as a swipe
    swipe_cooldown: float = 0.6

    # Palm-open menu: hold an open, still palm to toggle the menu.
    palm_thumb_min: float = 0.55
    palm_finger_min: float = 0.7
    palm_max_speed: float = 0.35   # frame widths per second
    menu_dwell: float = 0.7
    menu_block_after_swipe: float = 1.0

    # Cursor: the palm's central region of the camera maps to the whole screen.
    cursor_margin: float = 0.15
    cursor_min_cutoff: float = 1.2  # One Euro filter: lower = smoother at rest
    cursor_beta: float = 1.5        # One Euro filter: higher = less lag when moving fast

    lost_timeout: float = 0.3      # tolerate brief tracking dropouts

    # Custom gestures from the trained classifier.
    custom_min_confidence: float = 0.75
    custom_window: int = 8
    custom_min_votes: int = 6
    custom_ignore_labels: tuple[str, ...] = ("none",)


# Gesture -> UI action. Actions are listed in HandUI.ACTIONS.
# Custom gestures use the key "custom:<label>" with the label you recorded.
BINDINGS: dict[str, str] = {
    "swipe_left": "next_page",
    "swipe_right": "prev_page",
    "swipe_up": "new_card",
    "swipe_down": "tidy_cards",
    "custom:thumbs_up": "new_card",
    "custom:peace": "toggle_camera",
}
