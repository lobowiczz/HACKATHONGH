"""One frame of the three hand signals the screening tests measure."""
from __future__ import annotations

from dataclasses import dataclass

from .. import features as F
from ..tracker import Hand


@dataclass
class Sample:
    t: float
    pinch: float                 # thumb-tip to index-tip distance, in palm sizes
    fist: float                  # 0 = open hand .. 1 = tight fist
    palm: tuple[float, float]    # palm centre (x, y), in palm sizes, so it is camera-distance independent


def sample_from_hand(hand: Hand, t: float) -> Sample:
    lm = hand.landmarks
    size = F.palm_size(lm)
    x, y = (F.palm_center(lm)[:2] / size).tolist()
    return Sample(t, F.pinch_distance(lm), F.grab_strength(F.finger_extension(lm)), (x, y))
