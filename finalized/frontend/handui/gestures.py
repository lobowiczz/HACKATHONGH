"""Turns a stream of tracked hands into high-level gesture events.

GestureEngine.update(hand, t) is called once per frame and returns a list of
Event objects. All timing comes from the `t` argument (seconds), which keeps
the engine deterministic and easy to test with synthetic hands.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from . import features as F
from .config import GestureConfig

HAND_FOUND = "hand_found"
HAND_LOST = "hand_lost"
GRAB_START = "grab_start"
GRAB_MOVE = "grab_move"
GRAB_END = "grab_end"
SWIPE = "swipe"
MENU_OPEN = "menu_open"
MENU_CLOSE = "menu_close"
CUSTOM = "custom"


@dataclass
class Event:
    kind: str
    data: dict = field(default_factory=dict)


@dataclass
class GestureState:
    """Snapshot of the hand for the UI to draw from."""
    present: bool = False
    handedness: str = ""
    cursor: tuple[float, float] = (0.5, 0.5)  # normalized UI coordinates, smoothed
    extension: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    grab_strength: float = 0.0
    pinch: float = 1.0
    speed: float = 0.0
    grabbing: bool = False
    palm_open: bool = False
    menu_open: bool = False
    menu_progress: float = 0.0  # 0..1 while holding an open palm
    custom_label: str | None = None
    custom_confidence: float = 0.0


class OneEuroFilter:
    """Adaptive low-pass filter: smooth when still, responsive when moving fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x is None:
            self._x, self._t = x.astype(np.float64), t
            return self._x
        dt = max(t - self._t, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * (x - self._x) / dt + (1 - a_d) * self._dx
        a = self._alpha(self.min_cutoff + self.beta * float(np.linalg.norm(self._dx)), dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class LabelSmoother:
    """Majority vote over recent frames so a custom label doesn't flicker."""

    def __init__(self, window: int = 8, min_votes: int = 6):
        self.min_votes = min_votes
        self._votes: deque[str | None] = deque(maxlen=window)

    def push(self, label: str | None) -> str | None:
        self._votes.append(label)
        best, n = Counter(self._votes).most_common(1)[0]
        return best if best is not None and n >= self.min_votes else None

    def reset(self) -> None:
        self._votes.clear()


class GestureEngine:
    def __init__(self, config: GestureConfig | None = None, classifier=None):
        self.cfg = config or GestureConfig()
        self.classifier = classifier
        self.state = GestureState()
        self._cursor_filter = OneEuroFilter(self.cfg.cursor_min_cutoff, self.cfg.cursor_beta)
        self._smoother = LabelSmoother(self.cfg.custom_window, self.cfg.custom_min_votes)
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        self._cursor_filter.reset()
        self._smoother.reset()
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._last_seen: float | None = None
        self._prev_palm: np.ndarray | None = None
        self._prev_t = 0.0
        self._grab_frames = 0
        self._dwell_start: float | None = None
        self._dwell_latched = False
        self._swipe_block_until = 0.0
        self._menu_block_until = 0.0
        self._last_custom: str | None = None

    # ------------------------------------------------------------------ public

    def update(self, hand, t: float) -> list[Event]:
        if hand is None:
            if self.state.present and self._last_seen is not None and t - self._last_seen > self.cfg.lost_timeout:
                return self._lose_hand()
            return []

        events: list[Event] = []
        s = self.state
        if not s.present:
            s.present = True
            events.append(Event(HAND_FOUND))
        self._last_seen = t

        lm = hand.landmarks
        palm = F.palm_center(lm)[:2] / np.asarray(hand.image_size, dtype=np.float64)
        self._update_speed(palm, t)

        s.handedness = hand.handedness
        s.extension = F.finger_extension(lm)
        s.grab_strength = F.grab_strength(s.extension)
        s.pinch = F.pinch_distance(lm)
        s.cursor = tuple(self._map_cursor(palm, t))

        events += self._update_grab()
        events += self._update_swipe(palm, t)
        events += self._update_menu(t)
        events += self._update_custom(hand)
        return events

    def close_menu(self) -> None:
        """Called by the UI after a menu item is chosen."""
        self.state.menu_open = False

    # ---------------------------------------------------------------- internal

    def _lose_hand(self) -> list[Event]:
        s = self.state
        events = []
        if s.grabbing:
            events.append(Event(GRAB_END, {"pos": s.cursor, "cancelled": True}))
        if s.menu_open:
            events.append(Event(MENU_CLOSE, {"pos": s.cursor}))
        events.append(Event(HAND_LOST))
        self.state = GestureState(cursor=s.cursor)
        self._reset_tracking()
        return events

    def _update_speed(self, palm: np.ndarray, t: float) -> None:
        if self._prev_palm is not None and t > self._prev_t:
            inst = float(np.linalg.norm(palm - self._prev_palm)) / (t - self._prev_t)
            self.state.speed = 0.6 * self.state.speed + 0.4 * inst
        self._prev_palm, self._prev_t = palm, t

    def _map_cursor(self, palm: np.ndarray, t: float) -> np.ndarray:
        m = self.cfg.cursor_margin
        return self._cursor_filter(np.clip((palm - m) / (1 - 2 * m), 0.0, 1.0), t)

    def _update_grab(self) -> list[Event]:
        cfg, s = self.cfg, self.state
        engage = s.grab_strength >= cfg.grab_on or (cfg.pinch_grab and s.pinch <= cfg.pinch_on)
        release = s.grab_strength <= cfg.grab_off and (not cfg.pinch_grab or s.pinch >= cfg.pinch_off)

        if not s.grabbing:
            self._grab_frames = self._grab_frames + 1 if engage else 0
            if self._grab_frames >= cfg.grab_confirm_frames:
                s.grabbing = True
                return [Event(GRAB_START, {"pos": s.cursor})]
            return []
        if release:
            s.grabbing = False
            self._grab_frames = 0
            return [Event(GRAB_END, {"pos": s.cursor, "cancelled": False})]
        return [Event(GRAB_MOVE, {"pos": s.cursor})]

    def _update_swipe(self, palm: np.ndarray, t: float) -> list[Event]:
        cfg, s = self.cfg, self.state
        hist = self._history
        if s.grabbing or s.menu_open:
            hist.clear()  # motion while dragging or choosing must not count as a swipe later
            return []
        hist.append((t, palm))
        while t - hist[0][0] > cfg.swipe_window:
            hist.popleft()
        if t < self._swipe_block_until or s.extension[1:].mean() < cfg.swipe_min_open:
            return []

        t0, p0 = hist[0]
        if t - t0 < 0.05:
            return []
        dx, dy = palm - p0
        direction = None
        if abs(dx) >= cfg.swipe_distance and abs(dx) >= cfg.swipe_axis_ratio * abs(dy):
            direction = "right" if dx > 0 else "left"
        elif abs(dy) >= cfg.swipe_distance_y and abs(dy) >= cfg.swipe_axis_ratio * abs(dx):
            direction = "down" if dy > 0 else "up"
        if direction is None:
            return []

        hist.clear()
        self._swipe_block_until = t + cfg.swipe_cooldown
        self._menu_block_until = t + cfg.menu_block_after_swipe
        self._dwell_start = None
        s.menu_progress = 0.0
        return [Event(SWIPE, {"direction": direction, "speed": float(math.hypot(dx, dy) / (t - t0))})]

    def _update_menu(self, t: float) -> list[Event]:
        cfg, s = self.cfg, self.state
        s.palm_open = bool(s.extension[0] >= cfg.palm_thumb_min and s.extension[1:].min() >= cfg.palm_finger_min)
        if not s.palm_open:
            self._dwell_latched = False  # the palm has to close before it can toggle again
            self._dwell_start = None
            s.menu_progress = 0.0
            return []
        if self._dwell_latched or s.grabbing or t < self._menu_block_until or s.speed > cfg.palm_max_speed:
            self._dwell_start = None
            s.menu_progress = 0.0
            return []

        if self._dwell_start is None:
            self._dwell_start = t
        s.menu_progress = min(1.0, (t - self._dwell_start) / cfg.menu_dwell)
        if s.menu_progress < 1.0:
            return []
        self._dwell_latched = True
        self._dwell_start = None
        s.menu_progress = 0.0
        s.menu_open = not s.menu_open
        return [Event(MENU_OPEN if s.menu_open else MENU_CLOSE, {"pos": s.cursor})]

    def _update_custom(self, hand) -> list[Event]:
        if self.classifier is None:
            return []
        s = self.state
        label, conf = self.classifier.predict(F.gesture_features(hand.landmarks, hand.handedness))
        stable = self._smoother.push(label if conf >= self.cfg.custom_min_confidence else None)
        s.custom_label = stable
        s.custom_confidence = conf if stable == label else 0.0
        if stable == self._last_custom:
            return []
        self._last_custom = stable  # a held gesture fires once; it must change before firing again
        if stable is None or stable in self.cfg.custom_ignore_labels:
            return []
        return [Event(CUSTOM, {"label": stable, "confidence": conf})]
