"""Synthetic hand signals so the dashboard can be demonstrated and tested without a camera.

Everything produced here is labelled source="simulated" by the runner and kept apart
from real recordings.
"""
from __future__ import annotations

import math

import numpy as np

from .signals import Sample

PROFILES = {
    "healthy": "steady size and rhythm, no tremor",
    "slowing": "movements shrink and slow over the test, with one pause",
    "tremor": "steady movement plus a 5 Hz tremor in the held hand",
}


class Simulator:
    def __init__(self, profile: str = "healthy", seed: int = 0):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {', '.join(PROFILES)}")
        self.profile = profile
        self._rng = np.random.default_rng(seed)
        self._phase = {"pinch": 0.0, "fist": 0.0}
        self._last_t: float | None = None

    def sample(self, t: float, elapsed: float | None) -> Sample:
        dt = 0.0 if self._last_t is None else min(t - self._last_t, 0.1)
        self._last_t = t
        e = elapsed or 0.0
        slowing = self.profile == "slowing"
        scale = max(0.4, 1 - 0.06 * e) if slowing else 1.0
        freq_scale = max(0.5, 1 - 0.03 * e) if slowing else 1.0
        if slowing and 6.0 <= e < 6.8:
            freq_scale = 0.0  # a pause

        for key, hz in (("pinch", 4.5), ("fist", 2.2)):
            self._phase[key] += 2 * math.pi * hz * freq_scale * dt * (1 + self._rng.normal(0, 0.03))
        n = lambda s: float(self._rng.normal(0, s))  # noqa: E731
        pinch = 0.6 + 0.35 * scale * math.sin(self._phase["pinch"]) + n(0.01)
        fist = 0.5 + 0.45 * scale * math.sin(self._phase["fist"]) + n(0.01)
        tremor = 0.012 * math.sin(2 * math.pi * 5.0 * t) if self.profile == "tremor" else 0.0
        palm = (5.0 + tremor + n(0.002), 3.0 + 0.6 * tremor + n(0.002))
        return Sample(t, pinch, fist, palm)
