"""Guided test flow: wait for a hand, count down, record 10 seconds, analyse, store.

Driven by update(sample, t) once per frame with no clock of its own, so it can be
tested with synthetic samples. It returns the JSON messages to send to the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from . import analysis as A
from .signals import Sample
from .store import ResultStore

READY_HAND_S = 0.5      # hand must be steady in view this long before the countdown
COUNTDOWN_S = 3.0
WAIT_TIMEOUT_S = 20.0
LOST_ABORT_S = 2.0      # hand missing this long mid-test cancels it
PHASE_EVERY_S = 0.1


@dataclass(frozen=True)
class TestSpec:
    key: str
    title: str
    prompt: str
    signal: str            # which Sample field is analysed
    duration: float = 10.0


TESTS = {t.key: t for t in (
    TestSpec("tapping", "Finger tapping",
             "Tap your thumb and index finger together, as big and as fast as you can.", "pinch"),
    TestSpec("open_close", "Open and close",
             "Open your hand wide, then close it into a fist, again and again, as fully and quickly as you can.",
             "fist"),
    TestSpec("tremor", "Hold still",
             "Hold your hand out in front of the camera, fingers relaxed and spread, and keep it as still as you can.",
             "palm"),
)}


class ScreeningSession:
    def __init__(self, store: ResultStore, source: str = "camera", now=datetime.now):
        self.store, self.source, self._now = store, source, now
        self.phase = "idle"
        self.test: TestSpec | None = None
        self.hand = ""
        self._phase_t = 0.0
        self._present_since: float | None = None
        self._last_seen = 0.0
        self._samples: list[Sample] = []
        self._last_phase_msg = float("-inf")

    def elapsed(self, t: float) -> float | None:
        """Seconds into the recording, or None when not recording (used by the simulator)."""
        return t - self._phase_t if self.phase == "running" else None

    def start(self, test: str, t: float) -> list[dict]:
        if test not in TESTS or self.phase != "idle":
            return []
        self.test, self.phase, self._phase_t = TESTS[test], "waiting", t
        self._present_since, self._samples = None, []
        return [self._phase_msg(t)]

    def cancel(self, t: float, reason: str = "cancelled") -> list[dict]:
        if self.phase == "idle":
            return []
        self.phase, self.test = "idle", None
        return [{"type": "aborted", "reason": reason}, self._phase_msg(t)]

    def update(self, sample: Sample | None, t: float, hand: str = "") -> list[dict]:
        msgs = [self._signal_msg(sample, t)]
        if sample is not None:
            self._last_seen = t
            self.hand = hand or self.hand
        present = sample is not None

        if self.phase == "waiting":
            if not present:
                self._present_since = None
            elif self._present_since is None:
                self._present_since = t
            if self._present_since is not None and t - self._present_since >= READY_HAND_S:
                self.phase, self._phase_t = "countdown", t
            elif t - self._phase_t > WAIT_TIMEOUT_S:
                return msgs + self.cancel(t, "no_hand")
        elif self.phase == "countdown":
            if not present:
                self.phase, self._phase_t, self._present_since = "waiting", t, None
            elif t - self._phase_t >= COUNTDOWN_S:
                self.phase, self._phase_t, self._samples = "running", t, []
        elif self.phase == "running":
            if present:
                self._samples.append(sample)
            if t - self._last_seen > LOST_ABORT_S:
                return msgs + self.cancel(t, "hand_lost")
            if t - self._phase_t >= self.test.duration:
                return msgs + self._finish(t)

        if self.phase != "idle" and t - self._last_phase_msg >= PHASE_EVERY_S:
            msgs.append(self._phase_msg(t))
        return msgs

    def _phase_msg(self, t: float) -> dict:
        self._last_phase_msg = t
        total = {"countdown": COUNTDOWN_S, "running": self.test.duration if self.test else 0}.get(self.phase, 0)
        remaining = max(0.0, total - (t - self._phase_t)) if total else 0.0
        return {"type": "phase", "phase": self.phase, "test": self.test.key if self.test else None,
                "remaining": round(remaining, 2), "total": total}

    @staticmethod
    def _signal_msg(sample: Sample | None, t: float) -> dict:
        if sample is None:
            return {"type": "signal", "t": t, "present": False}
        return {"type": "signal", "t": t, "present": True, "pinch": sample.pinch, "fist": sample.fist,
                "palm": list(sample.palm)}

    def _finish(self, t: float) -> list[dict]:
        spec, t0 = self.test, self._phase_t
        ts = [s.t for s in self._samples]
        if spec.signal == "palm":
            result = A.analyze_tremor(ts, [s.palm for s in self._samples], t0, spec.duration)
        else:
            result = A.analyze_repetitive(ts, [getattr(s, spec.signal) for s in self._samples], t0, spec.duration)
        series = result.pop("series")
        for key in ("peak_t", "peak_v", "amplitudes"):
            if key in result:
                series[key] = result.pop(key)
        now = self._now()
        record = {"id": f"{now.timestamp():.3f}", "at": now.astimezone().isoformat(timespec="seconds"),
                  "test": spec.key, "hand": self.hand, "source": self.source, "duration": spec.duration,
                  "metrics": result, "observations": A.observations(spec.key, result), "series": series}
        self.store.append(record)
        self.phase, self.test = "idle", None
        return [{"type": "result", "record": record}, self._phase_msg(t)]
