import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import make_hand  # noqa: E402

from handui.screening import analysis as A  # noqa: E402
from handui.screening.session import COUNTDOWN_S, LOST_ABORT_S, TESTS, WAIT_TIMEOUT_S, ScreeningSession  # noqa: E402
from handui.screening.signals import Sample, sample_from_hand  # noqa: E402
from handui.screening.simulate import PROFILES, Simulator  # noqa: E402
from handui.screening.store import ResultStore  # noqa: E402

DT = 1 / 30
T = np.arange(0, 10, DT)


def kinds(msgs):
    return [m["type"] for m in msgs]


class AnalysisTests(unittest.TestCase):
    def test_steady_tapping(self):
        m = A.analyze_repetitive(T, 0.6 + 0.3 * np.sin(2 * np.pi * 4 * T), 0, 10)
        self.assertEqual(m["count"], 40)
        self.assertAlmostEqual(m["rate_hz"], 4.0)
        self.assertAlmostEqual(m["amplitude_mean"], 0.6, delta=0.05)
        self.assertLess(abs(m["amplitude_decrement_pct"]), 5)
        self.assertLess(m["rhythm_cv_pct"], 3)
        self.assertEqual(m["hesitations"], 0)
        self.assertTrue(m["quality"]["ok"])

    def test_shrinking_amplitude_gives_negative_decrement(self):
        x = 0.6 + 0.3 * np.sin(2 * np.pi * 4 * T) * np.linspace(1, 0.5, len(T))
        m = A.analyze_repetitive(T, x, 0, 10)
        self.assertLess(m["amplitude_decrement_pct"], -25)
        self.assertTrue(any("smaller" in n["text"] for n in A.observations("tapping", m)))

    def test_a_pause_is_counted(self):
        phase = np.cumsum(np.where((T > 5) & (T < 5.9), 0.0, 2 * np.pi * 4 * DT))
        m = A.analyze_repetitive(T, 0.6 + 0.3 * np.sin(phase), 0, 10)
        self.assertEqual(m["hesitations"], 1)

    def test_flat_signal_counts_nothing(self):
        m = A.analyze_repetitive(T, 0.5 + 0.001 * np.random.default_rng(0).normal(size=len(T)), 0, 10)
        self.assertEqual(m["count"], 0)
        self.assertEqual(A.observations("tapping", m)[0]["text"], "No repeated movement detected.")

    def test_tremor_found_at_its_frequency(self):
        rng = np.random.default_rng(1)
        xy = np.column_stack([0.01 * np.sin(2 * np.pi * 5 * T), 0.002 * rng.normal(size=len(T))])
        m = A.analyze_tremor(T, xy, 0, 10)
        self.assertAlmostEqual(m["dominant_hz"], 5.0, delta=0.5)
        self.assertTrue(m["tremor_like"])

    def test_sensor_noise_is_not_tremor(self):
        xy = 0.003 * np.random.default_rng(2).normal(size=(len(T), 2))
        self.assertFalse(A.analyze_tremor(T, xy, 0, 10)["tremor_like"])

    def test_slow_drift_is_not_tremor(self):
        xy = np.column_stack([0.2 * np.sin(2 * np.pi * 0.2 * T), 0.1 * T / 10])
        self.assertFalse(A.analyze_tremor(T, xy, 0, 10)["tremor_like"])

    def test_missing_data_lowers_quality(self):
        keep = (T < 3) | (T > 8)
        m = A.analyze_repetitive(T[keep], (0.6 + 0.3 * np.sin(2 * np.pi * 4 * T))[keep], 0, 10)
        self.assertFalse(m["quality"]["ok"])
        self.assertTrue(any("incomplete" in n["text"] for n in A.observations("tapping", m)))

    def test_results_are_json_serializable(self):
        json.dumps(A.analyze_repetitive(T, np.sin(T * 20), 0, 10))
        json.dumps(A.analyze_tremor(T, np.column_stack([np.sin(T * 30)] * 2), 0, 10))


class SignalTests(unittest.TestCase):
    def test_open_hand_vs_fist(self):
        open_s = sample_from_hand(make_hand(), 0.0)
        fist_s = sample_from_hand(make_hand(curls=(1, 1, 1, 1, 1)), 0.0)
        self.assertLess(open_s.fist, 0.1)
        self.assertGreater(fist_s.fist, 0.9)

    def test_pinch_is_small_when_fingers_touch(self):
        self.assertLess(sample_from_hand(make_hand(pinch=True), 0.0).pinch, 0.25)
        self.assertGreater(sample_from_hand(make_hand(), 0.0).pinch, 0.5)

    def test_palm_position_is_scale_invariant_after_removing_offset(self):
        a = sample_from_hand(make_hand(scale=100, pos=(400, 300)), 0.0)
        b = sample_from_hand(make_hand(scale=200, pos=(800, 600)), 0.0)
        self.assertAlmostEqual(a.palm[0], b.palm[0], places=3)
        self.assertAlmostEqual(a.palm[1], b.palm[1], places=3)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.store = ResultStore(Path(self.dir.name) / "r.jsonl")
        self.session = ScreeningSession(self.store, "simulated", now=lambda: datetime(2026, 1, 2, 3, 4, 5))

    def drive(self, seconds, t0, sample_fn):
        out, t = [], t0
        for _ in range(int(seconds / DT)):
            t += DT
            out += self.session.update(sample_fn(t), t, "Right")
        return out, t

    def test_full_run_for_every_test_and_profile(self):
        for profile in PROFILES:
            for test in TESTS:
                with self.subTest(profile=profile, test=test):
                    session = ScreeningSession(self.store, "simulated")
                    sim, t, out = Simulator(profile), 0.0, session.start(test, 0.0)
                    while session.phase != "idle":
                        t += DT
                        out += session.update(sim.sample(t, session.elapsed(t)), t, "Right")
                        self.assertLess(t, 30)
                    record = next(m for m in out if m["type"] == "result")["record"]
                    self.assertEqual((record["test"], record["hand"], record["source"]), (test, "Right", "simulated"))
                    json.dumps(record)

    def test_profiles_are_told_apart(self):
        def run(profile, test):
            session, sim, t = ScreeningSession(self.store, "simulated"), Simulator(profile), 0.0
            out = session.start(test, t)
            while session.phase != "idle":
                t += DT
                out += session.update(sim.sample(t, session.elapsed(t)), t, "Right")
            return next(m for m in out if m["type"] == "result")["record"]["metrics"]

        self.assertLess(run("slowing", "tapping")["amplitude_decrement_pct"], -20)
        self.assertGreater(run("healthy", "tapping")["amplitude_decrement_pct"], -10)
        self.assertTrue(run("tremor", "tremor")["tremor_like"])
        self.assertFalse(run("healthy", "tremor")["tremor_like"])

    def test_phases_progress_and_result_is_stored(self):
        sim = Simulator("healthy")
        self.session.start("tapping", 0.0)
        phases = set()
        t = 0.0
        while self.session.phase != "idle":
            t += DT
            for m in self.session.update(sim.sample(t, self.session.elapsed(t)), t, "Right"):
                if m["type"] == "phase":
                    phases.add(m["phase"])
        self.assertEqual(phases, {"waiting", "countdown", "running", "idle"})
        (saved,) = self.store.load()
        self.assertEqual(saved["at"][:19], "2026-01-02T03:04:05")
        self.assertEqual(self.store.history("simulated")[0].get("series"), None)

    def test_waits_for_a_hand_then_times_out(self):
        self.session.start("tapping", 0.0)
        out, t = self.drive(WAIT_TIMEOUT_S + 1, 0.0, lambda t: None)
        self.assertEqual(self.session.phase, "idle")
        self.assertIn({"type": "aborted", "reason": "no_hand"}, out)
        self.assertEqual(self.store.load(), [])

    def test_losing_the_hand_during_countdown_restarts_the_wait(self):
        sim = Simulator("healthy")
        self.session.start("tapping", 0.0)
        _, t = self.drive(COUNTDOWN_S / 2 + 1, 0.0, lambda t: sim.sample(t, None))
        self.assertEqual(self.session.phase, "countdown")
        _, t = self.drive(0.2, t, lambda t: None)
        self.assertEqual(self.session.phase, "waiting")

    def test_losing_the_hand_mid_test_aborts(self):
        sim = Simulator("healthy")
        self.session.start("tapping", 0.0)
        _, t = self.drive(COUNTDOWN_S + 2, 0.0, lambda t: sim.sample(t, self.session.elapsed(t)))
        self.assertEqual(self.session.phase, "running")
        out, _ = self.drive(LOST_ABORT_S + 0.5, t, lambda t: None)
        self.assertIn({"type": "aborted", "reason": "hand_lost"}, out)
        self.assertEqual(self.session.phase, "idle")
        self.assertEqual(self.store.load(), [])

    def test_cancel_and_ignored_starts(self):
        self.assertEqual(self.session.start("nonsense", 0.0), [])
        self.session.start("tapping", 0.0)
        self.assertEqual(self.session.start("tremor", 0.1), [])  # busy
        self.assertEqual(self.session.test.key, "tapping")
        self.assertEqual(kinds(self.session.cancel(0.2)), ["aborted", "phase"])
        self.assertEqual(self.session.phase, "idle")
        self.assertEqual(self.session.cancel(0.3), [])

    def test_signal_message_every_frame(self):
        out = self.session.update(Sample(0.0, 0.5, 0.2, (1.0, 2.0)), 0.0)
        self.assertEqual(out[0], {"type": "signal", "t": 0.0, "present": True, "pinch": 0.5, "fist": 0.2, "palm": [1.0, 2.0]})
        self.assertEqual(self.session.update(None, 0.03)[0]["present"], False)


class StoreTests(unittest.TestCase):
    def test_roundtrip_filter_and_corrupt_line(self):
        with tempfile.TemporaryDirectory() as d:
            store = ResultStore(Path(d) / "nested" / "r.jsonl")
            self.assertEqual(store.load(), [])
            store.append({"id": "1", "source": "camera", "series": {"t": [0]}})
            store.append({"id": "2", "source": "simulated"})
            with store.path.open("a") as f:
                f.write("{not json\n")
            store.append({"id": "3", "source": "camera"})
            self.assertEqual([r["id"] for r in store.load()], ["1", "2", "3"])
            self.assertEqual([r["id"] for r in store.load("camera")], ["1", "3"])
            self.assertNotIn("series", store.history("camera")[0])


if __name__ == "__main__":
    unittest.main()
