import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handui.brain import eeg as E
from handui.brain.runner import DEFAULT_APIS, VIEWER, BrainSession, hand_depth_message
from handui.gestures import GestureEngine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import make_hand  # noqa: E402


class MontageTests(unittest.TestCase):
    def test_positions_are_unit_vectors_on_the_correct_side(self):
        for code in E.CHANNELS:
            x, y, z = E.electrode_position(code)
            self.assertAlmostEqual(math.hypot(x, y, z), 1.0, places=6)
            if code[-1] == "z":
                self.assertAlmostEqual(x, 0.0, places=6)
            elif int(code[-1]) % 2:          # odd numbers are on the left
                self.assertLess(x, 0, code)
            else:
                self.assertGreater(x, 0, code)
        self.assertGreater(E.electrode_position("Fp1")[1], 0.9)   # front
        self.assertLess(E.electrode_position("O2")[1], -0.9)      # back
        self.assertEqual(E.electrode_position("Cz"), (0.0, 0.0, 1.0))

    def test_left_right_pairs_mirror(self):
        for left, right in (("Fp1", "Fp2"), ("F7", "F8"), ("C3", "C4"), ("T7", "T8"), ("P7", "P8"), ("O1", "O2")):
            lx, ly, lz = E.electrode_position(left)
            rx, ry, rz = E.electrode_position(right)
            np.testing.assert_allclose([lx, ly, lz], [-rx, ry, rz], atol=1e-9)

    def test_montage_covers_the_eegenius_channels(self):
        self.assertEqual(set(E.montage()), set(E.EEGENIUS_ORDER))
        self.assertEqual(E.montage()["C3"]["side"], "left")

    def test_channel_names(self):
        cases = {"EEG Fp1-REF": "Fp1", "FP1": "Fp1", "fz": "Fz", "T3": "T7", "T6": "P8", "O2-A1": "O2",
                 "time": None, "EEG": None, "ECG": None, "": None}
        for raw, want in cases.items():
            self.assertEqual(E.canonical(raw), want, raw)


class ParseTests(unittest.TestCase):
    def test_eegenius_request(self):
        rec = E.parse_text(json.dumps({"data": {c: i for i, c in enumerate(E.EEGENIUS_ORDER)}}))
        self.assertEqual(rec.channels, list(E.CHANNELS))
        self.assertEqual(rec.frames.shape, (1, 19))
        self.assertEqual(rec.frames[0, rec.channels.index("Pz")], 18)

    def test_csv_rows_are_frames_and_extra_columns_ignored(self):
        rec = E.parse_text("time,EEG Cz-REF,FP1,marker\n0,1,2,x\n0.1,3,4,y\n", "a.csv", rate=5)
        self.assertEqual(rec.channels, ["Fp1", "Cz"])
        np.testing.assert_array_equal(rec.frames, [[2, 1], [4, 3]])
        self.assertEqual((rec.rate, rec.source, rec.name), (5, "file", "a.csv"))

    def test_other_json_layouts(self):
        listed = E.parse_text(json.dumps([{"Fz": 1, "Cz": 2}, {"Fz": 3, "Cz": 4}]))
        np.testing.assert_array_equal(listed.frames, [[1, 2], [3, 4]])
        framed = E.parse_text(json.dumps({"channels": ["O1", "O2"], "frames": [[1, 2]], "rate": 2}))
        self.assertEqual((framed.channels, framed.rate), (["O1", "O2"], 2))

    def test_bad_input_raises_helpful_errors(self):
        with self.assertRaisesRegex(ValueError, "no 10-20 channel names"):
            E.parse_text("a,b\n1,2\n")
        with self.assertRaisesRegex(ValueError, "row 2, Cz"):
            E.parse_text("Cz\n1\nnope\n")
        with self.assertRaisesRegex(ValueError, "finite"):
            E.parse_text('{"Cz": NaN}')
        with self.assertRaisesRegex(ValueError, "no data rows"):
            E.parse_text("Cz,Pz\n")

    def test_raw_samples_are_summarised_to_rms(self):
        fs, t = 100, np.arange(200) / 100
        text = "Cz,Pz\n" + "\n".join(f"{10 * math.sin(2 * math.pi * 10 * s)},{2.0}" for s in t)
        rec = E.parse_text(text, fs=fs)
        self.assertEqual(rec.frames.shape, (4, 2))     # 2 s in 0.5 s windows
        self.assertEqual(rec.rate, 2.0)
        np.testing.assert_allclose(rec.frames[:, 0], 10 / math.sqrt(2), rtol=0.02)
        np.testing.assert_allclose(rec.frames[:, 1], 0, atol=1e-9)   # constant offset removed

    def test_long_recordings_are_thinned(self):
        text = "Cz\n" + "\n".join(str(i) for i in range(E.MAX_FRAMES * 2 + 10))
        rec = E.parse_text(text, rate=10)
        self.assertLessEqual(len(rec.frames), E.MAX_FRAMES)
        self.assertEqual(rec.rate, 10 / 3)
        self.assertIn("1 frame in 3", rec.note)

    def test_load_reads_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            path.write_text(json.dumps({"data": {"O1": 4, "O2": 5}}))
            rec = E.load(path)
        self.assertEqual((rec.name, rec.channels), ("snap.json", ["O1", "O2"]))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.engine = GestureEngine()
        self.session = BrainSession(E.parse_text("Fz,Cz\n1,2\n3,4\n", "preloaded.csv"), self.engine, camera=True)

    def test_hello_sends_montage_then_any_preloaded_data(self):
        init, data = self.session.handle({"cmd": "hello"})
        self.assertEqual((init["type"], init["camera"], data["type"]), ("init", True, "eeg"))
        self.assertEqual(len(init["montage"]), 19)
        self.assertEqual(init["eegenius_order"], list(E.EEGENIUS_ORDER))
        self.assertEqual((data["name"], data["frames"]), ("preloaded.csv", [[1, 2], [3, 4]]))
        json.dumps([init, data])   # must be serialisable for the bridge

    def test_without_preloaded_data_hello_sends_only_the_montage(self):
        msgs = BrainSession(None, camera=False).handle({"cmd": "hello"})
        self.assertEqual([m["type"] for m in msgs], ["init"])

    def test_uploaded_and_pushed_data_replace_the_recording(self):
        msg, = self.session.handle({"cmd": "eeg_text", "name": "mine.csv", "text": "Fz,Pz\n1,2\n"})
        self.assertEqual((msg["source"], msg["name"], msg["channels"]), ("upload", "mine.csv", ["Fz", "Pz"]))
        msg, = self.session.handle({"cmd": "eeg", "name": "EEGenius", "data": {"Cz": 3.5}, "units": "uV"})
        self.assertEqual((msg["source"], msg["units"], msg["frames"]), ("external", "uV", [[3.5]]))
        self.assertEqual(self.session.handle({"cmd": "hello"})[1]["name"], "EEGenius")

    def test_bad_data_reports_an_error_and_keeps_the_old_recording(self):
        msg, = self.session.handle({"cmd": "eeg", "data": {"Cz": "high"}})
        self.assertEqual(msg["type"], "error")
        self.assertEqual(self.session.recording.name, "preloaded.csv")
        self.assertEqual(self.session.handle({"cmd": "eeg_text"})[0]["type"], "error")
        self.assertEqual(self.session.handle({"cmd": "eeg", "data": [1, 2]})[0]["type"], "error")

    def test_menu_commands_reach_the_engine(self):
        self.session.handle({"cmd": "open_menu"})
        self.assertTrue(self.engine.state.menu_open)
        self.session.handle({"cmd": "close_menu"})
        self.assertFalse(self.engine.state.menu_open)
        self.assertEqual(self.session.handle({"cmd": "unknown"}), [])

    def test_init_carries_the_model_server_addresses(self):
        init = BrainSession(None, apis={"adhd": "http://127.0.0.1:9001"}).init_message()
        self.assertEqual(init["apis"]["adhd"], "http://127.0.0.1:9001")
        self.assertEqual(init["apis"]["parkinsons"], DEFAULT_APIS["parkinsons"])

    def test_viewer_page_is_self_contained_and_uses_the_model_servers(self):
        html = VIEWER.read_text()
        self.assertNotIn("<script src", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("localhost:8001", html)   # addresses come from the init message, not the page
        for endpoint in ("/heldout-samples", "/analyze"):
            self.assertIn(endpoint, html)
        self.assertIn('send({ cmd: "hello" })', html)


class HandDepthTests(unittest.TestCase):
    def test_palm_looks_bigger_when_closer(self):
        far = hand_depth_message(make_hand(scale=60))
        near = hand_depth_message(make_hand(scale=150))
        self.assertEqual(far["type"], "hand_depth")
        self.assertGreater(near["size"], far["size"] * 2)
        self.assertAlmostEqual(near["size"] / far["size"], 150 / 60, delta=0.05)
        json.dumps(near)


if __name__ == "__main__":
    unittest.main()
