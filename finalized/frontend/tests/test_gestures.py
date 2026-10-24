import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import IMAGE_SIZE, POSES, make_hand, pose  # noqa: E402

from handui import features as F  # noqa: E402
from handui import gestures as G  # noqa: E402
from handui.classifier import GestureClassifier, append_samples, label_counts, remove_label, train  # noqa: E402
from handui.ui import HandUI  # noqa: E402

DT = 1 / 30


def play(engine, frames, t0=0.0):
    """frames: list of Hand | None at 30 fps. Returns (events, end time)."""
    events, t = [], t0
    for hand in frames:
        t += DT
        events += engine.update(hand, t)
    return events, t


def kinds(events):
    return [e.kind for e in events]


def moving(name, x0, x1, n, y=360, **kw):
    return [pose(name, pos=(x0 + (x1 - x0) * i / max(n - 1, 1), y), **kw) for i in range(n)]


class FeatureTests(unittest.TestCase):
    def test_extension_open_and_fist(self):
        self.assertTrue((F.finger_extension(pose("open").landmarks) > 0.9).all())
        fist = F.finger_extension(pose("fist").landmarks)
        self.assertTrue((fist[1:] < 0.1).all())
        self.assertLess(fist[0], 0.55)

    def test_pointing_is_not_a_grab(self):
        self.assertLess(F.grab_strength(F.finger_extension(pose("point").landmarks)), 0.5)
        self.assertGreater(F.grab_strength(F.finger_extension(pose("fist").landmarks)), 0.9)

    def test_features_are_scale_position_and_hand_invariant(self):
        a = F.gesture_features(pose("peace").landmarks, "Right")
        b = F.gesture_features(pose("peace", pos=(300, 500), scale=70).landmarks, "Right")
        c = F.gesture_features(pose("peace", handedness="Left").landmarks, "Left")
        np.testing.assert_allclose(a, b, atol=1e-3)
        np.testing.assert_allclose(a, c, atol=1e-3)


class GrabTests(unittest.TestCase):
    def test_fist_grab_drag_release(self):
        engine = G.GestureEngine()
        frames = [pose("open")] * 5 + moving("fist", 640, 840, 20) + [pose("open", pos=(840, 360))] * 5
        events, _ = play(engine, frames)
        k = kinds(events)
        self.assertEqual(k.count(G.GRAB_START), 1)
        self.assertEqual(k.count(G.GRAB_END), 1)
        moves = [e.data["pos"] for e in events if e.kind == G.GRAB_MOVE]
        self.assertGreater(len(moves), 10)
        self.assertGreater(moves[-1][0], moves[0][0] + 0.1)
        self.assertNotIn(G.SWIPE, k)

    def test_pinch_grabs(self):
        engine = G.GestureEngine()
        events, _ = play(engine, [make_hand((0, 0.35, 0, 0, 0), pinch=True)] * 5)
        self.assertIn(G.GRAB_START, kinds(events))

    def test_pointing_does_not_grab(self):
        events, _ = play(G.GestureEngine(), [pose("point")] * 20)
        self.assertNotIn(G.GRAB_START, kinds(events))

    def test_losing_hand_cancels_grab(self):
        engine = G.GestureEngine()
        events, t = play(engine, [pose("fist")] * 10)
        self.assertTrue(engine.state.grabbing)
        events, _ = play(engine, [None] * 15, t)
        self.assertEqual(kinds(events), [G.GRAB_END, G.HAND_LOST])
        self.assertTrue(events[0].data["cancelled"])

    def test_brief_dropout_is_tolerated(self):
        engine = G.GestureEngine()
        _, t = play(engine, [pose("fist")] * 10)
        events, _ = play(engine, [None] * 4 + [pose("fist")] * 3, t)
        self.assertNotIn(G.GRAB_END, kinds(events))
        self.assertTrue(engine.state.grabbing)


class SwipeTests(unittest.TestCase):
    def test_fast_swipe_right_and_left(self):
        for x0, x1, direction in ((300, 900, "right"), (900, 300, "left")):
            events, _ = play(G.GestureEngine(), [pose("open", pos=(x0, 360))] * 3 + moving("open", x0, x1, 8))
            swipes = [e for e in events if e.kind == G.SWIPE]
            self.assertEqual(len(swipes), 1, direction)
            self.assertEqual(swipes[0].data["direction"], direction)

    def test_vertical_swipe(self):
        frames = [pose("open", pos=(640, y)) for y in np.linspace(150, 600, 8)]
        swipes = [e for e in play(G.GestureEngine(), frames)[0] if e.kind == G.SWIPE]
        self.assertEqual([s.data["direction"] for s in swipes], ["down"])

    def test_slow_move_is_not_a_swipe(self):
        events, _ = play(G.GestureEngine(), moving("open", 300, 900, 90))
        self.assertNotIn(G.SWIPE, kinds(events))

    def test_fist_motion_is_not_a_swipe(self):
        events, _ = play(G.GestureEngine(), moving("fist", 300, 900, 8))
        self.assertNotIn(G.SWIPE, kinds(events))


class MenuTests(unittest.TestCase):
    def test_hold_open_palm_toggles_menu_once(self):
        engine = G.GestureEngine()
        events, t = play(engine, [pose("open")] * 60)
        self.assertEqual(kinds(events).count(G.MENU_OPEN), 1)
        self.assertTrue(engine.state.menu_open)

        half_curl = make_hand((0, 0.4, 0.4, 0.4, 0.4))
        events, _ = play(engine, [half_curl] * 5 + [pose("open")] * 30, t)
        self.assertEqual(kinds(events), [G.MENU_CLOSE])

    def test_dwell_progress_reported(self):
        engine = G.GestureEngine()
        play(engine, [pose("open")] * 10)
        self.assertGreater(engine.state.menu_progress, 0.3)

    def test_swipe_does_not_open_menu_afterwards(self):
        engine = G.GestureEngine()
        frames = moving("open", 300, 900, 8) + [pose("open", pos=(900, 360))] * 36
        events, _ = play(engine, frames)
        self.assertIn(G.SWIPE, kinds(events))
        self.assertNotIn(G.MENU_OPEN, kinds(events))


class FakeClassifier:
    def __init__(self, labels):
        self.labels = list(labels)

    def predict(self, features):
        return self.labels.pop(0) if len(self.labels) > 1 else self.labels[0], 0.95


class CustomGestureTests(unittest.TestCase):
    def test_custom_event_fires_once_when_stable(self):
        engine = G.GestureEngine(classifier=FakeClassifier(["thumbs_up"]))
        events, _ = play(engine, [pose("open")] * 30)
        custom = [e for e in events if e.kind == G.CUSTOM]
        self.assertEqual(len(custom), 1)
        self.assertEqual(custom[0].data["label"], "thumbs_up")

    def test_none_label_is_ignored(self):
        engine = G.GestureEngine(classifier=FakeClassifier(["none"]))
        events, _ = play(engine, [pose("open")] * 20)
        self.assertNotIn(G.CUSTOM, kinds(events))


class ClassifierTests(unittest.TestCase):
    @staticmethod
    def samples(name, n, rng):
        out = []
        for _ in range(n):
            hand = make_hand(POSES[name], pos=rng.uniform((200, 150), (1080, 570)), scale=rng.uniform(70, 150),
                             rotation=rng.uniform(-0.25, 0.25), handedness=rng.choice(["Left", "Right"]),
                             noise=0.03, rng=rng)
            out.append((hand.landmarks, hand.handedness))
        return out

    def test_train_and_predict(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            data, model_path = Path(tmp) / "samples.csv", Path(tmp) / "model.joblib"
            for name in POSES:
                append_samples(name, self.samples(name, 80, rng), data)
            self.assertEqual(label_counts(data), {name: 80 for name in POSES})

            report = train(data, model_path, augment=2)
            self.assertGreaterEqual(report["holdout_accuracy"], 0.9)

            clf = GestureClassifier.load(model_path)
            correct = total = 0
            for name in POSES:
                for lm, handed in self.samples(name, 20, rng):
                    correct += clf.predict(F.gesture_features(lm, handed))[0] == name
                    total += 1
            self.assertGreaterEqual(correct / total, 0.9)

            self.assertEqual(remove_label("fist", data), 80)
            self.assertNotIn("fist", label_counts(data))
            self.assertEqual(sum(label_counts(data).values()), 240)

    def test_train_needs_two_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "samples.csv"
            append_samples("only", self.samples("open", 10, np.random.default_rng(0)), data)
            with self.assertRaises(ValueError):
                train(data, Path(tmp) / "m.joblib")


class UITests(unittest.TestCase):
    def setUp(self):
        self.engine = G.GestureEngine()
        self.ui = HandUI(*IMAGE_SIZE, engine=self.engine, bindings={"swipe_left": "next_page"})
        self.t = 0.0

    def step(self, hand):
        self.t += DT
        self.ui.handle(self.engine.update(hand, self.t), self.engine.state)
        return self.ui.render(np.zeros((720, 1280, 3), np.uint8), hand, self.engine.state, self.t)

    def camera_pos_for(self, x, y):
        """Camera-space palm position that maps to UI pixel (x, y)."""
        m = self.engine.cfg.cursor_margin
        return (m + x / 1280 * (1 - 2 * m)) * 1280, (m + y / 720 * (1 - 2 * m)) * 720

    def test_drag_card(self):
        card = next(c for c in self.ui.cards if c.page == 0)
        start = self.camera_pos_for(card.x + 40, card.y + 40)
        for _ in range(15):
            self.step(pose("point", pos=start))  # settle the cursor without grabbing
        x0 = card.x
        for i in range(20):
            self.step(pose("fist", pos=(start[0] + i * 8, start[1])))
        self.assertIs(self.ui.drag[0], card)
        for _ in range(5):
            self.step(pose("open", pos=(start[0] + 152, start[1])))
        self.assertIsNone(self.ui.drag)
        self.assertGreater(card.x, x0 + 100)

    def test_swipe_changes_page(self):
        for hand in moving("open", 900, 300, 8):
            self.step(hand)
        self.assertEqual(self.ui.page, 1)

    def test_menu_select_by_grab(self):
        for _ in range(30):
            frame = self.step(pose("open"))
        self.assertIsNotNone(self.ui.menu_center)
        self.assertEqual(frame.shape, (720, 1280, 3))
        # Move up (item 0, "New card") and make a fist.
        cx, cy = self.ui.menu_center
        target = self.camera_pos_for(cx, cy - 100)
        for _ in range(15):
            self.step(make_hand((0, 0.4, 0.4, 0.4, 0.4), pos=target))
        self.assertEqual(self.ui.menu_hover, 0)
        n = len(self.ui.cards)
        for _ in range(4):
            self.step(pose("fist", pos=target))
        self.assertIsNone(self.ui.menu_center)
        self.assertFalse(self.engine.state.menu_open)
        self.assertEqual(len(self.ui.cards), n + 1)
        self.assertIsNone(self.ui.drag)  # the selecting grab must not also drag a card


if __name__ == "__main__":
    unittest.main()
