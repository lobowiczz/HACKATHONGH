"""Main loop for the Brain Explorer: camera -> gesture engine -> bridge -> viewer.html.

The browser page does all the drawing and holds the input the user is working
on. Python tracks the hand, serves the page, parses EEG files and answers the
page's commands. Predictions come from the separate ADHD and Parkinson's model
servers, whose addresses the page receives in the init message.
"""
from __future__ import annotations

import time
import webbrowser
from pathlib import Path

import cv2

from .. import drawing as D
from .. import features as F
from ..bridge import DEFAULT_PORT, Bridge
from ..classifier import GestureClassifier
from ..gestures import GestureEngine
from ..tracker import HandTracker, open_camera
from .eeg import CHANNELS, EEGENIUS_ORDER, Recording, from_mapping, montage, parse_text

WINDOW = "Brain Explorer camera"
VIEWER = Path(__file__).with_name("viewer.html")
DEFAULT_APIS = {"adhd": "http://127.0.0.1:8011", "parkinsons": "http://127.0.0.1:8012"}


class BrainSession:
    """Answers viewer commands. Every reply is a message to broadcast."""

    def __init__(self, recording: Recording | None = None, engine: GestureEngine | None = None, camera: bool = True,
                 apis: dict[str, str] | None = None):
        self.recording = recording
        self.engine = engine
        self.camera = camera
        self.apis = {**DEFAULT_APIS, **(apis or {})}

    def init_message(self) -> dict:
        return {"type": "init", "camera": self.camera, "montage": montage(), "apis": self.apis,
                "channels": list(CHANNELS), "eegenius_order": list(EEGENIUS_ORDER)}

    def handle(self, cmd: dict) -> list[dict]:
        name = cmd.get("cmd")
        if name == "hello":
            return [self.init_message()] + ([self.recording.to_message()] if self.recording else [])
        if name == "close_menu" and self.engine:
            self.engine.close_menu()
        elif name == "open_menu" and self.engine:
            self.engine.state.menu_open = True
        elif name in ("eeg", "eeg_text"):
            try:
                self.recording = self._parse(cmd)
            except ValueError as e:
                return [{"type": "error", "message": str(e)}]
            return [self.recording.to_message()]
        return []

    @staticmethod
    def _parse(cmd: dict) -> Recording:
        label = str(cmd.get("name") or "data")[:80]
        if cmd["cmd"] == "eeg_text":  # a file the user dropped on the page
            if not isinstance(cmd.get("text"), str):
                raise ValueError("eeg_text needs a 'text' field")
            return parse_text(cmd["text"], label, "upload")
        if not isinstance(cmd.get("data"), dict):  # pushed by another program, e.g. EEGenius
            raise ValueError("eeg needs a 'data' object mapping channel names to values")
        return from_mapping(cmd["data"], label, "external", str(cmd.get("units") or "")[:12])


def hand_depth_message(hand) -> dict:
    """Apparent palm size as a fraction of frame width: bigger means the hand is closer to the camera."""
    return {"type": "hand_depth", "size": round(F.palm_size(hand.landmarks) / hand.image_size[0], 4)}


def _draw_preview(frame, hand) -> None:
    h, w = frame.shape[:2]
    if hand is not None:
        D.skeleton(frame, hand.landmarks[:, :2])
    D.panel(frame, 16, 16, 470, 46, 14, (20, 20, 20), 0.6)
    D.text(frame, "Brain Explorer: use the browser window   (q to quit)", 32, 46, 0.6)
    cv2.imshow(WINDOW, cv2.resize(frame, (w // 2, h // 2)))


def run_brain(recording: Recording | None = None, camera: int = 0, port: int = DEFAULT_PORT,
              open_browser: bool = True, preview: bool = True, use_camera: bool = True,
              apis: dict[str, str] | None = None) -> None:
    engine = GestureEngine(classifier=GestureClassifier.load()) if use_camera else None
    session = BrainSession(recording, engine, camera=use_camera, apis=apis)
    try:
        bridge = Bridge(port=port, routes={"/": VIEWER.read_bytes()}, local_only=True)
    except OSError as e:
        raise SystemExit(f"Port {port} is busy ({e.strerror}). Close the other Hand UI window or use --port.")
    url = f"http://127.0.0.1:{bridge.port}/"
    loaded = f"{recording.name}, {len(recording.frames)} row(s)" if recording else "enter data in the page"
    print(f"Brain Explorer: {url}   ({loaded})")
    if open_browser:
        webbrowser.open(url)

    tracker = cap = None
    try:
        if use_camera:
            tracker = HandTracker()
            cap = open_camera(camera)
        while True:
            t = time.monotonic()
            for cmd in bridge.poll_commands():
                for msg in session.handle(cmd):
                    bridge.send(msg)
            if not use_camera:
                time.sleep(1 / 30)
                continue

            ok, frame = cap.read()
            if not ok:
                print("Camera returned no frame.")
                break
            frame = cv2.flip(frame, 1)  # mirror view; MediaPipe handedness expects this
            hands = tracker.process(frame)
            hand = hands[0] if hands else None
            bridge.publish(engine.update(hand, t), engine.state, t)
            if hand is not None:
                bridge.send(hand_depth_message(hand))
            if preview:
                _draw_preview(frame, hand)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        if tracker:
            tracker.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
