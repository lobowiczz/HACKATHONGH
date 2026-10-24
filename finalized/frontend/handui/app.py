"""Main loop: camera -> hand tracker -> gesture engine -> UI."""
from __future__ import annotations

import time

import cv2

from .bridge import Bridge
from .classifier import GestureClassifier
from .gestures import GestureEngine
from .tracker import HandTracker, open_camera
from .ui import HandUI

WINDOW = "Hand UI"


def run(camera: int = 0, width: int = 1280, height: int = 720, bridge_port: int | None = None) -> None:
    classifier = GestureClassifier.load()
    engine = GestureEngine(classifier=classifier)
    bridge = Bridge(port=bridge_port) if bridge_port is not None else None
    ui = HandUI(width, height, engine=engine, classifier_labels=classifier.labels if classifier else None)
    if classifier:
        ui.toast(f"Custom gestures: {', '.join(classifier.labels)}")
    if bridge:
        print(f"Streaming gestures on ws://{bridge.host}:{bridge.port}")

    with HandTracker() as tracker:
        cap = open_camera(camera, width, height)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, width, height)
        try:
            while not ui.quit_requested:
                ok, frame = cap.read()
                if not ok:
                    print("Camera returned no frame.")
                    break
                frame = cv2.flip(frame, 1)  # mirror view; MediaPipe handedness expects this
                hands = tracker.process(frame)
                hand = hands[0] if hands else None
                t = time.monotonic()

                events = engine.update(hand, t)
                ui.handle(events, engine.state)
                if bridge:
                    bridge.publish(events, engine.state, t)
                cv2.imshow(WINDOW, ui.render(frame, hand, engine.state, t))
                ui.handle_key(cv2.waitKey(1) & 0xFF)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if bridge:
                bridge.close()
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
