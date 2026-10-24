"""Main loop for screening: camera (or simulator) -> session -> dashboard over the bridge."""
from __future__ import annotations

import time
import webbrowser
from pathlib import Path

import cv2

from .. import drawing as D
from ..bridge import DEFAULT_PORT, Bridge
from ..tracker import HandTracker, open_camera
from .session import TESTS, ScreeningSession
from .signals import sample_from_hand
from .simulate import PROFILES, Simulator
from .store import ResultStore

WINDOW = "MotorLens camera"
DASHBOARD = Path(__file__).with_name("dashboard.html")


def _init_message(session: ScreeningSession, simulate: str | None) -> dict:
    return {"type": "init", "source": session.source, "profile": simulate,
            "profiles": PROFILES if simulate else None,
            "tests": {k: {"title": t.title, "prompt": t.prompt, "duration": t.duration} for k, t in TESTS.items()},
            "history": session.store.history(session.source)}


def _handle_commands(bridge: Bridge, session: ScreeningSession, simulate: str | None, t: float) -> list[dict]:
    out = []
    for cmd in bridge.poll_commands():
        name = cmd.get("cmd")
        if name == "hello":
            out.append(_init_message(session, simulate))
        elif name == "start" and isinstance(cmd.get("test"), str):
            out += session.start(cmd["test"], t)
        elif name == "cancel":
            out += session.cancel(t)
    return out


def _draw_preview(frame, hand, session: ScreeningSession, msg: dict | None) -> None:
    h, w = frame.shape[:2]
    if hand is not None:
        D.skeleton(frame, hand.landmarks[:, :2])
    D.panel(frame, 16, 16, 420, 74, 14, (20, 20, 20), 0.6)
    title = session.test.title if session.test else "Choose a test in the dashboard"
    D.text(frame, title, 32, 48, 0.8, thickness=2)
    status = session.phase if session.phase == "idle" else session.phase + (
        f"  {msg['remaining']:.0f}s" if msg and msg["remaining"] else "")
    D.text(frame, f"{status}   (q to quit)", 32, 76, 0.55, color=(190, 190, 190))
    cv2.imshow(WINDOW, cv2.resize(frame, (w // 2, h // 2)))


def run_screening(camera: int = 0, simulate: str | None = None, port: int = DEFAULT_PORT,
                  open_browser: bool = True, preview: bool = True) -> None:
    session = ScreeningSession(ResultStore(), source="simulated" if simulate else "camera")
    sim = Simulator(simulate) if simulate else None
    bridge = Bridge(port=port, routes={"/": DASHBOARD.read_bytes()}, local_only=True)
    url = f"http://127.0.0.1:{bridge.port}/"
    print(f"MotorLens dashboard: {url}" + (f"  (simulated: {simulate})" if simulate else ""))
    if open_browser:
        webbrowser.open(url)

    tracker = cap = None
    try:
        if not sim:
            tracker = HandTracker()
            cap = open_camera(camera)
        next_frame = time.monotonic()
        while True:
            t = time.monotonic()
            hand, frame = None, None
            if sim:
                sample = sim.sample(t, session.elapsed(t))
                handedness = "Right"
                next_frame += 1 / 30
                time.sleep(max(0.0, next_frame - time.monotonic()))
            else:
                ok, frame = cap.read()
                if not ok:
                    print("Camera returned no frame.")
                    break
                frame = cv2.flip(frame, 1)  # mirror view; MediaPipe handedness expects this
                hands = tracker.process(frame)
                hand = hands[0] if hands else None
                sample = sample_from_hand(hand, t) if hand else None
                handedness = hand.handedness if hand else ""

            msgs = _handle_commands(bridge, session, simulate, t)
            msgs += session.update(sample, t, handedness)
            for m in msgs:
                bridge.send(m)

            if frame is not None and preview:
                phase = next((m for m in reversed(msgs) if m["type"] == "phase"), None)
                _draw_preview(frame, hand, session, phase)
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
