"""Record landmark samples for a custom gesture label from the webcam."""
from __future__ import annotations

import time

import cv2

from . import drawing as D
from .classifier import append_samples, label_counts
from .tracker import HandTracker, open_camera

WINDOW = "Hand UI - record"


def record(label: str, seconds: float = 8.0, camera: int = 0, countdown: float = 3.0) -> int:
    cap = open_camera(camera)
    samples = []
    paused = False
    recorded = 0.0
    start = last = time.monotonic()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    with HandTracker() as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)
            hand = hands[0] if hands else None
            now = time.monotonic()
            dt, last = now - last, now
            counting_down = now - start < countdown

            if not counting_down and not paused:
                recorded += dt
                if hand is not None:
                    samples.append((hand.landmarks.copy(), hand.handedness))
            if recorded >= seconds:
                break

            h, w = frame.shape[:2]
            if hand is not None:
                D.skeleton(frame, hand.landmarks[:, :2])
            D.panel(frame, 20, 20, w - 40, 90, 16, (30, 22, 18), 0.75)
            if counting_down:
                status = f"Get ready: show '{label}' in {countdown - (now - start):.0f}..."
            elif paused:
                status = "Paused - press SPACE to resume"
            else:
                status = f"Recording '{label}'  {seconds - recorded:4.1f}s left"
            D.text(frame, status, 40, 58, 0.8, (255, 255, 255), 2)
            D.text(frame, f"{len(samples)} samples   |   move, tilt and change distance for variety   |   "
                          "SPACE pause  Q stop", 40, 92, 0.5, (200, 200, 200))
            if not counting_down and not paused:
                cv2.circle(frame, (w - 50, 65), 10, (60, 60, 230), -1, D.AA)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused

    cap.release()
    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)
    append_samples(label, samples)
    total = label_counts()[label]
    print(f"Saved {len(samples)} samples for '{label}' ({total} total).")
    return len(samples)
