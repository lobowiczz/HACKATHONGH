"""Small OpenCV drawing helpers: rounded and translucent shapes, text, hand skeletons."""
from __future__ import annotations

import cv2
import numpy as np

AA = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    x, y, w, h = int(x), int(y), int(w), int(h)
    r = int(max(0, min(r, w // 2, h // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1, AA)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1, AA)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
        return
    cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, AA)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, AA)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, AA)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, AA)
    for (cx, cy), start in (((x + r, y + r), 180), ((x + w - r, y + r), 270),
                            ((x + w - r, y + h - r), 0), ((x + r, y + h - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), start, 0, 90, color, thickness, AA)


def blend(img, x0, y0, x1, y1, draw, alpha):
    """Run draw(overlay, ox, oy) on a copy of a region and alpha-blend it back in place."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    draw(overlay, -x0, -y0)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def panel(img, x, y, w, h, r, color, alpha):
    blend(img, x, y, x + w + 1, y + h + 1, lambda o, ox, oy: rounded_rect(o, x + ox, y + oy, w, h, r, color), alpha)


def shadow(img, x, y, w, h, r, spread=10, alpha=0.35):
    pad = spread * 2

    def draw(o, ox, oy):
        rounded_rect(o, x + ox + 2, y + oy + spread // 2, w, h, r, (0, 0, 0))
        k = spread * 2 + 1
        o[:] = cv2.GaussianBlur(o, (k, k), 0)

    blend(img, x - pad, y - pad, x + w + pad, y + h + pad + spread, draw, alpha)


def text(img, s, x, y, scale=0.6, color=(240, 240, 240), thickness=1, font=FONT, anchor="left"):
    (tw, th), _ = cv2.getTextSize(s, font, scale, thickness)
    if anchor == "center":
        x -= tw // 2
    elif anchor == "right":
        x -= tw
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thickness, AA)
    return tw, th


def skeleton(img, pts: np.ndarray, color=(255, 255, 255), joint_color=(120, 220, 255), thickness=2):
    """pts: (21, 2) pixel coordinates on img."""
    p = pts.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(p[a]), tuple(p[b]), color, thickness, AA)
    for i, q in enumerate(p):
        cv2.circle(img, tuple(q), 5 if i in (4, 8, 12, 16, 20) else 3, joint_color, -1, AA)
