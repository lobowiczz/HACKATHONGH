"""The hand-driven interface: draggable cards, swipeable pages and a radial palm menu.

HandUI consumes gesture events (handle) and draws a frame (render). It never
touches the camera or tracker, so it can be driven by synthetic input in tests.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import drawing as D
from . import gestures as G
from .config import BINDINGS

BG = (28, 20, 14)
CARD_BG = (58, 44, 34)
TEXT = (245, 240, 235)
MUTED = (178, 165, 155)
ACCENT = (255, 190, 90)
DANGER = (90, 90, 235)
PALETTE = ((99, 130, 239), (170, 190, 45), (240, 120, 150), (70, 185, 245), (240, 160, 80))
PAGES = ("Home", "Projects", "Ideas")

CARD_W, CARD_H = 250, 140
TOP = 160  # content starts below the header and HUD


@dataclass
class Card:
    title: str
    body: str
    color: tuple[int, int, int]
    page: int
    x: float  # top-left in page-local pixels
    y: float
    tx: float | None = None  # animation target
    ty: float | None = None

    def __post_init__(self):
        self.tx = self.x if self.tx is None else self.tx
        self.ty = self.y if self.ty is None else self.ty


def wrap(s: str, width: int, scale: float) -> list[str]:
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, D.FONT, scale, 1)[0][0] > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


class HandUI:
    ACTIONS = ("new_card", "tidy_cards", "next_page", "prev_page",
               "toggle_camera", "toggle_skeleton", "toggle_hud", "quit")
    MENU_ITEMS = (("New card", "new_card"), ("Tidy", "tidy_cards"), ("Camera", "toggle_camera"),
                  ("Skeleton", "toggle_skeleton"), ("HUD", "toggle_hud"), ("Quit", "quit"))
    MENU_R, MENU_INNER = 150, 46

    def __init__(self, width: int = 1280, height: int = 720, engine: G.GestureEngine | None = None,
                 bindings: dict[str, str] | None = None, classifier_labels: list[str] | None = None):
        self.W, self.H = width, height
        self.engine = engine
        self.bindings = BINDINGS if bindings is None else bindings
        self.classifier_labels = classifier_labels
        self.page = 0
        self.page_offset = 0.0
        self.cards = self._starter_cards()
        self.drag: tuple[Card, float, float] | None = None
        self.menu_center: tuple[int, int] | None = None
        self.menu_hover: int | None = None
        self.show_camera = self.show_skeleton = self.show_hud = True
        self.quit_requested = False
        self.toasts: list[tuple[str, float]] = []
        self.fps = 0.0
        self._last_t: float | None = None
        self._notes = 0
        self._view = (1.0, 0, 0)  # camera -> canvas transform: scale, x offset, y offset
        self._grid = self._make_grid()

    # ------------------------------------------------------------ setup

    def _starter_cards(self) -> list[Card]:
        spec = [
            (0, "Grab & drag", "Make a fist or pinch over a card, move, then open your hand."),
            (0, "Swipe", "Flick an open hand left or right to change page."),
            (0, "Palm menu", "Hold an open palm still to open the menu. Point at an item, then grab."),
            (0, "Delete", "Drag a card onto the bin in the corner to remove it."),
            (1, "Ocean data", "Drag me around. Swipe down to tidy this page."),
            (1, "Hand UI", "Built with MediaPipe + OpenCV."),
            (2, "Custom gestures", "Record: python -m handui record thumbs_up"),
            (2, "Then train", "python -m handui train, and bind it in config.py."),
        ]
        cards = []
        for i, (page, title, body) in enumerate(spec):
            cards.append(Card(title, body, PALETTE[i % len(PALETTE)], page, 0, 0))
        for p in range(len(PAGES)):
            self._tidy(cards, p, animate=False)
        return cards

    def _make_grid(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), BG, np.uint8)
        for y in range(20, self.H, 40):
            for x in range(20, self.W, 40):
                cv2.circle(img, (x, y), 1, (60, 48, 40), -1, D.AA)
        return img

    # ------------------------------------------------------------ input

    def cursor_px(self, state: G.GestureState) -> tuple[float, float]:
        return state.cursor[0] * self.W, state.cursor[1] * self.H

    def handle(self, events: list[G.Event], state: G.GestureState) -> None:
        for ev in events:
            if ev.kind == G.GRAB_START:
                self._grab_start(state)
            elif ev.kind == G.GRAB_MOVE:
                self._grab_move(state)
            elif ev.kind == G.GRAB_END:
                self._grab_end(state, ev.data.get("cancelled", False))
            elif ev.kind == G.SWIPE:
                d = ev.data["direction"]
                self._trigger(f"swipe_{d}", f"Swipe {d}")
            elif ev.kind == G.MENU_OPEN:
                self._open_menu(*self.cursor_px(state))
            elif ev.kind == G.MENU_CLOSE:
                self._close_menu()
            elif ev.kind == G.CUSTOM:
                self._trigger(f"custom:{ev.data['label']}",
                              f"Gesture '{ev.data['label']}' {ev.data['confidence']:.0%}")

    def handle_key(self, key: int) -> None:
        keys = {ord("c"): "toggle_camera", ord("s"): "toggle_skeleton", ord("h"): "toggle_hud",
                ord("n"): "new_card", ord("t"): "tidy_cards", ord("a"): "prev_page",
                ord("d"): "next_page", ord("q"): "quit", 27: "quit"}
        if key in keys:
            self.run_action(keys[key])
        elif key == ord("m"):
            if self.menu_center:
                self._close_menu()
            else:
                self._open_menu(self.W / 2, self.H / 2)

    def _trigger(self, binding: str, description: str) -> None:
        action = self.bindings.get(binding)
        self.toast(f"{description}  ->  {action.replace('_', ' ')}" if action else description)
        if action:
            self.run_action(action)

    def _page_shift(self, card: Card) -> float:
        return card.page * self.W - self.page_offset

    def _grab_start(self, state) -> None:
        x, y = self.cursor_px(state)
        if self.menu_center is not None:
            item = self._menu_item_at(x, y)
            self._close_menu()
            if item is not None:
                label, action = self.MENU_ITEMS[item]
                self.toast(label)
                self.run_action(action)
            return
        for card in reversed(self.cards):  # topmost first
            sx = card.x + self._page_shift(card)
            if card.page == self.page and sx <= x <= sx + CARD_W and card.y <= y <= card.y + CARD_H:
                self.cards.remove(card)
                self.cards.append(card)
                self.drag = (card, x - sx, y - card.y)
                return

    def _grab_move(self, state) -> None:
        if not self.drag:
            return
        card, ox, oy = self.drag
        x, y = self.cursor_px(state)
        card.x = card.tx = float(np.clip(x - ox - self._page_shift(card), -CARD_W / 2, self.W - CARD_W / 2))
        card.y = card.ty = float(np.clip(y - oy, 100, self.H - CARD_H / 2))

    def _grab_end(self, state, cancelled: bool) -> None:
        if not self.drag:
            return
        card = self.drag[0]
        self.drag = None
        if not cancelled and self._over_bin(*self.cursor_px(state)):
            self.cards.remove(card)
            self.toast(f"Deleted '{card.title}'")

    def _bin_center(self) -> tuple[int, int]:
        return self.W - 80, self.H - 90

    def _over_bin(self, x: float, y: float) -> bool:
        bx, by = self._bin_center()
        return math.hypot(x - bx, y - by) < 70

    def _open_menu(self, x: float, y: float) -> None:
        m = self.MENU_R + 12
        self.menu_center = (int(np.clip(x, m, self.W - m)), int(np.clip(y, m, self.H - m)))
        if self.engine:
            self.engine.state.menu_open = True

    def _close_menu(self) -> None:
        self.menu_center = None
        self.menu_hover = None
        if self.engine:
            self.engine.close_menu()

    def _menu_item_at(self, x: float, y: float) -> int | None:
        cx, cy = self.menu_center
        if math.hypot(x - cx, y - cy) < self.MENU_INNER:
            return None
        step = 360 / len(self.MENU_ITEMS)
        angle = math.degrees(math.atan2(y - cy, x - cx))
        return int(round(((angle + 90) % 360) / step)) % len(self.MENU_ITEMS)

    # ----------------------------------------------------------- actions

    def run_action(self, action: str) -> None:
        if action == "new_card":
            self._notes += 1
            x, y = self.W / 2 - CARD_W / 2, self.H / 2 - CARD_H / 2
            self.cards.append(Card(f"Note {self._notes}", "Fresh card. Grab it and put it somewhere.",
                                   PALETTE[(len(self.cards)) % len(PALETTE)], self.page, x, y + 60, x, y))
        elif action == "tidy_cards":
            self._tidy(self.cards, self.page)
        elif action in ("next_page", "prev_page"):
            target = self.page + (1 if action == "next_page" else -1)
            if 0 <= target < len(PAGES):
                self.page = target
            else:
                self.toast("No more pages")
        elif action == "toggle_camera":
            self.show_camera = not self.show_camera
        elif action == "toggle_skeleton":
            self.show_skeleton = not self.show_skeleton
        elif action == "toggle_hud":
            self.show_hud = not self.show_hud
        elif action == "quit":
            self.quit_requested = True

    def _tidy(self, cards: list[Card], page: int, animate: bool = True) -> None:
        gap = 30
        cols = max(1, (self.W - 2 * gap) // (CARD_W + gap))
        on_page = [c for c in cards if c.page == page]
        cols = min(cols, max(1, len(on_page)))
        left = (self.W - (cols * CARD_W + (cols - 1) * gap)) / 2
        for i, card in enumerate(on_page):
            card.tx = left + (i % cols) * (CARD_W + gap)
            card.ty = TOP + (i // cols) * (CARD_H + gap)
            if not animate:
                card.x, card.y = card.tx, card.ty

    def toast(self, message: str) -> None:
        self.toasts.append((message, time.monotonic()))
        self.toasts = self.toasts[-3:]

    # ------------------------------------------------------------ render

    def _animate(self, t: float) -> None:
        dt = 0.0 if self._last_t is None else min(t - self._last_t, 0.1)
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt
        self._last_t = t
        k = min(1.0, dt * 12)
        self.page_offset += (self.page * self.W - self.page_offset) * min(1.0, dt * 10)
        dragged = self.drag[0] if self.drag else None
        for card in self.cards:
            if card is not dragged:
                card.x += (card.tx - card.x) * k
                card.y += (card.ty - card.y) * k
        now = time.monotonic()
        self.toasts = [(m, t0) for m, t0 in self.toasts if now - t0 < 2.2]

    def render(self, frame: np.ndarray | None, hand, state: G.GestureState, t: float) -> np.ndarray:
        self._animate(t)
        canvas = self._background(frame)
        if hand is not None and frame is not None and self.show_camera and self.show_skeleton:
            s, ox, oy = self._view
            D.skeleton(canvas, hand.landmarks[:, :2] * s - (ox, oy), color=(200, 200, 200), thickness=1)
        self._draw_header(canvas)
        x, y = self.cursor_px(state)
        self._draw_cards(canvas, state)
        if self.drag:  # above the dragged card so the drop target stays visible
            self._draw_bin(canvas, state.present and self._over_bin(x, y))
        if self.menu_center:
            self.menu_hover = self._menu_item_at(x, y) if state.present else None
            self._draw_menu(canvas)
        if state.present:
            self._draw_cursor(canvas, state)
        else:
            D.text(canvas, "Show your hand to the camera", self.W // 2, self.H // 2 + 200, 0.8, MUTED, 1,
                   anchor="center")
        if self.show_hud:
            self._draw_hud(canvas, state)
        self._draw_toasts(canvas)
        D.text(canvas, "Fist / pinch: drag     Swipe: change page     Hold open palm: menu     Q: quit",
               self.W // 2, self.H - 16, 0.45, MUTED, anchor="center")
        return canvas

    def _background(self, frame) -> np.ndarray:
        if frame is None or not self.show_camera:
            return self._grid.copy()
        fh, fw = frame.shape[:2]
        s = max(self.W / fw, self.H / fh)
        rw, rh = int(round(fw * s)), int(round(fh * s))
        ox, oy = (rw - self.W) // 2, (rh - self.H) // 2
        self._view = (s, ox, oy)
        resized = cv2.resize(frame, (rw, rh))[oy:oy + self.H, ox:ox + self.W]
        return cv2.addWeighted(resized, 0.35, self._grid, 0.65, 0)

    def _draw_header(self, img) -> None:
        D.text(img, PAGES[self.page], self.W // 2, 52, 1.0, TEXT, 2, D.FONT_BOLD, anchor="center")
        n = len(PAGES)
        for i in range(n):
            c = (int(self.W / 2 + (i - (n - 1) / 2) * 22), 76)
            if i == self.page:
                cv2.circle(img, c, 5, ACCENT, -1, D.AA)
            else:
                cv2.circle(img, c, 4, MUTED, 1, D.AA)

    def _draw_cards(self, img, state) -> None:
        x, y = self.cursor_px(state)
        dragged = self.drag[0] if self.drag else None
        for card in self.cards:
            sx, sy = card.x + self._page_shift(card), card.y
            if sx > self.W or sx + CARD_W < 0:
                continue
            lifted = card is dragged
            hovered = (state.present and not lifted and self.menu_center is None and card.page == self.page
                       and sx <= x <= sx + CARD_W and sy <= y <= sy + CARD_H)
            if lifted:
                sy -= 6
            D.shadow(img, sx, sy, CARD_W, CARD_H, 16, spread=18 if lifted else 10, alpha=0.5 if lifted else 0.35)
            D.panel(img, sx, sy, CARD_W, CARD_H, 16, CARD_BG, 0.94)
            if lifted or hovered:
                D.rounded_rect(img, sx, sy, CARD_W, CARD_H, 16, ACCENT if lifted else TEXT, 2)
            D.rounded_rect(img, sx + 16, sy + 20, 5, CARD_H - 40, 2, card.color)
            D.text(img, card.title, sx + 34, sy + 42, 0.68, TEXT, 1, D.FONT_BOLD)
            for i, line in enumerate(wrap(card.body, CARD_W - 54, 0.48)[:3]):
                D.text(img, line, sx + 34, sy + 72 + i * 22, 0.48, MUTED)

    def _draw_bin(self, img, active: bool) -> None:
        bx, by = self._bin_center()
        D.blend(img, bx - 60, by - 60, bx + 61, by + 61,
                lambda o, ox, oy: cv2.circle(o, (bx + ox, by + oy), 52 if active else 44,
                                             DANGER if active else CARD_BG, -1, D.AA), 0.85)
        cv2.circle(img, (bx, by), 52 if active else 44, DANGER, 2, D.AA)
        D.rounded_rect(img, bx - 13, by - 10, 26, 28, 4, TEXT, 2)
        cv2.line(img, (bx - 18, by - 15), (bx + 18, by - 15), TEXT, 2, D.AA)
        cv2.line(img, (bx - 5, by - 20), (bx + 5, by - 20), TEXT, 2, D.AA)
        D.text(img, "Drop to delete", bx, by + 76, 0.45, TEXT if active else MUTED, anchor="center")

    def _draw_menu(self, img) -> None:
        cx, cy = self.menu_center
        R, r = self.MENU_R, self.MENU_INNER
        n = len(self.MENU_ITEMS)
        step = 360 / n

        def draw(o, ox, oy):
            c = (cx + ox, cy + oy)
            cv2.circle(o, c, R, (40, 30, 24), -1, D.AA)
            if self.menu_hover is not None:
                mid = -90 + self.menu_hover * step
                cv2.ellipse(o, c, (R, R), 0, mid - step / 2, mid + step / 2, ACCENT, -1, D.AA)
            cv2.circle(o, c, r, (24, 18, 14), -1, D.AA)

        D.shadow(img, cx - R, cy - R, 2 * R, 2 * R, R, spread=16, alpha=0.4)
        D.blend(img, cx - R - 2, cy - R - 2, cx + R + 3, cy + R + 3, draw, 0.92)
        for i, (label, _) in enumerate(self.MENU_ITEMS):
            edge = math.radians(-90 + (i + 0.5) * step)
            cv2.line(img, (int(cx + r * math.cos(edge)), int(cy + r * math.sin(edge))),
                     (int(cx + R * math.cos(edge)), int(cy + R * math.sin(edge))), (70, 58, 50), 1, D.AA)
            mid = math.radians(-90 + i * step)
            lx, ly = cx + (R + r) / 2 * math.cos(mid), cy + (R + r) / 2 * math.sin(mid)
            hot = i == self.menu_hover
            D.text(img, label, lx, ly + 6, 0.52, BG if hot else TEXT, 2 if hot else 1, anchor="center")
        cv2.circle(img, (cx, cy), R, (90, 75, 65), 1, D.AA)
        D.text(img, "close", cx, cy + 5, 0.42, MUTED, anchor="center")
        D.text(img, "Point at an item, then grab", cx, cy + R + 28, 0.48, TEXT, anchor="center")

    def _draw_cursor(self, img, state) -> None:
        c = tuple(int(v) for v in self.cursor_px(state))
        if state.grabbing:
            cv2.circle(img, c, 11, ACCENT, -1, D.AA)
            cv2.circle(img, c, 13, TEXT, 2, D.AA)
        else:
            cv2.circle(img, c, 15, TEXT, 2, D.AA)
            cv2.circle(img, c, 3, TEXT, -1, D.AA)
        if state.menu_progress > 0:
            cv2.ellipse(img, c, (24, 24), -90, 0, 360 * state.menu_progress, ACCENT, 3, D.AA)

    def _draw_hud(self, img, state) -> None:
        x, y, w, h = 16, 16, 230, 168
        D.panel(img, x, y, w, h, 14, (22, 16, 12), 0.78)
        D.text(img, f"{self.fps:4.0f} fps", x + 14, y + 26, 0.48, MUTED)
        D.text(img, f"{state.handedness} hand" if state.present else "no hand", x + w - 14, y + 26, 0.48,
               TEXT if state.present else MUTED, anchor="right")

        for i, (name, value) in enumerate(zip("TIMRP", state.extension if state.present else np.zeros(5))):
            bx, by, bh = x + 18 + i * 22, y + 92, 46
            D.rounded_rect(img, bx, by - bh, 12, bh, 4, (60, 48, 40))
            fill = int(bh * float(value))
            if fill > 2:
                D.rounded_rect(img, bx, by - fill, 12, fill, 4, ACCENT)
            D.text(img, name, bx + 6, by + 16, 0.4, MUTED, anchor="center")

        chips = (("GRAB", state.grabbing), ("PALM", state.palm_open), ("MENU", self.menu_center is not None))
        for i, (name, on) in enumerate(chips):
            cy = y + 44 + i * 22
            D.rounded_rect(img, x + 140, cy, 74, 18, 9, ACCENT if on else (60, 48, 40))
            D.text(img, name, x + 177, cy + 13, 0.4, BG if on else MUTED, 1, anchor="center")

        if self.classifier_labels is None:
            custom = "custom: no model"
        elif state.custom_label:
            custom = f"custom: {state.custom_label} {state.custom_confidence:.0%}"
        else:
            custom = "custom: -"
        D.text(img, custom, x + 14, y + 140, 0.48, TEXT if state.custom_label else MUTED)

    def _draw_toasts(self, img) -> None:
        now = time.monotonic()
        for i, (msg, t0) in enumerate(reversed(self.toasts)):
            age = now - t0
            alpha = min(1.0, (2.2 - age) / 0.4) * min(1.0, age / 0.12 + 0.2)
            (tw, _), _ = cv2.getTextSize(msg, D.FONT, 0.6, 1)
            w, h = tw + 40, 40
            x, y = (self.W - w) // 2, self.H - 90 - i * 50
            D.panel(img, x, y, w, h, 20, (60, 46, 36), 0.9 * alpha)
            if alpha > 0.3:
                D.text(img, msg, self.W // 2, y + 26, 0.6, TEXT, anchor="center")
