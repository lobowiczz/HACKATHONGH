"""Streams gesture events and hand state to other programs over a local WebSocket.

Standard library only. The server mostly sends JSON text frames, so a browser
page or a Python script can follow the hand with
`new WebSocket("ws://127.0.0.1:8765")`. Clients may also send small JSON
commands, which the host program collects with poll_commands(). The bridge sits
beside the engine and the UI and knows about neither; app.py hands it each
frame's output. It can also serve a few fixed pages over plain HTTP on the same
port, and with local_only=True it refuses other websites and foreign Host
headers.

Messages (all JSON):
  {"type": "hello", "version": 1, "state": {...}}                 on connect
  {"type": "event", "kind": "swipe", "data": {...}, "t": 12.3}    one per engine Event
  {"type": "state", "t": 12.3, ...state fields...}                throttled to state_hz
"""
from __future__ import annotations

import base64
import hashlib
import json
import queue
import select
import socket
import struct
import threading
import time
from urllib.parse import urlsplit

from .gestures import Event, GestureState

PROTOCOL_VERSION = 1
DEFAULT_PORT = 8765
_SEND_TIMEOUT = 0.25
_MAX_COMMAND_BYTES = 1 << 16
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_WS_GUID ="258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _jsonable(value):
    """Convert numpy scalars/arrays and tuples to plain JSON types."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def state_to_dict(state: GestureState) -> dict:
    return _jsonable({
        "present": state.present,
        "handedness": state.handedness,
        "cursor": state.cursor,
        "extension": state.extension,
        "grab_strength": state.grab_strength,
        "pinch": state.pinch,
        "speed": state.speed,
        "grabbing": state.grabbing,
        "palm_open": state.palm_open,
        "menu_open": state.menu_open,
        "menu_progress": state.menu_progress,
        "custom_label": state.custom_label,
        "custom_confidence": state.custom_confidence,
    })


def event_to_dict(event: Event, t: float) -> dict:
    return {"type": "event", "kind": event.kind, "data": _jsonable(event.data), "t": t}


def _frame(opcode: int, payload: bytes = b"") -> bytes:
    """Encode one unmasked server-to-client WebSocket frame."""
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", 0x80 | opcode, n)
    elif n < 1 << 16:
        head = struct.pack("!BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
    return head + payload


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def _read_frame(sock: socket.socket, max_len: int | None = None) -> tuple[int, bytes]:
    """Read one client frame (clients always mask). Returns (opcode, payload)."""
    b0, b1 = _recv_exact(sock, 2)
    n = b1 & 0x7F
    if n == 126:
        (n,) = struct.unpack("!H", _recv_exact(sock, 2))
    elif n == 127:
        (n,) = struct.unpack("!Q", _recv_exact(sock, 8))
    if max_len is not None and n > max_len:
        raise ConnectionError("frame too large")
    mask = _recv_exact(sock, 4) if b1 & 0x80 else b""
    data = _recv_exact(sock, n)
    if mask:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    return b0 & 0x0F, data


class Bridge:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, state_hz: float = 30.0,
                 routes: dict[str, bytes] | None = None, local_only: bool = False):
        """routes maps an HTTP path such as "/" to an HTML page served on the same port."""
        self.host, self.port = host, port
        self._routes = routes or {}
        self._local_only = local_only
        self._commands: queue.SimpleQueue[dict] = queue.SimpleQueue()
        self._min_state_gap = 1.0 / state_hz
        self._last_state = float("-inf")
        self._latest: dict = {}
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._closed = False
        self._server = socket.create_server((host, port))
        self.port = self._server.getsockname()[1]  # resolves port 0
        threading.Thread(target=self._accept_loop, daemon=True).start()

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def publish(self, events: list[Event], state: GestureState, t: float) -> None:
        """Call once per frame from the main loop; never blocks for long."""
        self._latest = state_to_dict(state)
        if not self._clients:
            return
        messages = [event_to_dict(e, t) for e in events]
        if t - self._last_state >= self._min_state_gap:
            self._last_state = t
            messages.append({"type": "state", "t": t, **self._latest})
        for msg in messages:
            self._broadcast(_frame(0x1, json.dumps(msg).encode()))

    def send(self, message: dict) -> None:
        """Broadcast any JSON-serializable message to every connected client."""
        if self._clients:
            self._broadcast(_frame(0x1, json.dumps(_jsonable(message)).encode()))

    def poll_commands(self) -> list[dict]:
        """JSON objects clients have sent since the last call."""
        out = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._closed = True
        self._server.close()
        with self._lock:
            clients, self._clients = self._clients, []
        for c in clients:
            try:
                c.sendall(_frame(0x8))
            except OSError:
                pass
            c.close()

    def _broadcast(self, data: bytes) -> None:
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            try:
                c.sendall(data)
            except OSError:  # includes send timeouts from a stalled client
                self._drop(c)

    def _drop(self, sock: socket.socket) -> None:
        with self._lock:
            if sock in self._clients:
                self._clients.remove(sock)
        try:
            sock.close()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                sock, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve_client, args=(sock,), daemon=True).start()

    def _origin_ok(self, headers: dict) -> bool:
        if not self._local_only:
            return True
        if urlsplit("//" + headers.get("host", "")).hostname not in _LOCAL_HOSTS:  # blocks DNS rebinding
            return False
        origin = headers.get("origin")
        return origin is None or urlsplit(origin).hostname in _LOCAL_HOSTS

    def _handshake(self, sock: socket.socket) -> bool:
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = sock.recv(4096)
            if not chunk or len(raw) > 16384:
                return False
            raw += chunk
        request_line, *header_lines = raw.decode("latin-1").split("\r\n")
        parts = request_line.split(" ")
        path = urlsplit(parts[1]).path if len(parts) > 1 else "/"
        headers = {}
        for line in header_lines:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        if not self._origin_ok(headers):
            sock.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return False
        key = headers.get("sec-websocket-key")
        if not key or headers.get("upgrade", "").lower() != "websocket":
            page = self._routes.get(path) if parts[0] == "GET" else None
            if page is None:
                sock.sendall(b"HTTP/1.1 426 Upgrade Required\r\nContent-Length: 0\r\n\r\n")
            else:
                sock.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\n"
                    b"Connection: close\r\nContent-Length: " + str(len(page)).encode() + b"\r\n\r\n" + page
                )
            return False
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        sock.sendall(
            ("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
             f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode()
        )
        return True

    def _serve_client(self, sock: socket.socket) -> None:
        try:
            sock.settimeout(5)
            if not self._handshake(sock):
                sock.close()
                return
            hello = {"type": "hello", "version": PROTOCOL_VERSION, "state": self._latest}
            sock.sendall(_frame(0x1, json.dumps(hello).encode()))
            sock.settimeout(_SEND_TIMEOUT)  # a stalled client is dropped instead of freezing the camera loop
            with self._lock:
                self._clients.append(sock)
            # Only reads from here on: answer pings, stop on close or error.
            while True:
                select.select([sock], [], [])
                opcode, payload = _read_frame(sock, _MAX_COMMAND_BYTES)
                if opcode == 0x8:
                    break
                if opcode == 0x1:
                    try:
                        msg = json.loads(payload)
                    except ValueError:
                        continue
                    if isinstance(msg, dict):
                        self._commands.put(msg)
                if opcode == 0x9:
                    sock.sendall(_frame(0xA, payload))
        except (OSError, ConnectionError):
            pass
        finally:
            self._drop(sock)
