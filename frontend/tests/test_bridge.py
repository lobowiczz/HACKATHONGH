import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handui import bridge as B  # noqa: E402
from handui.gestures import SWIPE, Event, GestureState  # noqa: E402


def connect(port, host="x", origin=None):
    """Minimal WebSocket client: returns the socket after a successful handshake."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=2)
    key = base64.b64encode(os.urandom(16)).decode()
    extra = f"Origin: {origin}\r\n" if origin else ""
    sock.sendall(
        (f"GET / HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n{extra}"
         f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode()
    )
    raw = b""
    while b"\r\n\r\n" not in raw:
        raw += sock.recv(1)  # byte at a time so the first frame stays in the socket
    expected = base64.b64encode(hashlib.sha1((key + B._WS_GUID).encode()).digest()).decode()
    if not (raw.startswith(b"HTTP/1.1 101") and expected in raw.decode()):
        sock.close()
        raise AssertionError(raw.split(b"\r\n")[0])
    return sock


def read_json(sock):
    opcode, payload = B._read_frame(sock)
    assert opcode == 0x1
    return json.loads(payload)


def wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = B.Bridge(port=0, state_hz=1000)
        self.addCleanup(self.bridge.close)

    def test_hello_then_events_and_state(self):
        sock = connect(self.bridge.port)
        self.addCleanup(sock.close)
        hello = read_json(sock)
        self.assertEqual(hello["type"], "hello")
        self.assertEqual(hello["version"], B.PROTOCOL_VERSION)

        self.assertTrue(wait_for(lambda: self.bridge.client_count == 1))
        state = GestureState(present=True, cursor=(np.float64(0.25), np.float64(0.75)),
                             extension=np.ones(5, dtype=np.float32), custom_label="peace")
        self.bridge.publish([Event(SWIPE, {"direction": "left", "speed": np.float32(2.5)})], state, 1.0)

        event = read_json(sock)
        self.assertEqual((event["type"], event["kind"], event["t"]), ("event", "swipe", 1.0))
        self.assertEqual(event["data"], {"direction": "left", "speed": 2.5})
        snapshot = read_json(sock)
        self.assertEqual(snapshot["type"], "state")
        self.assertEqual(snapshot["cursor"], [0.25, 0.75])
        self.assertEqual(snapshot["extension"], [1.0] * 5)
        self.assertEqual(snapshot["custom_label"], "peace")

    def test_state_is_throttled_but_events_are_not(self):
        bridge = B.Bridge(port=0, state_hz=10)
        self.addCleanup(bridge.close)
        sock = connect(bridge.port)
        self.addCleanup(sock.close)
        read_json(sock)  # hello
        self.assertTrue(wait_for(lambda: bridge.client_count == 1))
        state = GestureState()
        bridge.publish([], state, 0.00)
        bridge.publish([Event(SWIPE, {"direction": "up", "speed": 1.0})], state, 0.01)
        self.assertEqual(read_json(sock)["type"], "state")
        self.assertEqual(read_json(sock)["kind"], "swipe")  # the t=0.01 state frame was skipped

    def test_late_client_gets_latest_state_in_hello(self):
        self.bridge.publish([], GestureState(present=True, handedness="Right"), 0.0)
        sock = connect(self.bridge.port)
        self.addCleanup(sock.close)
        self.assertEqual(read_json(sock)["state"]["handedness"], "Right")

    def test_client_close_is_dropped_and_publish_survives(self):
        sock = connect(self.bridge.port)
        read_json(sock)
        self.assertTrue(wait_for(lambda: self.bridge.client_count == 1))
        mask = b"\x01\x02\x03\x04"
        sock.sendall(struct.pack("!BB", 0x88, 0x80) + mask)
        sock.close()
        self.assertTrue(wait_for(lambda: self.bridge.client_count == 0))
        self.bridge.publish([], GestureState(), 5.0)

    def test_plain_http_request_is_refused(self):
        sock = socket.create_connection(("127.0.0.1", self.bridge.port), timeout=2)
        self.addCleanup(sock.close)
        sock.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        self.assertTrue(sock.recv(64).startswith(b"HTTP/1.1 426"))
        self.assertEqual(self.bridge.client_count, 0)

    def test_large_payload_uses_extended_length(self):
        sock = connect(self.bridge.port)
        self.addCleanup(sock.close)
        read_json(sock)
        self.assertTrue(wait_for(lambda: self.bridge.client_count == 1))
        self.bridge._broadcast(B._frame(0x1, json.dumps({"pad": "x" * 70000}).encode()))
        self.assertEqual(len(read_json(sock)["pad"]), 70000)


def client_text(sock, obj):
    payload = json.dumps(obj).encode()
    mask = b"\x05\x06\x07\x08"
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    sock.sendall(struct.pack("!BB", 0x81, 0x80 | len(payload)) + mask + masked)


class BridgeCommandTests(unittest.TestCase):
    def test_client_commands_are_queued_and_send_broadcasts(self):
        bridge = B.Bridge(port=0)
        self.addCleanup(bridge.close)
        sock = connect(bridge.port)
        self.addCleanup(sock.close)
        read_json(sock)
        self.assertTrue(wait_for(lambda: bridge.client_count == 1))
        client_text(sock, {"cmd": "start", "test": "tapping"})
        self.assertTrue(wait_for(lambda: bridge._commands.qsize() == 1))
        self.assertEqual(bridge.poll_commands(), [{"cmd": "start", "test": "tapping"}])
        self.assertEqual(bridge.poll_commands(), [])
        bridge.send({"type": "phase", "x": np.float32(1.5)})
        self.assertEqual(read_json(sock), {"type": "phase", "x": 1.5})

    def test_invalid_command_payloads_are_ignored(self):
        bridge = B.Bridge(port=0)
        self.addCleanup(bridge.close)
        sock = connect(bridge.port)
        self.addCleanup(sock.close)
        read_json(sock)
        sock.sendall(struct.pack("!BB", 0x81, 0x80 | 3) + b"\0\0\0\0" + b"{{{")
        client_text(sock, [1, 2])
        client_text(sock, {"cmd": "ok"})
        self.assertTrue(wait_for(lambda: bridge._commands.qsize() == 1))
        self.assertEqual(bridge.poll_commands(), [{"cmd": "ok"}])


class BridgeHttpTests(unittest.TestCase):
    def setUp(self):
        self.bridge = B.Bridge(port=0, routes={"/": b"<h1>hi</h1>"}, local_only=True)
        self.addCleanup(self.bridge.close)

    def get(self, path, host="127.0.0.1"):
        sock = socket.create_connection(("127.0.0.1", self.bridge.port), timeout=2)
        self.addCleanup(sock.close)
        sock.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
        return sock.recv(4096)

    def test_route_is_served(self):
        reply = self.get("/")
        self.assertTrue(reply.startswith(b"HTTP/1.1 200"))
        self.assertTrue(reply.endswith(b"<h1>hi</h1>"))

    def test_unknown_path_is_refused(self):
        self.assertTrue(self.get("/nope").startswith(b"HTTP/1.1 426"))

    def test_foreign_host_header_is_forbidden(self):
        self.assertTrue(self.get("/", host="evil.example").startswith(b"HTTP/1.1 403"))

    def test_websocket_from_other_site_is_forbidden(self):
        with self.assertRaises(AssertionError):
            connect(self.bridge.port, host="127.0.0.1", origin="https://evil.example")
        self.assertEqual(self.bridge.client_count, 0)

    def test_websocket_from_localhost_page_is_accepted(self):
        sock = connect(self.bridge.port, host="localhost:8765", origin="http://localhost:8765")
        self.addCleanup(sock.close)
        self.assertEqual(read_json(sock)["type"], "hello")


if __name__ == "__main__":
    unittest.main()
