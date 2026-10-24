"""
Start the whole product: the ADHD and Parkinson's model servers, then the hand-tracked Brain Explorer.

    .venv/bin/python run.py              # camera, opens http://127.0.0.1:8765/
    .venv/bin/python run.py --no-camera  # mouse and keyboard only

Closing the Brain Explorer (q in the camera window, or Ctrl-C) stops the model servers too.
"""
import argparse
import json
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frontend"))


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def start_api(name, port):
    if not port_free(port):
        raise SystemExit(f"Port {port} for the {name} model is already in use. Stop that program or pass --{name}-port.")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port),
                             "--log-level", "warning"], cwd=ROOT / "backend" / name)
    return proc, f"http://127.0.0.1:{port}"


def wait_ready(name, proc, url, timeout=120):
    """Wait until the server answers; loading torch and the models takes a few seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"The {name} model server stopped during start-up (exit code {proc.returncode}).")
        try:
            with urllib.request.urlopen(url + "/", timeout=2) as r:
                print(f"  {name:<10} {url}  {json.load(r)['message']}")
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"The {name} model server did not answer within {timeout} s.")


def main():
    ap = argparse.ArgumentParser(description="Brain Explorer: model servers + hand-tracked 3D viewer")
    ap.add_argument("--adhd-port", type=int, default=8011)
    ap.add_argument("--parkinsons-port", type=int, default=8012)
    ap.add_argument("--port", type=int, default=8765, help="Brain Explorer page and gesture stream")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-camera", action="store_true", help="mouse and keyboard only")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    ap.add_argument("--no-preview", action="store_true", help="don't show the camera window")
    args = ap.parse_args()
    # a kill or a closed terminal should still stop the model servers, so turn SIGTERM into a normal exit
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))

    procs = {}
    try:
        print("Starting model servers...")
        apis = {}
        for name, port in (("adhd", args.adhd_port), ("parkinsons", args.parkinsons_port)):
            procs[name], apis[name] = start_api(name, port)
        for name in procs:
            wait_ready(name, procs[name], apis[name])

        from handui.brain.runner import run_brain
        run_brain(None, args.camera, args.port, not args.no_open, not args.no_preview, not args.no_camera, apis)
    finally:
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
