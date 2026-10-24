---
inclusion: always
---

# Tech stack and conventions

## Stack
- Python 3.11, venv at `.venv`. Dependencies: mediapipe, opencv-python, numpy, scikit-learn, scipy.
- Hand tracking: MediaPipe HandLandmarker (21 landmarks). Custom gestures: a small scikit-learn MLP.
- Browser side: a single self-contained HTML file with inline CSS and JS. No external libraries or CDNs, so it works offline.
- The WebSocket bridge uses the **standard library only**. `pip install websockets` failed in this environment, so do not add it.

## Commands
```bash
.venv/bin/python -m handui                              # gesture UI (OpenCV window)
.venv/bin/python -m handui --bridge                     # same, plus a WebSocket stream on ws://127.0.0.1:8765
.venv/bin/python -m handui screen                       # MotorLens, camera, dashboard at http://127.0.0.1:8765/
.venv/bin/python -m handui screen --simulate slowing    # no camera: healthy | slowing | tremor
.venv/bin/python -m handui brain [--eeg FILE] [--no-camera]  # Brain Explorer (input -> explore -> analyze) at http://127.0.0.1:8765/
.venv/bin/python -m handui record|train|labels|forget   # custom gestures
.venv/bin/python -m unittest discover -s tests          # all tests (74 at last count)
```

## Environment gotchas
- macOS camera permission belongs to the app that launched Python. Claude/Kiro may be denied; the user's own Terminal usually works. Camera runs fail with "not authorized to capture video" otherwise.
- The browser preview of a `file://` page cannot open WebSockets. Serve over http (`screen` serves its own dashboard).
- No camera available in CI or agent sessions: verify with `--simulate` and the unit tests.

## Conventions
- Match the surrounding style: type hints, `from __future__ import annotations`, short docstrings, few comments that explain *why*.
- **HandUI never touches the camera or tracker. The engine knows nothing about HandUI. The bridge and screening code know nothing about either.** Keep these boundaries.
- Time comes from a `t` argument (seconds), never from a clock inside logic. That keeps the engine, session and analysis deterministic and testable with synthetic data.
- Analysis functions are pure (arrays in, plain dicts out, JSON-serializable).
- Tests use `unittest`. Use `tests/synthetic.py` for synthetic hands. No test may need a camera or network.
- Tunable thresholds live in dataclasses (`GestureConfig` in `handui/config.py`, `Thresholds` in `handui/screening/analysis.py`), not inline.
- The user commits. Do not add Claude/Co-Authored-By trailers to commits.
