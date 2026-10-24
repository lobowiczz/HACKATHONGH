---
inclusion: always
---

# Project structure and data flow

```
camera -> tracker.py      MediaPipe HandLandmarker -> Hand (21 landmarks, handedness)
       -> features.py     finger extension, grab strength, pinch distance, palm centre/size
       -> gestures.py     GestureEngine.update(hand, t) -> [Event], keeps GestureState
       -> ui.py           HandUI: cards, pages, radial menu (pure OpenCV drawing)
classifier.py / record.py custom gesture samples, training, inference
bridge.py                 local WebSocket + tiny HTTP server (stdlib)
screening/                MotorLens (below)
brain/                    Brain Explorer: EEG on a 3D brain (below)
```

## Gesture engine facts
- Event kinds: `hand_found`, `hand_lost`, `grab_start`, `grab_move`, `grab_end` (data: `pos`, `cancelled`), `swipe` (`direction`, `speed`), `menu_open`, `menu_close`, `custom` (`label`, `confidence`).
- `GestureState`: `cursor` (smoothed palm centre, 0..1, already mirrored), `grabbing`, `palm_open`, `menu_open`, `menu_progress`, `grab_strength`, `pinch`, `custom_label`.
- Grab = fist or thumb-index pinch. Menu = still open palm held 0.7 s. Swipe cooldown 0.6 s. Thresholds in `GestureConfig`; gesture-to-action `BINDINGS` in `handui/config.py`.

## Bridge (`handui/bridge.py`)
One class, `Bridge(host, port, state_hz, routes, local_only)`.
- Server to client JSON: `{"type":"hello","version":1,"state":{}}` on connect; `{"type":"event","kind","data","t"}` per engine event; `{"type":"state","t",...GestureState fields}` throttled to 30 Hz via `publish(events, state, t)`; any other message via `send(dict)`.
- Client to server: small JSON objects, collected with `poll_commands()`. Frames over 64 KB drop the connection.
- `routes={"/": html_bytes}` serves pages over plain HTTP on the same port. `local_only=True` rejects non-localhost `Host` and `Origin` headers (DNS-rebinding and cross-site protection).
- A stalled client is dropped after 0.25 s so it cannot freeze the camera loop.

## MotorLens (`handui/screening/`)
| File | Role |
| --- | --- |
| `signals.py` | `Sample(t, pinch, fist, palm)` and `sample_from_hand`. Palm position is in palm sizes. |
| `analysis.py` | `analyze_repetitive`, `analyze_tremor`, `observations`, `Thresholds`. numpy and scipy only. |
| `session.py` | `ScreeningSession`: waiting, countdown (3 s), running (10 s), finish. Emits JSON messages. `TESTS` defines the three tests. |
| `store.py` | `ResultStore`: append-only JSONL at `data/screening/results.jsonl` (gitignored). |
| `simulate.py` | `Simulator(profile)`: healthy, slowing, tremor synthetic signals. |
| `runner.py` | Main loop: camera or simulator -> session -> bridge; handles `hello`, `start`, `cancel` commands. |
| `dashboard.html` | The whole UI: test picker, live trace, result tiles and charts, baseline deltas, history, print report. |

Dashboard protocol: client sends `{"cmd":"hello"|"start"|"cancel"}`. Server sends `init` (tests, history, source), `signal` (every frame), `phase`, `aborted`, `result`.

Tests live in `tests/` (`test_gestures.py`, `test_bridge.py`, `test_screening.py`, helper `synthetic.py`).

## Brain Explorer (`handui/brain/`)
This folder is `frontend/` inside `finalized/`, next to `backend/adhd` and `backend/parkinsons` (FastAPI model servers) and `run.py`, which starts all three. Workflow: 1 choose a held-out patient from a dropdown (the test split only, served by each backend's `/heldout-samples`) -> 2 explore on a transparent brain (ADHD: regions lit by channel amplitude; Parkinson's: voice data, speech areas lit as a labelled reference) -> 3 Analyze, which calls the backend's `/analyze` (the addresses arrive in the init message as `apis`).
| File | Role |
| --- | --- |
| `eeg.py` | `CHANNELS`, `EEGENIUS_ORDER`, `electrode_position` (BESA spherical -> unit vector, x right ear, y nose, z up), `montage()`, `canonical()` channel names, `parse_text`/`load`/`from_mapping` (CSV or JSON; each row is one input), `summarise` (raw samples -> RMS). |
| `runner.py` | `BrainSession.handle(cmd)` answers `hello`, `eeg`, `eeg_text`, `open_menu`, `close_menu`; `hand_depth_message` (palm size / frame width, for zoom); `run_brain` is the camera loop. |
| `viewer.html` | Self-contained WebGL2 page. Additive "hologram" shading (fresnel, gyri, wire mesh), per-region Gaussian glow on the cortex, point lights sized by strength, network lines, three-step side panel, hand/mouse/keyboard input, palm-distance zoom. |

Viewer protocol: client sends `{"cmd":"hello"}` and gets `init` (montage, camera flag, eegenius_order) then `eeg` if data was preloaded. Gestures arrive as the bridge's `event`/`state` messages; `{"type":"hand_depth","size":...}` streams while a hand is visible. Strength = min-max of the values within one input.
