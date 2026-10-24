# Hand UI

A desktop interface you drive with one hand, using a webcam. It's built on MediaPipe hand landmarks and OpenCV, plus a small neural network that learns custom gestures.

| Gesture | What it does |
| --- | --- |
| **Fist or pinch** over a card | Grab it. Move to drag, open your hand to drop. Drop on the bin to delete. |
| **Swipe** an open hand left / right | Next / previous page |
| **Swipe** up / down | New card / tidy the page |
| **Hold an open palm** still (~0.7 s) | Toggle the radial menu. Point at an item, then grab to choose it. |
| **Custom gestures** you record | Whatever you bind them to in `handui/config.py` |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads MediaPipe's hand landmark model (~7.5 MB) into `models/`.
On macOS, allow camera access for your terminal in **System Settings > Privacy & Security > Camera**.

## Run

```bash
.venv/bin/python -m handui
```

Keys: `q` quit · `c` camera background · `s` skeleton · `h` HUD · `m` menu · `n` new card · `t` tidy · `a`/`d` pages.

## Custom gestures

1. Record each gesture for about 8 seconds. While recording, move your hand around, tilt it, and change its distance from the camera.
   ```bash
   .venv/bin/python -m handui record thumbs_up
   .venv/bin/python -m handui record peace
   .venv/bin/python -m handui record none   # relaxed, random poses, so normal movement isn't misread
   ```
2. Train. Accuracy is measured on the last 20% of each recording, which the model doesn't see during training.
   ```bash
   .venv/bin/python -m handui train
   ```
3. Bind labels to actions in `BINDINGS` in `handui/config.py`, for example `"custom:thumbs_up": "new_card"`. Available actions are listed in `HandUI.ACTIONS`.

`python -m handui labels` lists what you've recorded. `python -m handui forget LABEL` deletes the samples for one label.

Features are invariant to position, scale and handedness, but not to rotation, so thumbs-up and thumbs-down stay different gestures.

## MotorLens: hand-function screening

A guided webcam check of how well one hand moves, built on the same tracker. Three 10 second tests, modelled on the finger-tapping, hand open-close and rest-tremor items clinicians use for Parkinson's, stroke recovery and arthritis:

| Test | Signal | Metrics |
| --- | --- | --- |
| Finger tapping | thumb-index distance | taps, speed, movement size, size change (first to last third), rhythm variability, pauses |
| Open and close | fist strength | same as tapping |
| Hold still | palm position | dominant frequency (3 to 12 Hz), tremor-band size, peak strength |

```bash
.venv/bin/python -m handui screen                    # camera; opens the dashboard at http://127.0.0.1:8765
.venv/bin/python -m handui screen --simulate slowing # no camera: healthy | slowing | tremor
```

Pick a test in the dashboard and press Start. It waits for your hand, counts down 3 seconds, records 10, then shows the result, charts and a comparison with your first session (the baseline). Results are appended to `data/screening/results.jsonl` (gitignored, local only). Simulated runs are stored with `source: "simulated"`, labelled in the dashboard, and never mixed with real ones. **Print report** saves a PDF.

All sizes are in palm widths, so distance from the camera doesn't matter. The thresholds behind the plain-language notes are illustrative defaults in `Thresholds` in `handui/screening/analysis.py`, not clinical cut-offs: use the trend against your own baseline, and don't read a single result as a diagnosis.

Code: `signals.py` (landmarks to three signals), `analysis.py` (metrics), `session.py` (test flow), `store.py`, `simulate.py`, `runner.py` (main loop), `dashboard.html`. The dashboard talks to Python over the bridge below; `local_only` mode refuses other websites.

## Brain Explorer: EEG on a hand-navigated 3D brain

The Brain Explorer is the frontend of the EEGenius product in this repo. See [`../README.md`](../README.md) for the whole system: the model servers in [`../backend`](../backend) and the single `run.py` launcher.

1. **Choose a patient:** pick ADHD (EEG, 19 channels) or Parkinson's (voice recording), then a person from a dropdown. It shows three at a time and you scroll for the rest. It lists only people from each model's test split, never seen in training. Picking one fills in the values.
2. **Explore:** a transparent, glowing brain. For ADHD each channel lights its region by amplitude, with network lines between regions. The Parkinson's data has no electrode positions, so the speech areas are lit as a labelled reference.
3. **Analyze:** *Run prediction* calls the model server's `/analyze`. It shows the prediction, what drove it, how the input compares with the training group, the model's real test record, the dataset label, and what the result means.

| Gesture | Action |
| --- | --- |
| Fist (or pinch) and move | Rotate the brain |
| Move your palm closer / further | Zoom in / out |
| Quick fist on a light | Select that region |
| Swipe left / right | Next / previous region, strongest first |
| Swipe up / down | Next / previous step |
| Hold an open palm | Menu: overview, strongest, labels, network, hand zoom on/off, analyze |

Mouse and keyboard work too (drag, click, scroll; arrows, 1-3, M, L, N, Z, R). To run the frontend alone, use `python -m handui brain --adhd-api URL --parkinsons-api URL`. The page gets the model server addresses from the init message.

Code: `handui/brain/eeg.py` (montage, loaders), `runner.py` (camera loop, hand-distance signal, viewer commands), `viewer.html` (WebGL hologram renderer, generated brain mesh and the three-step UI, in one offline file).

## Driving another program

Run with the bridge and any browser page or script can follow your hand over a local WebSocket (standard library only, no extra install):

```bash
.venv/bin/python -m handui --bridge        # ws://127.0.0.1:8765, or --bridge 9000 for another port
```

Clients can also send small JSON commands, which the host collects with `Bridge.poll_commands()`. Open `examples/bridge_demo.html` for a minimal client (a dot that follows the cursor, plus an event log). The server is one-way and listens on localhost only. Every message is JSON:

| Message | Fields |
| --- | --- |
| `{"type": "hello"}` | `version`, `state`; sent once on connect |
| `{"type": "event"}` | `kind` (`hand_found`, `hand_lost`, `grab_start`, `grab_move`, `grab_end`, `swipe`, `menu_open`, `menu_close`, `custom`), `data` (same as the engine's `Event.data`), `t` |
| `{"type": "state"}` | `t` plus every `GestureState` field: `present`, `handedness`, `cursor` (`[x, y]`, 0..1, mirrored), `extension`, `grab_strength`, `pinch`, `speed`, `grabbing`, `palm_open`, `menu_open`, `menu_progress`, `custom_label`, `custom_confidence`; throttled to 30 Hz |

```js
const ws = new WebSocket("ws://127.0.0.1:8765");
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.type === "event" && m.kind === "swipe") console.log("swipe", m.data.direction);
};
```

Bindings in `BINDINGS` only drive the built-in UI; a connected program decides for itself what each event means. The bridge lives in `handui/bridge.py` and is fed from `app.py`, so the engine and `HandUI` don't know about it.

## How it works

```
camera -> tracker.py      MediaPipe HandLandmarker -> 21 landmarks per hand
       -> features.py     finger extension, grab strength, pinch, classifier features
       -> gestures.py     GestureEngine: grab/drag, swipe, palm-dwell menu, custom labels -> Events
       -> ui.py           HandUI: cards, pages, radial menu, HUD (pure OpenCV drawing)
classifier.py             sample storage, augmentation, MLP training, inference
```

The cursor follows the palm centre, not a fingertip, so it stays put when you close your hand to grab. It's smoothed with a One Euro filter. Every threshold is in `GestureConfig` in `handui/config.py`.

## Tests

The tests drive the engine, classifier and UI with synthetic hands, so they don't need a camera:

```bash
.venv/bin/python -m unittest discover -s tests
```
