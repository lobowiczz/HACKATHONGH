# Design: MotorLens

## Overview
Python owns capture, timing and analysis; the browser owns presentation. They talk over the existing WebSocket bridge, and the bridge also serves the dashboard page.

```
camera/simulator -> signals.sample_from_hand -> ScreeningSession.update(sample, t)
                                                   |-> analysis.analyze_* -> ResultStore (JSONL)
                                                   '-> JSON messages -> Bridge.send -> dashboard.html
dashboard.html --{"cmd": hello|start|cancel}--> Bridge.poll_commands -> runner
```

## Components
- **`signals.py`**: `Sample(t, pinch, fist, palm)`. `pinch` = thumb-tip to index-tip distance over palm size. `fist` = `grab_strength` of finger extension (thumb ignored). `palm` = palm centre divided by palm size.
- **`analysis.py`**: pure functions.
  - `uniform()` resamples to a 30 Hz grid and reports coverage (share of grid points within 0.2 s of a real frame).
  - `analyze_repetitive()` uses `scipy.signal.find_peaks` with prominence at 25% of the 5th to 95th percentile range and a 0.1 s minimum spacing. Peak times get parabolic interpolation, because at 30 fps a 5 Hz cycle is 6 frames and raw peak times would inflate rhythm variability. Size change uses the inner cycles only.
  - `analyze_tremor()` uses Welch PSD (Hann, 128 samples, 50% overlap, linear detrend) summed over x and y. Peak search in 3 to 12 Hz; ratio against the median of 1 to 14 Hz; band RMS from the integrated PSD.
  - `observations()` turns metrics into plain-language notes from `Thresholds`.
- **`session.py`**: `ScreeningSession` state machine (`idle`, `waiting`, `countdown`, `running`). No clock of its own; `update()` returns the list of messages to send. `elapsed()` lets the simulator shape its signals per test.
- **`store.py`**: append-only JSONL; `history()` strips raw series.
- **`simulate.py`**: continuous synthetic signals. `slowing` shrinks amplitude to 40%, slows the rate and pauses once at 6.0 to 6.8 s. `tremor` adds a 5 Hz oscillation to palm position.
- **`runner.py`**: loop at camera rate (30 fps pacing in simulation), optional OpenCV preview window (`q` quits), opens the browser.

## Message protocol
Server to client: `init {source, profile, tests, history}`, `signal {t, present, pinch, fist, palm}` every frame, `phase {phase, test, remaining, total}` (about 10 Hz and on change), `aborted {reason}`, `result {record}`.
Result record: `{id, at, test, hand, source, duration, metrics, observations, series}`.

## Thresholds (illustrative, in `Thresholds`)
Size change of -20% or worse; hesitation = interval over 1.8x median; rhythm CV of 25% or more; coverage at least 0.8 and frame rate at least 20 fps; tremor needs peak ratio of 8 or more and band RMS of at least 0.5% of palm size. These are **not clinical cut-offs** and have only been checked against the simulator.

## Error handling
Hand loss, timeouts and low tracking quality are explicit (aborted reasons, quality flag). A damaged store line is skipped. A dropped dashboard connection reconnects after 1 s.

## Testing
`tests/test_screening.py`: analysis on known signals (rate, amplitude, decrement, pause, flat, tremor at 5 Hz, noise, drift, gaps), signal extraction from synthetic hands, every profile through every test, session abort and cancel paths, store round trip. `tests/test_bridge.py`: commands, HTTP routes, origin and host rejection.
