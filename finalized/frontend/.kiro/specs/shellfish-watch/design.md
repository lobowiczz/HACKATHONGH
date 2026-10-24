# Design: Shellfish Watch (DRAFT)

## Overview
Same shape as MotorLens: Python owns capture and logic, a self-contained HTML dashboard owns presentation, and the existing `Bridge` connects them and serves the page.

```
camera/simulator -> GestureEngine.update(hand, t) -> [Event]
                                                       -> FieldLog.handle(event, t) -> messages
sources.fetch(area) -> cache / bundled sample -> risk.assess(readings) -> messages
Bridge.send(...) -> dashboard.html ; dashboard --{"cmd": ...}--> Bridge.poll_commands
```

## New package `handui/shellfish/`
- **`areas.py`**: list of growing areas with name, coordinates and nearest data-source ids.
- **`sources.py`**: one small fetcher per source (buoy temperature and pH where available, bloom/closure information). Each returns `Reading(ts, kind, value, unit, source)`. Responses are cached to `data/shellfish/cache.json`. `sample.py` holds bundled sample series used when a fetch fails and no cache exists, flagged `sample=True`. Network code is isolated here so everything else is testable offline.
- **`risk.py`**: pure function `assess(readings, now) -> RiskReport(level, drivers[], stale_inputs[])`. Rule-based and transparent, with thresholds in a `RiskThresholds` dataclass. Placeholder values until reviewed by a domain expert.
- **`fieldlog.py`**: `FieldLog` state machine driven by engine `Event`s and a `t` argument: current tray, count, notes, undo stack, saved entries. Returns JSON messages. No clock or I/O inside.
- **`store.py`**: JSONL log at `data/shellfish/log.jsonl` (gitignored).
- **`runner.py`**: loop using `GestureEngine` (with the user's custom classifier for `thumbs_up` and `thumbs_down`), `FieldLog`, `Bridge(local_only=True, routes=...)`. Supports `--simulate` (scripted gesture events) and a keyboard fallback via dashboard commands.
- **`dashboard.html`**: area picker, condition cards with sparklines, risk banner with drivers, "SAMPLE DATA" and data-age labels, field-log panel with large current tray and count, mortality markers on the trend chart, CSV export.

## Gesture mapping
`swipe_left/right` select tray; `swipe_up/down` change count; `custom:thumbs_up` healthy; `custom:thumbs_down` mortality; `menu_open` opens the save/undo/export menu; grab confirms the highlighted menu item. Bindings are data in `handui/config.py`-style tables, not hard-coded.

## Message protocol (to be finalised)
Server: `init {areas, log}`, `conditions {area, readings, risk, stale, sample}`, `log_state {tray, count, status}`, `entry_saved {entry}`. Client: `{"cmd":"select_area"|"key"|"save"|"undo"|"export"}`.

## Error handling
Network failure gives cached data with age, then sample data labelled as such. Gesture ambiguity relies on the engine's existing hysteresis, cooldowns and custom-label voting. Every action has a keyboard path.

## Testing
Pure-function tests for `risk.assess` (boundaries, stale inputs, sample flag), `FieldLog` event sequences with synthetic events, store round trip, cache fallback with a fake fetcher. No test needs a camera or network.
