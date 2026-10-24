# Tasks: MotorLens

## Done
- [x] 1. Bridge: client commands, same-port HTTP routes, local-only origin and host check (_Requirements: 5.1, 5.3_)
- [x] 2. Signals and analysis: tapping, open-close, tremor metrics (_Requirements: 2.1 to 2.6_)
- [x] 3. Session state machine and abort paths (_Requirements: 1.1 to 1.7_)
- [x] 4. Result store and simulator (_Requirements: 4.1, 4.2, 4.4, 5.2_)
- [x] 5. `screen` CLI command and runner loop (_Requirements: 5.1, 5.2_)
- [x] 6. Dashboard: picker, live trace, tiles, charts, baseline deltas, history, print (_Requirements: 3.1 to 3.5, 5.4_)
- [x] 7. Unit tests for analysis, session, store and bridge
- [x] 8. Verify all three tests end to end in a browser with the simulator

## Open
- [ ] 9. **Validate on a real hand and camera.** Run `python -m handui screen` from the user's Terminal; do each test twice and sanity-check the numbers.
- [ ] 10. **Calibrate thresholds on real recordings** (MediaPipe jitter sets the tremor noise floor). Update `Thresholds` and the matching tests.
- [ ] 11. Commit the work, including the earlier `--bridge` CLI fix, with no Claude trailer.
- [ ] 12. Add CSV export of history for clinicians (_Requirements: 3.3_)
- [ ] 13. Handle left and right hands in trend charts explicitly (currently only baselines are per hand)
- [ ] 14. Add a short guided demo script for judges: baseline session, then a second session with deliberately slower, smaller taps to show the change chips
- [ ] 15. Optional: short usage notes and screenshots in the README
