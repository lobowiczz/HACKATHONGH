# Tasks: Shellfish Watch (DRAFT)

- [ ] 1. Confirm the idea with the team and decide whether MotorLens and Shellfish Watch are separate entries
- [ ] 2. Verify data sources: pull one real sample from each candidate feed (buoy temperature and pH, DMF closures, bloom advisories) and record format, update rate and terms (_Requirements: 1.2, 1.5_)
- [ ] 3. Ask a grower or extension scientist for sensible risk rules and thresholds (_Requirements: 1.3_)
- [ ] 4. Create `handui/shellfish/` with `areas.py`, `sample.py` and a first `sources.py` returning bundled sample data (_Requirements: 1.1, 1.5, 4.2_)
- [ ] 5. Implement `risk.assess` with `RiskThresholds` and tests (_Requirements: 1.3, 1.4_)
- [ ] 6. Implement real fetchers with caching and fallback, behind a fake in tests (_Requirements: 1.2, 1.5_)
- [ ] 7. Implement `FieldLog` state machine and tests with synthetic events (_Requirements: 2.1 to 2.4_)
- [ ] 8. Record `thumbs_up` and `thumbs_down` samples, train the classifier, check held-out accuracy (_Requirements: 2.3_)
- [ ] 9. Add the log store and CSV export (_Requirements: 3.2, 4.1_)
- [ ] 10. Build `runner.py` with `--simulate`, keyboard fallback and `local_only` bridge (_Requirements: 2.6, 4.2, 4.3_)
- [ ] 11. Build `dashboard.html`: area picker, condition cards, risk banner, field-log panel, mortality markers (_Requirements: 1.2, 1.3, 2.5, 3.1_)
- [ ] 12. End-to-end check in the browser with the simulator, then on a real camera from the user's Terminal
- [ ] 13. Pitch prep: one-page story (problem, user, data, why gestures), honest limits, 90-second demo script
