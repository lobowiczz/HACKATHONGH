# Requirements: MotorLens

Status: **implemented** (uncommitted at time of writing; verified with the simulator and 56 unit tests, not yet on a real hand and camera).

## Introduction
MotorLens is a webcam-based screening and tracking aid for hand motor function. A person performs three short guided tests; the system measures how they moved and tracks change against their own first session. It is not a medical device and does not diagnose.

## Requirements

### 1. Guided tests
**User story:** As a person tracking my hand function, I want short guided tests so that I can measure myself consistently.

1.1 WHEN the user starts a test THE SYSTEM SHALL wait until a hand has been steadily in view for 0.5 s, then run a 3 s countdown, then record for 10 s.
1.2 THE SYSTEM SHALL offer three tests: finger tapping (thumb-index distance), open and close (fist strength) and hold still (palm position).
1.3 WHEN no hand appears within 20 s of starting THE SYSTEM SHALL cancel the test with reason `no_hand`.
1.4 WHEN the hand is lost during the countdown THE SYSTEM SHALL return to waiting for a hand.
1.5 WHEN the hand is missing for more than 2 s during recording THE SYSTEM SHALL cancel the test with reason `hand_lost` and store nothing.
1.6 WHEN the user cancels THE SYSTEM SHALL return to idle and store nothing.
1.7 WHILE a test is not idle THE SYSTEM SHALL ignore requests to start another.

### 2. Movement metrics
**User story:** As a clinician or therapist, I want objective measures so that I can see change over time.

2.1 FOR tapping and open-close THE SYSTEM SHALL report cycle count, rate (Hz), mean movement size, size change from the first to the last third of cycles, rhythm variability (CV of inter-cycle intervals) and number of hesitations.
2.2 FOR hold-still THE SYSTEM SHALL report the dominant frequency in 3 to 12 Hz, band RMS size and peak ratio against the background spectrum.
2.3 THE SYSTEM SHALL express all sizes in palm sizes so results do not depend on camera distance.
2.4 THE SYSTEM SHALL exclude the first and last cycle when computing size change, because the recording window clips them.
2.5 THE SYSTEM SHALL report tracking quality (coverage and frame rate) and, WHEN quality is below threshold, advise repeating the test.
2.6 THE SYSTEM SHALL NOT flag a tremor for sensor noise or slow drift.

### 3. Results and trends
3.1 WHEN a test finishes THE SYSTEM SHALL show metric tiles, plain-language notes and charts (signal with counted cycles and per-cycle size; or position trace and spectrum).
3.2 THE SYSTEM SHALL compare each result with the first earlier session of the same test and hand, showing better or worse in colour.
3.3 THE SYSTEM SHALL list recent sessions and chart key metrics over time.
3.4 THE SYSTEM SHALL provide a print view for saving a report as PDF.
3.5 THE SYSTEM SHALL show a clear disclaimer that it is a screening aid and not a diagnosis.

### 4. Data handling
4.1 THE SYSTEM SHALL save each finished test as one JSON line in `data/screening/results.jsonl`, which is excluded from git.
4.2 THE SYSTEM SHALL tolerate a damaged line in that file without failing.
4.3 THE SYSTEM SHALL NOT transmit or store video.
4.4 THE SYSTEM SHALL label simulated runs with `source: "simulated"`, show a banner in the dashboard, and show only records matching the current source.

### 5. Operation
5.1 THE SYSTEM SHALL run with `python -m handui screen` and serve the dashboard on the same port as the bridge.
5.2 THE SYSTEM SHALL run without a camera using `--simulate healthy|slowing|tremor`.
5.3 THE SYSTEM SHALL refuse dashboard connections from other websites or foreign Host headers.
5.4 THE SYSTEM SHALL work offline, with no external scripts or fonts.

## Out of scope
Diagnosis, clinical scoring scales, multi-user accounts, cloud sync, two-hand tests.
