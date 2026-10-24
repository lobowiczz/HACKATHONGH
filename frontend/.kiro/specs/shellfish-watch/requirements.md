# Requirements: Shellfish Watch (DRAFT)

Status: **draft, not started.** Chosen only as the recommended climate idea; the team has not confirmed it. Data sources are unverified.

## Introduction
Shellfish Watch helps Massachusetts shellfish growers (oysters, quahogs, scallops) in places like Wellfleet, Duxbury and Barnstable. It combines an early-warning view of water conditions with a hands-free field log driven by the Hand UI gestures, for wet, gloved hands that cannot use a touchscreen. Track: Climate Tech, "understanding and protecting the ocean" and "sustainable blue economy".

## Requirements

### 1. Conditions and risk dashboard
**User story:** As a grower, I want to see today's water-condition risk for my growing area so that I can plan seeding, harvest and handling.

1.1 THE SYSTEM SHALL let the user choose a growing area from a built-in list of Massachusetts areas.
1.2 THE SYSTEM SHALL show recent water temperature, pH (where available), chlorophyll or bloom indicator, and closure status as a trend with the latest value.
1.3 THE SYSTEM SHALL compute a green, amber or red risk level from documented rules and show which inputs drove it.
1.4 THE SYSTEM SHALL always link to the official Division of Marine Fisheries closure information and state that its risk level is advisory and not regulatory.
1.5 WHEN a data source is unavailable THE SYSTEM SHALL show the last cached value with its age, and WHEN there is no cache SHALL use bundled sample data clearly labelled "SAMPLE DATA".

### 2. Hands-free field log
**User story:** As a grower on a dock or in a hatchery, I want to log counts and condition without touching a screen so that I can record while working.

2.1 WHEN the user swipes left or right THE SYSTEM SHALL move to the previous or next tray or line.
2.2 WHEN the user swipes up or down THE SYSTEM SHALL increase or decrease the count by a set step.
2.3 WHEN the user performs the custom gestures `thumbs_up` or `thumbs_down` THE SYSTEM SHALL mark the tray healthy or note mortality.
2.4 WHEN the user holds an open palm THE SYSTEM SHALL open a menu to save the entry, undo, or export.
2.5 THE SYSTEM SHALL confirm each logged action with a large on-screen message and a distinct sound-free visual cue.
2.6 THE SYSTEM SHALL allow every gesture action to be done with the keyboard as a fallback.

### 3. Linking log and conditions
3.1 THE SYSTEM SHALL plot logged mortality notes against water-condition trends for the same area and dates.
3.2 THE SYSTEM SHALL export the log as CSV.

### 4. Data and operation
4.1 THE SYSTEM SHALL store the log locally in a gitignored file and send nothing to any server.
4.2 THE SYSTEM SHALL run offline with bundled sample data and a gesture simulator.
4.3 THE SYSTEM SHALL reuse the existing `Bridge`, gesture engine and custom-gesture classifier.
4.4 THE SYSTEM SHALL keep real and sample data separate and labelled.

## Out of scope
Regulatory closure decisions, harvest-legality advice, multi-user sync, forecasting models beyond simple rules.

## Open questions
- Which datasets are actually reachable (and how often updated) within the hackathon window? Verify NERACOOS and NOAA buoy feeds, Mass DMF closure notices, and any bloom advisories before design is final.
- What are sensible risk thresholds? Ask a grower or extension scientist; do not invent numbers in the pitch.
