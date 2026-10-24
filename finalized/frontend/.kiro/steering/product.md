---
inclusion: always
---

# Product context

## What this repo is
**Hand UI** is a webcam hand-gesture interface written in Python. **MotorLens** is a hackathon project built on top of it: a guided hand-function screening tool with a browser dashboard. A second, undecided hackathon project (climate, Massachusetts) is drafted in `.kiro/specs/shellfish-watch/`.

## Hackathon
Software-only. Tracks considered: Education, Civic Education, Biotech/Healthcare, Energy, and **Track 3: Climate Tech**:
- Resilient communities in a changing climate
- Cleaner, more affordable buildings
- A more reliable clean energy system
- Understanding and protecting the ocean
- Sustainable blue economy and coastal livelihoods

MotorLens targets Biotech/Healthcare. The climate draft targets "understanding and protecting the ocean" plus "sustainable blue economy" for Massachusetts.

## Principles
- **The hand must earn its place.** Gesture input is only worth building where touch or keyboard fails (gloved, wet or sterile hands) or where the hand itself is the data (MotorLens).
- **Honest claims.** MotorLens is a screening and tracking aid, not a medical device, and never diagnoses. Its thresholds are illustrative and uncalibrated; the meaningful comparison is a person against their own earlier sessions.
- **Local and private.** Video never leaves the machine. Only hand landmarks are analysed. Results go to a gitignored local file.
- **Real vs simulated data are never mixed.** Anything synthetic is labelled and stored with `source: "simulated"`.
- **Demo-proof.** Everything must run offline with a simulator or sample data if the camera, wifi or an API fails.

## Users
- MotorLens: people tracking motor function (Parkinson's, stroke recovery, arthritis) and the clinicians or therapists who see their trends.
- Shellfish Watch (draft): Massachusetts shellfish growers working on docks and in hatcheries.
