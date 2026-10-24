---
inclusion: always
---

# Product context

## What this repo is
**EEGenius Brain Explorer** is a hackathon project for neurological screening. You pick a patient the model has never seen, explore their data on a hand-tracked holographic 3D brain, and get a transparent AI analysis for ADHD (from EEG) or Parkinson's (from voice features).

It is built on **Hand UI**, a webcam hand-gesture interface written in Python (`handui/`). The gesture engine drives the 3D brain so the whole exploration is touchless, with mouse and keyboard as fallbacks.

The repo root holds the full product: two FastAPI model servers (`backend/adhd`, `backend/parkinsons`), the hand-tracking frontend (`frontend/handui`), and `run.py`, which starts all three.

## Principles
- **The hand must earn its place.** Gesture input is only worth building where touch or keyboard fails, or where exploring in 3D is genuinely better by hand.
- **Honest claims.** EEGenius is a research and demonstration tool, not a medical device, and never diagnoses. The app shows each model's real measured test performance next to its answer, including where a model is weak (the ADHD model performs about at chance on its held-out children, and the app says so).
- **Only unseen patients.** The patient dropdown lists people from each model's test split only, never anyone the model trained on. The tests prove this.
- **Local and private.** Video never leaves the machine. Only hand landmarks are analysed.
- **Real vs simulated data are never mixed.** Anything synthetic is labelled and stored with `source: "simulated"`.
- **Demo-proof.** Everything must run offline with sample data and `--no-camera` if the camera, wifi or an API fails.

## Users
Judges and anyone exploring the intersection of AI and neuroscience: choose a held-out ADHD or Parkinson's patient, explore their signals on the brain, and read a plain-language analysis with the model's honest track record.
