"""
Build the held-out patient lists that the Brain Explorer's dropdown offers.

Only rows from the TEST people named in each model card are included: people the
model never saw (not in training, not in validation, not in early stopping). The
rows are rebuilt with the same code as training/train_adhd.py and
training/train_parkinsons*.py, and the script checks itself against the demo
samples those training scripts wrote.

Usage (from the repo root):
  .venv/bin/python tools/build_heldout.py \
      --adhd-csv ~/Pictures/EEG/eeg/adhdata.csv --pd-csv ~/Pictures/EEG/eeg/pd_EEG_features.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ADHD_DIR, PD_DIR = ROOT / "backend" / "adhd", ROOT / "backend" / "parkinsons"
CH = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
WIN = 256   # samples per window = 2 s at 128 Hz, as in train_adhd.py


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


def build_adhd(csv):
    card = load(ADHD_DIR / "adhd_model_card.json")
    test = set(card["test_subjects"])
    assert not test & (set(card["train_subjects"]) | set(card["val_subjects"])), "test subjects overlap training"
    assert card["window_samples"] == WIN

    df = pd.read_csv(csv)
    samples = []
    for s, g in df.groupby("ID", sort=False):
        if s not in test:
            continue
        a, label = g[CH].values.astype(np.float64), "ADHD" if g.Class.iloc[0] == "ADHD" else "Control"
        n = len(a) // WIN
        for i in range(n):
            std = a[i * WIN:(i + 1) * WIN].std(axis=0)   # amplitude (std dev, uV) per channel, exactly as in training
            samples.append({"id": f"{s}-w{i + 1:03d}", "subject": s, "window": i + 1, "windows": n,
                            "true_label": label, "data": {c: round(float(v), 2) for c, v in zip(CH, std)}})
    samples.sort(key=lambda x: (x["subject"], x["window"]))

    assert len(samples) == card["test_metrics"]["n"], (len(samples), card["test_metrics"]["n"])
    assert {x["subject"] for x in samples} == test
    for d in load(ADHD_DIR / "adhd_demo_samples.json"):   # every demo window must be one of ours
        assert any(x["subject"] == d["subject"] and all(abs(x["data"][c] - d["data"][c]) < 0.02 for c in CH)
                   for x in samples), f"demo window for {d['subject']} not reproduced"
    dump(ADHD_DIR / "adhd_heldout_samples.json", {
        "dataset": card["dataset"], "split": card["split"], "trained_at": card["trained_at"],
        "unit": "amplitude (std dev, uV) of each channel over a 2-second window",
        "n_people": len(test), "samples": samples})


def build_parkinsons(csv):
    card, full_card = load(PD_DIR / "parkinsons_model_card.json"), load(PD_DIR / "parkinsons_full_model_card.json")
    assert card["test_ids"] == full_card["test_ids"] and card["train_ids"] == full_card["train_ids"]
    test, train = set(card["test_ids"]), set(card["train_ids"])
    assert not test & (train | set(card["val_ids"])), "test people overlap training"
    key = load(PD_DIR / "scaling_params_v2.json")["features"]

    df = pd.read_csv(csv)
    full = [c for c in df.columns if c not in ("id", "class")]
    assert list(joblib.load(PD_DIR / full_card["model_file"])["features"]) == full, "full model feature order differs"

    samples, rows = [], {}
    for pid, g in df[df.id.isin(test)].groupby("id", sort=True):
        for k, (_, r) in enumerate(g.iterrows(), 1):
            sid = f"p{int(pid):03d}-r{k}"
            samples.append({"id": sid, "person_id": int(pid), "recording": k, "recordings": len(g),
                            "true_label": "Parkinsons" if int(r["class"]) == 1 else "Control",
                            "data": {f: float(r[f]) for f in key}})
            rows[sid] = [float(r[f]) for f in full]

    assert len(samples) == card["test_metrics"]["n"], (len(samples), card["test_metrics"]["n"])
    for d in load(PD_DIR / "parkinsons_demo_samples.json"):
        assert any(x["person_id"] == d["person_id"] and all(abs(x["data"][f] - d["data"][f]) < 1e-9 for f in key)
                   for x in samples), f"demo recording for person {d['person_id']} not reproduced"

    train_rows = df[df.id.isin(train)][full].values.astype(np.float64)
    dump(PD_DIR / "parkinsons_heldout_samples.json", {
        "dataset": card["dataset"], "split": card["split"], "trained_at": full_card["trained_at"],
        "unit": "voice-recording features (Sakar et al. Parkinson's speech dataset)",
        "n_people": len(test), "samples": samples})
    dump(PD_DIR / "parkinsons_heldout_full.json", {"features": full, "rows": rows})
    # typical training values, used to measure how much each feature moved a prediction
    dump(PD_DIR / "parkinsons_train_reference.json", {"features": full, "median": np.median(train_rows, 0).tolist(),
                                                      "iqr": (np.percentile(train_rows, 75, 0) - np.percentile(train_rows, 25, 0)).tolist()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adhd-csv", required=True)
    ap.add_argument("--pd-csv", required=True)
    args = ap.parse_args()
    build_adhd(args.adhd_csv)
    build_parkinsons(args.pd_csv)
    print("held-out lists built and checked against the training scripts' demo samples")
