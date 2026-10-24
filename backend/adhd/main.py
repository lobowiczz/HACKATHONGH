# main.py - ADHD prediction API (serves the model trained by training/train_adhd.py)
import json
import os

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))


# --- 1. MODEL (same layout as training/common.py: 19 -> 64 -> 32 -> 1) ---
class NeuralNetwork_drop(nn.Module):
    def __init__(self, input_size=19):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)   # inactive in eval(); no weights, so the saved file loads either way
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


model = NeuralNetwork_drop()
model.load_state_dict(torch.load(os.path.join(HERE, "adhd_model_v2.pth"), map_location="cpu"))
model.eval()

# --- 2. SCALER + MODEL CARD + DEMO SAMPLES (written by the training script) ---
with open(os.path.join(HERE, "adhd_scaler.json")) as f:
    scaler = json.load(f)
REGIONS = scaler["channels"]                       # exact training column order
MEAN = np.array(scaler["means"])
SCALE = np.array(scaler["scales"])

with open(os.path.join(HERE, "adhd_model_card.json")) as f:
    model_card = json.load(f)
with open(os.path.join(HERE, "adhd_demo_samples.json")) as f:
    demo_samples = json.load(f)


class EEGData(BaseModel):
    # channel code -> amplitude (std dev, microvolts) over a 2-second window
    data: Dict[str, float]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict_adhd(eeg_data: EEGData):
    try:
        raw = np.array([eeg_data.data[r] for r in REGIONS], dtype=np.float64)
    except KeyError as e:
        return {"error": f"Missing data for EEG region: {e}"}

    # identical to training: log(std + 1), then train-set standardisation
    x = (np.log(np.clip(raw, 0, None) + 1.0) - MEAN) / SCALE
    with torch.no_grad():
        probability = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()

    return {"prediction": "ADHD" if probability >= 0.5 else "Control", "probability": probability}


@app.get("/demo-samples")
def get_demo_samples():
    """Real windows from children the model never saw during training or validation."""
    return demo_samples


@app.get("/model-info")
def get_model_info():
    c = model_card
    return {
        "trained_at": c["trained_at"],
        "split": c["split"],
        "n_subjects": c["n_subjects"],
        "cv_accuracy": c["cv_on_dev_subjects"]["accuracy"]["mean"],
        "cv_auc": c["cv_on_dev_subjects"]["auc"]["mean"],
        "test_accuracy": c["test_metrics"]["accuracy"],
        "test_auc": c["test_metrics"]["auc"],
    }


@app.get("/")
def read_root():
    return {"message": "ADHD API is operational.", "model_trained_at": model_card["trained_at"]}


# --- 3. HELD-OUT PATIENTS + FULL ANALYSIS (used by the Brain Explorer) ---
# adhd_heldout_samples.json is written by finalized/tools/build_heldout.py: every 2-second window
# from the children in the TEST split of adhd_model_card.json.
with open(os.path.join(HERE, "adhd_heldout_samples.json")) as f:
    heldout = json.load(f)
HELDOUT = {s["id"]: s for s in heldout["samples"]}
TYPICAL = np.exp(MEAN) - 1.0          # typical amplitude per channel in the training group (back from log space)


class AnalyzeRequest(BaseModel):
    data: Dict[str, float]
    sample_id: Optional[str] = None   # a held-out window id, to add a summary over that child's other windows


def _probabilities(raw_rows):
    """Same transform as /predict, for a batch of rows of raw amplitudes in REGIONS order."""
    x = (np.log(np.clip(raw_rows, 0, None) + 1.0) - MEAN) / SCALE
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy().ravel()


def _margin(p):
    d = abs(p - 0.5)
    return "low" if d < 0.1 else "moderate" if d < 0.3 else "high"


def _performance():
    c, t = model_card, model_card["test_metrics"]
    return {
        "name": "3-layer neural network on 19 EEG channel amplitudes",
        "dataset": c["dataset"], "trained_at": c["trained_at"], "split": c["split"], "n_people": c["n_subjects"],
        "cv": {"accuracy": c["cv_on_dev_subjects"]["accuracy"], "auc": c["cv_on_dev_subjects"]["auc"],
               "folds": c["cv_on_dev_subjects"]["folds"]},
        "test": {"n": t["n"], "accuracy": t["accuracy"], "balanced_accuracy": t["balanced_accuracy"], "auc": t["auc"],
                 "sensitivity": t["sensitivity"], "specificity": t["specificity"],
                 "person_level_accuracy": c["test_subject_level_accuracy"]},
    }


@app.get("/heldout-samples")
def get_heldout_samples():
    """Every window from the children in the test split. The model never saw any of them."""
    return heldout


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Prediction plus what drove it, how the input compares with the training group, and the model's track record."""
    try:
        raw = np.array([req.data[r] for r in REGIONS], dtype=np.float64)
    except KeyError as e:
        return {"error": f"Missing data for EEG region: {e}"}

    p = float(_probabilities(raw[None])[0])
    z = (np.log(np.clip(raw, 0, None) + 1.0) - MEAN) / SCALE
    # Effect of each channel: replace it with the training group's typical value, re-run, see how far p moves.
    swapped = np.repeat(raw[None], len(REGIONS), axis=0)
    idx = np.arange(len(REGIONS))
    swapped[idx, idx] = TYPICAL
    effect = p - _probabilities(swapped)
    order = np.argsort(effect)

    result = {
        "condition": "adhd",
        "prediction": "ADHD" if p >= 0.5 else "Control",
        "probability": p, "threshold": 0.5, "margin": _margin(p),
        "channels": [{"code": c, "value": float(raw[i]), "typical": round(float(TYPICAL[i]), 2),
                      "z": float(z[i]), "effect": float(effect[i])} for i, c in enumerate(REGIONS)],
        "toward_positive": [REGIONS[i] for i in order[::-1] if effect[i] > 0.005][:5],
        "toward_negative": [REGIONS[i] for i in order if effect[i] < -0.005][:5],
        "model": _performance(),
    }
    sample = HELDOUT.get(req.sample_id or "")
    if sample:
        windows = [s for s in heldout["samples"] if s["subject"] == sample["subject"]]
        probs = _probabilities(np.array([[s["data"][r] for r in REGIONS] for s in windows]))
        result["person"] = {"id": sample["subject"], "windows": len(windows), "mean_probability": float(probs.mean()),
                            "share_positive": float((probs >= 0.5).mean()), "true_label": sample["true_label"]}
    return result
