# main.py - Parkinson's prediction API (serves the model trained by training/train_parkinsons.py)
# The prediction is the neural network's output. There is no hand-written scoring formula.
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


class NeuralNetwork_drop(nn.Module):
    def __init__(self, input_size=27):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)   # inactive in eval(); no weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


model = NeuralNetwork_drop(27)
model.load_state_dict(torch.load(os.path.join(HERE, "parkinsons_model_v2.pth"), map_location="cpu"))
model.eval()

with open(os.path.join(HERE, "scaling_params_v2.json")) as f:
    sp = json.load(f)
FEATURES = sp["features"]                          # exact training column order (27)
MEAN = np.array(sp["means"])
SCALE = np.array(sp["scales"])
assert len(FEATURES) == len(MEAN) == len(SCALE) == 27

with open(os.path.join(HERE, "parkinsons_model_card.json")) as f:
    model_card = json.load(f)
with open(os.path.join(HERE, "parkinsons_demo_samples.json")) as f:
    demo_samples = json.load(f)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientData(BaseModel):
    gender: float
    PPE: float
    EEG: float            # training column "EEG" (= DFA, detrended fluctuation analysis, in the source dataset)
    R_PDE: float          # training column "RPDE"
    numPulses: float
    numPeriodsPulses: float
    meanPeriodPulses: float
    stdDevPeriodPulses: float
    f1: float
    f2: float
    f3: float
    f4: float
    b1: float
    b2: float
    b3: float
    b4: float
    Ea: float
    Ed_1_coef: float
    Ed_2_coef: float
    Ed_3_coef: float
    Ed_4_coef: float
    Ed_5_coef: float
    Ed_6_coef: float
    Ed_7_coef: float
    Ed_8_coef: float
    Ed_9_coef: float
    Ed_10_coef: float


@app.post("/predict")
def predict_parkinsons(data: PatientData):
    values = data.model_dump()
    values["RPDE"] = values.pop("R_PDE")
    x = (np.array([values[f] for f in FEATURES], dtype=np.float64) - MEAN) / SCALE
    with torch.no_grad():
        probability = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
    return {"prediction": "Parkinsons" if probability >= 0.5 else "Control", "probability": probability}


@app.get("/demo-samples")
def get_demo_samples():
    """Real recordings from people the model never saw during training or validation."""
    return demo_samples


@app.get("/model-info")
def get_model_info():
    c = model_card
    return {
        "trained_at": c["trained_at"],
        "split": c["split"],
        "n_people": c["n_people"],
        "majority_baseline": c["majority_baseline"],
        "cv_accuracy": c["cv_on_dev_people"]["accuracy"]["mean"],
        "cv_auc": c["cv_on_dev_people"]["auc"]["mean"],
        "test_accuracy": c["test_metrics"]["accuracy"],
        "test_auc": c["test_metrics"]["auc"],
    }


# --- FULL-FEATURE MODEL (all 753 voice features; trained by training/train_parkinsons_full.py) ---
# Separate endpoints, so the 27-feature /predict above keeps working unchanged.
try:
    import joblib
    with open(os.path.join(HERE, "parkinsons_full_model_card.json")) as f:
        full_card = json.load(f)
    _bundle = joblib.load(os.path.join(HERE, full_card["model_file"]))
    full_model, FULL_FEATURES = _bundle["model"], _bundle["features"]
    with open(os.path.join(HERE, "parkinsons_full_demo_samples.json")) as f:
        full_demo = json.load(f)
except Exception as e:  # keep the 27-feature API alive even if the full model files are missing
    full_card, full_model, FULL_FEATURES, full_demo = None, None, [], []
    print(f"Full-feature model not loaded: {e}")


class FullPatientData(BaseModel):
    features: dict[str, float]    # feature name -> value; must contain all 753 names in /full-features


@app.post("/predict-full")
def predict_full(data: FullPatientData):
    if full_model is None:
        return {"error": "Full-feature model is not available on this server."}
    missing = [f for f in FULL_FEATURES if f not in data.features]
    if missing:
        return {"error": f"Missing {len(missing)} of {len(FULL_FEATURES)} features, e.g. {missing[:5]}"}
    x = np.array([[data.features[f] for f in FULL_FEATURES]], dtype=np.float64)
    probability = float(full_model.predict_proba(x)[0, 1])
    return {"prediction": "Parkinsons" if probability >= 0.5 else "Control", "probability": probability}


@app.get("/full-features")
def full_features():
    """The exact feature names (and order) /predict-full expects."""
    return FULL_FEATURES


@app.get("/demo-samples-full")
def get_demo_samples_full():
    """Real recordings from people the full model never saw during training or validation."""
    return full_demo


@app.get("/model-info-full")
def get_model_info_full():
    if full_card is None:
        return {"error": "Full-feature model is not available on this server."}
    c = full_card
    return {
        "trained_at": c["trained_at"], "split": c["split"], "n_people": c["n_people"],
        "n_features": c["n_features"], "model": c["selected_candidate"],
        "majority_baseline_test": c["majority_baseline_test"],
        "cv_accuracy": c["cv_on_dev_people_per_candidate"][c["selected_candidate"]]["accuracy"]["mean"],
        "cv_auc": c["cv_on_dev_people_per_candidate"][c["selected_candidate"]]["auc"]["mean"],
        "test_accuracy": c["test_metrics"]["accuracy"], "test_auc": c["test_metrics"]["auc"],
        "test_balanced_accuracy": c["test_metrics"]["balanced_accuracy"],
    }


@app.get("/")
def read_root():
    return {"message": "EEGenius Parkinson's API is operational.", "model_trained_at": model_card["trained_at"]}


# --- HELD-OUT PATIENTS + FULL ANALYSIS (used by the Brain Explorer) ---
# Written by finalized/tools/build_heldout.py from the TEST people in the model cards (same split for both models).
with open(os.path.join(HERE, "parkinsons_heldout_samples.json")) as f:
    heldout = json.load(f)
with open(os.path.join(HERE, "parkinsons_heldout_full.json")) as f:
    _held_full = json.load(f)
with open(os.path.join(HERE, "parkinsons_train_reference.json")) as f:
    _reference = json.load(f)
HELDOUT = {s["id"]: s for s in heldout["samples"]}
HELD_ROWS = _held_full["rows"]
TYPICAL = np.array(_reference["median"])     # training people's median of each of the 753 features
assert full_model is None or _held_full["features"] == _reference["features"] == list(FULL_FEATURES), \
    "held-out files don't match the model"

LABELS = {
    "gender": "Sex (1 = male)", "PPE": "Pitch period entropy", "EEG": "DFA (voice signal fractal scaling)",
    "RPDE": "Recurrence period density entropy", "numPulses": "Glottal pulses", "numPeriodsPulses": "Pitch periods",
    "meanPeriodPulses": "Mean pitch period (s)", "stdDevPeriodPulses": "Pitch period variation (s)",
    **{f"f{i}": f"Formant {i} (Hz)" for i in range(1, 5)}, **{f"b{i}": f"Formant {i} bandwidth (Hz)" for i in range(1, 5)},
    "Ea": "Wavelet approximation energy", **{f"Ed_{i}_coef": f"Wavelet detail energy, level {i}" for i in range(1, 11)},
}
FAMILIES = [   # (prefix test, family) checked in order
    (lambda n: n.startswith("tqwt"), "TQWT wavelet features"),
    (lambda n: n.startswith(("mean_", "std_")), "MFCC spectral shape"),
    (lambda n: n.startswith(("det_", "app_", "Ea", "Ed")), "Wavelet features"),
    (lambda n: n.startswith(("GQ_", "GNE_", "VFER_", "IMF_")), "Vocal fold function"),
    (lambda n: n.endswith("Jitter"), "Jitter (pitch instability)"),
    (lambda n: n.endswith("Shimmer"), "Shimmer (loudness instability)"),
    (lambda n: n.endswith("Harmonicity"), "Harmonicity (voice clarity)"),
    (lambda n: n.endswith("Intensity"), "Loudness"),
    (lambda n: n[:1] in "fb" and n[1:].isdigit(), "Formants (vocal tract)"),
    (lambda n: n == "gender", "Sex"),
]


def _family(name):
    return next((fam for test, fam in FAMILIES if test(name)), "Baseline voice measures")


class AnalyzeRequest(BaseModel):
    sample_id: Optional[str] = None             # a held-out recording id from /heldout-samples
    features: Optional[Dict[str, float]] = None  # or all 753 features from /full-features


def _margin(p):
    d = abs(p - 0.5)
    return "low" if d < 0.1 else "moderate" if d < 0.3 else "high"


def _performance():
    c, t = full_card, full_card["test_metrics"]
    return {
        "name": f"Gradient-boosted trees on {c['n_features']} voice features",
        "dataset": c["dataset"], "trained_at": c["trained_at"], "split": c["split"], "n_people": c["n_people"],
        "majority_baseline": c["majority_baseline_test"],
        "cv": {"accuracy": c["cv_on_dev_people_per_candidate"][c["selected_candidate"]]["accuracy"],
               "auc": c["cv_on_dev_people_per_candidate"][c["selected_candidate"]]["auc"], "folds": 5},
        "test": {"n": t["n"], "accuracy": t["accuracy"], "balanced_accuracy": t["balanced_accuracy"], "auc": t["auc"],
                 "sensitivity": t["sensitivity"], "specificity": t["specificity"]},
    }


@app.get("/heldout-samples")
def get_heldout_samples():
    """Every recording from the people in the test split, with the 27 headline features. The models never saw them."""
    return {**heldout, "labels": {f: LABELS[f] for f in FEATURES}}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Full-model prediction plus what drove it, the 27-feature model's second opinion and the model's track record."""
    if full_model is None:
        return {"error": "Full-feature model is not available on this server."}
    if req.sample_id is not None:
        if req.sample_id not in HELD_ROWS:
            return {"error": f"Unknown held-out recording: {req.sample_id}"}
        x = np.array(HELD_ROWS[req.sample_id], dtype=np.float64)
    elif req.features is not None:
        missing = [f for f in FULL_FEATURES if f not in req.features]
        if missing:
            return {"error": f"Missing {len(missing)} of {len(FULL_FEATURES)} features, e.g. {missing[:5]}"}
        x = np.array([req.features[f] for f in FULL_FEATURES], dtype=np.float64)
    else:
        return {"error": "Send a sample_id or all features."}

    p = float(full_model.predict_proba(x[None])[0, 1])
    # Effect of each feature: replace it with the training people's median, re-run, see how far p moves.
    swapped = np.repeat(x[None], len(FULL_FEATURES), axis=0)
    idx = np.arange(len(FULL_FEATURES))
    swapped[idx, idx] = TYPICAL
    effect = p - full_model.predict_proba(swapped)[:, 1]
    families = {}
    for name, e in zip(FULL_FEATURES, effect):
        families[_family(name)] = families.get(_family(name), 0.0) + float(e)

    # second opinion: the 27-feature neural network (the weaker model) on the same recording
    values = dict(zip(FULL_FEATURES, x))
    key = np.array([values[f] for f in FEATURES])
    with torch.no_grad():
        p27 = float(model(torch.tensor((key - MEAN) / SCALE, dtype=torch.float32).unsqueeze(0)).item())

    result = {
        "condition": "parkinsons",
        "prediction": "Parkinsons" if p >= 0.5 else "Control",
        "probability": p, "threshold": 0.5, "margin": _margin(p),
        "drivers": [{"name": FULL_FEATURES[i], "label": LABELS.get(FULL_FEATURES[i], FULL_FEATURES[i]),
                     "family": _family(FULL_FEATURES[i]), "value": float(x[i]), "typical": float(TYPICAL[i]),
                     "effect": float(effect[i])}
                    for i in np.argsort(-np.abs(effect))[:10] if abs(effect[i]) > 1e-4],
        "families": sorted(({"family": k, "effect": v} for k, v in families.items()), key=lambda d: -abs(d["effect"])),
        "key_features": [{"name": f, "label": LABELS[f], "value": float(key[i]), "z": float((key[i] - MEAN[i]) / SCALE[i])}
                         for i, f in enumerate(FEATURES)],
        "second_opinion": {"name": "Neural network on the 27 headline features", "probability": p27,
                           "prediction": "Parkinsons" if p27 >= 0.5 else "Control",
                           "test_auc": model_card["test_metrics"]["auc"]},
        "model": _performance(),
    }
    sample = HELDOUT.get(req.sample_id or "")
    if sample:
        ids = [s["id"] for s in heldout["samples"] if s["person_id"] == sample["person_id"]]
        probs = full_model.predict_proba(np.array([HELD_ROWS[i] for i in ids]))[:, 1]
        result["person"] = {"id": sample["person_id"], "recordings": len(ids), "mean_probability": float(probs.mean()),
                            "share_positive": float((probs >= 0.5).mean()), "true_label": sample["true_label"]}
    return result
