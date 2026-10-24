"""Backend tests: held-out lists, the new /analyze endpoints, and that nothing seen in training leaks in.

Run from the repo root:  .venv/bin/python -m unittest discover -s tests
"""
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent.parent / "backend"


def load_app(name):
    spec = importlib.util.spec_from_file_location(f"{name}_main", BACKEND / name / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, TestClient(module.app)


ADHD, adhd = load_app("adhd")
PD, pd_client = load_app("parkinsons")


class ADHDTests(unittest.TestCase):
    card = json.loads((BACKEND / "adhd" / "adhd_model_card.json").read_text())

    def test_heldout_list_is_exactly_the_test_children(self):
        samples = adhd.get("/heldout-samples").json()["samples"]
        subjects = {s["subject"] for s in samples}
        self.assertEqual(subjects, set(self.card["test_subjects"]))
        self.assertFalse(subjects & (set(self.card["train_subjects"]) | set(self.card["val_subjects"])))
        self.assertEqual(len(samples), self.card["test_metrics"]["n"])
        self.assertEqual(len({s["id"] for s in samples}), len(samples))

    def test_heldout_windows_reproduce_the_reported_test_accuracy(self):
        samples = ADHD.heldout["samples"]
        p = ADHD._probabilities(np.array([[s["data"][c] for c in ADHD.REGIONS] for s in samples]))
        y = np.array([s["true_label"] == "ADHD" for s in samples])
        self.assertAlmostEqual(float(((p >= 0.5) == y).mean()), self.card["test_metrics"]["accuracy"], delta=0.005)

    def test_analyze_matches_predict_and_explains_it(self):
        sample = ADHD.heldout["samples"][0]
        pred = adhd.post("/predict", json={"data": sample["data"]}).json()
        out = adhd.post("/analyze", json={"data": sample["data"], "sample_id": sample["id"]}).json()
        self.assertAlmostEqual(out["probability"], pred["probability"], places=6)
        self.assertEqual(out["prediction"], pred["prediction"])
        self.assertEqual([c["code"] for c in out["channels"]], ADHD.REGIONS)
        self.assertTrue(all(np.isfinite(c["effect"]) and np.isfinite(c["z"]) for c in out["channels"]))
        self.assertIn(out["margin"], ("low", "moderate", "high"))
        self.assertEqual(out["person"]["id"], sample["subject"])
        self.assertEqual(out["person"]["true_label"], sample["true_label"])
        self.assertAlmostEqual(out["model"]["test"]["auc"], self.card["test_metrics"]["auc"])
        json.dumps(out)

    def test_analyze_reports_missing_channels(self):
        self.assertIn("error", adhd.post("/analyze", json={"data": {"Fp1": 10}}).json())


class ParkinsonsTests(unittest.TestCase):
    card = json.loads((BACKEND / "parkinsons" / "parkinsons_full_model_card.json").read_text())

    def test_heldout_list_is_exactly_the_test_people(self):
        samples = pd_client.get("/heldout-samples").json()["samples"]
        people = {s["person_id"] for s in samples}
        self.assertEqual(people, set(self.card["test_ids"]))
        self.assertFalse(people & (set(self.card["train_ids"]) | set(self.card["val_ids"])))
        self.assertEqual(len(samples), self.card["test_metrics"]["n"])
        self.assertEqual(len(samples[0]["data"]), 27)

    def test_heldout_rows_reproduce_the_reported_test_metrics(self):
        ids = [s["id"] for s in PD.heldout["samples"]]
        p = PD.full_model.predict_proba(np.array([PD.HELD_ROWS[i] for i in ids]))[:, 1]
        y = np.array([PD.HELDOUT[i]["true_label"] == "Parkinsons" for i in ids])
        self.assertAlmostEqual(float(((p >= 0.5) == y).mean()), self.card["test_metrics"]["accuracy"], places=6)

    def test_analyze_by_id_matches_predict_full_and_the_27_feature_model(self):
        sample = PD.heldout["samples"][5]
        features = dict(zip(PD.FULL_FEATURES, PD.HELD_ROWS[sample["id"]]))
        full = pd_client.post("/predict-full", json={"features": features}).json()
        out = pd_client.post("/analyze", json={"sample_id": sample["id"]}).json()
        self.assertAlmostEqual(out["probability"], full["probability"], places=9)
        body = {("R_PDE" if k == "RPDE" else k): v for k, v in sample["data"].items()}
        small = pd_client.post("/predict", json=body).json()
        self.assertAlmostEqual(out["second_opinion"]["probability"], small["probability"], places=6)
        self.assertEqual(len(out["key_features"]), 27)
        self.assertLessEqual(len(out["drivers"]), 10)
        self.assertEqual(out["person"]["recordings"], sample["recordings"])
        self.assertEqual(out["person"]["true_label"], sample["true_label"])
        json.dumps(out)

    def test_analyze_with_raw_features_and_errors(self):
        sample = PD.heldout["samples"][0]
        features = dict(zip(PD.FULL_FEATURES, PD.HELD_ROWS[sample["id"]]))
        by_id = pd_client.post("/analyze", json={"sample_id": sample["id"]}).json()
        by_value = pd_client.post("/analyze", json={"features": features}).json()
        self.assertAlmostEqual(by_id["probability"], by_value["probability"], places=9)
        self.assertNotIn("person", by_value)
        self.assertIn("error", pd_client.post("/analyze", json={"sample_id": "p999-r1"}).json())
        self.assertIn("error", pd_client.post("/analyze", json={}).json())


if __name__ == "__main__":
    unittest.main()
