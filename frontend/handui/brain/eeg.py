"""10-20 electrode montage and EEG loading for the Brain Explorer.

Channel values are whatever one number per electrode the source provides (an
EEGenius snapshot, an amplitude, a band power). The viewer shows each channel
relative to the others, so the units only matter for the numbers in the panel.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The 19 channels EEGenius's ADHD model reads, in the order its /predict endpoint expects.
EEGENIUS_ORDER = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                  "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz")
# Display order: front to back, left to right.
CHANNELS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
            "P7", "P3", "Pz", "P4", "P8", "O1", "O2")
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}  # older 10-20 names

# BESA spherical coordinates (theta, phi) in degrees: theta from the vertex (Cz),
# negative on the left; phi the azimuth from the right ear (T8) toward the nose.
_BESA = {
    "Fp1": (-92, -72), "Fp2": (92, 72), "F7": (-92, -36), "F3": (-60, -51), "Fz": (46, 90),
    "F4": (60, 51), "F8": (92, 36), "T7": (-92, 0), "C3": (-46, 0), "Cz": (0, 0),
    "C4": (46, 0), "T8": (92, 0), "P7": (-92, 36), "P3": (-60, 51), "Pz": (46, -90),
    "P4": (60, -51), "P8": (92, -36), "O1": (-92, 72), "O2": (92, -72),
}

# Names match the EEGenius ADHD form.
NAMES = {
    "Fp1": "Left Prefrontal", "Fp2": "Right Prefrontal", "F7": "Left Frontal-Temporal",
    "F3": "Left Frontal", "Fz": "Frontal Midline", "F4": "Right Frontal",
    "F8": "Right Frontal-Temporal", "T7": "Left Mid-Temporal", "C3": "Left Central",
    "Cz": "Central Midline", "C4": "Right Central", "T8": "Right Mid-Temporal",
    "P7": "Left Posterior-Temporal", "P3": "Left Parietal", "Pz": "Parietal Midline",
    "P4": "Right Parietal", "P8": "Right Posterior-Temporal", "O1": "Left Occipital",
    "O2": "Right Occipital",
}

# What the cortex under each scalp site is generally associated with. Scalp EEG
# blurs over several centimetres, so this describes the area, not the exact source.
_REGIONS = {
    "prefrontal": ("Prefrontal cortex", "Planning, decision-making and keeping attention on a task."),
    "frontal": ("Frontal lobe", "Executive function, working memory and planning movement."),
    "frontal_temporal": ("Inferior frontal / anterior temporal", "Speech production on the left (near Broca's area); "
                         "emotion and social processing."),
    "central": ("Sensorimotor strip", "Movement and touch. Each side controls the opposite side of the body."),
    "central_mid": ("Sensorimotor strip, midline", "Leg and foot movement and touch; the top of the head."),
    "temporal": ("Temporal lobe", "Hearing and memory; language comprehension on the left."),
    "posterior_temporal": ("Posterior temporal", "Recognising objects and faces; language on the left."),
    "parietal": ("Parietal lobe", "Touch, spatial awareness and directing attention."),
    "occipital": ("Occipital lobe", "Vision. The alpha rhythm is strongest here when the eyes are closed."),
}
_REGION_OF = {
    "Fp1": "prefrontal", "Fp2": "prefrontal", "F3": "frontal", "F4": "frontal", "Fz": "frontal",
    "F7": "frontal_temporal", "F8": "frontal_temporal", "C3": "central", "C4": "central",
    "Cz": "central_mid", "T7": "temporal", "T8": "temporal", "P7": "posterior_temporal",
    "P8": "posterior_temporal", "P3": "parietal", "P4": "parietal", "Pz": "parietal",
    "O1": "occipital", "O2": "occipital",
}

MAX_FRAMES = 6000
_NAME_RE = re.compile(r"^(?:EEG[\s_-]*)?([A-Za-z]{1,2}\d{0,2}|[A-Za-z]{1,2}z)(?:[\s_-].*)?$")
_LOOKUP = {c.lower(): c for c in CHANNELS} | {k.lower(): v for k, v in ALIASES.items()}


def electrode_position(code: str) -> tuple[float, float, float]:
    """Unit vector in head coordinates: x toward the right ear, y toward the nose, z up."""
    theta, phi = (math.radians(a) for a in _BESA[code])
    return (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))


def montage() -> dict[str, dict]:
    out = {}
    for code in CHANNELS:
        region, about = _REGIONS[_REGION_OF[code]]
        x = electrode_position(code)[0]
        side = "midline" if abs(x) < 0.05 else ("left" if x < 0 else "right")
        out[code] = {"pos": [round(v, 4) for v in electrode_position(code)], "name": NAMES[code],
                     "region": region, "about": about, "side": side}
    return out


def canonical(name: str) -> str | None:
    """'EEG Fp1-REF', 'FP1' and 'fp1' -> 'Fp1'; old names like 'T3' -> 'T7'; anything else -> None."""
    m = _NAME_RE.match(str(name).strip())
    return _LOOKUP.get(m.group(1).lower()) if m else None


@dataclass
class Recording:
    channels: list[str]
    frames: np.ndarray       # (n_frames, n_channels)
    rate: float              # playback frames per second
    source: str              # "file" | "upload" | "external"
    name: str
    units: str = ""
    note: str | None = None

    def to_message(self) -> dict:
        return {"type": "eeg", "source": self.source, "name": self.name, "units": self.units,
                "rate": self.rate, "channels": self.channels, "note": self.note,
                "frames": np.round(self.frames, 3).tolist()}


# -------------------------------------------------------------------- parsing

def _numeric(value, where: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {value!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{where}: value must be finite")
    return v


def _from_columns(names: list, rows: list[list], where: str) -> tuple[list[str], np.ndarray]:
    picked: dict[str, int] = {}
    for i, n in enumerate(names):
        code = canonical(n)
        if code and code not in picked:
            picked[code] = i
    if not picked:
        raise ValueError(f"{where}: no 10-20 channel names found (expected names like Fp1, Cz, O2)")
    channels = [c for c in CHANNELS if c in picked]
    frames = np.array([[_numeric(row[picked[c]] if picked[c] < len(row) else "",
                                 f"{where} row {r + 1}, {c}") for c in channels]
                       for r, row in enumerate(rows)], dtype=np.float64)
    if len(frames) == 0:
        raise ValueError(f"{where}: no data rows")
    return channels, frames


def _from_snapshots(snapshots: list, where: str) -> tuple[list[str], np.ndarray]:
    if not all(isinstance(s, dict) for s in snapshots):
        raise ValueError(f"{where}: expected objects mapping channel names to values")
    names = list(dict.fromkeys(k for s in snapshots for k in s))
    return _from_columns(names, [[s.get(n, "") for n in names] for s in snapshots], where)


def parse_text(text: str, name: str = "data", source: str = "file", rate: float = 10.0,
               fs: float | None = None) -> Recording:
    """Read CSV (a header row of channel names, one row per frame) or JSON.

    JSON may be an EEGenius request {"data": {"Fp1": ...}}, a flat {"Fp1": ...},
    a list of such objects, or {"channels": [...], "frames": [[...], ...]}.
    With fs (Hz), rows are raw samples and are summarised into RMS amplitude
    over 0.5 s windows before display.
    """
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            obj = json.loads(stripped)
        except ValueError as e:
            raise ValueError(f"{name}: invalid JSON ({e})") from None
        if isinstance(obj, dict) and isinstance(obj.get("frames"), list) and isinstance(obj.get("channels"), list):
            channels, frames = _from_columns(obj["channels"], obj["frames"], name)
            rate = _numeric(obj.get("rate", rate), f"{name} rate")
        elif isinstance(obj, dict) and isinstance(obj.get("data"), dict):
            channels, frames = _from_snapshots([obj["data"]], name)
        elif isinstance(obj, dict):
            channels, frames = _from_snapshots([obj], name)
        elif isinstance(obj, list):
            channels, frames = _from_snapshots(obj, name)
        else:
            raise ValueError(f"{name}: unsupported JSON layout")
    else:
        rows = [r for r in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in r)]
        if not rows:
            raise ValueError(f"{name}: file is empty")
        channels, frames = _from_columns(rows[0], rows[1:], name)

    note = None
    if fs:
        frames, rate = summarise(frames, fs)
        note = f"RMS amplitude over 0.5 s windows of {fs:g} Hz samples"
    if len(frames) > MAX_FRAMES:
        step = math.ceil(len(frames) / MAX_FRAMES)
        frames, rate = frames[::step], rate / step
        note = f"{note + '; ' if note else ''}showing 1 frame in {step}"
    if not rate or rate <= 0:
        raise ValueError(f"{name}: playback rate must be positive")
    return Recording(channels, frames, float(rate), source, name, note=note)


def from_mapping(values: dict, name: str = "data", source: str = "external", units: str = "") -> Recording:
    """One snapshot such as EEGenius's {"Fp1": 12.0, ...}."""
    channels, frames = _from_snapshots([values], name)
    return Recording(channels, frames, 10.0, source, name, units)


def load(path: str | Path, rate: float = 10.0, fs: float | None = None) -> Recording:
    path = Path(path)
    return parse_text(path.read_text(encoding="utf-8-sig"), path.name, "file", rate, fs)


def summarise(samples: np.ndarray, fs: float, window: float = 0.5) -> tuple[np.ndarray, float]:
    """Raw samples (n, channels) -> RMS amplitude per window, after removing each window's mean."""
    n = max(1, int(round(fs * window)))
    usable = (len(samples) // n) * n
    if usable == 0:
        raise ValueError(f"need at least {n} samples for one {window} s window at {fs:g} Hz")
    blocks = samples[:usable].reshape(-1, n, samples.shape[1])
    blocks = blocks - blocks.mean(axis=1, keepdims=True)
    return np.sqrt((blocks ** 2).mean(axis=1)), 1.0 / window
