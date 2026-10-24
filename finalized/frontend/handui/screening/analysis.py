"""Movement metrics from short recordings of the hand.

Three tests, each a 10 second recording of one signal:
  tapping     thumb-index distance while tapping as big and fast as possible
  open_close  fist strength while opening and closing the hand
  tremor      palm position while holding the hand still

All amplitudes are in palm sizes (hand-width units) so they do not depend on how far
the hand is from the camera. The thresholds in Thresholds are illustrative defaults for
plain-language observations; they are not clinical cut-offs. The metrics are most
meaningful as change against the same person's earlier sessions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, welch

FPS = 30.0
MAX_GAP = 0.2  # seconds without a tracked frame that counts as missing data


@dataclass
class Thresholds:
    min_span: float = 0.15            # signal range below this means the hand barely moved
    prominence_frac: float = 0.25     # a cycle must reach this fraction of the signal range
    min_cycle_s: float = 0.1          # fastest cycle counted (10 Hz)
    decrement_pct: float = -20.0      # amplitude change first third -> last third worth noting
    hesitation_ratio: float = 1.8     # interval longer than this x the median = a pause
    rhythm_cv_pct: float = 25.0       # irregular rhythm
    min_coverage: float = 0.8         # fraction of the test with a tracked hand
    min_fps: float = 20.0
    tremor_band: tuple[float, float] = (3.0, 12.0)
    tremor_peak_ratio: float = 8.0    # spectral peak vs median spectrum
    tremor_min_rms_pct: float = 0.5   # band RMS, % of palm size


DEFAULTS = Thresholds()


def uniform(ts, values, t0: float, duration: float, fps: float = FPS):
    """Resample onto a uniform grid. Returns (seconds from t0, values, coverage in 0..1)."""
    ts = np.asarray(ts, dtype=float)
    values = np.asarray(values, dtype=float)
    grid = t0 + np.arange(int(round(duration * fps))) / fps
    if len(ts) < 2:
        return grid - t0, np.zeros((len(grid),) + values.shape[1:]), 0.0
    idx = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
    nearest = np.minimum(np.abs(grid - ts[idx - 1]), np.abs(ts[idx] - grid))
    coverage = float((nearest <= MAX_GAP).mean())
    if values.ndim == 1:
        out = np.interp(grid, ts, values)
    else:
        out = np.column_stack([np.interp(grid, ts, values[:, k]) for k in range(values.shape[1])])
    return grid - t0, out, coverage


def quality(ts, coverage: float, duration: float, th: Thresholds = DEFAULTS) -> dict:
    fps = len(ts) / duration if duration else 0.0
    return {"coverage": round(coverage, 3), "fps": round(fps, 1),
            "ok": bool(coverage >= th.min_coverage and fps >= th.min_fps)}


def analyze_repetitive(ts, values, t0: float, duration: float, th: Thresholds = DEFAULTS) -> dict:
    """Cycle count, rate, amplitude, amplitude decrement and rhythm from a back-and-forth signal."""
    rel, x, coverage = uniform(ts, values, t0, duration)
    span = float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) else 0.0
    out = {"count": 0, "rate_hz": 0.0, "amplitude_mean": None, "amplitude_decrement_pct": None,
           "rhythm_cv_pct": None, "hesitations": None, "peak_t": [], "peak_v": [], "amplitudes": [],
           "quality": quality(ts, coverage, duration, th), "series": {"t": rel, "v": x}}
    if span >= th.min_span:
        peaks, props = find_peaks(x, prominence=th.prominence_frac * span,
                                  distance=max(1, int(th.min_cycle_s * FPS)))
        amps = props["prominences"]
        n = len(peaks)
        # Parabolic interpolation: at 30 fps a 5 Hz cycle is only 6 frames, so raw peak
        # times jitter by a frame and would inflate the rhythm variability.
        y0, y1, y2 = x[peaks - 1], x[peaks], x[peaks + 1]
        shift = 0.5 * (y0 - y2) / np.where(y0 - 2 * y1 + y2 == 0, -1.0, y0 - 2 * y1 + y2)
        peak_t = rel[peaks] + np.clip(shift, -1, 1) / FPS
        out.update(count=n, rate_hz=n / duration, peak_t=peak_t, peak_v=x[peaks], amplitudes=amps)
        if n:
            out["amplitude_mean"] = float(amps.mean())
        if n >= 8:
            inner = amps[1:-1]  # the end cycles are clipped by the recording window
            k = len(inner) // 3
            first, last = float(inner[:k].mean()), float(inner[-k:].mean())
            out["amplitude_decrement_pct"] = (last - first) / first * 100 if first else None
        if n >= 4:
            gaps = np.diff(peak_t)
            out["rhythm_cv_pct"] = float(gaps.std() / gaps.mean() * 100)
            out["hesitations"] = int((gaps > th.hesitation_ratio * np.median(gaps)).sum())
    return _plain(out)


def analyze_tremor(ts, palm_xy, t0: float, duration: float, th: Thresholds = DEFAULTS) -> dict:
    """Dominant frequency and size of rhythmic oscillation in palm position while held still."""
    rel, xy, coverage = uniform(ts, palm_xy, t0, duration)
    out = {"dominant_hz": None, "band_rms_pct": None, "peak_ratio": None, "tremor_like": False,
           "quality": quality(ts, coverage, duration, th), "series": {"t": rel, "v": None}}
    if len(rel) < 64 or coverage == 0:
        return _plain(out)
    nper = min(128, len(rel))
    freqs, psd = welch(xy, fs=FPS, window="hann", nperseg=nper, noverlap=nper // 2, detrend="linear", axis=0)
    psd = psd.sum(axis=1)  # power from both axes
    lo, hi = th.tremor_band
    band = (freqs >= lo) & (freqs <= hi)
    ref = (freqs >= 1.0) & (freqs <= 14.0)
    peak = int(np.argmax(np.where(band, psd, -1)))
    ratio = float(psd[peak] / (np.median(psd[ref]) + 1e-12))
    rms_pct = float(np.sqrt(psd[band].sum() * (freqs[1] - freqs[0])) * 100)
    shown = freqs <= 14.0
    out.update(dominant_hz=float(freqs[peak]), band_rms_pct=rms_pct, peak_ratio=ratio,
               tremor_like=bool(ratio >= th.tremor_peak_ratio and rms_pct >= th.tremor_min_rms_pct))
    out["series"] = {"t": rel, "v": xy - xy.mean(axis=0), "freq": freqs[shown], "psd": psd[shown]}
    return _plain(out)


def observations(test: str, m: dict, th: Thresholds = DEFAULTS) -> list[dict]:
    """Plain-language notes on one result. Each is {"level": "ok"|"notice", "text": ...}."""
    notes = []
    if not m["quality"]["ok"]:
        notes.append({"level": "notice", "text": "Tracking was incomplete (hand out of view or low frame rate). "
                                                 "Repeat the test for a reliable result."})
    if test == "tremor":
        if m["tremor_like"]:
            notes.append({"level": "notice", "text": f"Rhythmic oscillation near {m['dominant_hz']:.1f} Hz while "
                                                     "holding still. Worth tracking over time."})
        else:
            notes.append({"level": "ok", "text": "No clear rhythmic oscillation while holding still."})
        return notes
    if m["count"] == 0:
        notes.append({"level": "notice", "text": "No repeated movement detected."})
        return notes
    dec = m["amplitude_decrement_pct"]
    if dec is not None and dec <= th.decrement_pct:
        notes.append({"level": "notice", "text": f"Movements got {abs(dec):.0f}% smaller over the test."})
    if m["hesitations"]:
        notes.append({"level": "notice", "text": f"{m['hesitations']} pause(s) or hesitation(s) in the rhythm."})
    if m["rhythm_cv_pct"] is not None and m["rhythm_cv_pct"] >= th.rhythm_cv_pct:
        notes.append({"level": "notice", "text": f"Irregular rhythm (variability {m['rhythm_cv_pct']:.0f}%)."})
    if not any(n["level"] == "notice" for n in notes):
        notes.append({"level": "ok", "text": "Steady size and rhythm across the test."})
    return notes


def _plain(obj):
    """Numpy -> plain Python so results serialize to JSON."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return np.round(obj, 5).tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
