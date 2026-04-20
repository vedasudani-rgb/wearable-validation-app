"""
Generate synthetic HR test data for all 4 protocols × multiple device scenarios.

Outputs CSV files to test_data/ with realistic HR profiles, noise, and artefacts.
Run with: python generate_test_data.py

Scenarios produced
──────────────────
Running – Steady Run (28 min / 1680 samples)
  • reference_chest_strap.csv
  • wearable_wrist_excellent.csv   MAPE ≈ 2.5%  (Excellent)
  • wearable_wrist_good.csv        MAPE ≈ 4.0%  (Good)
  • wearable_wrist_acceptable.csv  MAPE ≈ 7.5%  (Acceptable)
  • wearable_wrist_poor.csv        MAPE ≈ 13%   (Poor)
  • wearable_finger.csv            MAPE ≈ 5.0%  (Good/Acceptable)
  • combined_wrist_excellent.csv   (combined format)
  • combined_wrist_good.csv        (combined format)

Running – Interval Session (34 min / 2040 samples)
  • reference_chest_strap.csv
  • wearable_wrist.csv             MAPE ≈ 5% overall, worse at high intensity
  • combined.csv

Cycling – Steady Ride (32 min / 1920 samples)
  • reference_chest_strap.csv
  • wearable_wrist.csv             MAPE ≈ 4%
  • wearable_finger.csv            MAPE ≈ 8% (grip artefact at high intensity)
  • combined_wrist.csv

Cycling – Interval Ride (34 min / 2040 samples)
  • reference_chest_strap.csv
  • wearable_wrist.csv
  • combined.csv

Multi-athlete – Running Steady (3 athletes, same protocol)
  • athleteN_reference.csv / athleteN_wearable.csv
    Athlete 1: MAPE ≈ 3%   (Excellent)
    Athlete 2: MAPE ≈ 6%   (Acceptable)
    Athlete 3: MAPE ≈ 12%  (Poor — e.g. loose band, darker skin tone)

Device Comparison – Running Steady
  • reference.csv
  • device_a_wrist_garmin.csv      MAPE ≈ 2.8%  rank 1
  • device_b_wrist_apple.csv       MAPE ≈ 5.1%  rank 2
  • device_c_finger_polar.csv      MAPE ≈ 8.4%  rank 3
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)


# ── HR profile generators ──────────────────────────────────────────────────────

def _smooth_transition(start: float, end: float, n: int, tau_frac: float = 0.25) -> np.ndarray:
    """Exponential approach from start → end over n samples."""
    t = np.linspace(0, 1, n)
    tau = tau_frac
    curve = end + (start - end) * np.exp(-t / tau)
    return curve


def _make_reference(segments: list[tuple[int, float, float]], noise_std: float = 1.5) -> np.ndarray:
    """
    Build a clean reference HR array.
    segments: list of (duration_sec, hr_start, hr_end)
    """
    parts = []
    for dur, hr_s, hr_e in segments:
        parts.append(_smooth_transition(hr_s, hr_e, dur))
    ref = np.concatenate(parts)
    ref += RNG.normal(0, noise_std, len(ref))
    return np.clip(np.round(ref, 1), 50, 220)


def _add_ppg_noise(
    ref: np.ndarray,
    bias: float = 0.0,
    noise_std: float = 5.0,
    lag_samples: int = 3,
    spike_prob: float = 0.005,
    intensity_noise_scale: float = 1.0,   # multiplier above 160 BPM
    intensity_threshold: float = 160.0,
) -> np.ndarray:
    """
    Apply realistic PPG noise model to reference HR.
      bias              — systematic offset (positive = overestimate)
      noise_std         — base Gaussian noise SD
      lag_samples       — smoothing lag (cardiac response lag simulation)
      spike_prob        — probability of a spike/dropout per sample
      intensity_noise_scale — noise multiplier for HR > intensity_threshold
    """
    n = len(ref)

    # 1. Lag: moving average simulates PPG response delay
    if lag_samples > 1:
        kernel = np.ones(lag_samples) / lag_samples
        smoothed = np.convolve(ref, kernel, mode="same")
    else:
        smoothed = ref.copy()

    # 2. Intensity-dependent noise
    noise_sd = np.where(smoothed > intensity_threshold,
                        noise_std * intensity_noise_scale,
                        noise_std)
    noise = RNG.normal(0, noise_sd)

    # 3. Systematic bias
    wearable = smoothed + bias + noise

    # 4. Random spikes / dropouts
    spike_mask = RNG.random(n) < spike_prob
    spike_direction = RNG.choice([-1, 1], size=n)
    spike_magnitude = RNG.uniform(15, 35, size=n)
    wearable[spike_mask] += spike_direction[spike_mask] * spike_magnitude[spike_mask]

    return np.clip(np.round(wearable, 1), 40, 220)


def _timestamps(n: int) -> np.ndarray:
    return np.arange(n, dtype=float)


def _save(path: Path, timestamps, hr_col, col_name="hr", ref=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if ref is not None:
        df = pd.DataFrame({"timestamp": timestamps, "hr_wearable": hr_col, "hr_reference": ref})
    else:
        df = pd.DataFrame({"timestamp": timestamps, col_name: hr_col})
    df.to_csv(path, index=False)
    print(f"  Wrote {path}  ({len(df)} rows)")


# ── Protocol HR segment definitions ───────────────────────────────────────────

def running_steady_segments():
    """28 min / 1680 s — Continuous Graded Run."""
    return [
        (300,  68,  128),   # Warm-Up
        (300, 128,  132),   # Easy
        (300, 132,  152),   # Moderate
        (300, 152,  167),   # Comfortably Hard
        (180, 167,  180),   # Hard
        (300, 180,   88),   # Cool-Down
    ]


def running_interval_segments():
    """34 min / 2040 s — 5 min WU + 6×[2 min hard / 2 min recovery] + 5 min CD."""
    segs = [(300, 68, 130)]   # Warm-Up
    hr = 130.0
    for _ in range(6):
        segs.append((120, hr, 178))   # Hard interval
        segs.append((120, 178, 128))  # Recovery
        hr = 128.0
    segs.append((300, 128, 85))       # Cool-Down
    return segs


def cycling_steady_segments():
    """32 min / 1920 s — 6-min stages (HR stabilises more slowly in cycling)."""
    return [
        (360,  65,  122),   # Warm-Up
        (360, 122,  132),   # Easy
        (360, 132,  150),   # Moderate
        (360, 150,  163),   # Comfortably Hard
        (180, 163,  174),   # Hard
        (300, 174,   85),   # Cool-Down
    ]


def cycling_interval_segments():
    """34 min / 2040 s — same interval structure as running."""
    segs = [(300, 65, 128)]
    hr = 128.0
    for _ in range(6):
        segs.append((120, hr, 172))
        segs.append((120, 172, 125))
        hr = 125.0
    segs.append((300, 125, 82))
    return segs


def _mape(wearable, reference):
    diff = wearable - reference
    return float(np.mean(np.abs(diff) / reference) * 100)


# ── Generate all files ────────────────────────────────────────────────────────

base = Path("test_data")

# ────────────────────────────────────────────────────────────────────────────
# 1. RUNNING — STEADY RUN
# ────────────────────────────────────────────────────────────────────────────
print("\n── Running / Steady Run ──")
d = base / "running_steady"
ref = _make_reference(running_steady_segments())
ts  = _timestamps(len(ref))
_save(d / "reference_chest_strap.csv", ts, ref, col_name="hr_reference")

configs = [
    ("wearable_wrist_excellent",  1.5,  3.5, 2, 0.003, 1.1),
    ("wearable_wrist_good",       2.0,  6.0, 3, 0.004, 1.3),
    ("wearable_wrist_acceptable", 3.0, 11.5, 4, 0.010, 1.6),
    ("wearable_wrist_poor",       4.5, 18.0, 5, 0.025, 2.0),
    ("wearable_finger",           2.5,  7.5, 3, 0.006, 1.4),
]
for name, bias, noise, lag, spike, intensity_scale in configs:
    w = _add_ppg_noise(ref, bias, noise, lag, spike, intensity_scale)
    _save(d / f"{name}.csv", ts, w, col_name="hr_wearable")
    print(f"    {name}: MAPE = {_mape(w, ref):.1f}%")

# Combined files (timestamp + both HR columns)
for name in ("wearable_wrist_excellent", "wearable_wrist_good"):
    w = pd.read_csv(d / f"{name}.csv")["hr_wearable"].values
    _save(d / f"combined_{name.replace('wearable_', '')}.csv", ts, w, ref=ref)

# ────────────────────────────────────────────────────────────────────────────
# 2. RUNNING — INTERVAL SESSION
# ────────────────────────────────────────────────────────────────────────────
print("\n── Running / Interval Session ──")
d = base / "running_interval"
ref = _make_reference(running_interval_segments())
ts  = _timestamps(len(ref))
_save(d / "reference_chest_strap.csv", ts, ref, col_name="hr_reference")

# Wrist PPG — good at moderate intensity, degrades >165 BPM (motion artefact)
w = _add_ppg_noise(ref, bias=2.0, noise_std=7.0, lag_samples=3, spike_prob=0.008,
                   intensity_noise_scale=2.0, intensity_threshold=162.0)
_save(d / "wearable_wrist.csv", ts, w, col_name="hr_wearable")
print(f"    wearable_wrist: MAPE = {_mape(w, ref):.1f}%")
_save(d / "combined.csv", ts, w, ref=ref)

# ────────────────────────────────────────────────────────────────────────────
# 3. CYCLING — STEADY RIDE
# ────────────────────────────────────────────────────────────────────────────
print("\n── Cycling / Steady Ride ──")
d = base / "cycling_steady"
ref = _make_reference(cycling_steady_segments())
ts  = _timestamps(len(ref))
_save(d / "reference_chest_strap.csv", ts, ref, col_name="hr_reference")

# Wrist PPG — reasonable accuracy in cycling (less arm motion)
w_wrist = _add_ppg_noise(ref, bias=1.5, noise_std=5.5, lag_samples=2, spike_prob=0.004,
                          intensity_noise_scale=1.4, intensity_threshold=155.0)
_save(d / "wearable_wrist.csv", ts, w_wrist, col_name="hr_wearable")
print(f"    wearable_wrist: MAPE = {_mape(w_wrist, ref):.1f}%")

# Finger PPG — handlebar grip increases artefact above ~155 BPM
w_finger = _add_ppg_noise(ref, bias=3.0, noise_std=7.0, lag_samples=4, spike_prob=0.012,
                           intensity_noise_scale=2.8, intensity_threshold=150.0)
_save(d / "wearable_finger.csv", ts, w_finger, col_name="hr_wearable")
print(f"    wearable_finger: MAPE = {_mape(w_finger, ref):.1f}%")

_save(d / "combined_wrist.csv", ts, w_wrist, ref=ref)

# ────────────────────────────────────────────────────────────────────────────
# 4. CYCLING — INTERVAL RIDE
# ────────────────────────────────────────────────────────────────────────────
print("\n── Cycling / Interval Ride ──")
d = base / "cycling_interval"
ref = _make_reference(cycling_interval_segments())
ts  = _timestamps(len(ref))
_save(d / "reference_chest_strap.csv", ts, ref, col_name="hr_reference")

w = _add_ppg_noise(ref, bias=2.0, noise_std=6.5, lag_samples=3, spike_prob=0.007,
                   intensity_noise_scale=1.9, intensity_threshold=158.0)
_save(d / "wearable_wrist.csv", ts, w, col_name="hr_wearable")
print(f"    wearable_wrist: MAPE = {_mape(w, ref):.1f}%")
_save(d / "combined.csv", ts, w, ref=ref)

# ────────────────────────────────────────────────────────────────────────────
# 5. MULTI-ATHLETE — Running Steady (3 athletes, same protocol)
# ────────────────────────────────────────────────────────────────────────────
print("\n── Multi-Athlete / Running Steady ──")
d = base / "multi_athlete"
athlete_configs = [
    # (name,        bias, noise, lag, spike, int_scale)  target MAPE
    ("athlete1",    1.5,  3.5,   2,  0.003, 1.1),   # Excellent ~2.5%
    ("athlete2",    3.0,  9.0,   4,  0.008, 1.5),   # Acceptable ~6%
    ("athlete3",    5.0, 17.0,   5,  0.022, 2.2),   # Poor ~12%
]
for name, bias, noise, lag, spike, int_scale in athlete_configs:
    # Each athlete gets a slightly different physiological HR profile
    segs = running_steady_segments()
    ref = _make_reference(segs)
    ts  = _timestamps(len(ref))
    w   = _add_ppg_noise(ref, bias, noise, lag, spike, int_scale)
    _save(d / f"{name}_reference.csv", ts, ref, col_name="hr_reference")
    _save(d / f"{name}_wearable.csv",  ts, w,   col_name="hr_wearable")
    print(f"    {name}: MAPE = {_mape(w, ref):.1f}%")

# ────────────────────────────────────────────────────────────────────────────
# 6. DEVICE COMPARISON — Running Steady (3 wearables, 1 reference)
# ────────────────────────────────────────────────────────────────────────────
print("\n── Device Comparison / Running Steady ──")
d = base / "device_comparison"
ref = _make_reference(running_steady_segments())
ts  = _timestamps(len(ref))
_save(d / "reference_chest_strap.csv", ts, ref, col_name="hr_reference")

device_configs = [
    # (filename,                bias, noise, lag, spike, int_scale)
    ("device_a_garmin_wrist",   1.5,  3.5,  2,  0.003, 1.1),  # rank 1 ≈2.5% MAPE
    ("device_b_apple_wrist",    3.0,  8.0,  4,  0.006, 1.4),  # rank 2 ≈5% MAPE
    ("device_c_polar_finger",   4.0, 12.0,  4,  0.012, 1.8),  # rank 3 ≈8% MAPE
]
for name, bias, noise, lag, spike, int_scale in device_configs:
    w = _add_ppg_noise(ref, bias, noise, lag, spike, int_scale)
    _save(d / f"{name}.csv", ts, w, col_name="hr_wearable")
    print(f"    {name}: MAPE = {_mape(w, ref):.1f}%")

# ────────────────────────────────────────────────────────────────────────────
# 7. LONGITUDINAL — Same device across 6 test dates (drift + firmware recovery)
#    Narrative: Good baseline → sensor drift → firmware fix → Excellent
# ────────────────────────────────────────────────────────────────────────────
print("\n── Longitudinal / Running Steady (6 sessions) ──")
d = base / "longitudinal"

# (date, bias, noise_std, lag, spike_prob, intensity_scale)  → target MAPE
longitudinal_sessions = [
    ("2024-01-15", 2.0,  5.0, 3, 0.003, 1.2),  # Good baseline ~4%
    ("2024-02-19", 1.8,  4.5, 2, 0.003, 1.1),  # Good — still clean ~3.5%
    ("2024-04-03", 3.5,  9.5, 4, 0.010, 1.6),  # Acceptable — drift begins ~6.5%
    ("2024-05-14", 4.0, 12.0, 5, 0.014, 1.9),  # Acceptable — degraded ~7.5%
    ("2024-06-20", 2.5,  6.5, 3, 0.005, 1.3),  # Good — firmware fix applied ~5%
    ("2024-08-08", 1.5,  3.5, 2, 0.003, 1.1),  # Excellent — further improved ~2.5%
]
for date_str, bias, noise, lag, spike, int_scale in longitudinal_sessions:
    segs = running_steady_segments()
    ref  = _make_reference(segs)
    ts   = _timestamps(len(ref))
    w    = _add_ppg_noise(ref, bias, noise, lag, spike, int_scale)
    session_dir = d / date_str
    _save(session_dir / "reference.csv", ts, ref, col_name="hr_reference")
    _save(session_dir / "wearable.csv",  ts, w,   col_name="hr_wearable")
    print(f"    {date_str}: MAPE = {_mape(w, ref):.1f}%")

print(f"\nDone. All files written to: {base.resolve()}/")
print("\nQuick reference — expected MAPE values:")
print("  running_steady/wearable_wrist_excellent  → ~2–3%   Excellent")
print("  running_steady/wearable_wrist_good       → ~4–5%   Good")
print("  running_steady/wearable_wrist_acceptable → ~7–9%   Acceptable")
print("  running_steady/wearable_wrist_poor       → ~12–15% Poor")
print("  running_steady/wearable_finger           → ~5–6%   Good/Acceptable")
print("  running_interval/wearable_wrist          → ~5–7%   (worse at peak HR)")
print("  cycling_steady/wearable_wrist            → ~4–5%   Good")
print("  cycling_steady/wearable_finger           → ~8–10%  Acceptable/Poor (grip artefact)")
print("  cycling_interval/wearable_wrist          → ~5–7%")
print("  multi_athlete: athlete1 ~2.5%, athlete2 ~6%, athlete3 ~12%")
print("  device_comparison: Garmin ~2.5% > Apple ~5% > Polar finger ~8%")
print("  longitudinal: Jan ~4%, Feb ~3.5%, Apr ~6.5%, May ~7.5%, Jun ~5%, Aug ~2.5%")
