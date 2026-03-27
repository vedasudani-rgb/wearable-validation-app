"""
main.py — CLI demonstration script.
Runs both protocol types with synthetic HR data and prints full reports.
Usage: python main.py
"""
from __future__ import annotations
import numpy as np

from wearable_validation import (
    ProtocolParams,
    Protocol,
    TestRunMetadata,
    HRDataSeries,
    generate_protocol,
    analyze_hr_validation,
    analyze_group,
    format_report,
    format_group_report,
)
from wearable_validation.constants import INTENSITY_BASELINE_HR


# ── Synthetic data generation ─────────────────────────────────────────────────

def simulate_hr_data(
    protocol: Protocol,
    bias_bpm: float = 2.5,
    noise_sd: float = 4.0,
    seed: int = 42,
) -> HRDataSeries:
    """
    Generate synthetic HR data for demonstration.
    Reference HR follows intensity-label baselines; wearable adds bias + Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    ref_segments: list[np.ndarray] = []
    for step in protocol.steps:
        base = INTENSITY_BASELINE_HR.get(step.intensity_label, 140.0)
        ref_segments.append(
            np.full(step.duration_sec, base) + rng.normal(0, 2.0, step.duration_sec)
        )

    hr_ref  = np.concatenate(ref_segments)
    hr_wear = hr_ref + bias_bpm + rng.normal(0, noise_sd, len(hr_ref))
    timestamps = np.arange(len(hr_ref), dtype=float)
    return HRDataSeries(hr_wearable=hr_wear, hr_reference=hr_ref, timestamps=timestamps)


# ── Protocol printer ──────────────────────────────────────────────────────────

def print_protocol(protocol: Protocol) -> None:
    print(f"\n{'='*62}")
    print(f"  FIELD PROTOCOL — {protocol.context.upper().replace('_', ' ')}")
    print(f"{'='*62}")
    print(f"  Estimated duration  : {protocol.estimated_duration_min} min")
    print(f"  Expected samples    : {protocol.n_expected_samples:,} (at 1 Hz)")
    if protocol.hrmax:
        print(f"  Estimated HRmax     : {protocol.hrmax} BPM (Tanaka formula)")
    print()
    print(protocol.device_instructions)
    print("Step-by-step:")
    for i, step in enumerate(protocol.steps, 1):
        mins = step.duration_sec // 60
        secs = step.duration_sec % 60
        bpm_str = f"  BPM: {step.target_hr_bpm}" if step.target_hr_bpm else ""
        print(f"  {i:2}. [{mins}:{secs:02d}] {step.name}")
        print(f"       {step.target_rpe} | {step.target_hr_pct}{bpm_str}")
        print(f"       → {step.instructions}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    base_params = dict(
        sport="running",
        metric="heart_rate",
        wearable_type="wrist_based_ppg",
        reference_type="chest_strap_ecg",
        wearable_device_name="Garmin Forerunner 265",
        reference_device_name="Polar H10",
        test_date="2026-03-26",
        age=30,
    )

    # ── Demo 1: Continuous graded run, single athlete ─────────────────────
    print("\n" + "#"*62)
    print("  DEMO 1: Single Athlete — Continuous Graded Run")
    print("#"*62)

    params_A = ProtocolParams(**base_params, context="steady_run")
    protocol_A = generate_protocol(params_A)
    print_protocol(protocol_A)

    data_A = simulate_hr_data(protocol_A, bias_bpm=2.5, noise_sd=4.0)
    print(f"Simulated {len(data_A.hr_reference):,} paired HR samples.\n")

    meta_A = TestRunMetadata(
        test_date="2026-03-26",
        athlete_name="Demo Athlete",
        wearable_device_name="Garmin Forerunner 265",
        reference_device_name="Polar H10",
        sport="running",
        metric="heart_rate",
        wearable_type="wrist_based_ppg",
        reference_type="chest_strap_ecg",
        conditions="Outdoor flat road, 18°C, low wind",
    )
    report_A = analyze_hr_validation(data_A, meta_A)
    print(format_report(report_A))

    # ── Demo 2: Interval session ──────────────────────────────────────────
    print("\n" + "#"*62)
    print("  DEMO 2: Single Athlete — Interval Session")
    print("#"*62)

    params_B = ProtocolParams(**base_params, context="interval_session")
    protocol_B = generate_protocol(params_B)
    print_protocol(protocol_B)

    data_B = simulate_hr_data(protocol_B, bias_bpm=5.0, noise_sd=6.0)
    meta_B = TestRunMetadata(**{**meta_A.__dict__, "athlete_name": "Demo Athlete"})
    report_B = analyze_hr_validation(data_B, meta_B)
    print(format_report(report_B))

    # ── Demo 3: Multi-athlete group (3 athletes, varying bias) ───────────
    print("\n" + "#"*62)
    print("  DEMO 3: Multi-Athlete Group (3 athletes)")
    print("#"*62)

    athlete_configs = [
        ("Alice",   2.0, 3.5, 10),
        ("Bob",     4.5, 5.0, 20),
        ("Charlie", -1.0, 4.0, 30),
    ]

    all_reports = []
    all_data    = []
    for name, bias, noise, seed in athlete_configs:
        d = simulate_hr_data(protocol_A, bias_bpm=bias, noise_sd=noise, seed=seed)
        m = TestRunMetadata(
            test_date="2026-03-26",
            athlete_name=name,
            wearable_device_name="Garmin Forerunner 265",
            reference_device_name="Polar H10",
            sport="running",
            metric="heart_rate",
            wearable_type="wrist_based_ppg",
            reference_type="chest_strap_ecg",
            conditions="Outdoor track, 15°C",
        )
        r = analyze_hr_validation(d, m)
        all_reports.append(r)
        all_data.append(d)
        print(f"  {name:<10} MAPE={r.mape:.1f}%  Bias={r.bias:+.1f} BPM  {r.quality_label}")

    group = analyze_group(all_reports, all_data)
    print()
    print(format_group_report(group))


if __name__ == "__main__":
    main()
