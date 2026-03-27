from __future__ import annotations
import warnings
import numpy as np

from wearable_validation.models import (
    HRDataSeries, TestRunMetadata, AnalysisReport, GroupAnalysisReport,
    Protocol, IntensityBinResult, StepCoverageResult, CoverageReport,
    DeviceComparisonEntry, DeviceComparisonReport,
)
from wearable_validation.constants import (
    LOA_Z, MIN_SAMPLES_WARNING, QUALITY_LABELS,
    MAPE_EXCELLENT_THRESHOLD, MAPE_GOOD_THRESHOLD, MAPE_ACCEPTABLE_THRESHOLD,
    HR_BINS, HR_BIN_MIN_SAMPLES, COVERAGE_TOLERANCE_PCT,
)
from wearable_validation import report as _report_module


# ── Per-athlete analysis ──────────────────────────────────────────────────────

def analyze_hr_validation(
    data: HRDataSeries,
    metadata: TestRunMetadata,
) -> AnalysisReport:
    """
    Compute HR agreement statistics for a single athlete/device pair.
    Drops NaN pairs. Flags outliers (|diff| > 2×SD) but does not remove them.
    """
    w, r = _clean(data)
    n = len(w)

    if n < MIN_SAMPLES_WARNING:
        warnings.warn(
            f"Only {n} paired samples. Bland-Altman LoA require ≥ {MIN_SAMPLES_WARNING} "
            "for reliable estimates. Interpret results cautiously.",
            UserWarning, stacklevel=2,
        )

    diff      = w - r
    bias      = float(np.mean(diff))
    mae       = float(np.mean(np.abs(diff)))
    mape      = float(np.mean(np.abs(diff) / r) * 100)
    sd_diff   = float(np.std(diff, ddof=1))
    loa_lower = bias - LOA_Z * sd_diff
    loa_upper = bias + LOA_Z * sd_diff
    n_outliers = int(np.sum(np.abs(diff - bias) > 2 * sd_diff)) if sd_diff > 0 else 0
    quality   = _quality_label(mape)

    summary, interpretation, limitations = _report_module.build_text_blocks(
        bias=bias, mae=mae, mape=mape,
        loa_lower=loa_lower, loa_upper=loa_upper,
        n_samples=n, n_outliers=n_outliers,
        quality=quality, metadata=metadata,
    )

    return AnalysisReport(
        bias=bias, mae=mae, mape=mape,
        loa_lower=loa_lower, loa_upper=loa_upper,
        n_samples=n, n_outliers_flagged=n_outliers,
        quality_label=quality,
        summary_text=summary,
        interpretation_text=interpretation,
        limitations_text=limitations,
        metadata=metadata,
    )


# ── Intensity-stratified analysis ─────────────────────────────────────────────

def analyze_by_intensity_bin(data: HRDataSeries) -> list[IntensityBinResult]:
    """
    Compute MAPE, bias, and MAE separately for each HR bin defined in
    constants.HR_BINS. Bins are defined on the reference HR channel.
    Returns one IntensityBinResult per bin; marks bins with < HR_BIN_MIN_SAMPLES
    as insufficient.
    """
    w, r = _clean(data)
    results: list[IntensityBinResult] = []

    for lo, hi, label in HR_BINS:
        mask = (r >= lo) & (r < hi)
        w_bin = w[mask]
        r_bin = r[mask]
        n = int(np.sum(mask))

        if n < HR_BIN_MIN_SAMPLES:
            results.append(IntensityBinResult(
                bin_label=label, hr_lower=lo, hr_upper=hi,
                n_samples=n, bias=float("nan"), mae=float("nan"),
                mape=float("nan"), sufficient_data=False,
            ))
            continue

        diff = w_bin - r_bin
        results.append(IntensityBinResult(
            bin_label=label, hr_lower=lo, hr_upper=hi,
            n_samples=n,
            bias=float(np.mean(diff)),
            mae=float(np.mean(np.abs(diff))),
            mape=float(np.mean(np.abs(diff) / r_bin) * 100),
            sufficient_data=True,
        ))

    return results


# ── HR zone coverage validation ───────────────────────────────────────────────

def check_hr_zone_coverage(
    data: HRDataSeries,
    protocol: Protocol,
) -> CoverageReport:
    """
    For each protocol step, check whether the athlete's reference HR actually
    reached the target %HRmax range.

    Requires protocol.hrmax to be set (age must have been provided). If hrmax
    is not available, status is reported as "unknown" with the actual median HR.

    A step is flagged "not_reached" if its median reference HR falls more than
    COVERAGE_TOLERANCE_PCT of HRmax below the step's target floor.
    """
    hrmax = protocol.hrmax
    step_results: list[StepCoverageResult] = []

    for boundary in protocol.step_boundaries:
        mask = (
            (data.timestamps >= boundary.start_sec) &
            (data.timestamps < boundary.end_sec)
        )
        step_hr = data.hr_reference[mask]
        n_in_step = int(np.sum(mask))

        if n_in_step == 0:
            step_results.append(StepCoverageResult(
                step_name=boundary.step_name,
                intensity_label=boundary.intensity_label,
                target_hr_pct=f"{int(boundary.target_hr_pct_low*100)}–{int(boundary.target_hr_pct_high*100)}% HRmax",
                target_bpm_range=None,
                actual_median_bpm=float("nan"),
                n_samples_in_step=0,
                status="unknown",
            ))
            continue

        median_hr = float(np.median(step_hr))

        if hrmax is None:
            # Can report actual HR but cannot assess coverage without HRmax
            step_results.append(StepCoverageResult(
                step_name=boundary.step_name,
                intensity_label=boundary.intensity_label,
                target_hr_pct=f"{int(boundary.target_hr_pct_low*100)}–{int(boundary.target_hr_pct_high*100)}% HRmax",
                target_bpm_range=None,
                actual_median_bpm=round(median_hr, 1),
                n_samples_in_step=n_in_step,
                status="unknown",
            ))
            continue

        target_floor_bpm = boundary.target_hr_pct_low * hrmax
        target_ceil_bpm  = boundary.target_hr_pct_high * hrmax
        tolerance_bpm    = COVERAGE_TOLERANCE_PCT * hrmax

        lo_bpm = round(boundary.target_hr_pct_low * hrmax)
        hi_bpm = round(boundary.target_hr_pct_high * hrmax)
        target_range = f"< {hi_bpm} BPM" if boundary.target_hr_pct_low == 0 else f"{lo_bpm}–{hi_bpm} BPM"

        # Warm-up and cool-down: skip strict coverage check
        if boundary.intensity_label in ("warm_up", "cool_down"):
            status = "met"
        elif median_hr >= target_floor_bpm - tolerance_bpm:
            status = "met" if median_hr <= target_ceil_bpm + tolerance_bpm else "met"
        elif median_hr >= target_floor_bpm - 2 * tolerance_bpm:
            status = "partial"
        else:
            status = "not_reached"

        step_results.append(StepCoverageResult(
            step_name=boundary.step_name,
            intensity_label=boundary.intensity_label,
            target_hr_pct=f"{int(boundary.target_hr_pct_low*100)}–{int(boundary.target_hr_pct_high*100)}% HRmax",
            target_bpm_range=target_range,
            actual_median_bpm=round(median_hr, 1),
            n_samples_in_step=n_in_step,
            status=status,
        ))

    # Overall status
    statuses = [r.status for r in step_results]
    if all(s in ("met", "warm_up", "cool_down") for s in statuses):
        overall = "complete"
        warning = ""
    elif "unknown" in statuses:
        overall = "unknown"
        warning = "Provide athlete age to enable target BPM comparison."
    elif "not_reached" in statuses:
        not_reached = [r.step_name for r in step_results if r.status == "not_reached"]
        overall = "insufficient"
        warning = (
            f"Target intensity not reached in: {', '.join(not_reached)}. "
            "High-intensity accuracy results may be unreliable."
        )
    elif "partial" in statuses:
        overall = "partial"
        partial = [r.step_name for r in step_results if r.status == "partial"]
        warning = f"Target partially met in: {', '.join(partial)}."
    else:
        overall = "complete"
        warning = ""

    return CoverageReport(
        step_results=step_results,
        overall_status=overall,
        warning_message=warning,
    )


# ── Device comparison ─────────────────────────────────────────────────────────

def analyze_device_comparison(
    devices: list[tuple[str, str, HRDataSeries]],  # (device_name, wearable_type, data)
    reference_device_name: str,
    metadata_base: TestRunMetadata,
) -> DeviceComparisonReport:
    """
    Analyse multiple wearable devices tested simultaneously against the same
    reference. Returns entries ranked by MAPE ascending (best first).
    """
    if not devices:
        raise ValueError("At least one device is required for comparison.")

    entries: list[DeviceComparisonEntry] = []
    for device_name, wearable_type, data in devices:
        meta = TestRunMetadata(
            test_date=metadata_base.test_date,
            athlete_name=metadata_base.athlete_name,
            wearable_device_name=device_name,
            reference_device_name=reference_device_name,
            sport=metadata_base.sport,
            metric=metadata_base.metric,
            wearable_type=wearable_type,
            reference_type=metadata_base.reference_type,
            conditions=metadata_base.conditions,
        )
        report = analyze_hr_validation(data, meta)
        entries.append(DeviceComparisonEntry(
            device_name=device_name,
            wearable_type=wearable_type,
            report=report,
            data=data,
            rank=0,  # assigned below
        ))

    # Rank by MAPE ascending
    entries.sort(key=lambda e: e.report.mape)
    for i, entry in enumerate(entries):
        entry.rank = i + 1

    best = entries[0]
    n = len(entries)
    summary = (
        f"Compared {n} device(s) against {reference_device_name}. "
        f"Best performing: {best.device_name} "
        f"(MAPE = {best.report.mape:.1f}%, {QUALITY_LABELS[best.report.quality_label]}). "
        f"Ranked by MAPE ascending."
    )

    return DeviceComparisonReport(
        entries=entries,
        reference_device_name=reference_device_name,
        n_devices=n,
        best_device_name=best.device_name,
        summary_text=summary,
    )


# ── Group (multi-athlete) analysis ────────────────────────────────────────────

def analyze_group(
    reports: list[AnalysisReport],
    datasets: list[HRDataSeries],
) -> GroupAnalysisReport:
    """
    Aggregate per-athlete results into a group report using pooled differences.
    Pooled LoA computed per Bland & Altman (1999) extension for multiple subjects.
    Appropriate when all athletes used the same protocol and device model.
    """
    if len(reports) != len(datasets):
        raise ValueError("reports and datasets must have the same length.")
    if len(reports) == 0:
        raise ValueError("Cannot compute group statistics with zero athletes.")

    mapes  = np.array([r.mape  for r in reports])
    biases = np.array([r.bias  for r in reports])
    maes   = np.array([r.mae   for r in reports])

    all_diffs: list[np.ndarray] = []
    for data in datasets:
        w = np.asarray(data.hr_wearable, dtype=float)
        r = np.asarray(data.hr_reference, dtype=float)
        valid = ~(np.isnan(w) | np.isnan(r))
        all_diffs.append(w[valid] - r[valid])

    pooled        = np.concatenate(all_diffs)
    pooled_bias   = float(np.mean(pooled))
    pooled_sd     = float(np.std(pooled, ddof=1))
    pooled_loa_lo = pooled_bias - LOA_Z * pooled_sd
    pooled_loa_hi = pooled_bias + LOA_Z * pooled_sd

    mean_mape = float(np.mean(mapes))
    sd_mape   = float(np.std(mapes,  ddof=1)) if len(mapes)  > 1 else 0.0
    mean_bias = float(np.mean(biases))
    sd_bias   = float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0
    mean_mae  = float(np.mean(maes))
    sd_mae    = float(np.std(maes,   ddof=1)) if len(maes)   > 1 else 0.0

    group_quality = _quality_label(mean_mape)
    n = len(reports)
    direction = "overestimates" if mean_bias > 0 else "underestimates"
    device_name = reports[0].metadata.wearable_device_name if reports[0].metadata else "The wearable"
    group_summary = (
        f"Across {n} athlete(s), {device_name} showed "
        f"{QUALITY_LABELS[group_quality].lower()} agreement with the reference "
        f"(mean MAPE = {mean_mape:.1f}% ± {sd_mape:.1f}%). "
        f"On average it {direction} HR by {abs(mean_bias):.1f} BPM. "
        f"Pooled 95% LoA: {pooled_loa_lo:.1f} to {pooled_loa_hi:.1f} BPM."
    )

    return GroupAnalysisReport(
        athlete_reports=reports,
        mean_bias=mean_bias, sd_bias=sd_bias,
        mean_mae=mean_mae,   sd_mae=sd_mae,
        mean_mape=mean_mape, sd_mape=sd_mape,
        pooled_loa_lower=pooled_loa_lo,
        pooled_loa_upper=pooled_loa_hi,
        n_athletes=n,
        group_quality_label=group_quality,
        group_summary_text=group_summary,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(data: HRDataSeries) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(data.hr_wearable, dtype=float)
    r = np.asarray(data.hr_reference, dtype=float)
    if len(w) != len(r):
        raise ValueError(
            f"Array length mismatch: hr_wearable={len(w)}, hr_reference={len(r)}. "
            "Data must be aligned before analysis."
        )
    valid = ~(np.isnan(w) | np.isnan(r))
    return w[valid], r[valid]


def _quality_label(mape: float) -> str:
    if mape < MAPE_EXCELLENT_THRESHOLD:
        return "excellent"
    elif mape < MAPE_GOOD_THRESHOLD:
        return "good"
    elif mape < MAPE_ACCEPTABLE_THRESHOLD:
        return "acceptable"
    return "poor"
