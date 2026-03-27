from __future__ import annotations
from wearable_validation.models import TestRunMetadata, AnalysisReport, GroupAnalysisReport
from wearable_validation.constants import QUALITY_LABELS


# ── Public functions ──────────────────────────────────────────────────────────

def build_text_blocks(
    bias: float,
    mae: float,
    mape: float,
    loa_lower: float,
    loa_upper: float,
    n_samples: int,
    n_outliers: int,
    quality: str,
    metadata: TestRunMetadata | None,
) -> tuple[str, str, str]:
    """Return (summary_text, interpretation_text, limitations_text)."""
    return (
        _summary(bias, mape, quality, metadata),
        _interpretation(mape, loa_lower, loa_upper, quality),
        _limitations(n_samples, n_outliers, metadata),
    )


def format_report(report: AnalysisReport) -> str:
    """Render a complete plain-text report from a fully populated AnalysisReport."""
    meta = report.metadata
    device_line = ""
    if meta:
        device_line = (
            f"  Athlete  : {meta.athlete_name}\n"
            f"  Wearable : {meta.wearable_device_name} ({meta.wearable_type})\n"
            f"  Reference: {meta.reference_device_name} ({meta.reference_type})\n"
            f"  Date     : {meta.test_date}\n"
        )
        if meta.conditions:
            device_line += f"  Conditions: {meta.conditions}\n"

    return (
        f"{'='*62}\n"
        f"  WEARABLE HR VALIDATION REPORT\n"
        f"{'='*62}\n"
        f"{device_line}"
        f"\n--- SUMMARY ---\n{report.summary_text}\n"
        f"\n--- NUMERICAL RESULTS ---\n"
        f"  Bias (mean error)        : {report.bias:+.2f} BPM\n"
        f"  MAE                      : {report.mae:.2f} BPM\n"
        f"  MAPE                     : {report.mape:.2f}%\n"
        f"  Limits of Agreement (95%): [{report.loa_lower:.2f}, {report.loa_upper:.2f}] BPM\n"
        f"  Paired samples (N)       : {report.n_samples}\n"
        f"  Outliers flagged (>2 SD) : {report.n_outliers_flagged}\n"
        f"  Quality label            : {QUALITY_LABELS[report.quality_label]}\n"
        f"\n--- INTERPRETATION ---\n{report.interpretation_text}\n"
        f"\n--- LIMITATIONS ---\n{report.limitations_text}\n"
        f"{'='*62}\n"
    )


def format_group_report(group_report: GroupAnalysisReport) -> str:
    """Render a plain-text group summary report."""
    per_athlete = ""
    for r in group_report.athlete_reports:
        name = r.metadata.athlete_name if r.metadata else "Unknown"
        per_athlete += (
            f"  {name:<20} MAPE={r.mape:.1f}%  Bias={r.bias:+.1f} BPM  "
            f"MAE={r.mae:.1f} BPM  LoA=[{r.loa_lower:.1f}, {r.loa_upper:.1f}]  "
            f"{QUALITY_LABELS[r.quality_label]}\n"
        )

    return (
        f"{'='*62}\n"
        f"  GROUP HR VALIDATION REPORT ({group_report.n_athletes} athletes)\n"
        f"{'='*62}\n"
        f"\n--- GROUP SUMMARY ---\n{group_report.group_summary_text}\n"
        f"\n--- PER-ATHLETE RESULTS ---\n{per_athlete}"
        f"\n--- GROUP STATISTICS ---\n"
        f"  Mean MAPE         : {group_report.mean_mape:.2f}% ± {group_report.sd_mape:.2f}%\n"
        f"  Mean Bias         : {group_report.mean_bias:+.2f} ± {group_report.sd_bias:.2f} BPM\n"
        f"  Mean MAE          : {group_report.mean_mae:.2f} ± {group_report.sd_mae:.2f} BPM\n"
        f"  Pooled LoA (95%)  : [{group_report.pooled_loa_lower:.2f}, {group_report.pooled_loa_upper:.2f}] BPM\n"
        f"  Group quality     : {QUALITY_LABELS[group_report.group_quality_label]}\n"
        f"\n--- LIMITATIONS ---\n"
        f"  - Results based on a single session per athlete. Replication is required.\n"
        f"  - Pooled LoA assume all athletes used the same device model and protocol.\n"
        f"  - Between-athlete variability in MAPE (SD={group_report.sd_mape:.1f}%) may reflect\n"
        f"    differences in skin tone, wrist circumference, or movement patterns.\n"
        f"  - These thresholds are heuristic. LoA are the primary validity indicator.\n"
        f"{'='*62}\n"
    )


# ── Private text builders ─────────────────────────────────────────────────────

def _summary(
    bias: float,
    mape: float,
    quality: str,
    meta: TestRunMetadata | None,
) -> str:
    device = meta.wearable_device_name if meta else "The wearable device"
    direction = "overestimates" if bias > 0 else "underestimates"
    return (
        f"{device} showed {QUALITY_LABELS[quality].lower()} agreement with the "
        f"reference standard (MAPE = {mape:.1f}%). "
        f"On average it {direction} HR by {abs(bias):.1f} BPM."
    )


def _interpretation(
    mape: float,
    loa_lower: float,
    loa_upper: float,
    quality: str,
) -> str:
    use_case_map = {
        "excellent": (
            "The device is suitable for research-grade HR monitoring and "
            "precision training-load applications."
        ),
        "good": (
            "The device is suitable for most training and health monitoring uses. "
            "Caution is warranted for clinical or research contexts where MAPE < 3% is required."
        ),
        "acceptable": (
            "The device provides a general indication of HR zones but should not be used "
            "for precise pace or load prescription. Suitable for recreational monitoring only."
        ),
        "poor": (
            "Accuracy is insufficient for reliable HR monitoring. "
            "Investigate sensor placement, motion artefact, or device malfunction before relying on data."
        ),
    }
    return (
        f"The 95% limits of agreement span {loa_lower:.1f} to {loa_upper:.1f} BPM, "
        f"meaning 95% of individual measurements fall within this range of the reference.\n"
        f"{use_case_map[quality]}\n"
        f"Note: MAPE thresholds are heuristic (see Navalta et al. 2020, INTERLIVE 2020). "
        f"LoA are the primary validity indicator."
    )


def _limitations(
    n_samples: int,
    n_outliers: int,
    meta: TestRunMetadata | None,
) -> str:
    lines = [
        "- Single-session results. Independent replication is required before drawing "
        "generalisable conclusions.",
        f"- {n_outliers} data point(s) exceeded 2 SD of the mean difference and were flagged "
        "but not excluded. Manual review is recommended.",
        "- Accuracy may differ across HR ranges, populations, and environmental conditions "
        "not tested here.",
        "- Alignment assumes the two timestamps are synchronised. Any clock drift between "
        "devices will inflate bias estimates.",
    ]
    if meta and meta.wearable_type in ("wrist_based_ppg", "finger_based_ppg"):
        if meta.sport == "cycling" and meta.wearable_type == "finger_based_ppg":
            lines.append(
                "- Finger-based PPG during cycling: handlebar grip pressure can compress the "
                "sensor and introduce motion artefact. Results should be interpreted with caution."
            )
        elif meta.sport == "cycling":
            lines.append(
                "- PPG-based devices are susceptible to motion artefact during cycling, "
                "particularly at high cadences, and may show reduced accuracy at >85% HRmax."
            )
        else:
            lines.append(
                "- PPG-based devices are susceptible to motion artefact during high-cadence "
                "running and may show reduced accuracy at >85% HRmax."
            )
    return "\n".join(lines)
