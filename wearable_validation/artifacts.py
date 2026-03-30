"""
Artifact detection for aligned wearable HR timeseries.

Three checks are applied to each channel independently:
  1. Out-of-range  — HR below or above physiological bounds
  2. Spike         — instantaneous ΔHR exceeding a rate-of-change threshold;
                     characteristic of PPG motion artifacts which appear as
                     sudden jumps rather than gradual changes
  3. Flatline      — identical value held for ≥ N consecutive seconds;
                     indicates signal dropout or disconnected sensor

Threshold rationale and references
───────────────────────────────────
HR_MIN_BPM (25 BPM)
  Lowest resting HR ever recorded in elite endurance athletes is ~27–28 BPM.
  During exercise, values below 25 BPM are physiologically impossible.
  Basis: Camm et al. (2003) ESC Guidelines; McArdle, Katch & Katch (2015)
  Exercise Physiology.

HR_MAX_BPM (220 BPM)
  The canonical upper bound for human exercise HR used throughout sports
  science. No healthy individual sustains above 220 BPM.
  Basis: Tanaka et al. (2001) Med Sci Sports Exerc; Fox et al. (1971)
  Am Heart J.

SPIKE_THRESHOLD_BPM_PER_SEC (20 BPM/s)
  The rate-of-change approach as a signal quality criterion is established
  in Orphanidou et al. (2015) IEEE J Biomed Health Inform. The physiological
  upper bound for real HR transitions is ~3–5 BPM/s even during maximal
  sprint efforts (Jones et al. 1999 J Physiol; Bellenger et al. 2016
  Sports Med Review). 20 BPM/s is 4× this maximum, meaning any
  sample-to-sample jump above this threshold is almost certainly artifactual
  rather than a real cardiac response.
  Note: the specific 20 BPM/s value is a pragmatic choice grounded in the
  physiological rate-of-change literature above. No single paper specifies
  this exact cutoff; it should be treated as a conservative heuristic.

FLATLINE_MIN_SECONDS (8 s)
  After 1 Hz linear interpolation, genuinely identical consecutive values
  indicate raw-signal dropout (sensor disconnected, firmware freeze). 8 s
  is long enough to exclude true steady-state plateaus (which show small
  sample-to-sample variation even at constant pace) and short enough to
  catch real dropouts promptly.
  Methodology basis: Orphanidou et al. (2015) IEEE J Biomed Health Inform;
  Charlton et al. (2022) Physiol. Meas. (PPG quality assessment review).
"""
from __future__ import annotations
import numpy as np

from wearable_validation.models import HRDataSeries, ArtifactReport

# ── Thresholds ────────────────────────────────────────────────────────────────

# Physiological range — conservative bounds; values outside these are impossible
# for a human to sustain, regardless of fitness level.
HR_MIN_BPM: float = 25.0    # below → dropout or sensor-off artefact
HR_MAX_BPM: float = 220.0   # above → spike artefact (no human HR exceeds this)

# Rate-of-change spike threshold (BPM per second at 1 Hz sampling).
# Real cardiac output changes < 5 BPM/s even during maximal transitions
# (Robergs & Landwehr 2002). PPG motion spikes are instantaneous by comparison.
SPIKE_THRESHOLD_BPM_PER_SEC: float = 20.0

# Flatline detection — number of consecutive identical-value samples at 1 Hz.
# 8 s is long enough to distinguish a genuine momentary plateau from a dropout.
FLATLINE_MIN_SECONDS: int = 8


# ── Core detection ────────────────────────────────────────────────────────────

def _detect_channel(
    hr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Run all three artifact checks on a single HR channel.

    Returns
    -------
    combined_mask, oor_mask, spike_mask, flatline_mask : np.ndarray[bool]
    reasons : list[str]
    """
    n = len(hr)
    reasons: list[str] = []

    # 1 — Out-of-range
    oor_mask = (hr < HR_MIN_BPM) | (hr > HR_MAX_BPM)
    if oor_mask.sum():
        reasons.append(
            f"{int(oor_mask.sum())} out-of-range sample(s) "
            f"(< {HR_MIN_BPM:.0f} or > {HR_MAX_BPM:.0f} BPM)"
        )

    # 2 — Spike (rate-of-change)
    spike_mask = np.zeros(n, dtype=bool)
    if n > 1:
        delta = np.abs(np.diff(hr))
        spike_transitions = np.where(delta > SPIKE_THRESHOLD_BPM_PER_SEC)[0]
        for idx in spike_transitions:
            spike_mask[idx] = True
            if idx + 1 < n:
                spike_mask[idx + 1] = True
        new_spikes = spike_mask & ~oor_mask
        if new_spikes.sum():
            reasons.append(
                f"{int(new_spikes.sum())} spike sample(s) "
                f"(ΔHR > {SPIKE_THRESHOLD_BPM_PER_SEC:.0f} BPM/s)"
            )

    # 3 — Flatline (signal dropout)
    flatline_mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j = i + 1
        while j < n and hr[j] == hr[i]:
            j += 1
        if j - i >= FLATLINE_MIN_SECONDS:
            flatline_mask[i:j] = True
        i = j
    new_flatlines = flatline_mask & ~oor_mask & ~spike_mask
    if new_flatlines.sum():
        reasons.append(
            f"{int(new_flatlines.sum())} flatline sample(s) "
            f"(≥ {FLATLINE_MIN_SECONDS}s constant value)"
        )

    combined_mask = oor_mask | spike_mask | flatline_mask
    return combined_mask, oor_mask, spike_mask, flatline_mask, reasons


def detect_artifacts(data: HRDataSeries) -> ArtifactReport:
    """
    Detect physiologically implausible samples in both HR channels.

    The wearable channel is expected to produce more artifacts (motion, poor
    contact, ambient light); the reference channel is checked separately so
    users can identify problems with the chest strap too.

    Parameters
    ----------
    data : HRDataSeries
        Aligned 1 Hz timeseries from ``align_timeseries()``.

    Returns
    -------
    ArtifactReport
        Contains per-channel boolean masks and human-readable reason lists.
        Use ``apply_artifact_exclusion()`` to get a cleaned HRDataSeries.
    """
    w = np.asarray(data.hr_wearable, dtype=float)
    r = np.asarray(data.hr_reference, dtype=float)

    w_mask, w_oor, w_spike, w_flat, w_reasons = _detect_channel(w)
    r_mask, r_oor, r_spike, r_flat, r_reasons = _detect_channel(r)
    combined = w_mask | r_mask

    return ArtifactReport(
        n_total=len(w),
        n_flagged_wearable=int(w_mask.sum()),
        n_flagged_reference=int(r_mask.sum()),
        wearable_mask=w_mask,
        reference_mask=r_mask,
        combined_mask=combined,
        flag_reasons_wearable=w_reasons,
        flag_reasons_reference=r_reasons,
        oor_mask_wearable=w_oor,
        spike_mask_wearable=w_spike,
        flatline_mask_wearable=w_flat,
        oor_mask_reference=r_oor,
        spike_mask_reference=r_spike,
        flatline_mask_reference=r_flat,
    )


def apply_artifact_exclusion(
    data: HRDataSeries,
    artifact_report: ArtifactReport,
) -> HRDataSeries:
    """
    Return a new HRDataSeries with all flagged samples removed.

    Both channels are cleaned together — a sample flagged in either channel
    is excluded from both, preserving pair-wise alignment.
    """
    keep = ~artifact_report.combined_mask
    return HRDataSeries(
        hr_wearable=data.hr_wearable[keep],
        hr_reference=data.hr_reference[keep],
        timestamps=data.timestamps[keep],
    )
