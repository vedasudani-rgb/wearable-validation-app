from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── Protocol inputs ───────────────────────────────────────────────────────────

@dataclass
class ProtocolParams:
    sport: str                        # "running" | "cycling"
    metric: str                       # "heart_rate"
    wearable_type: str                # "wrist_based_ppg" | "finger_based_ppg"
    reference_type: str               # "chest_strap_ecg" | "chest_strap_hr"
    context: str                      # "steady_run" | "interval_session" | "steady_ride" | "interval_ride"
    wearable_device_name: str
    reference_device_name: str
    test_date: str                    # ISO-8601 date string
    age: Optional[int] = None         # used for Tanaka HRmax calculation


# ── Protocol building blocks ──────────────────────────────────────────────────

@dataclass
class ProtocolStep:
    name: str
    duration_sec: int
    intensity_label: str
    target_rpe: str
    target_hr_pct: str
    instructions: str
    target_hr_bpm: Optional[str] = None  # set when age provided


@dataclass
class ProtocolStepBoundary:
    """Time window for one protocol step — used for HR zone coverage validation."""
    step_name: str
    intensity_label: str
    start_sec: float
    end_sec: float
    target_hr_pct_low: float          # fraction of HRmax (e.g. 0.70)
    target_hr_pct_high: float


@dataclass
class Protocol:
    context: str
    steps: list[ProtocolStep]
    device_instructions: str
    estimated_duration_min: float
    n_expected_samples: int
    hrmax: Optional[int] = None
    step_boundaries: list[ProtocolStepBoundary] = field(default_factory=list)


# ── Test run metadata ─────────────────────────────────────────────────────────

@dataclass
class TestRunMetadata:
    test_date: str
    athlete_name: str
    wearable_device_name: str
    reference_device_name: str
    sport: str
    metric: str
    wearable_type: str
    reference_type: str
    conditions: str = ""


# ── Raw aligned data ──────────────────────────────────────────────────────────

@dataclass
class HRDataSeries:
    hr_wearable: np.ndarray           # shape (N,), BPM floats
    hr_reference: np.ndarray          # shape (N,), BPM floats
    timestamps: np.ndarray            # shape (N,), seconds from start


# ── Per-athlete analysis output ───────────────────────────────────────────────

@dataclass
class AnalysisReport:
    bias: float
    mae: float
    mape: float
    loa_lower: float
    loa_upper: float
    n_samples: int
    n_outliers_flagged: int
    quality_label: str
    summary_text: str
    interpretation_text: str
    limitations_text: str
    metadata: Optional[TestRunMetadata] = None


# ── Intensity-stratified accuracy ─────────────────────────────────────────────

@dataclass
class IntensityBinResult:
    bin_label: str
    hr_lower: float
    hr_upper: float
    n_samples: int
    bias: float
    mae: float
    mape: float
    sufficient_data: bool             # False if n_samples < HR_BIN_MIN_SAMPLES


# ── HR zone coverage validation ───────────────────────────────────────────────

@dataclass
class StepCoverageResult:
    step_name: str
    intensity_label: str
    target_hr_pct: str
    target_bpm_range: Optional[str]   # e.g. "150–159 BPM", None if no hrmax
    actual_median_bpm: float
    n_samples_in_step: int
    status: str                       # "met" | "partial" | "not_reached" | "unknown"


@dataclass
class CoverageReport:
    step_results: list[StepCoverageResult]
    overall_status: str               # "complete" | "partial" | "insufficient" | "unknown"
    warning_message: str


# ── Onboarding recommendation ─────────────────────────────────────────────────

@dataclass
class UseCaseRecommendation:
    use_case_key: str
    use_case_label: str
    status: str                       # "suitable" | "caution" | "not_recommended"
    reason: str


@dataclass
class OnboardingRecommendation:
    recommendations: list[UseCaseRecommendation]
    overall_verdict: str              # "suitable" | "caution" | "not_recommended"
    summary_text: str


# ── Device comparison ─────────────────────────────────────────────────────────

@dataclass
class DeviceComparisonEntry:
    device_name: str
    wearable_type: str
    report: AnalysisReport
    data: HRDataSeries
    rank: int                         # 1 = best (lowest MAPE)


@dataclass
class DeviceComparisonReport:
    entries: list[DeviceComparisonEntry]   # sorted by MAPE ascending
    reference_device_name: str
    n_devices: int
    best_device_name: str
    summary_text: str


# ── Artifact detection ────────────────────────────────────────────────────────

@dataclass
class ArtifactReport:
    n_total: int
    n_flagged_wearable: int           # samples flagged in wearable channel
    n_flagged_reference: int          # samples flagged in reference channel
    wearable_mask: np.ndarray         # bool (N,), True = flagged (any type)
    reference_mask: np.ndarray        # bool (N,), True = flagged (any type)
    combined_mask: np.ndarray         # True if flagged in either channel
    flag_reasons_wearable: list[str]  # human-readable breakdown per flag type
    flag_reasons_reference: list[str]
    # Per-type masks for plot legend differentiation
    oor_mask_wearable: np.ndarray     # out-of-range
    spike_mask_wearable: np.ndarray   # rate-of-change spike
    flatline_mask_wearable: np.ndarray
    oor_mask_reference: np.ndarray
    spike_mask_reference: np.ndarray
    flatline_mask_reference: np.ndarray

    @property
    def n_flagged_combined(self) -> int:
        return int(self.combined_mask.sum())

    @property
    def pct_flagged(self) -> float:
        return 100.0 * self.n_flagged_combined / self.n_total if self.n_total > 0 else 0.0

    @property
    def has_artifacts(self) -> bool:
        return self.n_flagged_combined > 0


# ── Group (multi-athlete) analysis output ────────────────────────────────────

@dataclass
class GroupAnalysisReport:
    athlete_reports: list[AnalysisReport]
    mean_bias: float
    sd_bias: float
    mean_mae: float
    sd_mae: float
    mean_mape: float
    sd_mape: float
    pooled_loa_lower: float
    pooled_loa_upper: float
    n_athletes: int
    group_quality_label: str
    group_summary_text: str
