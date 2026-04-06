"""
Central store for all magic numbers, labels, and supported-value sets.
Nothing in the codebase hardcodes these values directly.
"""

# ── Sampling ──────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE_HZ: float = 1.0

# ── Bland-Altman ─────────────────────────────────────────────────────────────
LOA_Z: float = 1.96
MIN_SAMPLES_WARNING: int = 100
MIN_OVERLAP_SECONDS: int = 60

# ── MAPE quality thresholds ───────────────────────────────────────────────────
# Synthesised from Navalta et al. (2020) JFMK, Gillinov et al. (2017),
# Passler et al. (2019), INTERLIVE Düking et al. (2020).
# Literature most commonly uses a single ≤5% threshold; this 4-level scheme
# adds practitioner granularity and is explicitly labelled as heuristic.
MAPE_EXCELLENT_THRESHOLD: float = 3.0   # < 3%
MAPE_GOOD_THRESHOLD: float = 5.0        # 3–5%
MAPE_ACCEPTABLE_THRESHOLD: float = 10.0 # 5–10%
# > 10% → "poor"

QUALITY_LABELS: dict[str, str] = {
    "excellent": "Excellent (MAPE < 3%)",
    "good":      "Good (MAPE 3–5%)",
    "acceptable": "Acceptable (MAPE 5–10%)",
    "poor":      "Poor (MAPE > 10%)",
}

QUALITY_COLOURS: dict[str, str] = {
    "excellent": "#2ecc71",
    "good":      "#3498db",
    "acceptable": "#f39c12",
    "poor":      "#e74c3c",
}

# ── HRmax formula ─────────────────────────────────────────────────────────────
# Tanaka et al. (2001) Medicine & Science in Sports & Exercise
# Most cited HRmax formula in sports science literature (>2000 citations)
TANAKA_INTERCEPT: float = 208.0
TANAKA_SLOPE: float = 0.7
HRMAX_SE_BPM: int = 7  # approximate population-level SE

# ── Supported values ──────────────────────────────────────────────────────────
SUPPORTED_SPORTS: frozenset[str] = frozenset({"running", "cycling"})
SUPPORTED_METRICS: frozenset[str] = frozenset({"heart_rate"})
SUPPORTED_CONTEXTS: frozenset[str] = frozenset({
    "steady_run", "interval_session",
    "steady_ride", "interval_ride",
})
SPORT_CONTEXTS: dict[str, list[str]] = {
    "running": ["steady_run", "interval_session"],
    "cycling": ["steady_ride", "interval_ride"],
}
SUPPORTED_WEARABLE_TYPES: frozenset[str] = frozenset({
    "wrist_based_ppg",
    "finger_based_ppg",
})
SUPPORTED_REFERENCE_TYPES: frozenset[str] = frozenset({
    "chest_strap_ecg",
    "chest_strap_hr",
})

# ── Intensity → approximate baseline HR (BPM) for synthetic data / simulation ─
INTENSITY_BASELINE_HR: dict[str, float] = {
    "warm_up":          70.0,
    "easy":             130.0,
    "moderate":         150.0,
    "comfortably_hard": 165.0,
    "hard":             178.0,
    "recovery":         125.0,
    "cool_down":        100.0,
}

# ── Intensity zone %HRmax bounds ─────────────────────────────────────────────
INTENSITY_HR_PCT: dict[str, tuple[float, float]] = {
    "warm_up":          (0.50, 0.60),
    "easy":             (0.60, 0.65),
    "moderate":         (0.70, 0.75),
    "comfortably_hard": (0.80, 0.85),
    "hard":             (0.88, 0.92),
    "recovery":         (0.55, 0.65),
    "cool_down":        (0.00, 0.60),
}

# ── HR bins for intensity-stratified accuracy analysis ────────────────────────
# Each entry: (lower_bpm_inclusive, upper_bpm_exclusive, display_label)
# Bins are defined on the reference HR channel.
HR_BINS: list[tuple[float, float, str]] = [
    (0,   130, "< 130 BPM (Easy)"),
    (130, 150, "130–150 BPM (Moderate)"),
    (150, 170, "150–170 BPM (Hard)"),
    (170, 999, "> 170 BPM (Very Hard)"),
]
HR_BIN_MIN_SAMPLES: int = 10  # bins with fewer samples show as insufficient

# ── Use-case accuracy thresholds for onboarding recommendation ────────────────
#
# MAPE thresholds
#   ≤5%: synthesised from Navalta et al. (2020) JFMK (most comprehensive review,
#        n=57 studies); Gillinov et al. (2017) MSSE; Passler et al. (2019) Sensors;
#        Düking et al. (2020) INTERLIVE consensus. ≤5% is the most widely cited
#        single-threshold for exercise HR validity in the literature.
#   ≤3%: stricter threshold for high-precision applications (Buchheit & Laursen
#        2013 SJMSS Parts I & II; INTERLIVE Düking et al. 2020).
#
# LoA span thresholds  (span = LoA_upper − LoA_lower = 2 × 1.96 × SD_diff)
#   ≤20 BPM (span = ±10 BPM): load monitoring tolerates random error because
#        session-average HR is used (TRIMP, Edwards load); errors cancel across
#        a session. Basis: Borresen & Lambert (2009) Int J Sports Physiol Perform;
#        Gillinov et al. (2017) found most acceptable devices within ±10 BPM.
#   ≤15 BPM (span = ±7.5 BPM): HR zone width in the 5-zone Coggan/ACSM model
#        is ~10–15 BPM; device error must be < zone width to reliably classify
#        effort. Basis: Seiler & Kjerland (2006) Scand J Med Sci Sports;
#        Lucia et al. (2001) Med Sci Sports Exerc; Gillinov et al. (2017).
#   ≤10 BPM (span = ±5 BPM): real-time interval pacing requires knowing effort
#        within ±5 BPM of target; wider LoA renders interval prescription
#        unreliable. Basis: Buchheit & Laursen (2013) SJMSS; Navalta et al. (2020).
#   ≤15 BPM (recovery): post-exercise HR recovery (HRR60, HRR120) is clinically
#        meaningful at ~10 BPM resolution; a span >15 BPM masks real recovery
#        differences. Basis: Buchheit et al. (2010) Eur J Appl Physiol;
#        Plews et al. (2013) Int J Sports Physiol Perform.
#
USE_CASES: dict[str, dict] = {
    "load_monitoring": {
        "label":               "Daily Load Monitoring",
        "description":         (
            "Tracks session training load over time using average HR or metrics like TRIMP. "
            "Errors average out across a full session, so a ±5% tolerance is acceptable."
        ),
        "mape_threshold":      5.0,
        "loa_span_threshold":  20.0,   # Borresen & Lambert (2009); Gillinov et al. (2017)
    },
    "zone_training": {
        "label":               "HR Zone Training",
        "description":         (
            "Keeps athletes within a target effort zone (e.g. Zone 2 aerobic, Zone 4 threshold). "
            "HR zones are ~10–15 BPM wide — device error must stay below that to reliably identify the correct zone."
        ),
        "mape_threshold":      5.0,
        "loa_span_threshold":  15.0,   # Seiler & Kjerland (2006); Lucia et al. (2001)
    },
    "interval_pacing": {
        "label":               "Interval Pacing",
        "description":         (
            "Provides real-time HR feedback during high-intensity intervals. "
            "Requires the strictest accuracy (±5 BPM) because a misread at peak effort directly affects pacing decisions."
        ),
        "mape_threshold":      3.0,
        "loa_span_threshold":  10.0,   # Buchheit & Laursen (2013); Navalta et al. (2020)
    },
    "recovery_tracking": {
        "label":               "Recovery Tracking",
        "description":         (
            "Monitors resting and post-exercise HR to gauge readiness for the next session. "
            "Recovery differences of ~10 BPM are clinically meaningful, so device error must stay below that threshold."
        ),
        "mape_threshold":      5.0,
        "loa_span_threshold":  15.0,   # Buchheit et al. (2010); Plews et al. (2013)
    },
}

# Within this margin above the threshold → "caution" rather than "not recommended"
RECOMMENDATION_CAUTION_MAPE_BUFFER: float = 2.0   # percentage points
RECOMMENDATION_CAUTION_LOA_BUFFER: float = 5.0    # BPM

# ── HR zone coverage check ────────────────────────────────────────────────────
# A step is flagged if its median reference HR falls more than this fraction of
# HRmax below the step's target floor (e.g. 0.05 = 5% HRmax below target)
COVERAGE_TOLERANCE_PCT: float = 0.05
