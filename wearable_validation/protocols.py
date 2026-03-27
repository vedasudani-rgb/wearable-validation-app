from __future__ import annotations
from wearable_validation.models import (
    ProtocolParams, Protocol, ProtocolStep, ProtocolStepBoundary,
)
from wearable_validation.constants import (
    DEFAULT_SAMPLE_RATE_HZ,
    SUPPORTED_SPORTS,
    SUPPORTED_METRICS,
    SUPPORTED_CONTEXTS,
    SUPPORTED_WEARABLE_TYPES,
    SPORT_CONTEXTS,
    TANAKA_INTERCEPT,
    TANAKA_SLOPE,
    HRMAX_SE_BPM,
    INTENSITY_HR_PCT,
)


def compute_hrmax(age: int) -> int:
    """
    Tanaka et al. (2001): HRmax = 208 − 0.7 × age.
    Most-cited HRmax formula in sports science (Medicine & Science in Sports
    & Exercise, >2000 citations). Population SE ≈ ±7 BPM.
    """
    return round(TANAKA_INTERCEPT - TANAKA_SLOPE * age)


def generate_protocol(params: ProtocolParams) -> Protocol:
    """
    Validate params and return a fully populated Protocol.
    Dispatches to sport- and context-specific builders.
    """
    _validate_params(params)
    hrmax = compute_hrmax(params.age) if params.age is not None else None

    builders = {
        "steady_run":       _build_continuous_graded_run,
        "interval_session": _build_interval_run,
        "steady_ride":      _build_continuous_graded_ride,
        "interval_ride":    _build_interval_ride,
    }
    steps = builders[params.context](hrmax)

    total_sec         = sum(s.duration_sec for s in steps)
    estimated_min     = round(total_sec / 60, 1)
    n_expected        = int(total_sec * DEFAULT_SAMPLE_RATE_HZ)
    step_boundaries   = _compute_step_boundaries(steps)

    return Protocol(
        context=params.context,
        steps=steps,
        device_instructions=_device_instructions(params, hrmax),
        estimated_duration_min=estimated_min,
        n_expected_samples=n_expected,
        hrmax=hrmax,
        step_boundaries=step_boundaries,
    )


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_params(params: ProtocolParams) -> None:
    errors = []
    if params.sport not in SUPPORTED_SPORTS:
        errors.append(f"Sport '{params.sport}' not supported. Supported: {sorted(SUPPORTED_SPORTS)}")
    if params.metric not in SUPPORTED_METRICS:
        errors.append(f"Metric '{params.metric}' not supported.")
    valid_contexts = SPORT_CONTEXTS.get(params.sport, [])
    if params.context not in valid_contexts:
        errors.append(
            f"Context '{params.context}' is not valid for sport '{params.sport}'. "
            f"Valid options: {valid_contexts}"
        )
    if params.wearable_type not in SUPPORTED_WEARABLE_TYPES:
        errors.append(
            f"Wearable type '{params.wearable_type}' not supported. "
            f"Supported: {sorted(SUPPORTED_WEARABLE_TYPES)}"
        )
    if params.age is not None and not (10 <= params.age <= 100):
        errors.append(f"Age {params.age} is outside the plausible range (10–100).")
    if errors:
        raise ValueError("\n".join(errors))


# ── Step boundaries ───────────────────────────────────────────────────────────

def _compute_step_boundaries(steps: list[ProtocolStep]) -> list[ProtocolStepBoundary]:
    boundaries = []
    t = 0.0
    for step in steps:
        lo, hi = INTENSITY_HR_PCT.get(step.intensity_label, (0.0, 1.0))
        boundaries.append(ProtocolStepBoundary(
            step_name=step.name,
            intensity_label=step.intensity_label,
            start_sec=t,
            end_sec=t + step.duration_sec,
            target_hr_pct_low=lo,
            target_hr_pct_high=hi,
        ))
        t += step.duration_sec
    return boundaries


# ── BPM range helper ──────────────────────────────────────────────────────────

def _bpm_range(intensity_label: str, hrmax: int | None) -> str | None:
    if hrmax is None:
        return None
    lo_pct, hi_pct = INTENSITY_HR_PCT.get(intensity_label, (0.0, 1.0))
    lo_bpm = round(lo_pct * hrmax)
    hi_bpm = round(hi_pct * hrmax)
    if lo_pct == 0.0:
        return f"< {hi_bpm} BPM"
    return f"{lo_bpm}–{hi_bpm} BPM"


# ── Device setup instructions ─────────────────────────────────────────────────

def _device_instructions(params: ProtocolParams, hrmax: int | None) -> str:
    ref_type  = params.reference_type
    wear_type = params.wearable_type
    sport     = params.sport

    # ── Reference setup ──────────────────────────────────────────────────────
    if ref_type == "chest_strap_ecg":
        ref_section = (
            "Reference device setup (chest strap — ECG-validated HR):\n"
            "  1. Moisten the electrode zones on the chest strap with water or electrode gel\n"
            "     to ensure good electrical contact with the skin.\n"
            "  2. Position the strap snugly below the pectoral muscles, with the sensor\n"
            "     module centred on the sternum.\n"
            "  3. The strap should be firm but comfortable — it must not shift during exercise.\n"
            "  4. Confirm the device is transmitting a HR signal before starting the session.\n"
            "  Ref: Stahl et al. (2016) Front. Physiol.; Gillinov et al. (2017) MSSE.\n"
        )
    else:  # chest_strap_hr
        ref_section = (
            "Reference device setup (chest strap — optical HR):\n"
            "  1. Position the strap snugly below the pectoral muscles, with the sensor\n"
            "     pod centred on the sternum or slightly left of centre.\n"
            "  2. Ensure firm, consistent skin contact — the strap must not shift or bounce\n"
            "     during exercise.\n"
            "  3. If the strap has a fabric component, dampen it slightly to improve contact.\n"
            "  4. Confirm the device is transmitting a HR signal before starting the session.\n"
            "  Ref: Gillinov et al. (2017) MSSE; Pasadyn et al. (2019) npj Digit. Med.\n"
        )

    # ── Wearable setup ────────────────────────────────────────────────────────
    if wear_type == "wrist_based_ppg":
        if sport == "running":
            wear_section = (
                "Wearable device setup (wrist-based PPG):\n"
                "  1. Wear the device 1–2 finger-widths above the wrist bone, on the same\n"
                "     wrist each session.\n"
                "  2. The sensor should sit flush against the skin — snug but not so tight\n"
                "     as to restrict circulation.\n"
                "  3. Avoid a loose fit: motion artefact from a loose band is the primary\n"
                "     source of PPG error during running.\n"
                "  4. Remove any jewellery or watches on the same wrist.\n"
                "  Ref: Gillinov et al. (2017) MSSE; Stahl et al. (2016) Front. Physiol.\n"
            )
        else:  # cycling
            wear_section = (
                "Wearable device setup (wrist-based PPG):\n"
                "  1. Wear the device 1–2 finger-widths above the wrist bone.\n"
                "  2. Tighten the strap one notch firmer than your normal daily wear —\n"
                "     handlebar vibration is a known source of motion artefact for wrist-based\n"
                "     PPG during cycling.\n"
                "  3. The sensor must remain flush against the skin throughout the session.\n"
                "  4. Keep the wrist in a consistent position on the handlebar — avoid\n"
                "     resting the wrist on the bar, which can compress the sensor.\n"
                "  5. Remove any jewellery or watches on the same wrist.\n"
                "  Ref: Gillinov et al. (2017) MSSE; Wallen et al. (2016) JSCR.\n"
            )
    else:  # finger_based_ppg
        if sport == "running":
            wear_section = (
                "Wearable device setup (finger-based PPG):\n"
                "  1. Place the device on the index or middle finger of the non-dominant hand.\n"
                "  2. Ensure firm, flush contact between the sensor and the finger pad.\n"
                "  3. The device should not rotate or slide during activity.\n"
                "  4. Avoid rings or jewellery on the same finger.\n"
                "  Ref: Pasadyn et al. (2019) npj Digit. Med.\n"
            )
        else:  # cycling
            wear_section = (
                "Wearable device setup (finger-based PPG):\n"
                "  1. Place the device on the index or middle finger of the hand that grips\n"
                "     most lightly during cycling.\n"
                "  2. Ensure firm, flush contact between the sensor and the finger pad.\n"
                "  3. Note: handlebar grip pressure during cycling can compress the sensor\n"
                "     and introduce motion artefact — this is a significant limitation that\n"
                "     should be recorded in the test limitations and reported alongside results.\n"
                "  4. Avoid rings or jewellery on the same finger.\n"
                "  Ref: Pasadyn et al. (2019) npj Digit. Med.; Gillinov et al. (2017) MSSE.\n"
            )

    # ── Sport-specific general notes ─────────────────────────────────────────
    if sport == "running":
        general_section = (
            "General session instructions:\n"
            "  • Start recording on BOTH devices simultaneously (within 2 seconds).\n"
            "  • Conduct the test on a flat, measured course or treadmill to minimise\n"
            "    speed variation and GPS artefact.\n"
            "  • Minimise arm movements unrelated to running (no phone use, no gesturing).\n"
            "  • Export data with timestamps at the finest available resolution —\n"
            "    the app will align and resample both series automatically.\n"
            "  • Stop recordings on BOTH devices simultaneously at the end of the session.\n"
            "  Ref: Düking et al. (2020) INTERLIVE consensus; Stahl et al. (2016) Front. Physiol.\n"
        )
    else:  # cycling
        general_section = (
            "General session instructions:\n"
            "  • Start recording on BOTH devices simultaneously (within 2 seconds).\n"
            "  • Use a smooth tarmac road or indoor turbo trainer to minimise vibration\n"
            "    artefact — avoid cobblestones or gravel surfaces.\n"
            "  • Maintain a consistent riding position throughout (do not switch between\n"
            "    aero and upright positions — positional changes alter HR response).\n"
            "  • Target cadence: 70–90 RPM throughout the session.\n"
            "  • Export data with timestamps at the finest available resolution —\n"
            "    the app will align and resample both series automatically.\n"
            "  • Stop recordings on BOTH devices simultaneously at the end of the session.\n"
            "  Ref: Lucia et al. (2001) MSSE; Düking et al. (2020) INTERLIVE consensus;\n"
            "       Bouillod et al. (2016) IJSM.\n"
        )

    # ── HRmax note ────────────────────────────────────────────────────────────
    hrmax_note = ""
    if hrmax is not None:
        hrmax_note = (
            f"\nEstimated HRmax (Tanaka et al. 2001): {hrmax} BPM "
            f"(208 − 0.7 × age). Population SE ≈ ±{HRMAX_SE_BPM} BPM — "
            "actual HRmax may differ.\n"
        )

    header = (
        f"Device Setup Instructions\n"
        f"  Wearable  : {params.wearable_device_name} ({params.wearable_type})\n"
        f"  Reference : {params.reference_device_name} ({params.reference_type})\n"
        f"{hrmax_note}"
    )

    return f"{header}\n{ref_section}\n{wear_section}\n{general_section}"


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNING PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_continuous_graded_run(hrmax: int | None) -> list[ProtocolStep]:
    """
    Continuous Graded Run — ~28 min / ~1680 samples at 1 Hz.

    Multiple steady-state intensity blocks covering the full HR range.
    5-min stages allow HR to stabilise before the next increment.
    Evidence base: INTERLIVE consensus (Düking et al. 2020); Stahl et al.
    (2016) Front. Physiol.; Gillinov et al. (2017) MSSE.
    RPE anchors: Borg (1982) CR10 scale.
    """
    spec = [
        ("Warm-Up Walk / Easy Jog",  300, "warm_up",          "RPE 2–3/10", "50–60% HRmax",
         "Walk briskly or jog very easily. This allows both devices to settle and HR to rise "
         "gradually. Confirm both units are recording before moving off."),
        ("Easy Pace",                300, "easy",             "RPE 3/10",   "60–65% HRmax",
         "Maintain a conversational easy jog. You should be able to speak in full sentences "
         "comfortably throughout this block. (Borg 1982; Seiler & Kjerland 2006 IJSPP)"),
        ("Moderate Pace",            300, "moderate",         "RPE 5/10",   "70–75% HRmax",
         "Increase to a steady aerobic pace. Breathing is noticeably deeper but still "
         "controlled. Speaking in short sentences remains possible."),
        ("Comfortably Hard Pace",    300, "comfortably_hard", "RPE 7/10",   "80–85% HRmax",
         "Push to a threshold-adjacent effort. Speech is limited to a few words. "
         "Breathing is heavy but rhythmic. Maintain even foot strike and arm carriage."),
        ("Hard Effort",              180, "hard",             "RPE 8–9/10", "88–92% HRmax",
         "Run hard but in control — approximately 10-K race effort. Conversation is not "
         "possible. Reduce duration if safety is a concern."),
        ("Cool-Down",                300, "cool_down",        "RPE 2/10",   "< 60% HRmax",
         "Return to easy jog or walk. Allow HR to drop below 120 BPM before stopping. "
         "Stop recordings on BOTH devices simultaneously."),
    ]
    return [
        ProtocolStep(
            name=name, duration_sec=dur, intensity_label=label,
            target_rpe=rpe, target_hr_pct=pct, instructions=instr,
            target_hr_bpm=_bpm_range(label, hrmax),
        )
        for name, dur, label, rpe, pct, instr in spec
    ]


def _build_interval_run(hrmax: int | None) -> list[ProtocolStep]:
    """
    Interval Run — ~34 min / ~2040 samples at 1 Hz.

    Six hard/easy intervals assess how quickly wrist PPG tracks rapid HR
    transitions — a known limitation of optical sensors.
    Evidence base: Passler et al. (2019) Sensors; Düking et al. (2020)
    INTERLIVE consensus; Borg (1982) RPE scale.
    """
    steps: list[ProtocolStep] = [
        ProtocolStep(
            name="Warm-Up Jog",
            duration_sec=300,
            intensity_label="warm_up",
            target_rpe="RPE 3/10",
            target_hr_pct="55–65% HRmax",
            instructions=(
                "Easy jog to elevate HR gradually. Confirm both devices are recording "
                "and transmitting before leaving the start point."
            ),
            target_hr_bpm=_bpm_range("warm_up", hrmax),
        ),
    ]
    for i in range(1, 7):
        steps.append(ProtocolStep(
            name=f"Interval {i} – Hard",
            duration_sec=120,
            intensity_label="hard",
            target_rpe="RPE 8/10",
            target_hr_pct="85–92% HRmax",
            instructions=(
                f"Interval {i} of 6: Run hard at a near-maximal sustainable effort. "
                "Maintain consistent pace across all intervals."
            ),
            target_hr_bpm=_bpm_range("hard", hrmax),
        ))
        steps.append(ProtocolStep(
            name=f"Interval {i} – Recovery",
            duration_sec=120,
            intensity_label="recovery",
            target_rpe="RPE 3/10",
            target_hr_pct="55–65% HRmax",
            instructions=(
                f"Recovery jog after interval {i}. Slow to allow genuine HR recovery. "
                "Do not stop — continue moving."
            ),
            target_hr_bpm=_bpm_range("recovery", hrmax),
        ))
    steps.append(ProtocolStep(
        name="Cool-Down",
        duration_sec=300,
        intensity_label="cool_down",
        target_rpe="RPE 2/10",
        target_hr_pct="< 60% HRmax",
        instructions=(
            "Easy jog or walk. Stop recordings on BOTH devices simultaneously "
            "after HR has settled below 120 BPM."
        ),
        target_hr_bpm=_bpm_range("cool_down", hrmax),
    ))
    return steps


# ═══════════════════════════════════════════════════════════════════════════════
# CYCLING PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_continuous_graded_ride(hrmax: int | None) -> list[ProtocolStep]:
    """
    Continuous Graded Ride — ~32 min / ~1920 samples at 1 Hz.

    Stages are 6 min (vs. 5 min for running) because HR stabilisation
    during cycling at submaximal intensities takes longer than running
    (Buchheit & Laursen 2013 Sports Med).
    Evidence base: Gillinov et al. (2017) MSSE; Wallen et al. (2016) JSCR;
    INTERLIVE consensus (Düking et al. 2020).
    Target cadence: 70–90 RPM throughout (Lucia et al. 2001 MSSE).
    """
    spec = [
        ("Warm-Up Easy Spin",        300, "warm_up",          "RPE 2–3/10", "50–60% HRmax",
         "Spin at an easy, comfortable cadence (70–90 RPM). Remain seated. Confirm both "
         "devices are recording before moving off. (Lucia et al. 2001 MSSE)"),
        ("Easy Pace",                360, "easy",             "RPE 3/10",   "60–65% HRmax",
         "Maintain an easy conversational pace. Cadence 80–90 RPM. You should be able to "
         "speak in full sentences comfortably. (Borg 1982 CR10; Seiler & Kjerland 2006 IJSPP)"),
        ("Moderate Pace",            360, "moderate",         "RPE 5/10",   "70–75% HRmax",
         "Steady aerobic effort. Cadence 80–90 RPM. Breathing is deeper but still controlled. "
         "Short sentences possible. Do not change riding position."),
        ("Comfortably Hard Pace",    360, "comfortably_hard", "RPE 7/10",   "80–85% HRmax",
         "Threshold-adjacent effort. Cadence 80–90 RPM. Speech limited to a few words. "
         "Breathing is heavy but rhythmic. Maintain consistent position."),
        ("Hard Effort",              240, "hard",             "RPE 8–9/10", "88–92% HRmax",
         "Near-maximal sustainable effort. Cadence 85–95 RPM. Conversation is not possible. "
         "Maintain consistent riding position throughout this block."),
        ("Cool-Down Spin",           300, "cool_down",        "RPE 2/10",   "< 60% HRmax",
         "Return to easy spin. Allow HR to drop below 120 BPM before stopping. "
         "Stop recordings on BOTH devices simultaneously."),
    ]
    return [
        ProtocolStep(
            name=name, duration_sec=dur, intensity_label=label,
            target_rpe=rpe, target_hr_pct=pct, instructions=instr,
            target_hr_bpm=_bpm_range(label, hrmax),
        )
        for name, dur, label, rpe, pct, instr in spec
    ]


def _build_interval_ride(hrmax: int | None) -> list[ProtocolStep]:
    """
    Interval Ride — ~34 min / ~2040 samples at 1 Hz.

    Six hard/easy intervals assess how accurately wrist PPG tracks rapid HR
    transitions during cycling — particularly relevant given handlebar
    vibration and positional artefact.
    Evidence base: Bouillod et al. (2016) IJSM; Passler et al. (2019) Sensors;
    INTERLIVE consensus (Düking et al. 2020).
    """
    steps: list[ProtocolStep] = [
        ProtocolStep(
            name="Warm-Up Jog",
            duration_sec=300,
            intensity_label="warm_up",
            target_rpe="RPE 3/10",
            target_hr_pct="55–65% HRmax",
            instructions=(
                "Easy spin at 70–80 RPM to elevate HR gradually. Confirm both devices are "
                "recording before leaving the start point."
            ),
            target_hr_bpm=_bpm_range("warm_up", hrmax),
        ),
    ]
    for i in range(1, 7):
        steps.append(ProtocolStep(
            name=f"Interval {i} – Hard",
            duration_sec=120,
            intensity_label="hard",
            target_rpe="RPE 8/10",
            target_hr_pct="85–92% HRmax",
            instructions=(
                f"Interval {i} of 6: Ride hard at a near-maximal sustainable effort. "
                "Cadence 85–95 RPM. Maintain consistent position across all intervals. "
                "(Bouillod et al. 2016 IJSM)"
            ),
            target_hr_bpm=_bpm_range("hard", hrmax),
        ))
        steps.append(ProtocolStep(
            name=f"Interval {i} – Recovery",
            duration_sec=120,
            intensity_label="recovery",
            target_rpe="RPE 3/10",
            target_hr_pct="55–65% HRmax",
            instructions=(
                f"Recovery spin after interval {i}. Reduce effort significantly. "
                "Cadence 70–80 RPM. Do not stop pedalling."
            ),
            target_hr_bpm=_bpm_range("recovery", hrmax),
        ))
    steps.append(ProtocolStep(
        name="Cool-Down Spin",
        duration_sec=300,
        intensity_label="cool_down",
        target_rpe="RPE 2/10",
        target_hr_pct="< 60% HRmax",
        instructions=(
            "Easy spin. Stop recordings on BOTH devices simultaneously "
            "after HR has settled below 120 BPM."
        ),
        target_hr_bpm=_bpm_range("cool_down", hrmax),
    ))
    return steps
