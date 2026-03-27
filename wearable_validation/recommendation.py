"""
Use-case based onboarding recommendation engine.

Maps accuracy statistics (MAPE, LoA span) to structured
go / caution / no-go recommendations per coach-selected use case.

Thresholds informed by:
  Navalta et al. (2020) JFMK; Gillinov et al. (2017) MSSE;
  Passler et al. (2019) Sensors; INTERLIVE Düking et al. (2020).
"""
from __future__ import annotations

from wearable_validation.models import (
    AnalysisReport,
    UseCaseRecommendation,
    OnboardingRecommendation,
)
from wearable_validation.constants import (
    USE_CASES,
    RECOMMENDATION_CAUTION_MAPE_BUFFER,
    RECOMMENDATION_CAUTION_LOA_BUFFER,
)


def generate_recommendation(
    report: AnalysisReport,
    selected_use_cases: list[str],
) -> OnboardingRecommendation | None:
    """
    Generate a structured onboarding recommendation for each selected use case.

    Returns None if no use cases are selected.
    Each use case is assessed against MAPE and LoA span thresholds.
    A caution buffer is applied so borderline results are flagged rather than
    hard-classified as not recommended.
    """
    if not selected_use_cases:
        return None

    loa_span = report.loa_upper - report.loa_lower
    recommendations: list[UseCaseRecommendation] = []

    for key in selected_use_cases:
        if key not in USE_CASES:
            continue
        uc = USE_CASES[key]
        mape_thresh = uc["mape_threshold"]
        loa_thresh  = uc["loa_span_threshold"]

        mape_ok      = report.mape < mape_thresh
        loa_ok       = loa_span   < loa_thresh
        mape_caution = report.mape < mape_thresh + RECOMMENDATION_CAUTION_MAPE_BUFFER
        loa_caution  = loa_span   < loa_thresh  + RECOMMENDATION_CAUTION_LOA_BUFFER

        if mape_ok and loa_ok:
            status = "suitable"
            reason = (
                f"MAPE {report.mape:.1f}% meets the ≤{mape_thresh:.0f}% threshold "
                f"and LoA span {loa_span:.1f} BPM meets the ≤{loa_thresh:.0f} BPM threshold."
            )
        elif mape_caution and loa_caution:
            status = "caution"
            issues = []
            if not mape_ok:
                issues.append(
                    f"MAPE {report.mape:.1f}% slightly exceeds the {mape_thresh:.0f}% threshold"
                )
            if not loa_ok:
                issues.append(
                    f"LoA span {loa_span:.1f} BPM slightly exceeds the {loa_thresh:.0f} BPM threshold"
                )
            reason = "; ".join(issues) + ". Individual measurement variability should be communicated to athletes."
        else:
            status = "not_recommended"
            issues = []
            if not mape_caution:
                issues.append(
                    f"MAPE {report.mape:.1f}% is above the {mape_thresh:.0f}% accuracy "
                    f"threshold required for this use case"
                )
            if not loa_caution:
                issues.append(
                    f"LoA span {loa_span:.1f} BPM is above the {loa_thresh:.0f} BPM "
                    f"precision threshold required for this use case"
                )
            reason = "; ".join(issues) + ". Consider validating under more controlled conditions or evaluating an alternative device for this application."

        recommendations.append(UseCaseRecommendation(
            use_case_key=key,
            use_case_label=uc["label"],
            status=status,
            reason=reason,
        ))

    if not recommendations:
        return None

    statuses = [r.status for r in recommendations]
    n = len(statuses)
    n_suitable = statuses.count("suitable")
    n_not      = statuses.count("not_recommended")

    if n_suitable == n:
        overall = "suitable"
        summary = "This device meets accuracy requirements for all selected use cases."
    elif n_not == n:
        overall = "not_recommended"
        summary = (
            "This device's accuracy falls below the required threshold for all selected "
            "use cases. It may still be suitable for lower-precision applications not listed here."
        )
    else:
        overall = "caution"
        summary = (
            f"This device meets requirements for {n_suitable} of {n} selected use case(s). "
            "Review individual use case results below."
        )

    return OnboardingRecommendation(
        recommendations=recommendations,
        overall_verdict=overall,
        summary_text=summary,
    )
