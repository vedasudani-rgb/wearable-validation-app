"""
Wearable HR Validation Tool — Streamlit MVP
Run with: streamlit run app.py --browser.gatherUsageStats false
"""
import io
import warnings
from collections import Counter
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from wearable_validation import (
    ProtocolParams,
    TestRunMetadata,
    generate_protocol,
    compute_hrmax,
    parse_combined_file,
    parse_two_files,
    trim_session,
    detect_artifacts,
    apply_artifact_exclusion,
    analyze_hr_validation,
    analyze_group,
    analyze_by_intensity_bin,
    check_hr_zone_coverage,
    analyze_device_comparison,
    analyze_longitudinal,
    generate_recommendation,
    format_report,
    format_group_report,
    format_longitudinal_report,
)
from wearable_validation.plots import (
    plot_timeseries,
    plot_bland_altman,
    plot_scatter,
    plot_group_bland_altman,
    plot_intensity_bins,
    plot_data_preview,
    plot_longitudinal_trend,
)
from wearable_validation.constants import (
    QUALITY_COLOURS, QUALITY_LABELS, HRMAX_SE_BPM, SPORT_CONTEXTS, USE_CASES,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wearable HR Validation Tool",
    page_icon="🏃",
    layout="wide",
)

st.title("Wearable HR Validation Tool")
st.caption(
    "Generate a field protocol for validating wearable heart rate accuracy "
    "against a reference standard, then upload session data for analysis."
)

# ── Global style fixes ────────────────────────────────────────────────────────
# Normalize the gap between metric label text and its tooltip (?) button so
# it is consistent across all st.metric() calls on the page.
st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] > div {
        gap: 0.25rem;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

SPORT_DISPLAY = {"running": "Running", "cycling": "Cycling"}
SPORT_KEY = {v: k for k, v in SPORT_DISPLAY.items()}

CONTEXT_DISPLAY = {
    "steady_run":       "Continuous Graded Run (~28 min)",
    "interval_session": "Interval Session (~34 min)",
    "steady_ride":      "Continuous Graded Ride (~32 min)",
    "interval_ride":    "Interval Ride (~34 min)",
}

COVERAGE_STATUS_COLOUR = {
    "met":         "#2ecc71",
    "partial":     "#f39c12",
    "not_reached": "#e74c3c",
    "unknown":     "#95a5a6",
    "trimmed":     "#bdc3c7",
}

RECOMMENDATION_STATUS_COLOUR = {
    "suitable":        "#2ecc71",
    "caution":         "#f39c12",
    "not_recommended": "#e74c3c",
}

RECOMMENDATION_ICON = {
    "suitable":        "✅",
    "caution":         "⚠️",
    "not_recommended": "⛔",
}

RECOMMENDATION_LABEL = {
    "suitable":        "Meets Requirements",
    "caution":         "Use with Caution",
    "not_recommended": "Accuracy Insufficient",
}


def _parse_device_instructions(text: str) -> dict:
    """Parse the device_instructions string into structured sections."""
    import re
    result = {"header": {}, "hrmax": None, "reference": None, "wearable": None, "general": None}
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]

    for section in sections:
        lines = [l for l in section.split("\n") if l.strip()]
        if not lines:
            continue
        first = lines[0].strip()

        if first.startswith("Device Setup Instructions"):
            for line in lines[1:]:
                if "Wearable" in line and ":" in line:
                    result["header"]["wearable"] = line.split(":", 1)[1].strip()
                elif "Reference" in line and ":" in line:
                    result["header"]["reference"] = line.split(":", 1)[1].strip()

        elif "Estimated HRmax" in first:
            result["hrmax"] = first

        elif first.lower().startswith("reference device setup"):
            result["reference"] = _parse_numbered_section(lines)

        elif first.lower().startswith("wearable device setup"):
            result["wearable"] = _parse_numbered_section(lines)

        elif first.lower().startswith("general session instructions"):
            result["general"] = _parse_bullets_section(lines)

    return result


def _parse_numbered_section(lines: list) -> dict:
    import re
    header = lines[0].strip().rstrip(":")
    steps, ref, current = [], None, None
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("Ref:"):
            if current:
                steps.append(current); current = None
            ref = s[4:].strip()
        elif re.match(r"^\d+\.", s):
            if current:
                steps.append(current)
            current = re.sub(r"^\d+\.\s*", "", s)
        elif current is not None:
            current += " " + s
    if current:
        steps.append(current)
    return {"header": header, "steps": steps, "ref": ref}


def _parse_bullets_section(lines: list) -> dict:
    header = lines[0].strip().rstrip(":")
    bullets, ref, current = [], None, None
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("Ref:"):
            if current:
                bullets.append(current)
                current = None
            ref = s[4:].strip()
        elif s.startswith("•"):
            if current:
                bullets.append(current)
            current = s[1:].strip()
        elif current is not None:
            current += " " + s   # continuation line — join to current bullet
    if current:
        bullets.append(current)
    return {"header": header, "bullets": bullets, "ref": ref}


def _render_device_instructions(instructions: str) -> None:
    """Render device setup instructions in a two-column card layout."""
    parsed = _parse_device_instructions(instructions)

    if parsed["hrmax"]:
        st.info(parsed["hrmax"])

    col_ref, col_wear = st.columns(2)

    with col_ref:
        with st.container(border=True):
            ref_name = parsed["header"].get("reference", "Reference Device")
            st.markdown(f"**📡 {ref_name}**")
            sec = parsed.get("reference")
            if sec:
                st.caption(sec["header"])
                for i, step in enumerate(sec["steps"], 1):
                    st.markdown(f"{i}. {step}")
                if sec["ref"]:
                    st.caption(f"*{sec['ref']}*")

    with col_wear:
        with st.container(border=True):
            wear_name = parsed["header"].get("wearable", "Wearable Device")
            st.markdown(f"**⌚ {wear_name}**")
            sec = parsed.get("wearable")
            if sec:
                st.caption(sec["header"])
                for i, step in enumerate(sec["steps"], 1):
                    st.markdown(f"{i}. {step}")
                if sec["ref"]:
                    st.caption(f"*{sec['ref']}*")

    sec = parsed.get("general")
    if sec:
        st.markdown("---")
        st.markdown(f"**📋 {sec['header']}**")
        for bullet in sec["bullets"]:
            st.markdown(f"- {bullet}")
        if sec["ref"]:
            st.caption(f"*{sec['ref']}*")


def _make_pdf(figures: list, text_pages: list[str] | None = None) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for text in (text_pages or []):
            lines = text.splitlines()
            for chunk_start in range(0, max(1, len(lines)), 85):
                chunk = "\n".join(lines[chunk_start:chunk_start + 85])
                fig_t = plt.figure(figsize=(8.5, 11))
                fig_t.patch.set_facecolor("white")
                fig_t.text(0.05, 0.95, chunk, transform=fig_t.transFigure,
                           fontsize=7, fontfamily="monospace", verticalalignment="top")
                pdf.savefig(fig_t, bbox_inches="tight")
                plt.close(fig_t)
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _adv_stat_fields(r) -> dict:
    """Return the 7 optional advanced-stat CSV columns for an AnalysisReport."""
    return {
        "pearson_r":          r.pearson_r     if r.pearson_r     is not None else "",
        "r_squared":          r.r_squared     if r.r_squared     is not None else "",
        "see_bpm":            r.see           if r.see           is not None else "",
        "bias_ci_lower_bpm":  r.bias_ci_lower if r.bias_ci_lower is not None else "",
        "bias_ci_upper_bpm":  r.bias_ci_upper if r.bias_ci_upper is not None else "",
        "mape_ci_lower_pct":  r.mape_ci_lower if r.mape_ci_lower is not None else "",
        "mape_ci_upper_pct":  r.mape_ci_upper if r.mape_ci_upper is not None else "",
    }


def _make_single_csv(report) -> bytes:
    r = report
    m = r.metadata
    row = {
        "athlete_name":       m.athlete_name if m else "",
        "device_name":        m.wearable_device_name if m else "",
        "date":               m.test_date if m else "",
        "bias_bpm":           r.bias,
        "mae_bpm":            r.mae,
        "mape_pct":           r.mape,
        "loa_lower_bpm":      r.loa_lower,
        "loa_upper_bpm":      r.loa_upper,
        "n_samples":          r.n_samples,
        "n_outliers_flagged": r.n_outliers_flagged,
        "quality_label":      r.quality_label,
        **_adv_stat_fields(r),
    }
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def _make_group_csv(group) -> bytes:
    rows = []
    for r in group.athlete_reports:
        m = r.metadata
        rows.append({
            "athlete_name":       m.athlete_name if m else "",
            "device_name":        m.wearable_device_name if m else "",
            "date":               m.test_date if m else "",
            "bias_bpm":           r.bias,
            "mae_bpm":            r.mae,
            "mape_pct":           r.mape,
            "loa_lower_bpm":      r.loa_lower,
            "loa_upper_bpm":      r.loa_upper,
            "n_samples":          r.n_samples,
            "n_outliers_flagged": r.n_outliers_flagged,
            "quality_label":      r.quality_label,
            **_adv_stat_fields(r),
        })
    rows.append({
        "athlete_name":       "GROUP_SUMMARY",
        "device_name":        "",
        "date":               "",
        "bias_bpm":           group.mean_bias,
        "mae_bpm":            group.mean_mae,
        "mape_pct":           group.mean_mape,
        "loa_lower_bpm":      group.pooled_loa_lower,
        "loa_upper_bpm":      group.pooled_loa_upper,
        "n_samples":          "",
        "n_outliers_flagged": "",
        "quality_label":      group.group_quality_label,
        "pearson_r":          group.mean_pearson_r  if group.mean_pearson_r  is not None else "",
        "r_squared":          group.mean_r_squared  if group.mean_r_squared  is not None else "",
        "see_bpm":            "",
        "bias_ci_lower_bpm":  "",
        "bias_ci_upper_bpm":  "",
        "mape_ci_lower_pct":  "",
        "mape_ci_upper_pct":  "",
    })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _make_comparison_csv(cmp_report) -> bytes:
    rows = []
    for entry in cmp_report.entries:
        r = entry.report
        rows.append({
            "rank":               entry.rank,
            "device_name":        entry.device_name,
            "wearable_type":      entry.wearable_type,
            "bias_bpm":           r.bias,
            "mae_bpm":            r.mae,
            "mape_pct":           r.mape,
            "loa_lower_bpm":      r.loa_lower,
            "loa_upper_bpm":      r.loa_upper,
            "n_samples":          r.n_samples,
            "n_outliers_flagged": r.n_outliers_flagged,
            "quality_label":      r.quality_label,
            **_adv_stat_fields(r),
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _make_longitudinal_csv(long_report) -> bytes:
    rows = []
    for s in long_report.sessions:
        r = s.report
        rows.append({
            "date":               s.date,
            "device_name":        long_report.device_name,
            "athlete_name":       long_report.athlete_name,
            "bias_bpm":           r.bias,
            "mae_bpm":            r.mae,
            "mape_pct":           r.mape,
            "loa_lower_bpm":      r.loa_lower,
            "loa_upper_bpm":      r.loa_upper,
            "n_samples":          r.n_samples,
            "n_outliers_flagged": r.n_outliers_flagged,
            "quality_label":      r.quality_label,
            **_adv_stat_fields(r),
        })
    rows.append({
        "date":               "LONGITUDINAL_SUMMARY",
        "device_name":        long_report.device_name,
        "athlete_name":       long_report.athlete_name,
        "bias_bpm":           long_report.mean_bias,
        "mae_bpm":            "",
        "mape_pct":           long_report.mean_mape,
        "loa_lower_bpm":      "",
        "loa_upper_bpm":      "",
        "n_samples": "", "n_outliers_flagged": "", "quality_label": "",
        "pearson_r": "", "r_squared": "", "see_bpm": "",
        "bias_ci_lower_bpm": "", "bias_ci_upper_bpm": "",
        "mape_ci_lower_pct": "", "mape_ci_upper_pct": "",
    })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _quality_badge_html(quality_key: str, label: str | None = None) -> str:
    """Return a small coloured <span> badge for a quality label key."""
    colour = QUALITY_COLOURS.get(quality_key, "#555")
    text = label if label is not None else quality_key.capitalize()
    return (
        f"<span style='background:{colour};padding:2px 10px;"
        f"border-radius:4px;color:white;font-size:0.85em;font-weight:600;'>"
        f"{text}</span>"
    )


def _format_device_comparison_text(cmp_report) -> str:
    W = 26
    lines = [
        "=" * 62,
        f"  DEVICE COMPARISON REPORT ({cmp_report.n_devices} devices)",
        "=" * 62,
        f"\n  Reference  : {cmp_report.reference_device_name}",
        f"  Best Device: {cmp_report.best_device_name}",
        f"\n  SUMMARY\n  {cmp_report.summary_text}",
        "\n--- PER-DEVICE RESULTS ---",
    ]
    for entry in cmp_report.entries:
        r = entry.report
        adv_lines = []
        if r.pearson_r is not None:
            adv_lines = [
                f"    {'Pearson r':<{W}}: {r.pearson_r:.4f}",
                f"    {'R²':<{W}}: {r.r_squared:.4f}",
                f"    {'SEE':<{W}}: {r.see:.3f} BPM",
                f"    {'Bias 95% CI (parametric)':<{W}}: [{r.bias_ci_lower:+.2f}, {r.bias_ci_upper:+.2f}] BPM",
                f"    {'MAPE 95% CI (bootstrap)':<{W}}: [{r.mape_ci_lower:.2f}%, {r.mape_ci_upper:.2f}%]",
            ]
        lines += [
            f"\n  Rank {entry.rank}: {entry.device_name} ({entry.wearable_type})",
            f"    {'Bias':<{W}}: {r.bias:+.2f} BPM",
            f"    {'MAE':<{W}}: {r.mae:.2f} BPM",
            f"    {'MAPE':<{W}}: {r.mape:.2f}%",
            f"    {'LoA (95%)':<{W}}: [{r.loa_lower:.2f}, {r.loa_upper:.2f}] BPM",
            f"    {'N samples':<{W}}: {r.n_samples}",
            f"    {'Outliers flagged':<{W}}: {r.n_outliers_flagged}",
            f"    {'Quality':<{W}}: {QUALITY_LABELS[r.quality_label]}",
        ] + adv_lines
    lines.append("\n" + "=" * 62)
    return "\n".join(lines)


def _build_metadata(
    athlete_name: str,
    conditions: str,
    sport_key: str,
    wearable_type: str,
    reference_type: str,
    wearable_name: str,
    reference_name: str,
    test_date_str: str,
) -> TestRunMetadata:
    return TestRunMetadata(
        test_date=test_date_str,
        athlete_name=athlete_name,
        wearable_device_name=wearable_name or "Wearable",
        reference_device_name=reference_name or "Reference",
        sport=sport_key,
        metric="heart_rate",
        wearable_type=wearable_type,
        reference_type=reference_type,
        conditions=conditions,
    )


def _render_data_preview(data, artifact_report, metadata=None, key_prefix="single"):
    """Render artifact summary and preview plot. Returns True if user wants artifacts excluded."""
    n_flagged = artifact_report.n_flagged_combined
    n_total   = artifact_report.n_total
    pct       = artifact_report.pct_flagged

    if not artifact_report.has_artifacts:
        st.success(f"No artifacts detected across {n_total:,} samples. Data looks clean.")
        exclude = False
    else:
        # Severity colour
        colour = "#f39c12" if pct < 5 else "#e74c3c"
        st.markdown(
            f"<div style='background:{colour};padding:8px 14px;border-radius:6px;"
            f"color:white;margin-bottom:8px;'>"
            f"<b>{n_flagged} of {n_total:,} samples flagged as potential artifacts ({pct:.1f}%)</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Per-channel breakdown
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                if artifact_report.flag_reasons_wearable:
                    st.markdown("**⌚ Wearable channel — ⚠️ flagged**")
                    for r in artifact_report.flag_reasons_wearable:
                        st.caption(f"• {r}")
                else:
                    st.markdown("**⌚ Wearable channel — ✅ clean**")
        with c2:
            with st.container(border=True):
                if artifact_report.flag_reasons_reference:
                    st.markdown("**📡 Reference channel — ⚠️ flagged**")
                    for r in artifact_report.flag_reasons_reference:
                        st.caption(f"• {r}")
                else:
                    st.markdown("**📡 Reference channel — ✅ clean**")

        exclude = st.checkbox(
            "Exclude flagged samples from analysis",
            value=True,
            key=f"{key_prefix}_exclude_artifacts",
            help=(
                "Removes samples flagged as out-of-range, motion spikes, or flatlines "
                "from both channels before running analysis. Recommended unless you have "
                "a specific reason to include them."
            ),
        )

    st.pyplot(plot_data_preview(data, artifact_report, metadata))
    return exclude


def _render_coverage(coverage):
    """Render HR zone coverage results as a compact table."""
    status_emoji = {
        "met":         "✅ Met",
        "partial":     "⚠️ Partial",
        "not_reached": "❌ Not Reached",
        "unknown":     "— Unknown",
        "trimmed":     "✂️ Trimmed",
    }
    rows = []
    for s in coverage.step_results:
        rows.append({
            "Step": s.step_name,
            "Target %HRmax": s.target_hr_pct,
            "Target BPM": s.target_bpm_range or "—",
            "Actual Median (BPM)": f"{s.actual_median_bpm:.1f}" if not np.isnan(s.actual_median_bpm) else "—",
            "Samples": s.n_samples_in_step,
            "Status": status_emoji.get(s.status, s.status),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if coverage.warning_message:
        st.warning(coverage.warning_message)




def _render_recommendation(rec):
    """Render onboarding recommendation per use case."""
    overall_colour = RECOMMENDATION_STATUS_COLOUR.get(rec.overall_verdict, "#555")
    overall_label  = RECOMMENDATION_LABEL.get(rec.overall_verdict, rec.overall_verdict)
    st.markdown(
        f"<div style='background:{overall_colour};padding:10px 16px;"
        f"border-radius:8px;color:white;margin-bottom:8px;'>"
        f"<b>Onboarding Verdict: {overall_label}</b> — "
        f"{rec.summary_text}</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(rec.recommendations))
    for col, uc_rec in zip(cols, rec.recommendations):
        icon   = RECOMMENDATION_ICON.get(uc_rec.status, "")
        colour = RECOMMENDATION_STATUS_COLOUR.get(uc_rec.status, "#555")
        label  = RECOMMENDATION_LABEL.get(uc_rec.status, uc_rec.status)
        col.markdown(
            f"<div style='border:1px solid {colour};border-radius:6px;padding:10px;'>"
            f"<b>{icon} {uc_rec.use_case_label}</b><br>"
            f"<span style='color:{colour};font-size:0.85em;'>{label}</span><br>"
            f"<span style='font-size:0.8em;color:#555;'>{uc_rec.reason}</span></div>",
            unsafe_allow_html=True,
        )


# ── SECTION 1: Protocol Generator ────────────────────────────────────────────
st.header("1. Generate Field Protocol")

col1, col2 = st.columns(2)

with col1:
    sport_display = st.selectbox("Sport", list(SPORT_DISPLAY.values()))
    sport_key = SPORT_KEY[sport_display]

    metric = st.selectbox("Metric", ["Heart Rate (Exercise)"], index=0)

    wearable_type = st.selectbox(
        "Wearable Type",
        options=["wrist_based_ppg", "finger_based_ppg"],
        format_func=lambda x: {
            "wrist_based_ppg": "Wrist-Based PPG",
            "finger_based_ppg": "Finger-Based PPG",
        }[x],
    )
    reference_type = st.selectbox(
        "Reference (Gold Standard)",
        options=["chest_strap_ecg", "chest_strap_hr"],
        format_func=lambda x: {
            "chest_strap_ecg": "Chest Strap (ECG-validated)",
            "chest_strap_hr": "Chest Strap (HR)",
        }[x],
    )

with col2:
    context_options = SPORT_CONTEXTS[sport_key]
    context = st.selectbox(
        "Protocol Type",
        options=context_options,
        format_func=lambda x: CONTEXT_DISPLAY.get(x, x),
    )
    wearable_name = st.text_input("Wearable Device Name", placeholder="e.g. Garmin Forerunner 265")
    reference_name = st.text_input("Reference Device Name", placeholder="e.g. Polar H10")
    test_date = st.date_input("Test Date")

st.markdown("---")
col_age, col_hrmax = st.columns([1, 2])
with col_age:
    age_input = st.number_input(
        "Athlete Age (optional — for personalised HR targets)",
        min_value=10, max_value=100, value=None, step=1,
        help=f"Uses the Tanaka formula: HRmax = 208 − 0.7 × age. Population SE ≈ ±{HRMAX_SE_BPM} BPM.",
    )
with col_hrmax:
    if age_input:
        hrmax_val = compute_hrmax(int(age_input))
        st.metric(
            "Estimated HRmax (Tanaka)",
            f"{hrmax_val} BPM",
            help=f"HRmax = 208 − 0.7 × {age_input}.",
        )
        st.caption(f"Formula-based estimate. Actual HRmax may differ by ±{HRMAX_SE_BPM}–10 BPM.")

if st.button("Generate Protocol", type="primary"):
    params = ProtocolParams(
        sport=sport_key,
        metric="heart_rate",
        wearable_type=wearable_type,
        reference_type=reference_type,
        context=context,
        wearable_device_name=wearable_name or "Wearable Device",
        reference_device_name=reference_name or "Reference Device",
        test_date=str(test_date),
        age=int(age_input) if age_input else None,
    )
    try:
        protocol = generate_protocol(params)
        st.session_state["protocol"] = protocol
        st.session_state["params"] = params
    except ValueError as e:
        st.error(str(e))

if "protocol" in st.session_state:
    protocol = st.session_state["protocol"]

    st.success(
        f"Protocol generated: **{protocol.estimated_duration_min:.0f} min** / "
        f"~{protocol.n_expected_samples:,} expected samples at 1 Hz"
    )

    rows = []
    for step in protocol.steps:
        mins = step.duration_sec // 60
        secs = step.duration_sec % 60
        row = {
            "Step": step.name,
            "Duration": f"{mins}:{secs:02d}",
            "RPE": step.target_rpe,
            "% HRmax": step.target_hr_pct,
        }
        if protocol.hrmax:
            row["Target BPM"] = step.target_hr_bpm or "—"
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Device Setup Instructions", expanded=True):
        _render_device_instructions(protocol.device_instructions)

# Use-case selection (always visible, applied at analysis time)
st.markdown("---")
st.subheader("Intended Use Cases (for onboarding recommendation)")
st.caption(
    "Select the use cases you intend to evaluate this device for. "
    "The tool will assess whether device accuracy meets established thresholds for each use case."
)
uc_cols = st.columns(len(USE_CASES))
selected_use_cases: list[str] = []
for col, (uc_key, uc_meta) in zip(uc_cols, USE_CASES.items()):
    with col:
        if st.checkbox(uc_meta["label"], key=f"uc_{uc_key}",
                       help=uc_meta["description"]):
            selected_use_cases.append(uc_key)


# ── SECTION 2: Upload Session Data ───────────────────────────────────────────
st.header("2. Upload Session Data")

mode = st.radio(
    "Session mode",
    options=["Single Athlete", "Multiple Athletes", "Compare Devices", "Track Over Time"],
    horizontal=True,
)

if mode == "Single Athlete":
    file_mode = st.radio(
        "File format",
        ["Combined file (timestamp + both HR columns)", "Two separate files (one per device)"],
        horizontal=True,
    )
    athlete_name = st.text_input("Athlete Name (for report)", value="Athlete 1")
    conditions = st.text_input("Conditions (optional)", placeholder="e.g. outdoor flat, 18°C")

    parsed_data = None  # will hold HRDataSeries if files are ready

    if file_mode.startswith("Combined"):
        uploaded = st.file_uploader("Upload combined CSV or JSON", type=["csv", "json"])
        if uploaded:
            with st.expander("Column Mapping (auto-detected — override if needed)", expanded=False):
                df_preview = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_json(uploaded)
                cols = list(df_preview.columns)
                st.caption(f"Detected columns: {cols}")
                ts_col = st.selectbox("Timestamp column", cols, index=0)
                w_col  = st.selectbox("Wearable HR column", cols, index=min(1, len(cols) - 1))
                r_col  = st.selectbox("Reference HR column", cols, index=min(2, len(cols) - 1))
            st.session_state["single_upload"] = (
                uploaded, "combined", ts_col, w_col, r_col, athlete_name, conditions
            )
            try:
                uploaded.seek(0)
                parsed_data = parse_combined_file(uploaded, timestamp_col=ts_col,
                                                  wearable_col=w_col, reference_col=r_col)
            except Exception as e:
                st.error(f"Could not parse file: {e}")
    else:
        w_file = st.file_uploader("Wearable device file (CSV or JSON)", type=["csv", "json"], key="wfile")
        r_file = st.file_uploader("Reference device file (CSV or JSON)", type=["csv", "json"], key="rfile")
        if w_file and r_file:
            st.session_state["single_upload"] = (w_file, r_file, "two", athlete_name, conditions)
            try:
                w_file.seek(0); r_file.seek(0)
                parsed_data = parse_two_files(w_file, r_file)
            except Exception as e:
                st.error(f"Could not parse files: {e}")

    # If files were cleared, wipe stale analysis results immediately
    if parsed_data is None:
        for _k in ["single_parsed_data", "single_artifact_report",
                   "single_report", "single_data", "single_bins",
                   "single_coverage", "single_rec"]:
            st.session_state.pop(_k, None)

    # Data preview + artifact detection (shown as soon as files are parsed)
    if parsed_data is not None:
        st.markdown("---")
        st.subheader("Data Preview")
        preview_meta = _build_metadata(
            athlete_name, conditions, sport_key, wearable_type,
            reference_type, wearable_name, reference_name, str(test_date),
        )
        artifact_report = detect_artifacts(parsed_data)
        exclude_artifacts = _render_data_preview(
            parsed_data, artifact_report, preview_meta, key_prefix="single"
        )
        # Store parsed data + artifact report for the Analyse step
        # (exclude_artifacts is stored automatically by Streamlit via the widget key)
        st.session_state["single_parsed_data"]     = parsed_data
        st.session_state["single_artifact_report"] = artifact_report

        # Session trim
        with st.expander("Session trim (optional)"):
            st.caption(
                "Trim leading warm-up and/or trailing cool-down from analysis. "
                "Only start/end trims are supported to preserve accuracy of intensity-specific recommendations."
            )
            _tc1, _tc2 = st.columns(2)
            _tc1.number_input(
                "Exclude warm-up (minutes)", min_value=0.0, max_value=30.0,
                value=0.0, step=0.5, key="single_warmup_minutes",
            )
            _tc2.number_input(
                "Exclude cool-down (minutes)", min_value=0.0, max_value=30.0,
                value=0.0, step=0.5, key="single_cooldown_minutes",
            )

elif mode == "Multiple Athletes":
    st.markdown("Add one row per athlete. Each athlete needs a wearable file and a reference file.")

    if "athlete_rows" not in st.session_state:
        st.session_state["athlete_rows"] = [{"name": "Athlete 1", "w_file": None, "r_file": None}]

    if st.button("+ Add Athlete"):
        n = len(st.session_state["athlete_rows"]) + 1
        st.session_state["athlete_rows"].append({"name": f"Athlete {n}", "w_file": None, "r_file": None})

    for i, row in enumerate(st.session_state["athlete_rows"]):
        c1, c2, c3 = st.columns([2, 3, 3])
        with c1:
            st.session_state["athlete_rows"][i]["name"] = st.text_input(
                f"Name {i + 1}", value=row["name"], key=f"name_{i}"
            )
        with c2:
            st.session_state["athlete_rows"][i]["w_file"] = st.file_uploader(
                f"Wearable file {i + 1}", type=["csv", "json"], key=f"wfile_{i}"
            )
        with c3:
            st.session_state["athlete_rows"][i]["r_file"] = st.file_uploader(
                f"Reference file {i + 1}", type=["csv", "json"], key=f"rfile_{i}"
            )

    conditions_group = st.text_input(
        "Conditions (applies to all athletes)", placeholder="e.g. outdoor flat, 18°C"
    )

elif mode == "Compare Devices":
    st.markdown(
        "Upload one reference file and multiple wearable device files. "
        "Each device will be ranked by MAPE against the same reference."
    )
    athlete_name_cmp = st.text_input("Athlete / Session Name", value="Athlete 1")
    conditions_cmp = st.text_input("Conditions (optional)", placeholder="e.g. outdoor flat, 18°C")
    ref_file_cmp = st.file_uploader(
        "Reference device file (CSV or JSON)", type=["csv", "json"], key="ref_cmp"
    )

    if "device_rows" not in st.session_state:
        st.session_state["device_rows"] = [{
            "name": "Device 1",
            "wearable_type": "wrist_based_ppg",
            "file": None,
        }]

    if st.button("+ Add Device"):
        n = len(st.session_state["device_rows"]) + 1
        st.session_state["device_rows"].append({
            "name": f"Device {n}",
            "wearable_type": "wrist_based_ppg",
            "file": None,
        })

    for i, drow in enumerate(st.session_state["device_rows"]):
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            st.session_state["device_rows"][i]["name"] = st.text_input(
                f"Device name {i + 1}", value=drow["name"], key=f"dname_{i}"
            )
        with c2:
            st.session_state["device_rows"][i]["wearable_type"] = st.selectbox(
                f"Type {i + 1}",
                options=["wrist_based_ppg", "finger_based_ppg"],
                format_func=lambda x: {
                    "wrist_based_ppg": "Wrist PPG",
                    "finger_based_ppg": "Finger PPG",
                }[x],
                key=f"dtype_{i}",
            )
        with c3:
            st.session_state["device_rows"][i]["file"] = st.file_uploader(
                f"Wearable file {i + 1}", type=["csv", "json"], key=f"dfile_{i}"
            )


else:
    # Track Over Time
    st.markdown(
        "Upload wearable and reference files for the **same device** tested on multiple dates. "
        "The tool plots accuracy trends to show whether device performance is improving, "
        "stable, or degrading over time."
    )
    long_device_name = st.text_input("Device Name", value="Wearable Device", key="long_device_name")
    long_athlete_name = st.text_input("Athlete / Subject Name", value="Athlete 1", key="long_athlete_name")
    long_wearable_type = st.selectbox(
        "Wearable Type",
        options=["wrist_based_ppg", "finger_based_ppg"],
        format_func=lambda x: {"wrist_based_ppg": "Wrist PPG", "finger_based_ppg": "Finger PPG"}[x],
        key="long_wearable_type",
    )

    if "long_rows" not in st.session_state:
        st.session_state["long_rows"] = [
            {"date": "", "w_file": None, "r_file": None},
            {"date": "", "w_file": None, "r_file": None},
        ]

    if st.button("+ Add Session"):
        st.session_state["long_rows"].append({"date": "", "w_file": None, "r_file": None})

    st.caption(
        "Each session needs: a test date (YYYY-MM-DD), a wearable file, and a reference file. "
        "Files must contain `timestamp` and HR columns."
    )
    for i, lrow in enumerate(st.session_state["long_rows"]):
        c1, c2, c3 = st.columns([2, 3, 3])
        with c1:
            st.session_state["long_rows"][i]["date"] = st.text_input(
                f"Date {i + 1} (YYYY-MM-DD)", value=lrow["date"], key=f"ldate_{i}"
            )
        with c2:
            st.session_state["long_rows"][i]["w_file"] = st.file_uploader(
                f"Wearable file {i + 1}", type=["csv", "json"], key=f"lw_{i}"
            )
        with c3:
            st.session_state["long_rows"][i]["r_file"] = st.file_uploader(
                f"Reference file {i + 1}", type=["csv", "json"], key=f"lr_{i}"
            )

    with st.expander("Session Trim — optional (applies to all sessions)", expanded=False):
        st.caption(
            "Trim the same amount from every session to keep the longitudinal comparison consistent. "
            "Varying trim per session would confound accuracy trends."
        )
        _ltc1, _ltc2 = st.columns(2)
        _ltc1.number_input(
            "Warm-up to remove (seconds)", min_value=0, step=5,
            value=0, key="long_warmup_input",
        )
        _ltc2.number_input(
            "Cool-down to remove (seconds)", min_value=0, step=5,
            value=0, key="long_cooldown_input",
        )


# ── SECTION 3: Analyse & Results ─────────────────────────────────────────────
st.header("3. Analysis Results")


if mode == "Single Athlete":
    has_data = "single_parsed_data" in st.session_state
    if st.button("Analyse", type="primary", disabled=not has_data):
        try:
            data = st.session_state["single_parsed_data"]
            artifact_report = st.session_state.get("single_artifact_report")
            exclude = st.session_state.get("single_exclude_artifacts", True)

            if exclude and artifact_report and artifact_report.has_artifacts:
                data = apply_artifact_exclusion(data, artifact_report)

            warmup_s = st.session_state.get("single_warmup_minutes", 0.0) * 60
            cooldown_s = st.session_state.get("single_cooldown_minutes", 0.0) * 60
            if warmup_s > 0 or cooldown_s > 0:
                data = trim_session(data, warmup_seconds=warmup_s, cooldown_seconds=cooldown_s)

            upload = st.session_state.get("single_upload")
            if upload and upload[1] == "combined":
                ath_name, conds = upload[5], upload[6]
            elif upload:
                ath_name, conds = upload[3], upload[4]
            else:
                ath_name, conds = athlete_name, conditions

            meta = _build_metadata(
                ath_name, conds, sport_key, wearable_type,
                reference_type, wearable_name, reference_name, str(test_date),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = analyze_hr_validation(data, meta)
                for w in caught:
                    st.warning(str(w.message))

            bins = analyze_by_intensity_bin(data)

            coverage = None
            if "protocol" in st.session_state:
                coverage = check_hr_zone_coverage(data, st.session_state["protocol"])

            rec = generate_recommendation(report, selected_use_cases)

            st.session_state["single_report"]   = report
            st.session_state["single_data"]     = data
            st.session_state["single_bins"]     = bins
            st.session_state["single_coverage"] = coverage
            st.session_state["single_rec"]      = rec

        except Exception as e:
            st.error(f"Analysis failed: {e}")

    if "single_report" in st.session_state:
        report   = st.session_state["single_report"]
        data     = st.session_state["single_data"]
        bins     = st.session_state["single_bins"]
        coverage = st.session_state.get("single_coverage")
        rec      = st.session_state.get("single_rec")

        # HR Zone Coverage
        if coverage:
            st.subheader("HR Zone Coverage")
            overall_col = COVERAGE_STATUS_COLOUR.get(coverage.overall_status, "#555")
            st.markdown(
                f"<div style='background:{overall_col};padding:8px 14px;"
                f"border-radius:6px;color:white;margin-bottom:8px;'>"
                f"<b>Coverage: {coverage.overall_status.replace('_', ' ').title()}</b></div>",
                unsafe_allow_html=True,
            )
            _render_coverage(coverage)

        # Summary card
        q_colour = QUALITY_COLOURS.get(report.quality_label, "#555")
        st.markdown(
            f"<div style='background:{q_colour};padding:12px 18px;"
            f"border-radius:8px;color:white;margin-top:8px;'>"
            f"<b>{QUALITY_LABELS[report.quality_label]}</b> — {report.summary_text}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Metrics row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            "Bias",
            f"{report.bias:+.2f} BPM",
            help="Average difference (wearable \u2212 reference). Positive = reads too high; negative = reads too low. A well-calibrated device should be close to 0 BPM. Bland & Altman (1986).",
        )
        m2.metric(
            "MAE",
            f"{report.mae:.2f} BPM",
            help="Mean Absolute Error \u2014 average size of errors regardless of direction. Unlike Bias, opposing errors do not cancel each other out. Lower is better.",
        )
        m3.metric(
            "MAPE",
            f"{report.mape:.2f}%",
            help="Mean Absolute Percentage Error \u2014 error as a % of true HR, making accuracy comparable across effort levels. < 3% = excellent, < 5% = good, < 10% = acceptable. Navalta et al. (2020), INTERLIVE (2020).",
        )
        m4.metric(
            "LoA Lower",
            f"{report.loa_lower:.1f} BPM",
            help="Lower bound of Bland-Altman 95% Limits of Agreement. 95% of individual readings fall above this value relative to the reference. The LoA span is the primary validity indicator. Bland & Altman (1986).",
        )
        m5.metric(
            "LoA Upper",
            f"{report.loa_upper:.1f} BPM",
            help="Upper bound of Bland-Altman 95% Limits of Agreement. 95% of individual readings fall below this value relative to the reference. The LoA span is the primary validity indicator. Bland & Altman (1986).",
        )
        m6.metric(
            "N samples",
            f"{report.n_samples:,}",
            help="Number of valid paired HR readings used in the analysis. Reliable LoA estimates require \u2265 100 samples. Results from fewer samples should be interpreted with caution.",
        )

        # Advanced statistics (collapsed by default — primary metrics stay prominent)
        with st.expander("Advanced Statistics", expanded=False):
            if report.pearson_r is not None:
                # Row 1: correlation stats — 3 equal columns
                ac1, ac2, ac3 = st.columns(3)
                ac1.metric(
                    "Pearson r", f"{report.pearson_r:.4f}",
                    help="Correlation coefficient measuring how well the wearable tracks the shape of reference HR. High r (close to 1.0) indicates good tracking but does not confirm accurate values \u2014 secondary to LoA. Atkinson & Nevill (1998) Sports Med 26(4):217\u2013238.",
                )
                ac2.metric(
                    "R\u00b2", f"{report.r_squared:.4f}",
                    help="Proportion of reference HR variance explained by the wearable (= Pearson r\u00b2). A value of 0.95 means 95% of the variation is captured. Standard form of r for publication reporting.",
                )
                ac3.metric(
                    "SEE", f"{report.see:.2f} BPM",
                    help="Standard Error of the Estimate: SD_diff \u00d7 \u221a(1\u2212r\u00b2). Combines tracking quality and error spread into a single BPM value. Closer to 0 is better. Hopkins (2000) Sports Med 30(1):1\u201315.",
                )
                # Row 2: confidence intervals — same 3-column grid so CIs align under Pearson r and R²
                bc1, bc2, _ = st.columns(3)
                bc1.metric(
                    "Bias 95% CI",
                    f"{report.bias_ci_lower:+.2f} to {report.bias_ci_upper:+.2f} BPM",
                    help="Confidence interval for the bias estimate. A narrow CI means the bias is well-characterised; a wide CI means more data is needed. Parametric method: bias \u00b1 1.96\u00d7(SD/\u221an). Bland & Altman (1999) Stat Methods Med Res 8(2):135\u2013160.",
                )
                bc2.metric(
                    "MAPE 95% CI",
                    f"{report.mape_ci_lower:.2f}% to {report.mape_ci_upper:.2f}%",
                    help="Confidence interval for MAPE, calculated by bootstrapping (1000 resamples). A narrow CI confirms the MAPE estimate is reliable. Non-parametric percentile method avoids normality assumption on MAPE. Efron & Tibshirani (1993).",
                )
                st.caption(
                    "r and R\u00b2 are secondary indicators \u2014 LoA is the primary validity measure "
                    "(Atkinson & Nevill 1998, *Sports Med* 26(4):217\u2013238)."
                )
            else:
                st.info("Advanced statistics require \u2265 3 paired samples.")

        # Onboarding recommendation
        if rec:
            st.subheader("Onboarding Recommendation")
            _render_recommendation(rec)

        # Plots
        tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Bland-Altman", "Scatter", "By Intensity"])
        meta_obj = report.metadata
        fig_ts = plot_timeseries(data, meta_obj)
        fig_ba = plot_bland_altman(report, data)
        fig_sc = plot_scatter(data, meta_obj)
        fig_bi = plot_intensity_bins(bins)
        with tab1:
            st.pyplot(fig_ts)
        with tab2:
            st.pyplot(fig_ba)
        with tab3:
            st.pyplot(fig_sc)
        with tab4:
            st.pyplot(fig_bi)

        # Export
        pdf_bytes = _make_pdf([fig_ts, fig_ba, fig_sc, fig_bi], text_pages=[format_report(report)])
        csv_bytes = _make_single_csv(report)
        athlete_slug = (report.metadata.athlete_name if report.metadata else "report").replace(" ", "_")
        _col_pdf, _col_csv = st.columns(2)
        with _col_pdf:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"hr_validation_{athlete_slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with _col_csv:
            st.download_button(
                "📊 Download CSV",
                data=csv_bytes,
                file_name=f"hr_validation_{athlete_slug}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("Full Text Report"):
            st.text(format_report(report))

elif mode == "Multiple Athletes":
    with st.expander("Session trim (optional)"):
        st.caption(
            "Trim leading warm-up and/or trailing cool-down from analysis. "
            "Only start/end trims are supported to preserve accuracy of intensity-specific recommendations."
        )
        _mc1, _mc2 = st.columns(2)
        _mc1.number_input(
            "Exclude warm-up (minutes)", min_value=0.0, max_value=30.0,
            value=0.0, step=0.5, key="multi_warmup_minutes",
        )
        _mc2.number_input(
            "Exclude cool-down (minutes)", min_value=0.0, max_value=30.0,
            value=0.0, step=0.5, key="multi_cooldown_minutes",
        )

    # Clear stale group results when the file set changes
    _current_multi_sig = tuple(
        (r["name"], r["w_file"].name if r["w_file"] else None, r["r_file"].name if r["r_file"] else None)
        for r in st.session_state.get("athlete_rows", [])
    )
    if st.session_state.get("_multi_sig") != _current_multi_sig:
        for _k in ["group_report", "group_data", "group_artifact_counts"]:
            st.session_state.pop(_k, None)

    if st.button("Analyse All", type="primary"):
        rows = st.session_state.get("athlete_rows", [])
        valid_rows = [r for r in rows if r["w_file"] and r["r_file"]]
        if not valid_rows:
            st.warning("Please upload files for at least one athlete.")
        else:
            all_reports, all_data = [], []
            errors = []
            for row in valid_rows:
                try:
                    row["w_file"].seek(0)
                    row["r_file"].seek(0)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        d = parse_two_files(row["w_file"], row["r_file"])
                        for w in caught:
                            st.warning(f"{row['name']}: {w.message}")
                    multi_warmup_s = st.session_state.get("multi_warmup_minutes", 0.0) * 60
                    multi_cooldown_s = st.session_state.get("multi_cooldown_minutes", 0.0) * 60
                    if multi_warmup_s > 0 or multi_cooldown_s > 0:
                        d = trim_session(d, warmup_seconds=multi_warmup_s, cooldown_seconds=multi_cooldown_s)
                    meta = _build_metadata(
                        row["name"], conditions_group, sport_key, wearable_type,
                        reference_type, wearable_name, reference_name, str(test_date),
                    )
                    r = analyze_hr_validation(d, meta)
                    all_reports.append(r)
                    all_data.append(d)
                except Exception as e:
                    errors.append(f"{row['name']}: {e}")

            for err in errors:
                st.error(err)

            if all_reports:
                group = analyze_group(all_reports, all_data)
                st.session_state["group_report"] = group
                st.session_state["group_data"]   = all_data
                st.session_state["group_artifact_counts"] = [
                    detect_artifacts(d).n_flagged_combined for d in all_data
                ]
                st.session_state["_multi_sig"] = _current_multi_sig

    if "group_report" in st.session_state:
        group    = st.session_state["group_report"]
        all_data = st.session_state["group_data"]

        # Group summary card
        q_colour = QUALITY_COLOURS.get(group.group_quality_label, "#555")
        st.markdown(
            f"<div style='background:{q_colour};padding:12px 18px;"
            f"border-radius:8px;color:white;'>"
            f"<b>{QUALITY_LABELS[group.group_quality_label]}</b> — {group.group_summary_text}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Per-athlete table
        st.subheader("Per-Athlete Results")
        artifact_counts = st.session_state.get("group_artifact_counts", [])
        table_rows = []
        for i, r in enumerate(group.athlete_reports):
            row = {
                "Athlete":    r.metadata.athlete_name if r.metadata else "—",
                "MAPE (%)":   round(r.mape, 2),
                "Bias (BPM)": round(r.bias, 2),
                "MAE (BPM)":  round(r.mae, 2),
                "LoA Lower":  round(r.loa_lower, 2),
                "LoA Upper":  round(r.loa_upper, 2),
                "Quality":    QUALITY_LABELS[r.quality_label],
                "Pearson r":  round(r.pearson_r, 4) if r.pearson_r is not None else "\u2014",
                "R\u00b2":        round(r.r_squared, 4) if r.r_squared is not None else "\u2014",
                "SEE (BPM)":  round(r.see, 3)       if r.see       is not None else "\u2014",
            }
            if artifact_counts:
                row["Artifacts"] = artifact_counts[i]
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        with st.expander("Column definitions", expanded=False):
            st.markdown(
                "- **MAPE (%)** — Mean Absolute Percentage Error: error as % of true HR. "
                "< 3% Excellent · < 5% Good · < 10% Acceptable. Navalta et al. (2020), INTERLIVE (2020).  \n"
                "- **Bias (BPM)** — Average systematic offset (wearable − reference). "
                "Positive = reads too high. Bland & Altman (1986).  \n"
                "- **MAE (BPM)** — Mean Absolute Error: average magnitude of errors ignoring direction.  \n"
                "- **LoA Lower / Upper** — 95% Limits of Agreement: the range within which 95% of "
                "individual errors fall. Wider span = less consistent. Bland & Altman (1986).  \n"
                "- **Pearson r** — Linear correlation coefficient. Secondary to LoA for validity assessment. "
                "Atkinson & Nevill (1998).  \n"
                "- **R²** — Coefficient of determination. Proportion of variance explained.  \n"
                "- **SEE (BPM)** — Standard Error of the Estimate. Hopkins (2000).  \n"
                "- **Artifacts** — Number of samples flagged as out-of-range, spikes, or flatlines "
                "and excluded before analysis."
            )

        # Group metrics
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Mean MAPE",  f"{group.mean_mape:.2f}% ± {group.sd_mape:.2f}%")
        g2.metric("Mean Bias",  f"{group.mean_bias:+.2f} ± {group.sd_bias:.2f} BPM")
        g3.metric("Pooled LoA", f"[{group.pooled_loa_lower:.1f}, {group.pooled_loa_upper:.1f}]")
        g4.metric("Athletes",   group.n_athletes)

        # Group advanced statistics (collapsed)
        with st.expander("Group Advanced Statistics", expanded=False):
            if group.mean_pearson_r is not None:
                ga1, ga2 = st.columns(2)
                ga1.metric("Mean Pearson r", f"{group.mean_pearson_r:.4f}")
                ga2.metric("Mean R\u00b2",       f"{group.mean_r_squared:.4f}")
                st.caption("Per-athlete bias/MAPE CIs are shown in the Full Group Report.")
            else:
                st.info("No advanced statistics available.")

        # Group plots
        tab1, tab2, tab3, tab4 = st.tabs([
            "Per-Athlete Bland-Altman", "Pooled Bland-Altman",
            "Individual Scatter", "Intensity Zones",
        ])
        all_figs = []
        with tab1:
            for r, d in zip(group.athlete_reports, all_data):
                fig = plot_bland_altman(r, d)
                st.pyplot(fig)
                all_figs.append(fig)
        with tab2:
            fig_grp = plot_group_bland_altman(group, all_data)
            st.pyplot(fig_grp)
            all_figs.append(fig_grp)
        with tab3:
            for r, d in zip(group.athlete_reports, all_data):
                fig = plot_scatter(d, r.metadata)
                st.pyplot(fig)
                all_figs.append(fig)
        with tab4:
            for r, d in zip(group.athlete_reports, all_data):
                name = r.metadata.athlete_name if r.metadata else "Athlete"
                st.caption(f"**{name}**")
                bins_a = analyze_by_intensity_bin(d)
                fig_bi = plot_intensity_bins(bins_a)
                st.pyplot(fig_bi)
                all_figs.append(fig_bi)

        # Export
        pdf_bytes = _make_pdf(all_figs, text_pages=[format_group_report(group)])
        csv_bytes = _make_group_csv(group)
        _col_pdf, _col_csv = st.columns(2)
        with _col_pdf:
            st.download_button(
                "📄 Download Group PDF Report",
                data=pdf_bytes,
                file_name="hr_validation_group.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with _col_csv:
            st.download_button(
                "📊 Download Group CSV",
                data=csv_bytes,
                file_name="hr_validation_group.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("Full Group Report"):
            st.text(format_group_report(group))

elif mode == "Compare Devices":
    with st.expander("Session trim (optional)"):
        st.caption(
            "Trim leading warm-up and/or trailing cool-down from analysis. "
            "Only start/end trims are supported to preserve accuracy of intensity-specific recommendations."
        )
        _cc1, _cc2 = st.columns(2)
        _cc1.number_input(
            "Exclude warm-up (minutes)", min_value=0.0, max_value=30.0,
            value=0.0, step=0.5, key="comp_warmup_minutes",
        )
        _cc2.number_input(
            "Exclude cool-down (minutes)", min_value=0.0, max_value=30.0,
            value=0.0, step=0.5, key="comp_cooldown_minutes",
        )

    # Clear stale comparison results when the file set changes
    _current_cmp_sig = (
        ref_file_cmp.name if ref_file_cmp else None,
        tuple(
            (d["name"], d["file"].name if d["file"] else None)
            for d in st.session_state.get("device_rows", [])
        ),
    )
    if st.session_state.get("_cmp_sig") != _current_cmp_sig:
        st.session_state.pop("cmp_report", None)

    if st.button("Analyse Devices", type="primary"):
        device_rows = st.session_state.get("device_rows", [])
        valid_devices = [d for d in device_rows if d["file"]]
        if not ref_file_cmp:
            st.warning("Please upload the reference file.")
        elif not valid_devices:
            st.warning("Please upload at least one wearable device file.")
        else:
            try:
                devices_input = []
                ref_file_cmp.seek(0)
                comp_warmup_s = st.session_state.get("comp_warmup_minutes", 0.0) * 60
                comp_cooldown_s = st.session_state.get("comp_cooldown_minutes", 0.0) * 60
                for drow in valid_devices:
                    drow["file"].seek(0)
                    ref_file_cmp.seek(0)
                    d = parse_two_files(drow["file"], ref_file_cmp)
                    if comp_warmup_s > 0 or comp_cooldown_s > 0:
                        d = trim_session(d, warmup_seconds=comp_warmup_s, cooldown_seconds=comp_cooldown_s)
                    devices_input.append((drow["name"], drow["wearable_type"], d))

                meta_base = _build_metadata(
                    athlete_name_cmp, conditions_cmp, sport_key, wearable_type,
                    reference_type, wearable_name, reference_name, str(test_date),
                )
                cmp_report = analyze_device_comparison(
                    devices_input,
                    reference_device_name=reference_name or "Reference",
                    metadata_base=meta_base,
                )
                st.session_state["cmp_report"] = cmp_report
                st.session_state["_cmp_sig"] = _current_cmp_sig

            except Exception as e:
                st.error(f"Comparison failed: {e}")

    if "cmp_report" in st.session_state:
        cmp_report = st.session_state["cmp_report"]

        # Summary banner
        best = cmp_report.entries[0]
        b_colour = QUALITY_COLOURS.get(best.report.quality_label, "#555")
        st.markdown(
            f"<div style='background:{b_colour};padding:12px 18px;"
            f"border-radius:8px;color:white;'>"
            f"<b>Best Device: {best.device_name}</b> — {cmp_report.summary_text}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Ranking table
        st.subheader("Device Ranking (Best → Worst MAPE)")
        rank_rows = []
        for entry in cmp_report.entries:
            r = entry.report
            rank_rows.append({
                "Rank":           entry.rank,
                "Device":         entry.device_name,
                "Type":           entry.wearable_type.replace("_", " ").title(),
                "MAPE (%)":       round(r.mape, 2),
                "Bias (BPM)":     round(r.bias, 2),
                "MAE (BPM)":      round(r.mae, 2),
                "LoA Lower":      round(r.loa_lower, 2),
                "LoA Upper":      round(r.loa_upper, 2),
                "Quality":        QUALITY_LABELS[r.quality_label],
                "Pearson r":      round(r.pearson_r, 4) if r.pearson_r is not None else "\u2014",
                "R\u00b2":            round(r.r_squared, 4) if r.r_squared is not None else "\u2014",
                "SEE (BPM)":      round(r.see, 3) if r.see is not None else "\u2014",
                "Bias 95% CI":    (
                    f"{r.bias_ci_lower:+.2f} to {r.bias_ci_upper:+.2f} BPM"
                    if r.bias_ci_lower is not None else "\u2014"
                ),
                "MAPE 95% CI":    (
                    f"{r.mape_ci_lower:.2f}% to {r.mape_ci_upper:.2f}%"
                    if r.mape_ci_lower is not None else "\u2014"
                ),
                "Artifacts":      r.n_outliers_flagged,
            })
        st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)
        with st.expander("Column definitions", expanded=False):
            st.markdown(
                "- **MAPE (%)** — Mean Absolute Percentage Error: error as % of true HR. "
                "< 3% Excellent · < 5% Good · < 10% Acceptable. Navalta et al. (2020), INTERLIVE (2020).  \n"
                "- **Bias (BPM)** — Average systematic offset (wearable − reference). "
                "Positive = reads too high. Bland & Altman (1986).  \n"
                "- **MAE (BPM)** — Mean Absolute Error: average magnitude of errors ignoring direction.  \n"
                "- **LoA Lower / Upper** — 95% Limits of Agreement span within which 95% of individual "
                "errors fall. Bland & Altman (1986).  \n"
                "- **Pearson r / R²** — Correlation statistics. Secondary to LoA. "
                "Atkinson & Nevill (1998).  \n"
                "- **SEE (BPM)** — Standard Error of the Estimate. Hopkins (2000).  \n"
                "- **Bias 95% CI** — Parametric 95% confidence interval for the mean bias. "
                "Bland & Altman (1999).  \n"
                "- **MAPE 95% CI** — Percentile bootstrap 95% CI for MAPE (1000 resamples). "
                "Efron & Tibshirani (1993).  \n"
                "- **Artifacts** — Samples flagged as out-of-range, spikes, or flatlines and excluded."
            )

        # Onboarding recommendation for best device
        if selected_use_cases:
            st.subheader(f"Onboarding Recommendation — {best.device_name}")
            rec = generate_recommendation(best.report, selected_use_cases)
            if rec:
                _render_recommendation(rec)

        # Per-device plots — one tab per device
        st.subheader("Per-Device Analysis")
        all_figs = []
        device_tabs = st.tabs([e.device_name for e in cmp_report.entries])
        for tab, entry in zip(device_tabs, cmp_report.entries):
            with tab:
                fig_ba = plot_bland_altman(entry.report, entry.data)
                st.pyplot(fig_ba)
                all_figs.append(fig_ba)
                bins_entry = analyze_by_intensity_bin(entry.data)
                fig_bi = plot_intensity_bins(bins_entry)
                st.pyplot(fig_bi)
                all_figs.append(fig_bi)

        # Export
        pdf_bytes = _make_pdf(all_figs, text_pages=[_format_device_comparison_text(cmp_report)])
        csv_bytes = _make_comparison_csv(cmp_report)
        _col_pdf, _col_csv = st.columns(2)
        with _col_pdf:
            st.download_button(
                "📄 Download Comparison PDF",
                data=pdf_bytes,
                file_name="hr_validation_device_comparison.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with _col_csv:
            st.download_button(
                "📊 Download Comparison CSV",
                data=csv_bytes,
                file_name="hr_validation_device_comparison.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    # Track Over Time
    # Clear stale results when file set changes
    _current_long_sig = tuple(
        (r["date"], r["w_file"].name if r["w_file"] else None, r["r_file"].name if r["r_file"] else None)
        for r in st.session_state.get("long_rows", [])
    )
    if st.session_state.get("_long_sig") != _current_long_sig:
        st.session_state.pop("long_report", None)

    long_rows = st.session_state.get("long_rows", [])
    valid_long_rows = [r for r in long_rows if r["date"] and r["w_file"] and r["r_file"]]
    can_analyse = len(valid_long_rows) >= 2

    if not can_analyse and long_rows:
        st.info("Complete at least 2 sessions (date + wearable file + reference file) to enable analysis.")

    if st.button("Analyse Sessions", type="primary", disabled=not can_analyse):
        sessions_input = []
        errors = []
        _l_warmup = st.session_state.get("long_warmup_input", 0)
        _l_cooldown = st.session_state.get("long_cooldown_input", 0)
        for lrow in valid_long_rows:
            try:
                lrow["w_file"].seek(0)
                lrow["r_file"].seek(0)
                d = parse_two_files(lrow["w_file"], lrow["r_file"])
                if _l_warmup or _l_cooldown:
                    d = trim_session(d, _l_warmup, _l_cooldown)
                meta = _build_metadata(
                    st.session_state.get("long_athlete_name", "Athlete"),
                    "",
                    sport_key,
                    st.session_state.get("long_wearable_type", "wrist_based_ppg"),
                    reference_type,
                    st.session_state.get("long_device_name", "Wearable Device"),
                    reference_name or "Reference",
                    lrow["date"],
                )
                sessions_input.append((lrow["date"], d, meta))
            except Exception as e:
                errors.append(f"{lrow['date']}: {e}")

        for err in errors:
            st.error(err)

        if len(sessions_input) >= 2:
            try:
                long_report = analyze_longitudinal(sessions_input)
                st.session_state["long_report"] = long_report
                st.session_state["_long_sig"] = _current_long_sig
            except Exception as e:
                st.error(f"Longitudinal analysis failed: {e}")

    if "long_report" in st.session_state:
        long_report = st.session_state["long_report"]

        # Overall quality — use the most common quality label as the banner colour
        overall_q = Counter(long_report.quality_trend).most_common(1)[0][0]
        q_colour = QUALITY_COLOURS.get(overall_q, "#555")
        date_range = f"{long_report.dates[0]} → {long_report.dates[-1]}"
        st.markdown(
            f"<div style='background:{q_colour};padding:12px 18px;"
            f"border-radius:8px;color:white;margin-bottom:4px;'>"
            f"<b>{long_report.device_name}</b>  ·  {long_report.athlete_name}  ·  "
            f"{len(long_report.sessions)} sessions  ·  {date_range}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Metric row — 3 columns + full-width quality trend below
        lm1, lm2, lm3 = st.columns(3)
        lm1.metric(
            "Sessions",
            len(long_report.sessions),
            help="Number of test sessions included in the longitudinal analysis.",
        )
        lm2.metric(
            "Mean MAPE",
            f"{long_report.mean_mape:.1f}% ± {long_report.sd_mape:.1f}%",
            help=(
                "Mean Absolute Percentage Error averaged across all sessions. "
                "SD reflects how much accuracy varied between test dates. "
                "Navalta et al. (2020), INTERLIVE (2020)."
            ),
        )
        lm3.metric(
            "Mean Bias",
            f"{long_report.mean_bias:+.1f} ± {long_report.sd_bias:.1f} BPM",
            help=(
                "Average systematic offset (wearable − reference) across sessions. "
                "Positive = device consistently reads too high. Bland & Altman (1986)."
            ),
        )
        # Quality trend as full-width coloured badge row
        _badge_html = " &nbsp;→&nbsp; ".join(
            _quality_badge_html(q) for q in long_report.quality_trend
        )
        st.markdown(
            f"<div style='margin-top:6px;'><span style='font-size:0.85em;color:#666;'>"
            f"Quality Trend</span><br/>{_badge_html}</div>",
            unsafe_allow_html=True,
        )

        # Main trend plot — full width
        st.markdown("---")
        fig_trend = plot_longitudinal_trend(long_report)
        st.pyplot(fig_trend)

        # Per-session expandable details (collapsed by default)
        st.markdown("---")
        st.subheader("Per-Session Details")
        session_figs = [fig_trend]
        for session in long_report.sessions:
            q_label = QUALITY_LABELS[session.report.quality_label]
            q_c = QUALITY_COLOURS.get(session.report.quality_label, "#555")
            with st.expander(
                f"{session.date}  ·  MAPE {session.report.mape:.1f}%  ·  {q_label}",
                expanded=False,
            ):
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric(
                    "MAPE",
                    f"{session.report.mape:.2f}%",
                    help="Mean Absolute Percentage Error for this session.",
                )
                sc2.metric(
                    "Bias",
                    f"{session.report.bias:+.2f} BPM",
                    help="Mean systematic offset for this session.",
                )
                sc3.metric(
                    "MAE",
                    f"{session.report.mae:.2f} BPM",
                    help="Mean Absolute Error for this session.",
                )
                sc4.metric(
                    "N Samples",
                    session.report.n_samples,
                    help="Paired samples used in analysis after artifact exclusion.",
                )
                st.markdown(
                    _quality_badge_html(session.report.quality_label, q_label),
                    unsafe_allow_html=True,
                )
                fig_ba = plot_bland_altman(session.report, session.data)
                # Append session date to Bland-Altman title for PDF clarity
                fig_ba.axes[0].set_title(
                    fig_ba.axes[0].get_title() + f"  ·  {session.date}"
                )
                st.pyplot(fig_ba)
                session_figs.append(fig_ba)

        # Export
        st.markdown("---")
        pdf_bytes = _make_pdf(session_figs, text_pages=[format_longitudinal_report(long_report)])
        csv_bytes = _make_longitudinal_csv(long_report)
        _lc_pdf, _lc_csv = st.columns(2)
        with _lc_pdf:
            st.download_button(
                "📄 Download Longitudinal PDF Report",
                data=pdf_bytes,
                file_name=f"hr_longitudinal_{long_report.device_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with _lc_csv:
            st.download_button(
                "📊 Download Longitudinal CSV",
                data=csv_bytes,
                file_name=f"hr_longitudinal_{long_report.device_name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("Full Longitudinal Report"):
            st.text(format_longitudinal_report(long_report))
