"""
Matplotlib figure generators for the analysis report.
All functions return a matplotlib Figure that can be passed directly to
Streamlit's st.pyplot() or saved with fig.savefig().
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from wearable_validation.models import (
    HRDataSeries, TestRunMetadata, AnalysisReport, GroupAnalysisReport,
    IntensityBinResult,
)
from wearable_validation.constants import QUALITY_COLOURS


def plot_timeseries(
    data: HRDataSeries,
    metadata: TestRunMetadata | None = None,
) -> plt.Figure:
    """
    Dual-line HR time series overlay.
    X: time in minutes, Y: HR in BPM.
    """
    t_min = data.timestamps / 60.0
    wearable_label = metadata.wearable_device_name if metadata else "Wearable"
    reference_label = metadata.reference_device_name if metadata else "Reference"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_min, data.hr_reference, color="#e74c3c", linewidth=1.2,
            label=reference_label, alpha=0.9)
    ax.plot(t_min, data.hr_wearable, color="#3498db", linewidth=1.2,
            label=wearable_label, alpha=0.85, linestyle="--")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Heart Rate (BPM)")
    title = "HR Time Series"
    if metadata:
        title += f" — {metadata.athlete_name}"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bland_altman(
    report: AnalysisReport,
    data: HRDataSeries,
) -> plt.Figure:
    """
    Bland-Altman (Tukey mean-difference) plot.
    X: mean of wearable + reference, Y: wearable − reference.
    Horizontal lines: bias (solid blue), LoA (dashed red).
    """
    w = np.asarray(data.hr_wearable, dtype=float)
    r = np.asarray(data.hr_reference, dtype=float)
    valid = ~(np.isnan(w) | np.isnan(r))
    w, r = w[valid], r[valid]

    means = (w + r) / 2.0
    diffs = w - r

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(means, diffs, alpha=0.4, s=8, color="#555555", label="Data points")

    ax.axhline(report.bias, color="#3498db", linewidth=1.8,
               label=f"Bias = {report.bias:+.2f} BPM")
    ax.axhline(report.loa_upper, color="#e74c3c", linewidth=1.4, linestyle="--",
               label=f"Upper LoA = {report.loa_upper:.2f} BPM")
    ax.axhline(report.loa_lower, color="#e74c3c", linewidth=1.4, linestyle="--",
               label=f"Lower LoA = {report.loa_lower:.2f} BPM")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

    # Shaded LoA band
    ax.fill_between(
        [means.min(), means.max()],
        report.loa_lower, report.loa_upper,
        alpha=0.06, color="#e74c3c",
    )

    quality_colour = QUALITY_COLOURS.get(report.quality_label, "#555555")
    ax.text(
        0.98, 0.97,
        f"MAPE = {report.mape:.1f}%\n{report.quality_label.capitalize()}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=quality_colour, alpha=0.85),
    )

    ax.set_xlabel("Mean HR — (Wearable + Reference) / 2 (BPM)")
    ax.set_ylabel("Difference — Wearable − Reference (BPM)")
    title = "Bland-Altman Plot"
    if report.metadata:
        title += f" — {report.metadata.athlete_name}"
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_scatter(
    data: HRDataSeries,
    metadata: TestRunMetadata | None = None,
) -> plt.Figure:
    """
    Wearable vs. reference scatter plot with identity line and linear regression.
    Pearson r and regression equation annotated on the plot.
    """
    w = np.asarray(data.hr_wearable, dtype=float)
    r = np.asarray(data.hr_reference, dtype=float)
    valid = ~(np.isnan(w) | np.isnan(r))
    w, r = w[valid], r[valid]

    # Linear regression
    slope, intercept = np.polyfit(r, w, 1)
    r_pearson = float(np.corrcoef(r, w)[0, 1])

    x_line = np.linspace(r.min(), r.max(), 200)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(r, w, alpha=0.35, s=8, color="#555555", label="Paired samples")

    # Identity line (y = x)
    ax.plot(x_line, x_line, color="grey", linewidth=1.2, linestyle=":", label="Identity (y = x)")

    # Regression line
    ax.plot(x_line, slope * x_line + intercept, color="#3498db", linewidth=1.6,
            label=f"Regression: y = {slope:.2f}x + {intercept:+.1f}")

    ax.text(
        0.05, 0.95,
        f"r = {r_pearson:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    wearable_label = metadata.wearable_device_name if metadata else "Wearable HR (BPM)"
    reference_label = metadata.reference_device_name if metadata else "Reference HR (BPM)"
    ax.set_xlabel(f"Reference HR — {reference_label} (BPM)")
    ax.set_ylabel(f"Wearable HR — {wearable_label} (BPM)")
    title = "Wearable vs. Reference Scatter"
    if metadata:
        title += f" — {metadata.athlete_name}"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_group_bland_altman(
    group_report: GroupAnalysisReport,
    datasets: list[HRDataSeries],
) -> plt.Figure:
    """
    Pooled Bland-Altman plot across all athletes.
    Each athlete's points are plotted in a distinct colour.
    Pooled bias and LoA are shown as horizontal lines.
    """
    # Generate distinct colours for up to ~12 athletes
    cmap = plt.cm.get_cmap("tab10", max(len(datasets), 1))
    colours = [cmap(i) for i in range(len(datasets))]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (report, data) in enumerate(zip(group_report.athlete_reports, datasets)):
        w = np.asarray(data.hr_wearable, dtype=float)
        r = np.asarray(data.hr_reference, dtype=float)
        valid = ~(np.isnan(w) | np.isnan(r))
        w, r = w[valid], r[valid]
        means = (w + r) / 2.0
        diffs = w - r
        name = report.metadata.athlete_name if report.metadata else f"Athlete {i+1}"
        ax.scatter(means, diffs, alpha=0.35, s=8, color=colours[i], label=name)

    ax.axhline(group_report.pooled_loa_upper, color="#e74c3c", linewidth=1.4, linestyle="--",
               label=f"Upper LoA = {group_report.pooled_loa_upper:.2f} BPM")
    ax.axhline(group_report.mean_bias, color="#3498db", linewidth=1.8,
               label=f"Pooled Bias = {group_report.mean_bias:+.2f} BPM")
    ax.axhline(group_report.pooled_loa_lower, color="#e74c3c", linewidth=1.4, linestyle="--",
               label=f"Lower LoA = {group_report.pooled_loa_lower:.2f} BPM")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")

    ax.fill_between(
        ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else [50, 200],
        group_report.pooled_loa_lower, group_report.pooled_loa_upper,
        alpha=0.05, color="#e74c3c",
    )

    ax.set_xlabel("Mean HR — (Wearable + Reference) / 2 (BPM)")
    ax.set_ylabel("Difference — Wearable − Reference (BPM)")
    ax.set_title(f"Group Bland-Altman Plot ({group_report.n_athletes} athletes)")
    ax.legend(loc="lower left", fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_intensity_bins(
    bin_results: list[IntensityBinResult],
) -> plt.Figure:
    """
    Grouped bar chart showing MAPE and MAE per HR intensity bin.
    Bins with insufficient data are greyed out and labelled.
    """
    labels = [
        f"{b.bin_label}\n(n={b.n_samples})" for b in bin_results
    ]
    mapes = [b.mape if b.sufficient_data else 0.0 for b in bin_results]
    maes  = [b.mae  if b.sufficient_data else 0.0 for b in bin_results]

    x = np.arange(len(labels))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars_mape = ax.bar(x - bar_w / 2, mapes, bar_w, label="MAPE (%)", color="#3498db", alpha=0.85)
    bars_mae  = ax.bar(x + bar_w / 2, maes,  bar_w, label="MAE (BPM)", color="#e67e22", alpha=0.85)

    for i, result in enumerate(bin_results):
        if not result.sufficient_data:
            bars_mape[i].set_facecolor("#cccccc")
            bars_mae[i].set_facecolor("#cccccc")
            ax.text(x[i], 0.4, "Insufficient\ndata",
                    ha="center", va="bottom", fontsize=7, color="#777777")
        else:
            ax.text(bars_mape[i].get_x() + bars_mape[i].get_width() / 2,
                    bars_mape[i].get_height() + 0.1,
                    f"{result.mape:.1f}%", ha="center", va="bottom", fontsize=8)
            ax.text(bars_mae[i].get_x() + bars_mae[i].get_width() / 2,
                    bars_mae[i].get_height() + 0.1,
                    f"{result.mae:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("HR Intensity Zone")
    ax.set_ylabel("Error")
    ax.set_title("Accuracy by HR Intensity Zone")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
