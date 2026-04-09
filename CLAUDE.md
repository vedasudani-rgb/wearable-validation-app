# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py --browser.gatherUsageStats false

# Run all tests
python -m unittest discover tests/

# Run a single test file
python -m unittest tests/test_analysis.py

# Run a single test method
python -m unittest tests.test_analysis.TestQualityLabel.test_excellent

# CLI demo (single + multi-athlete with synthetic data, no UI)
python main.py

# Regenerate all synthetic test CSV files
python generate_test_data.py
```

## Architecture

The codebase is split into a pure Python library (`wearable_validation/`) and a Streamlit front-end (`app.py`). The library has no Streamlit dependency and can be used standalone.

### Data flow

```
ProtocolParams → generate_protocol() → Protocol
                                           ↓ (step_boundaries used for coverage check)
CSV/JSON files → parse_*() → align_timeseries() → HRDataSeries
                                                        ↓
                                          analyze_hr_validation() → AnalysisReport
                                          analyze_by_intensity_bin() → [IntensityBinResult]
                                          check_hr_zone_coverage()  → CoverageReport
                                          generate_recommendation() → OnboardingRecommendation
                                          analyze_group()           → GroupAnalysisReport
                                          analyze_device_comparison() → DeviceComparisonReport
```

### Library modules

- **`models.py`** — all dataclasses; no logic. The single source of truth for types.
- **`constants.py`** — every magic number and threshold. Nothing is hardcoded elsewhere. Use-case LoA thresholds here are literature-cited (see inline comments for references).
- **`protocols.py`** — `generate_protocol()` dispatches to sport/context-specific builders. `_compute_step_boundaries()` produces cumulative timestamps used by `check_hr_zone_coverage()`. Device instructions are fully conditional on `reference_type` (ECG vs HR), `wearable_type` (wrist vs finger), and `sport`.
- **`io.py`** — `normalise_timestamps()` auto-detects ISO-8601, Unix epoch (seconds/ms), and HH:MM:SS. Decimal minutes are **not** auto-detected (removed: ambiguous with integer seconds). `align_timeseries()` resamples both series to a common 1 Hz grid via linear interpolation. `trim_session(data, warmup_seconds, cooldown_seconds)` removes leading/trailing windows; timestamps are **not** re-zeroed so protocol step boundaries stay aligned.
- **`analysis.py`** — stateless functions; all accept `HRDataSeries` and return dataclasses. `check_hr_zone_coverage()` uses a `"trimmed"` step status (distinct from `"unknown"`) for steps with zero samples so trimmed warm-up/cool-down windows don't pollute the overall coverage verdict. `analyze_hr_validation()` computes advanced statistics (Pearson r, R², SEE, parametric bias CI, bootstrap MAPE CI) for n ≥ 3; all are `None` for smaller samples. `_bootstrap_mape_ci()` is a pure-numpy percentile bootstrap (1000 resamples, fixed seed=42 for reproducibility). `analyze_group()` aggregates `mean_pearson_r` and `mean_r_squared` across athletes.
- **`recommendation.py`** — maps `AnalysisReport` + selected use-case keys to `OnboardingRecommendation`. Thresholds pulled from `constants.USE_CASES`. Internal status key is `"not_recommended"` but displayed as "Accuracy Insufficient" in the UI.
- **`report.py`** — plain-text formatters and `build_text_blocks()` (called by `analyze_hr_validation`). PPG limitation note is sport-aware. `_advanced_stats_block()` appends an `--- ADVANCED STATISTICS ---` section to `format_report()` when advanced stats are present. All label columns use Python `:<N` f-string alignment (metadata: width 10, numerical results + advanced stats: width 26, group stats: width 20) so colons are always at a fixed column.
- **`plots.py`** — all return `matplotlib.Figure`. Uses `Agg` backend for Streamlit compatibility.

### Streamlit app (`app.py`)

Three sections rendered top-to-bottom on every run (Streamlit re-runs the full script on each interaction):

1. **Protocol Generator** — builds `Protocol`, stores in `st.session_state["protocol"]`. Use-case checkboxes (`selected_use_cases`) are read at analysis time, not stored in session state.
2. **Upload Session Data** — three modes: Single Athlete, Multiple Athletes, Compare Devices. Files and column mappings stored in session state.
3. **Analysis Results** — reads from session state; calls analysis functions on "Analyse" button press; stores results back into session state.

Key helpers in `app.py`: `_parse_device_instructions()` / `_render_device_instructions()` parse the plain-text protocol string into sections and render a two-column card layout. `_parse_numbered_section()` and `_parse_bullets_section()` handle multi-line continuation. Session trim (warm-up + cool-down) is applied after artifact exclusion, before analysis.

**Advanced Statistics expander** (Single Athlete mode): collapsed by default, sits between the primary 6-column metrics row and the Onboarding Recommendation. Row 1 — 3 columns: Pearson r, R², SEE. Row 2 — 2 columns: Bias 95% CI, MAPE 95% CI (2-column layout prevents value truncation). Cards are rendered via `_adv_metric_card(col, label, value, help_text)` — custom HTML rather than `st.metric()` — so centering and tooltip-icon spacing are pixel-perfect without fighting Streamlit's internal flex layout. The `?` icon uses `inline-flex` with `align-items: center` so it sits flush with the label text. A global `<style>` block injected once after `st.set_page_config` normalises the gap between metric label text and its tooltip button for all `st.metric()` calls on the page. All `st.metric()` calls throughout the analysis section carry a `help=` tooltip with a plain-English description and literature citation. Multiple Athletes mode adds Pearson r, R², SEE columns to the per-athlete table and a "Group Advanced Statistics" expander. Device Comparison mode adds those three columns to the device ranking table.

### Supported values

Sports: `running`, `cycling`. Contexts are sport-gated via `SPORT_CONTEXTS`. Adding a new sport requires: a constant entry, a protocol builder function, conditional branches in `_device_instructions()`, and a `report.py` limitations note.

### Test data

`test_data/` contains pre-generated CSVs organised by scenario (running_steady, running_interval, cycling_steady, cycling_interval, multi_athlete, device_comparison). Regenerate with `python generate_test_data.py`. All files use plain integer seconds as timestamps (column: `timestamp`), with `hr_wearable` or `hr_reference` as HR column names.

## Deployment (Streamlit Cloud)

Live at `wearable-validation-app.streamlit.app` (private repo: `vedasudani-rgb/wearable-validation-app`).

- `runtime.txt` pins Python 3.11. Streamlit Cloud defaults to Python 3.14+ which is too new; without this file the app breaks.
- `requirements.txt` caps `numpy<2.0` and `pandas<3.0` to avoid breaking API changes in those major versions.
- After pushing commits, Streamlit Cloud does a `git pull` (not a fresh clone). If a file update doesn't take effect, go to the Streamlit Cloud dashboard → three-dot menu → **Reboot app** to force a fresh `git clone`.

## Backlog

Approved features not yet implemented:

- **Structured export** — Excel/CSV download of analysis results (per-athlete metrics + group summary).
- **Longitudinal device tracking** — compare the same device across multiple test dates to track performance over time.
