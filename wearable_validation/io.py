"""
Data ingestion and timestamp normalisation.

Supports:
  - Combined CSV/JSON: columns timestamp, hr_wearable, hr_reference
  - Two separate CSV/JSON files: each with columns timestamp, hr

Timestamp formats detected automatically (see normalise_timestamps docstring).
Both series are resampled to a common 1 Hz grid via linear interpolation.
"""
from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd

from wearable_validation.models import HRDataSeries
from wearable_validation.constants import DEFAULT_SAMPLE_RATE_HZ, MIN_OVERLAP_SECONDS, MIN_SAMPLES_WARNING


# ── Public parse functions ────────────────────────────────────────────────────

def parse_combined_file(
    file,
    timestamp_col: str = "timestamp",
    wearable_col: str = "hr_wearable",
    reference_col: str = "hr_reference",
) -> HRDataSeries:
    """
    Parse a single combined CSV or JSON file containing both HR series.
    Returns an aligned HRDataSeries resampled to DEFAULT_SAMPLE_RATE_HZ.
    """
    df = _read_file(file)
    df = _normalise_column_names(df)
    timestamp_col, wearable_col, reference_col = (
        _resolve_col(df, timestamp_col, _TIMESTAMP_ALIASES, "timestamp"),
        _resolve_col(df, wearable_col, _WEARABLE_ALIASES, "hr_wearable"),
        _resolve_col(df, reference_col, _REFERENCE_ALIASES, "hr_reference"),
    )
    t  = normalise_timestamps(df[timestamp_col])
    hw = df[wearable_col].to_numpy(dtype=float)
    hr = df[reference_col].to_numpy(dtype=float)
    return align_timeseries(t, hw, t, hr)


def parse_two_files(
    wearable_file,
    reference_file,
    timestamp_col: str = "timestamp",
    wearable_hr_col: str = "hr",
    reference_hr_col: str = "hr",
) -> HRDataSeries:
    """
    Parse two separate files (one per device). Each must contain a timestamp
    column and an HR column. Returns an aligned HRDataSeries.
    """
    dfw = _read_file(wearable_file)
    dfr = _read_file(reference_file)
    dfw = _normalise_column_names(dfw)
    dfr = _normalise_column_names(dfr)

    tc_w = _resolve_col(dfw, timestamp_col, _TIMESTAMP_ALIASES, "timestamp")
    hc_w = _resolve_col(dfw, wearable_hr_col, _HR_ALIASES, "hr")
    tc_r = _resolve_col(dfr, timestamp_col, _TIMESTAMP_ALIASES, "timestamp")
    hc_r = _resolve_col(dfr, reference_hr_col, _HR_ALIASES, "hr")

    tw = normalise_timestamps(dfw[tc_w])
    hw = dfw[hc_w].to_numpy(dtype=float)
    tr = normalise_timestamps(dfr[tc_r])
    hr = dfr[hc_r].to_numpy(dtype=float)

    return align_timeseries(tw, hw, tr, hr)


# ── Timestamp normalisation ───────────────────────────────────────────────────

def normalise_timestamps(raw: pd.Series) -> np.ndarray:
    """
    Convert any of the following timestamp formats to a float array of
    seconds elapsed from t=0 (first sample):

      - ISO-8601 datetime strings: "2026-03-26T10:30:00Z", "2026-03-26 10:30:00"
      - US datetime strings:       "03/26/2026 10:30:00"
      - Time-only strings:         "10:30:00", "10:30:00.500"
      - Unix epoch seconds:        1711452600   (values > 1e8 heuristic)
      - Unix epoch milliseconds:   1711452600000 (values > 1e11 heuristic)
      - Numeric seconds from start: 0, 1, 2, ...

    Note: decimal minutes are NOT auto-detected because they are
    indistinguishable from integer seconds. Convert decimal minutes to
    seconds before uploading.

    Raises ValueError with diagnostic info if the format cannot be inferred.
    """
    raw = raw.copy()

    # ── String timestamps ──────────────────────────────────────────────────
    if raw.dtype == object or pd.api.types.is_string_dtype(raw):
        raw_str = raw.astype(str).str.strip()

        # Time-only: HH:MM:SS or HH:MM:SS.fff
        if raw_str.str.match(r"^\d{1,2}:\d{2}:\d{2}").all():
            return _time_only_to_seconds(raw_str)

        # Try pandas datetime parsing (handles ISO-8601, US formats, etc.)
        try:
            parsed = pd.to_datetime(raw_str, utc=False)
            return _datetime_series_to_seconds(parsed)
        except Exception:
            pass

        raise ValueError(
            f"Could not parse timestamp column. First few values: {raw_str.head(3).tolist()}\n"
            "Expected: ISO-8601, US datetime, HH:MM:SS, Unix epoch, or numeric seconds."
        )

    # ── Numeric timestamps ─────────────────────────────────────────────────
    values = pd.to_numeric(raw, errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"Timestamp column contains non-numeric values that could not be parsed. "
            f"Sample: {raw.head(3).tolist()}"
        )

    arr = values.to_numpy(dtype=float)
    median_val = float(np.median(arr))

    if median_val > 1e11:
        # Unix epoch milliseconds → convert to seconds
        arr = arr / 1000.0
        arr -= arr[0]
    elif median_val > 1e8:
        # Unix epoch seconds
        arr -= arr[0]
    else:
        # Treat as seconds from start (or seconds-from-offset)
        arr -= arr[0]

    return arr.astype(float)


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_timeseries(
    w_time: np.ndarray,
    w_hr: np.ndarray,
    r_time: np.ndarray,
    r_hr: np.ndarray,
    hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> HRDataSeries:
    """
    Resample both HR series onto a shared regular time grid using linear
    interpolation (Stahl et al. 2016, INTERLIVE Düking et al. 2020).

    Steps:
    1. Find the overlapping time window.
    2. Build a common axis at `hz` Hz.
    3. Linear interpolate both signals onto the common axis.
    4. Drop any NaN pairs.
    """
    t_start = max(float(w_time[0]), float(r_time[0]))
    t_end   = min(float(w_time[-1]), float(r_time[-1]))

    overlap_sec = t_end - t_start
    if overlap_sec < MIN_OVERLAP_SECONDS:
        raise ValueError(
            f"Overlapping recording time is only {overlap_sec:.1f} s "
            f"(minimum required: {MIN_OVERLAP_SECONDS} s). "
            "Check that both files are from the same session."
        )

    step = 1.0 / hz
    # Add half-step epsilon so the endpoint is included (np.arange is half-open)
    t_common = np.arange(t_start, t_end + step * 0.5, step)

    hw = np.interp(t_common, w_time, w_hr)
    hr = np.interp(t_common, r_time, r_hr)

    # Drop NaN pairs (can arise from out-of-bounds extrapolation edge cases)
    valid = ~(np.isnan(hw) | np.isnan(hr))
    hw, hr, t_common = hw[valid], hr[valid], t_common[valid]

    # Re-zero timestamps
    t_common = t_common - t_common[0]

    n = len(hw)
    if n < MIN_SAMPLES_WARNING:
        warnings.warn(
            f"Only {n} aligned samples after overlap. "
            f"Bland-Altman LoA require ≥ {MIN_SAMPLES_WARNING} for reliable estimates.",
            UserWarning,
            stacklevel=2,
        )

    return HRDataSeries(hr_wearable=hw, hr_reference=hr, timestamps=t_common)


# ── Internal helpers ──────────────────────────────────────────────────────────

# Column name aliases (lowercase, stripped)
_TIMESTAMP_ALIASES = {"timestamp", "time", "ts", "datetime", "date_time", "elapsed", "elapsed_s",
                      "elapsed_sec", "time_s", "time_sec", "seconds"}
_WEARABLE_ALIASES  = {"hr_wearable", "wearable_hr", "hr_watch", "watch_hr", "hr_device",
                      "device_hr", "hr1", "heart_rate_1", "bpm_wearable"}
_REFERENCE_ALIASES = {"hr_reference", "reference_hr", "hr_strap", "strap_hr", "hr_ref",
                      "ref_hr", "hr2", "heart_rate_2", "bpm_reference", "gold_standard_hr"}
_HR_ALIASES        = {"hr", "heart_rate", "bpm", "heartrate", "hr_bpm", "heart_rate_bpm"}


def _read_file(file) -> pd.DataFrame:
    """Read CSV or JSON from a file path or file-like object."""
    # Detect format from name attribute or assume CSV
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith(".json"):
        return pd.read_json(file)
    return pd.read_csv(file)


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip whitespace from column names."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _resolve_col(
    df: pd.DataFrame,
    preferred: str,
    aliases: set[str],
    role: str,
) -> str:
    """
    Return the actual column name to use for a given role.
    Prefers `preferred` if present; otherwise finds the first alias match.
    Raises ValueError if no match is found.
    """
    preferred_lower = preferred.strip().lower().replace(" ", "_")
    if preferred_lower in df.columns:
        return preferred_lower
    for alias in aliases:
        if alias in df.columns:
            return alias
    raise ValueError(
        f"Could not find a {role} column in the file. "
        f"Available columns: {list(df.columns)}. "
        f"Known aliases: {sorted(aliases)}."
    )


def _datetime_series_to_seconds(parsed: pd.Series) -> np.ndarray:
    """Convert a parsed datetime Series to seconds from first timestamp."""
    epoch = parsed.min()
    return (parsed - epoch).dt.total_seconds().to_numpy(dtype=float)


def _time_only_to_seconds(raw_str: pd.Series) -> np.ndarray:
    """Convert HH:MM:SS[.fff] strings to seconds from first value."""
    def _parse_one(s: str) -> float:
        parts = s.split(":")
        h, m = int(parts[0]), int(parts[1])
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return h * 3600 + m * 60 + sec

    arr = np.array([_parse_one(s) for s in raw_str], dtype=float)
    return arr - arr[0]
