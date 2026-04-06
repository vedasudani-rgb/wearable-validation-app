import io
import json
import unittest
import warnings
import numpy as np
import pandas as pd

from wearable_validation.io import normalise_timestamps, align_timeseries, parse_combined_file, parse_two_files, trim_session
from wearable_validation.models import HRDataSeries


class TestNormaliseTimestamps(unittest.TestCase):

    def _series(self, values) -> pd.Series:
        return pd.Series(values)

    def test_seconds_from_start(self):
        ts = self._series([0.0, 1.0, 2.0, 3.0])
        result = normalise_timestamps(ts)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0, 3.0])

    def test_seconds_from_offset(self):
        ts = self._series([100.0, 101.0, 102.0])
        result = normalise_timestamps(ts)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_unix_epoch_seconds(self):
        base = 1_711_452_600
        ts = self._series([base, base + 1, base + 2])
        result = normalise_timestamps(ts)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_unix_epoch_milliseconds(self):
        base = 1_711_452_600_000
        ts = self._series([base, base + 1000, base + 2000])
        result = normalise_timestamps(ts)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_iso8601_strings(self):
        ts = self._series([
            "2026-03-26T10:00:00",
            "2026-03-26T10:00:01",
            "2026-03-26T10:00:02",
        ])
        result = normalise_timestamps(ts)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_time_only_strings(self):
        ts = self._series(["10:00:00", "10:00:01", "10:00:02"])
        result = normalise_timestamps(ts)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 2.0])

    def test_time_only_with_fractional_seconds(self):
        ts = self._series(["10:00:00.000", "10:00:00.500", "10:00:01.000"])
        result = normalise_timestamps(ts)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_unrecognised_raises(self):
        ts = self._series(["abc", "def", "ghi"])
        with self.assertRaises(ValueError):
            normalise_timestamps(ts)


class TestAlignTimeseries(unittest.TestCase):

    def test_same_length_same_time(self):
        t = np.arange(200, dtype=float)
        w = np.ones(200) * 150.0
        r = np.ones(200) * 148.0
        data = align_timeseries(t, w, t, r)
        self.assertGreaterEqual(len(data.hr_wearable), 199)
        np.testing.assert_array_almost_equal(data.hr_wearable, 150.0)
        np.testing.assert_array_almost_equal(data.hr_reference, 148.0)

    def test_different_start_times_trims_to_overlap(self):
        # Wearable starts 10 s earlier
        tw = np.arange(0, 200, dtype=float)
        tr = np.arange(10, 210, dtype=float)
        w = np.ones(200) * 150.0
        r = np.ones(200) * 148.0
        data = align_timeseries(tw, w, tr, r)
        # Overlap is t=10 to t=199 → 189 seconds
        self.assertAlmostEqual(data.timestamps[0], 0.0, places=5)
        self.assertLessEqual(len(data.hr_wearable), 190)

    def test_too_short_overlap_raises(self):
        tw = np.arange(0, 30, dtype=float)
        tr = np.arange(0, 30, dtype=float)
        with self.assertRaises(ValueError):
            align_timeseries(tw, np.ones(30), tr, np.ones(30))

    def test_timestamps_start_at_zero(self):
        t = np.arange(200, dtype=float)
        data = align_timeseries(t, np.ones(200), t, np.ones(200))
        self.assertAlmostEqual(data.timestamps[0], 0.0, places=6)

    def test_warns_on_few_samples(self):
        t = np.arange(70, dtype=float)
        with self.assertWarns(UserWarning):
            align_timeseries(t, np.ones(70) * 150, t, np.ones(70) * 148)


class TestParseCombinedFile(unittest.TestCase):

    def _make_csv(self, fmt: str = "seconds") -> io.StringIO:
        if fmt == "seconds":
            rows = "\n".join(
                f"{i},{150 + (i % 5)},{148 + (i % 4)}"
                for i in range(200)
            )
            return io.StringIO(f"timestamp,hr_wearable,hr_reference\n{rows}")
        elif fmt == "iso8601":
            import datetime
            base = datetime.datetime(2026, 3, 26, 10, 0, 0)
            rows = "\n".join(
                f"{(base + datetime.timedelta(seconds=i)).isoformat()},{150 + (i % 5)},{148 + (i % 4)}"
                for i in range(200)
            )
            buf = io.StringIO(f"timestamp,hr_wearable,hr_reference\n{rows}")
            buf.name = "test.csv"
            return buf

    def test_parses_seconds_timestamps(self):
        buf = self._make_csv("seconds")
        buf.name = "test.csv"
        data = parse_combined_file(buf)
        self.assertGreaterEqual(len(data.hr_wearable), 199)
        self.assertAlmostEqual(data.timestamps[0], 0.0)

    def test_parses_iso8601_timestamps(self):
        buf = self._make_csv("iso8601")
        data = parse_combined_file(buf)
        self.assertGreaterEqual(len(data.hr_wearable), 199)
        np.testing.assert_array_almost_equal(data.timestamps[:3], [0.0, 1.0, 2.0])


class TestParseTwoFiles(unittest.TestCase):

    def _make_device_csv(self, hr_values, t_offset=0) -> io.StringIO:
        rows = "\n".join(
            f"{i + t_offset},{hr_values[i]}"
            for i in range(len(hr_values))
        )
        buf = io.StringIO(f"timestamp,hr\n{rows}")
        buf.name = "device.csv"
        return buf

    def test_parses_two_aligned_files(self):
        hr_w = [150.0 + i % 5 for i in range(200)]
        hr_r = [148.0 + i % 4 for i in range(200)]
        fw = self._make_device_csv(hr_w)
        fr = self._make_device_csv(hr_r)
        data = parse_two_files(fw, fr)
        self.assertGreaterEqual(len(data.hr_wearable), 199)

    def test_parses_files_with_time_offset(self):
        # Note: plain numeric (elapsed-second) timestamps are zeroed per-file
        # so a t_offset on one file is lost — both appear synchronised.
        # Offset detection requires absolute timestamps (ISO-8601 / epoch).
        # This test verifies the file parses successfully; alignment covers the
        # full overlapping duration (both files zeroed to 0..199 → 200 samples).
        hr_w = [150.0] * 200
        hr_r = [148.0] * 200
        fw = self._make_device_csv(hr_w, t_offset=0)
        fr = self._make_device_csv(hr_r, t_offset=10)
        data = parse_two_files(fw, fr)
        self.assertGreaterEqual(len(data.hr_wearable), 199)


class TestTrimSession(unittest.TestCase):

    def _make_data(self, n: int = 600) -> HRDataSeries:
        """600-sample HRDataSeries with timestamps 0..599 s."""
        t = np.arange(n, dtype=float)
        return HRDataSeries(
            hr_wearable=np.full(n, 150.0),
            hr_reference=np.full(n, 148.0),
            timestamps=t,
        )

    def test_trim_session_removes_warmup(self):
        data = self._make_data()
        result = trim_session(data, warmup_seconds=60.0)
        self.assertTrue(np.all(result.timestamps >= 0.0))
        # Original had 600 samples; 60 s warm-up removed → ~540 remain
        self.assertEqual(len(result.hr_wearable), 540)

    def test_trim_session_removes_cooldown(self):
        data = self._make_data()
        result = trim_session(data, cooldown_seconds=60.0)
        self.assertEqual(len(result.hr_wearable), 540)

    def test_trim_session_retains_original_timestamps(self):
        # Timestamps are NOT re-zeroed — they stay in the original timebase
        # so that protocol step boundaries (used by check_hr_zone_coverage) still align.
        data = self._make_data()
        result = trim_session(data, warmup_seconds=120.0)
        self.assertAlmostEqual(result.timestamps[0], 120.0)

    def test_trim_session_zero_leaves_data_unchanged(self):
        data = self._make_data()
        result = trim_session(data, warmup_seconds=0.0, cooldown_seconds=0.0)
        np.testing.assert_array_equal(result.hr_wearable, data.hr_wearable)
        np.testing.assert_array_equal(result.hr_reference, data.hr_reference)
        np.testing.assert_array_almost_equal(result.timestamps, data.timestamps)


if __name__ == "__main__":
    unittest.main()
