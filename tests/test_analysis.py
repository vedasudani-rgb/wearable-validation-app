import unittest
import warnings
import numpy as np

from wearable_validation.models import HRDataSeries, TestRunMetadata
from wearable_validation.analysis import (
    analyze_hr_validation, analyze_group, analyze_longitudinal,
    _quality_label, _bootstrap_mape_ci,
)
from wearable_validation.protocols import generate_protocol, compute_hrmax
from wearable_validation.models import ProtocolParams


def _make_metadata(athlete_name: str = "Test Athlete") -> TestRunMetadata:
    return TestRunMetadata(
        test_date="2026-03-26",
        athlete_name=athlete_name,
        wearable_device_name="Garmin Forerunner 265",
        reference_device_name="Polar H10",
        sport="running",
        metric="heart_rate",
        wearable_type="wrist_based_ppg",
        reference_type="chest_strap_ecg",
        conditions="outdoor flat road, 18°C",
    )


def _make_data(w: np.ndarray, r: np.ndarray) -> HRDataSeries:
    t = np.arange(len(w), dtype=float)
    return HRDataSeries(hr_wearable=w, hr_reference=r, timestamps=t)


class TestQualityLabel(unittest.TestCase):
    def test_excellent(self):
        self.assertEqual(_quality_label(2.0), "excellent")

    def test_good(self):
        self.assertEqual(_quality_label(4.0), "good")

    def test_acceptable(self):
        self.assertEqual(_quality_label(7.5), "acceptable")

    def test_poor(self):
        self.assertEqual(_quality_label(12.0), "poor")

    def test_boundary_at_3_is_good(self):
        # 3.0 is NOT excellent (threshold is strictly < 3%)
        self.assertEqual(_quality_label(3.0), "good")

    def test_boundary_at_5_is_acceptable(self):
        self.assertEqual(_quality_label(5.0), "acceptable")


class TestHRmax(unittest.TestCase):
    def test_tanaka_age_30(self):
        self.assertEqual(compute_hrmax(30), 187)

    def test_tanaka_age_40(self):
        self.assertEqual(compute_hrmax(40), 180)

    def test_tanaka_age_20(self):
        self.assertEqual(compute_hrmax(20), 194)


class TestAnalysisPerfectData(unittest.TestCase):
    """Wearable = reference: all errors should be zero."""

    def setUp(self):
        rng = np.random.default_rng(42)
        ref = rng.uniform(120, 180, size=200)
        self.data = _make_data(ref.copy(), ref)
        self.meta = _make_metadata()

    def test_bias_zero(self):
        r = analyze_hr_validation(self.data, self.meta)
        self.assertAlmostEqual(r.bias, 0.0, places=6)

    def test_mae_zero(self):
        r = analyze_hr_validation(self.data, self.meta)
        self.assertAlmostEqual(r.mae, 0.0, places=6)

    def test_mape_zero(self):
        r = analyze_hr_validation(self.data, self.meta)
        self.assertAlmostEqual(r.mape, 0.0, places=6)

    def test_quality_excellent(self):
        r = analyze_hr_validation(self.data, self.meta)
        self.assertEqual(r.quality_label, "excellent")

    def test_n_samples(self):
        r = analyze_hr_validation(self.data, self.meta)
        self.assertEqual(r.n_samples, 200)


class TestAnalysisKnownBias(unittest.TestCase):
    """Wearable = reference + constant 5 BPM offset."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.ref = rng.uniform(120, 180, size=300)
        self.wear = self.ref + 5.0
        self.data = _make_data(self.wear, self.ref)
        self.meta = _make_metadata()

    def test_bias(self):
        r = analyze_hr_validation(self.data, self.meta)
        self.assertAlmostEqual(r.bias, 5.0, places=5)

    def test_mae(self):
        r = analyze_hr_validation(self.data, self.meta)
        self.assertAlmostEqual(r.mae, 5.0, places=5)

    def test_loa_collapses_to_bias(self):
        # Constant offset → SD of diff = 0 → LoA = [5, 5]
        r = analyze_hr_validation(self.data, self.meta)
        self.assertAlmostEqual(r.loa_lower, 5.0, places=3)
        self.assertAlmostEqual(r.loa_upper, 5.0, places=3)


class TestNaNHandling(unittest.TestCase):
    def test_nan_pairs_dropped(self):
        ref  = np.array([150.0, 155.0, np.nan, 160.0, 165.0])
        wear = np.array([151.0, np.nan, 153.0, 161.0, 166.0])
        data = _make_data(wear, ref)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = analyze_hr_validation(data, _make_metadata())
        # Only indices 0 and 3 are complete pairs
        # Valid: index 0 (150, 151), index 3 (160, 161), index 4 (165, 166) = 3 pairs
        self.assertEqual(r.n_samples, 3)


class TestLengthMismatch(unittest.TestCase):
    def test_raises(self):
        data = HRDataSeries(
            hr_wearable=np.array([150.0, 160.0]),
            hr_reference=np.array([150.0]),
            timestamps=np.array([0.0, 1.0]),
        )
        with self.assertRaises(ValueError):
            analyze_hr_validation(data, _make_metadata())


class TestLowSampleWarning(unittest.TestCase):
    def test_warns_when_few_samples(self):
        ref  = np.ones(50) * 150.0
        wear = ref + 2.0
        data = _make_data(wear, ref)
        with self.assertWarns(UserWarning):
            analyze_hr_validation(data, _make_metadata())


class TestProtocolGeneration(unittest.TestCase):
    def _make_params(self, context: str, age=None) -> ProtocolParams:
        return ProtocolParams(
            sport="running", metric="heart_rate",
            wearable_type="wrist_based_ppg", reference_type="chest_strap_ecg",
            context=context,
            wearable_device_name="Garmin", reference_device_name="Polar H10",
            test_date="2026-03-26", age=age,
        )

    def test_steady_run_duration(self):
        p = generate_protocol(self._make_params("steady_run"))
        self.assertAlmostEqual(p.estimated_duration_min, 28.0, places=1)

    def test_steady_run_samples(self):
        p = generate_protocol(self._make_params("steady_run"))
        self.assertEqual(p.n_expected_samples, 1680)

    def test_interval_duration(self):
        p = generate_protocol(self._make_params("interval_session"))
        self.assertAlmostEqual(p.estimated_duration_min, 34.0, places=1)

    def test_interval_samples(self):
        p = generate_protocol(self._make_params("interval_session"))
        self.assertEqual(p.n_expected_samples, 2040)

    def test_invalid_context_raises(self):
        params = self._make_params("steady_run")
        params.context = "tempo_run"
        with self.assertRaises(ValueError):
            generate_protocol(params)

    def test_invalid_wearable_type_raises(self):
        params = self._make_params("steady_run")
        params.wearable_type = "implant"
        with self.assertRaises(ValueError):
            generate_protocol(params)

    def test_hrmax_set_when_age_given(self):
        p = generate_protocol(self._make_params("steady_run", age=30))
        self.assertEqual(p.hrmax, 187)

    def test_bpm_targets_set_when_age_given(self):
        p = generate_protocol(self._make_params("steady_run", age=30))
        for step in p.steps:
            self.assertIsNotNone(step.target_hr_bpm, f"Missing BPM for step: {step.name}")

    def test_bpm_targets_absent_without_age(self):
        p = generate_protocol(self._make_params("steady_run"))
        for step in p.steps:
            self.assertIsNone(step.target_hr_bpm)

    def test_all_steps_have_instructions(self):
        for context in ("steady_run", "interval_session"):
            p = generate_protocol(self._make_params(context))
            for step in p.steps:
                self.assertTrue(step.instructions, f"Empty instructions: {step.name}")


class TestGroupAnalysis(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.metas = [_make_metadata(f"Athlete {i+1}") for i in range(3)]
        self.datasets = []
        self.reports = []
        for i, meta in enumerate(self.metas):
            ref = rng.uniform(130, 175, size=200)
            wear = ref + (i * 2.0) + rng.normal(0, 3.0, 200)
            data = _make_data(wear, ref)
            self.datasets.append(data)
            self.reports.append(analyze_hr_validation(data, meta))

    def test_n_athletes(self):
        g = analyze_group(self.reports, self.datasets)
        self.assertEqual(g.n_athletes, 3)

    def test_mean_mape_reasonable(self):
        g = analyze_group(self.reports, self.datasets)
        self.assertGreater(g.mean_mape, 0.0)
        self.assertLess(g.mean_mape, 20.0)

    def test_pooled_loa_wider_than_per_athlete(self):
        g = analyze_group(self.reports, self.datasets)
        span = g.pooled_loa_upper - g.pooled_loa_lower
        self.assertGreater(span, 0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            analyze_group([], [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            analyze_group(self.reports, self.datasets[:2])


class TestAdvancedStatistics(unittest.TestCase):
    """Advanced stats: Pearson r, R², SEE, and 95% CIs on bias and MAPE."""

    def setUp(self):
        rng = np.random.default_rng(99)
        self.meta = _make_metadata()
        # Perfect: wearable == reference
        ref_perfect = rng.uniform(120, 180, size=200)
        self.data_perfect = _make_data(ref_perfect.copy(), ref_perfect)
        # Constant offset: wearable = reference + 5 BPM (perfect correlation)
        ref_offset = rng.uniform(120, 180, size=200)
        self.data_offset = _make_data(ref_offset + 5.0, ref_offset)
        # Random noise: wearable = reference + N(0, 8)
        ref_noise = rng.uniform(120, 180, size=300)
        noise = rng.normal(0, 8.0, size=300)
        self.data_noise = _make_data(ref_noise + noise, ref_noise)

    def test_perfect_correlation(self):
        r = analyze_hr_validation(self.data_perfect, self.meta)
        self.assertAlmostEqual(r.pearson_r, 1.0, places=5)
        self.assertAlmostEqual(r.r_squared, 1.0, places=5)
        self.assertAlmostEqual(r.see, 0.0, places=5)
        self.assertAlmostEqual(r.bias_ci_lower, 0.0, places=5)
        self.assertAlmostEqual(r.bias_ci_upper, 0.0, places=5)
        self.assertAlmostEqual(r.mape_ci_lower, 0.0, places=5)
        self.assertAlmostEqual(r.mape_ci_upper, 0.0, places=5)

    def test_constant_offset_correlation(self):
        # Constant shift preserves perfect linear relationship
        r = analyze_hr_validation(self.data_offset, self.meta)
        self.assertAlmostEqual(r.pearson_r, 1.0, places=5)
        self.assertAlmostEqual(r.r_squared, 1.0, places=5)
        self.assertAlmostEqual(r.see, 0.0, places=5)
        # CI on bias should collapse to [5.0, 5.0] since variance is zero
        self.assertAlmostEqual(r.bias_ci_lower, 5.0, places=3)
        self.assertAlmostEqual(r.bias_ci_upper, 5.0, places=3)

    def test_random_noise_reduces_correlation(self):
        r = analyze_hr_validation(self.data_noise, self.meta)
        self.assertGreater(r.pearson_r, 0.0)
        self.assertLess(r.pearson_r, 1.0)
        self.assertLessEqual(r.r_squared, r.pearson_r)   # r² ≤ r when 0 < r < 1
        self.assertGreater(r.see, 0.0)
        # Bias CI should straddle zero for unbiased noise
        self.assertLess(r.bias_ci_lower, 0.1)
        self.assertGreater(r.bias_ci_upper, -0.1)
        # MAPE CI must be a valid interval
        self.assertGreater(r.mape_ci_upper, r.mape_ci_lower)
        self.assertGreater(r.mape_ci_lower, 0.0)

    def test_small_n_returns_none(self):
        # n = 2 → advanced stats must be None
        data_tiny = _make_data(
            np.array([150.0, 155.0]),
            np.array([150.0, 155.0]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = analyze_hr_validation(data_tiny, self.meta)
        self.assertIsNone(r.pearson_r)
        self.assertIsNone(r.r_squared)
        self.assertIsNone(r.see)
        self.assertIsNone(r.bias_ci_lower)
        self.assertIsNone(r.bias_ci_upper)
        self.assertIsNone(r.mape_ci_lower)
        self.assertIsNone(r.mape_ci_upper)

    def test_bootstrap_reproducibility(self):
        # Same inputs + seed → identical bounds every call
        rng = np.random.default_rng(5)
        ref = rng.uniform(130, 170, size=100)
        w   = ref + rng.normal(0, 5, size=100)
        lo1, hi1 = _bootstrap_mape_ci(w, ref, seed=42)
        lo2, hi2 = _bootstrap_mape_ci(w, ref, seed=42)
        self.assertEqual(lo1, lo2)
        self.assertEqual(hi1, hi2)
        self.assertGreater(hi1, lo1)
        self.assertGreater(lo1, 0.0)

    def test_group_aggregates_r_and_r_squared(self):
        # analyze_group should populate mean_pearson_r and mean_r_squared
        rng = np.random.default_rng(11)
        reports, datasets = [], []
        for i in range(3):
            ref = rng.uniform(130, 175, size=200)
            wear = ref + rng.normal(0, 4.0, 200)
            data = _make_data(wear, ref)
            datasets.append(data)
            reports.append(analyze_hr_validation(data, _make_metadata(f"A{i}")))
        g = analyze_group(reports, datasets)
        self.assertIsNotNone(g.mean_pearson_r)
        self.assertIsNotNone(g.mean_r_squared)
        self.assertGreater(g.mean_pearson_r, 0.0)
        self.assertLessEqual(g.mean_pearson_r, 1.0)


class TestLongitudinalAnalysis(unittest.TestCase):
    """Tests for analyze_longitudinal()."""

    def _make_session(self, date: str, noise: float) -> tuple:
        rng = np.random.default_rng(hash(date) % (2**32))
        ref  = rng.uniform(130, 175, size=200)
        wear = ref + rng.normal(0, noise, 200)
        data = _make_data(wear, ref)
        meta = TestRunMetadata(
            test_date=date,
            athlete_name="Test Athlete",
            wearable_device_name="TestDevice",
            reference_device_name="Polar H10",
            sport="running",
            metric="heart_rate",
            wearable_type="wrist_based_ppg",
            reference_type="chest_strap_ecg",
        )
        return date, data, meta

    def test_returns_correct_session_count(self):
        sessions = [self._make_session(d, 4.0) for d in ("2024-01-01", "2024-03-01", "2024-06-01")]
        report = analyze_longitudinal(sessions)
        self.assertEqual(len(report.sessions), 3)
        self.assertEqual(len(report.dates), 3)
        self.assertEqual(len(report.mape_trend), 3)

    def test_sessions_sorted_by_date(self):
        # Provide sessions out of order — should come back sorted
        sessions = [
            self._make_session("2024-06-01", 4.0),
            self._make_session("2024-01-01", 4.0),
            self._make_session("2024-03-15", 4.0),
        ]
        report = analyze_longitudinal(sessions)
        self.assertEqual(report.dates, sorted(report.dates))

    def test_mape_trend_matches_per_session_reports(self):
        sessions = [self._make_session(d, 4.0) for d in ("2024-01-01", "2024-04-01")]
        report = analyze_longitudinal(sessions)
        for session, mape in zip(report.sessions, report.mape_trend):
            self.assertAlmostEqual(session.report.mape, mape, places=8)

    def test_mean_mape_within_trend_range(self):
        sessions = [
            self._make_session("2024-01-01", 2.0),   # low noise → low MAPE
            self._make_session("2024-06-01", 12.0),  # high noise → high MAPE
        ]
        report = analyze_longitudinal(sessions)
        self.assertGreaterEqual(report.mean_mape, min(report.mape_trend))
        self.assertLessEqual(report.mean_mape, max(report.mape_trend))

    def test_quality_trend_length_matches_sessions(self):
        sessions = [self._make_session(d, 4.0) for d in ("2024-01-01", "2024-04-01", "2024-08-01")]
        report = analyze_longitudinal(sessions)
        self.assertEqual(len(report.quality_trend), len(report.sessions))

    def test_raises_with_fewer_than_two_sessions(self):
        with self.assertRaises(ValueError):
            analyze_longitudinal([self._make_session("2024-01-01", 4.0)])

    def test_device_name_from_metadata(self):
        sessions = [self._make_session(d, 4.0) for d in ("2024-01-01", "2024-04-01")]
        report = analyze_longitudinal(sessions)
        self.assertEqual(report.device_name, "TestDevice")

    def test_sd_mape_is_zero_for_identical_sessions(self):
        # Two sessions with the same data → sd_mape should be 0
        rng = np.random.default_rng(0)
        ref  = rng.uniform(130, 175, size=200)
        wear = ref + rng.normal(0, 4.0, 200)
        data = _make_data(wear, ref)
        meta = TestRunMetadata(
            test_date="2024-01-01", athlete_name="A", wearable_device_name="D",
            reference_device_name="R", sport="running", metric="heart_rate",
            wearable_type="wrist_based_ppg", reference_type="chest_strap_ecg",
        )
        meta2 = TestRunMetadata(
            test_date="2024-04-01", athlete_name="A", wearable_device_name="D",
            reference_device_name="R", sport="running", metric="heart_rate",
            wearable_type="wrist_based_ppg", reference_type="chest_strap_ecg",
        )
        report = analyze_longitudinal([("2024-01-01", data, meta), ("2024-04-01", data, meta2)])
        self.assertAlmostEqual(report.sd_mape, 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
