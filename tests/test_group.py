import unittest
import numpy as np

from wearable_validation.models import HRDataSeries, TestRunMetadata
from wearable_validation.analysis import analyze_hr_validation, analyze_group


def _meta(athlete_name: str) -> TestRunMetadata:
    return TestRunMetadata(
        test_date="2026-03-26",
        athlete_name=athlete_name,
        wearable_device_name="Garmin Forerunner 265",
        reference_device_name="Polar H10",
        sport="running",
        metric="heart_rate",
        wearable_type="wrist_based_ppg",
        reference_type="chest_strap_ecg",
    )


def _data(bias: float, n: int = 300, seed: int = 0) -> HRDataSeries:
    rng = np.random.default_rng(seed)
    ref  = rng.uniform(130, 175, size=n)
    wear = ref + bias + rng.normal(0, 3.0, n)
    t    = np.arange(n, dtype=float)
    return HRDataSeries(hr_wearable=wear, hr_reference=ref, timestamps=t)


class TestGroupAggregation(unittest.TestCase):

    def setUp(self):
        biases = [2.0, 4.0, -1.0]
        self.metas    = [_meta(f"Athlete {i+1}") for i in range(3)]
        self.datasets = [_data(b, seed=i) for i, b in enumerate(biases)]
        self.reports  = [
            analyze_hr_validation(d, m)
            for d, m in zip(self.datasets, self.metas)
        ]
        self.group = analyze_group(self.reports, self.datasets)

    def test_n_athletes(self):
        self.assertEqual(self.group.n_athletes, 3)

    def test_mean_bias_approx(self):
        # Rough check: mean of [2, 4, -1] ≈ 1.67
        self.assertAlmostEqual(self.group.mean_bias, 1.67, delta=0.5)

    def test_sd_mape_nonnegative(self):
        self.assertGreaterEqual(self.group.sd_mape, 0.0)

    def test_pooled_loa_ordered(self):
        self.assertLess(self.group.pooled_loa_lower, self.group.pooled_loa_upper)

    def test_group_summary_non_empty(self):
        self.assertGreater(len(self.group.group_summary_text), 20)

    def test_quality_label_valid(self):
        valid = {"excellent", "good", "acceptable", "poor"}
        self.assertIn(self.group.group_quality_label, valid)

    def test_athlete_reports_preserved(self):
        self.assertEqual(len(self.group.athlete_reports), 3)
        for i, r in enumerate(self.group.athlete_reports):
            self.assertEqual(r.metadata.athlete_name, f"Athlete {i+1}")


class TestGroupSingleAthlete(unittest.TestCase):
    """Group of 1 athlete — should work, sd=0 handled gracefully."""

    def test_single_athlete_no_error(self):
        d = _data(3.0)
        m = _meta("Solo")
        r = analyze_hr_validation(d, m)
        g = analyze_group([r], [d])
        self.assertEqual(g.n_athletes, 1)
        self.assertAlmostEqual(g.sd_mape, 0.0, places=6)

    def test_single_athlete_pooled_loa_match_individual(self):
        d = _data(3.0)
        m = _meta("Solo")
        r = analyze_hr_validation(d, m)
        g = analyze_group([r], [d])
        # Pooled LoA from one athlete should match that athlete's LoA closely
        self.assertAlmostEqual(g.pooled_loa_lower, r.loa_lower, delta=0.05)
        self.assertAlmostEqual(g.pooled_loa_upper, r.loa_upper, delta=0.05)


class TestGroupEdgeCases(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            analyze_group([], [])

    def test_mismatched_length_raises(self):
        d = _data(2.0)
        m = _meta("A")
        r = analyze_hr_validation(d, m)
        with self.assertRaises(ValueError):
            analyze_group([r, r], [d])


if __name__ == "__main__":
    unittest.main()
