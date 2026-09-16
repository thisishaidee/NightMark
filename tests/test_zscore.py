"""
Tests for src/zscore.py: rolling same-session z-score.
"""

import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import SessionType
from src.observation import MarketObservation, ReferenceQuality, ReferenceSource
from src.zscore import rolling_same_session_zscore
from tests.fixtures import et


def make_obs(timestamp, session, basis_bps, symbol="X"):
    """Minimal synthetic MarketObservation builder for z-score tests only.

    Only `session` and `basis_bps` matter to src/zscore.py; the rest of
    the fields are filled with harmless placeholders.
    """
    return MarketObservation(
        timestamp=timestamp,
        symbol=symbol,
        session=session,
        rtoken_bid=None,
        rtoken_ask=None,
        rtoken_mid=None,
        spread_bps=None,
        available_depth=None,
        reference_price=None,
        reference_source=ReferenceSource.BITGET_OFFICIAL,
        reference_quality=ReferenceQuality.OFFICIAL,
        basis=None,
        basis_bps=basis_bps,
        is_valid=basis_bps is not None,
        invalid_reason=None if basis_bps is not None else "missing_for_test",
    )


class TestBasicZScoreFormula(unittest.TestCase):
    def test_known_values_match_manual_zscore_formula(self):
        values = [10.0, 12.0, 11.0, 13.0, 50.0]
        obs = [make_obs(et(2026, 9, 15, 12, i), SessionType.RTH, v) for i, v in enumerate(values)]
        results = rolling_same_session_zscore(obs, window=10, min_observations=2)

        self.assertEqual(results[0].reason, "insufficient_history")
        self.assertEqual(results[1].reason, "insufficient_history")

        baseline3 = [10.0, 12.0]
        expected3 = (11.0 - statistics.fmean(baseline3)) / statistics.stdev(baseline3)
        self.assertAlmostEqual(results[2].value, expected3, places=9)

        baseline5 = [10.0, 12.0, 11.0, 13.0]
        expected5 = (50.0 - statistics.fmean(baseline5)) / statistics.stdev(baseline5)
        self.assertAlmostEqual(results[4].value, expected5, places=9)


class TestInsufficientHistory(unittest.TestCase):
    def test_first_observation_always_insufficient(self):
        obs = [make_obs(et(2026, 9, 15, 12, 0), SessionType.RTH, 10.0)]
        results = rolling_same_session_zscore(obs, window=5, min_observations=1)
        self.assertIsNone(results[0].value)
        self.assertEqual(results[0].reason, "insufficient_history")

    def test_min_observations_of_one_still_needs_two_points_for_stdev(self):
        obs = [
            make_obs(et(2026, 9, 15, 12, 0), SessionType.RTH, 10.0),
            make_obs(et(2026, 9, 15, 12, 1), SessionType.RTH, 12.0),
        ]
        results = rolling_same_session_zscore(obs, window=5, min_observations=1)
        # Second point has exactly 1 prior point -> still insufficient,
        # since a sample stdev needs at least 2 baseline points.
        self.assertEqual(results[1].reason, "insufficient_history")


class TestZeroVariance(unittest.TestCase):
    def test_identical_baseline_values_report_zero_variance(self):
        values = [10.0, 10.0, 10.0]
        obs = [make_obs(et(2026, 9, 15, 12, i), SessionType.RTH, v) for i, v in enumerate(values)]
        results = rolling_same_session_zscore(obs, window=5, min_observations=2)
        self.assertEqual(results[2].reason, "zero_variance")
        self.assertIsNone(results[2].value)


class TestMissingValue(unittest.TestCase):
    def test_missing_current_value_reports_missing_value(self):
        obs = [
            make_obs(et(2026, 9, 15, 12, 0), SessionType.RTH, 10.0),
            make_obs(et(2026, 9, 15, 12, 1), SessionType.RTH, 11.0),
            make_obs(et(2026, 9, 15, 12, 2), SessionType.RTH, None),
        ]
        results = rolling_same_session_zscore(obs, window=5, min_observations=2)
        self.assertEqual(results[2].reason, "missing_value")
        self.assertIsNone(results[2].value)

    def test_missing_value_never_enters_future_baseline(self):
        obs = [
            make_obs(et(2026, 9, 15, 12, 0), SessionType.RTH, 10.0),
            make_obs(et(2026, 9, 15, 12, 1), SessionType.RTH, 11.0),
            make_obs(et(2026, 9, 15, 12, 2), SessionType.RTH, None),  # missing, must be skipped
            make_obs(et(2026, 9, 15, 12, 3), SessionType.RTH, 12.0),
        ]
        results = rolling_same_session_zscore(obs, window=5, min_observations=2)
        self.assertEqual(results[3].observations_used, 2)


class TestNoLookAhead(unittest.TestCase):
    def test_future_outlier_does_not_affect_earlier_zscores(self):
        leading_values = [10.0, 11.0, 12.0, 10.5]
        obs_a = [make_obs(et(2026, 9, 15, 12, i), SessionType.RTH, v) for i, v in enumerate(leading_values)]
        results_a = rolling_same_session_zscore(obs_a, window=10, min_observations=2)

        obs_b = obs_a + [make_obs(et(2026, 9, 15, 13, 30), SessionType.RTH, 999999.0)]
        results_b = rolling_same_session_zscore(obs_b, window=10, min_observations=2)

        for i in range(len(obs_a)):
            self.assertEqual(results_a[i].value, results_b[i].value)
            self.assertEqual(results_a[i].reason, results_b[i].reason)

    def test_current_observation_excluded_from_its_own_baseline(self):
        obs = [
            make_obs(et(2026, 9, 15, 12, 0), SessionType.RTH, 5.0),
            make_obs(et(2026, 9, 15, 12, 1), SessionType.RTH, 5.0),
            make_obs(et(2026, 9, 15, 12, 2), SessionType.RTH, 5.0),
        ]
        results = rolling_same_session_zscore(obs, window=10, min_observations=2)
        # Baseline for the 3rd point must be exactly the first two 5.0s
        # (size 2), not size 3 - which would mean the point counted itself.
        self.assertEqual(results[2].observations_used, 2)
        self.assertEqual(results[2].reason, "zero_variance")


class TestSessionSeparation(unittest.TestCase):
    def test_rth_and_weekend_never_share_a_baseline(self):
        obs = [
            make_obs(et(2026, 9, 15, 12, 0), SessionType.RTH, 10.0),
            make_obs(et(2026, 9, 19, 13, 0), SessionType.WEEKEND, 1000.0),
            make_obs(et(2026, 9, 15, 12, 1), SessionType.RTH, 11.0),
            make_obs(et(2026, 9, 19, 13, 1), SessionType.WEEKEND, 1010.0),
            make_obs(et(2026, 9, 15, 12, 2), SessionType.RTH, 12.0),
            make_obs(et(2026, 9, 19, 13, 2), SessionType.WEEKEND, 1020.0),
        ]
        results = rolling_same_session_zscore(obs, window=10, min_observations=2)

        # 3rd RTH point (index 4): baseline must be exactly [10.0, 11.0].
        self.assertEqual(results[4].observations_used, 2)
        expected_rth = (12.0 - statistics.fmean([10.0, 11.0])) / statistics.stdev([10.0, 11.0])
        self.assertAlmostEqual(results[4].value, expected_rth, places=9)

        # 3rd WEEKEND point (index 5): baseline must be exactly [1000.0, 1010.0].
        self.assertEqual(results[5].observations_used, 2)
        expected_weekend = (1020.0 - statistics.fmean([1000.0, 1010.0])) / statistics.stdev([1000.0, 1010.0])
        self.assertAlmostEqual(results[5].value, expected_weekend, places=9)

    def test_overnight_and_post_also_kept_separate(self):
        obs = [
            make_obs(et(2026, 9, 15, 20, 0), SessionType.OVERNIGHT, 5.0),
            make_obs(et(2026, 9, 15, 18, 0), SessionType.POST, 500.0),
            make_obs(et(2026, 9, 15, 21, 0), SessionType.OVERNIGHT, 6.0),
            make_obs(et(2026, 9, 15, 18, 1), SessionType.POST, 510.0),
            make_obs(et(2026, 9, 15, 22, 0), SessionType.OVERNIGHT, 7.0),
        ]
        results = rolling_same_session_zscore(obs, window=10, min_observations=2)
        self.assertEqual(results[4].observations_used, 2)


class TestConfigurableWindowAndMinObservations(unittest.TestCase):
    def test_window_truncates_older_history(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        obs = [make_obs(et(2026, 9, 15, 12, i), SessionType.RTH, v) for i, v in enumerate(values)]
        results = rolling_same_session_zscore(obs, window=2, min_observations=2)
        self.assertEqual(results[4].observations_used, 2)
        expected = (50.0 - statistics.fmean([30.0, 40.0])) / statistics.stdev([30.0, 40.0])
        self.assertAlmostEqual(results[4].value, expected, places=9)

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            rolling_same_session_zscore([], window=0, min_observations=1)

    def test_min_observations_exceeding_window_raises(self):
        with self.assertRaises(ValueError):
            rolling_same_session_zscore([], window=2, min_observations=3)

    def test_zero_min_observations_raises(self):
        with self.assertRaises(ValueError):
            rolling_same_session_zscore([], window=2, min_observations=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
