"""
Tests for src/mark_quality.py.

Every gate is tested independently (via directly constructed configs, not
the shipped config/mark_quality.yaml, so these tests are stable regardless
of what real thresholds get tuned later) plus boundary cases at each
threshold edge.
"""

import datetime as dt
import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import SessionType
from src.mark_quality import (
    CorporateActionWindow,
    DepthQualityConfig,
    MarkQualityConfig,
    ReferenceStalenessConfig,
    SpreadQualityConfig,
    evaluate_mark_quality,
)
from src.observation import MarketObservation, ReferenceQuality, ReferenceSource

ET = ZoneInfo("America/New_York")


def et(y, m, d, h, mi, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=ET)


def make_obs(
    *,
    timestamp,
    symbol="RNVDA",
    session=SessionType.WEEKEND,
    spread_bps=10.0,
    available_depth=1000.0,
    reference_price=100.0,
    reference_source=ReferenceSource.BITGET_OFFICIAL,
    reference_quality=ReferenceQuality.OFFICIAL,
    is_valid=True,
    invalid_reason=None,
):
    return MarketObservation(
        timestamp=timestamp,
        symbol=symbol,
        session=session,
        rtoken_bid=99.5,
        rtoken_ask=100.5,
        rtoken_mid=100.0,
        spread_bps=spread_bps,
        available_depth=available_depth,
        reference_price=reference_price,
        reference_source=reference_source,
        reference_quality=reference_quality,
        basis=0.0,
        basis_bps=0.0,
        is_valid=is_valid,
        invalid_reason=invalid_reason,
    )


def base_config(**overrides):
    defaults = dict(
        spread=SpreadQualityConfig(
            percentile=0.90,
            min_history_for_percentile=5,
            fallback_max_spread_bps=50.0,
        ),
        depth=DepthQualityConfig(default_min_depth=100.0),
        staleness=ReferenceStalenessConfig(
            max_age_by_session={
                SessionType.OVERNIGHT: dt.timedelta(hours=6),
                SessionType.WEEKEND: dt.timedelta(hours=48),
            },
            weekend_corroboration_required_beyond=dt.timedelta(hours=48),
        ),
        corporate_action_windows=(),
    )
    defaults.update(overrides)
    return MarkQualityConfig(**defaults)


WEEKEND_TS = et(2026, 9, 19, 13, 0)  # verified Phase 1 WEEKEND timestamp


class TestSpreadQuality(unittest.TestCase):
    def test_spread_within_fallback_threshold_passes(self):
        obs = make_obs(timestamp=WEEKEND_TS, spread_bps=49.9)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(result.passed)
        self.assertEqual(result.spread_threshold_source, "FALLBACK_STATIC")

    def test_spread_at_fallback_threshold_boundary_passes(self):
        # Boundary: exactly at threshold is inclusive (pass).
        obs = make_obs(timestamp=WEEKEND_TS, spread_bps=50.0)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(result.passed)

    def test_spread_just_above_fallback_threshold_fails(self):
        obs = make_obs(timestamp=WEEKEND_TS, spread_bps=50.01)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("spread_too_wide", result.reasons)

    def test_missing_spread_fails_closed(self):
        obs = make_obs(timestamp=WEEKEND_TS, spread_bps=None)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("spread_unavailable", result.reasons)

    def test_historical_percentile_used_when_enough_history(self):
        history = [10.0, 12.0, 14.0, 16.0, 18.0]  # 5 points, min_history_for_percentile=5
        cfg = base_config()
        obs_ok = make_obs(timestamp=WEEKEND_TS, spread_bps=17.0)
        result_ok = evaluate_mark_quality(
            obs_ok, config=cfg, reference_timestamp=WEEKEND_TS, spread_history_bps=history
        )
        self.assertEqual(result_ok.spread_threshold_source, "HISTORICAL_PERCENTILE")
        # 90th percentile (linear interpolation) of [10,12,14,16,18] = 17.2
        self.assertAlmostEqual(result_ok.spread_threshold_used, 17.2, places=6)
        self.assertTrue(result_ok.passed)

        obs_bad = make_obs(timestamp=WEEKEND_TS, spread_bps=17.5)
        result_bad = evaluate_mark_quality(
            obs_bad, config=cfg, reference_timestamp=WEEKEND_TS, spread_history_bps=history
        )
        self.assertFalse(result_bad.passed)
        self.assertIn("spread_too_wide", result_bad.reasons)

    def test_insufficient_history_falls_back_to_static_not_more_permissive(self):
        history = [10.0, 12.0]  # only 2 points, below min_history_for_percentile=5
        cfg = base_config()
        obs = make_obs(timestamp=WEEKEND_TS, spread_bps=49.9)
        result = evaluate_mark_quality(
            obs, config=cfg, reference_timestamp=WEEKEND_TS, spread_history_bps=history
        )
        self.assertEqual(result.spread_threshold_source, "FALLBACK_STATIC")
        self.assertEqual(result.spread_threshold_used, 50.0)

    def test_per_symbol_fallback_override(self):
        cfg = base_config(
            spread=SpreadQualityConfig(
                min_history_for_percentile=100,
                fallback_max_spread_bps=50.0,
                fallback_max_spread_bps_by_symbol={"RNVDA": 5.0},
            )
        )
        obs = make_obs(timestamp=WEEKEND_TS, symbol="RNVDA", spread_bps=6.0)
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertEqual(result.spread_threshold_used, 5.0)
        self.assertFalse(result.passed)


class TestDepthQuality(unittest.TestCase):
    def test_depth_above_minimum_passes(self):
        obs = make_obs(timestamp=WEEKEND_TS, available_depth=150.0)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(result.passed)

    def test_depth_at_minimum_boundary_passes(self):
        obs = make_obs(timestamp=WEEKEND_TS, available_depth=100.0)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(result.passed)

    def test_depth_just_below_minimum_fails(self):
        obs = make_obs(timestamp=WEEKEND_TS, available_depth=99.99)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("depth_below_minimum", result.reasons)

    def test_missing_depth_fails_closed(self):
        obs = make_obs(timestamp=WEEKEND_TS, available_depth=None)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("depth_unavailable", result.reasons)

    def test_unconfigured_symbol_fails_closed_when_no_default(self):
        cfg = base_config(depth=DepthQualityConfig(default_min_depth=None))
        obs = make_obs(timestamp=WEEKEND_TS, available_depth=1_000_000.0)
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("depth_minimum_not_configured", result.reasons)

    def test_per_symbol_minimum_override(self):
        cfg = base_config(
            depth=DepthQualityConfig(default_min_depth=100.0, min_depth_by_symbol={"RNVDA": 500.0})
        )
        obs = make_obs(timestamp=WEEKEND_TS, symbol="RNVDA", available_depth=200.0)
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("depth_below_minimum", result.reasons)


class TestReferenceStaleness(unittest.TestCase):
    def test_fresh_reference_passes(self):
        ref_ts = WEEKEND_TS - dt.timedelta(hours=1)
        obs = make_obs(timestamp=WEEKEND_TS)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=ref_ts)
        self.assertTrue(result.passed)

    def test_reference_exactly_at_max_age_boundary_passes(self):
        ref_ts = WEEKEND_TS - dt.timedelta(hours=48)
        obs = make_obs(timestamp=WEEKEND_TS)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=ref_ts)
        self.assertTrue(result.passed)

    def test_missing_reference_timestamp_fails_closed(self):
        obs = make_obs(timestamp=WEEKEND_TS)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=None)
        self.assertFalse(result.passed)
        self.assertIn("reference_timestamp_unavailable", result.reasons)

    def test_unconfigured_session_staleness_fails_closed(self):
        cfg = base_config(staleness=ReferenceStalenessConfig(max_age_by_session={}))
        obs = make_obs(timestamp=WEEKEND_TS)
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("reference_staleness_threshold_not_configured_for_session", result.reasons)

    def test_weekend_over_48h_uncorroborated_fails(self):
        ref_ts = WEEKEND_TS - dt.timedelta(hours=49)
        obs = make_obs(timestamp=WEEKEND_TS)
        cfg = base_config()
        result = evaluate_mark_quality(
            obs, config=cfg, reference_timestamp=ref_ts, has_corroborating_witness_move=False
        )
        self.assertFalse(result.passed)
        self.assertIn("stale_reference_weekend_uncorroborated", result.reasons)

    def test_weekend_over_48h_corroborated_passes_staleness_gate(self):
        ref_ts = WEEKEND_TS - dt.timedelta(hours=49)
        obs = make_obs(timestamp=WEEKEND_TS)
        cfg = base_config()
        result = evaluate_mark_quality(
            obs, config=cfg, reference_timestamp=ref_ts, has_corroborating_witness_move=True
        )
        self.assertTrue(result.passed)
        self.assertNotIn("stale_reference_weekend_uncorroborated", result.reasons)

    def test_weekend_between_max_age_and_corroboration_threshold_fails_plain_stale(self):
        # 48h max_age already exceeded at, say, 48.5h, but the 48h
        # corroboration threshold in this config is ALSO 48h - so anything
        # over 48h on WEEKEND routes through the corroboration branch, not
        # a separate "plain stale" branch. This test instead uses a config
        # where the corroboration threshold is set higher (72h) than the
        # max_age (48h), to prove the intermediate zone (48h-72h) is
        # rejected as plain stale_reference, not silently passed.
        cfg = base_config(
            staleness=ReferenceStalenessConfig(
                max_age_by_session={SessionType.WEEKEND: dt.timedelta(hours=48)},
                weekend_corroboration_required_beyond=dt.timedelta(hours=72),
            )
        )
        ref_ts = WEEKEND_TS - dt.timedelta(hours=60)  # 48h < 60h < 72h
        obs = make_obs(timestamp=WEEKEND_TS)
        result = evaluate_mark_quality(
            obs, config=cfg, reference_timestamp=ref_ts, has_corroborating_witness_move=True
        )
        self.assertFalse(result.passed)
        self.assertIn("stale_reference", result.reasons)
        self.assertNotIn("stale_reference_weekend_uncorroborated", result.reasons)

    def test_overnight_staleness_has_no_corroboration_exception(self):
        overnight_ts = et(2026, 9, 15, 22, 0)  # verified Phase 1 OVERNIGHT timestamp
        ref_ts = overnight_ts - dt.timedelta(hours=7)  # exceeds 6h max_age
        obs = make_obs(timestamp=overnight_ts, session=SessionType.OVERNIGHT)
        cfg = base_config()
        result = evaluate_mark_quality(
            obs, config=cfg, reference_timestamp=ref_ts, has_corroborating_witness_move=True
        )
        self.assertFalse(result.passed)
        self.assertIn("stale_reference", result.reasons)

    def test_missing_reference_price_fails_closed(self):
        obs = make_obs(timestamp=WEEKEND_TS, reference_price=None, reference_source=ReferenceSource.UNAVAILABLE, reference_quality=ReferenceQuality.UNAVAILABLE, is_valid=False, invalid_reason="no_reference_price_available")
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("reference_unavailable", result.reasons)

    def test_future_reference_timestamp_rejected(self):
        ref_ts = WEEKEND_TS + dt.timedelta(hours=1)
        obs = make_obs(timestamp=WEEKEND_TS)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=ref_ts)
        self.assertFalse(result.passed)
        self.assertIn("reference_timestamp_in_future", result.reasons)


class TestStaleOrMissingMarketData(unittest.TestCase):
    def test_invalid_observation_fails_closed(self):
        obs = make_obs(
            timestamp=WEEKEND_TS,
            is_valid=False,
            invalid_reason="invalid_or_missing_rtoken_quote",
        )
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("unusable_observation:invalid_or_missing_rtoken_quote", result.reasons)

    def test_valid_observation_does_not_trigger_this_reason(self):
        obs = make_obs(timestamp=WEEKEND_TS, is_valid=True)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(all("unusable_observation" not in r for r in result.reasons))


class TestSessionValidity(unittest.TestCase):
    def test_rth_rejected(self):
        obs = make_obs(timestamp=et(2026, 9, 15, 12, 0), session=SessionType.RTH)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=et(2026, 9, 15, 12, 0))
        self.assertFalse(result.passed)
        self.assertIn("session_not_tradeable:RTH", result.reasons)

    def test_post_rejected(self):
        obs = make_obs(timestamp=et(2026, 9, 15, 18, 0), session=SessionType.POST)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=et(2026, 9, 15, 18, 0))
        self.assertFalse(result.passed)
        self.assertIn("session_not_tradeable:POST", result.reasons)

    def test_halt_rejected(self):
        obs = make_obs(timestamp=WEEKEND_TS, session=SessionType.HALT)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("session_not_tradeable:HALT", result.reasons)

    def test_unknown_rejected(self):
        obs = make_obs(timestamp=WEEKEND_TS, session=SessionType.UNKNOWN)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("session_not_tradeable:UNKNOWN", result.reasons)

    def test_overnight_allowed_by_this_gate(self):
        overnight_ts = et(2026, 9, 15, 22, 0)
        obs = make_obs(timestamp=overnight_ts, session=SessionType.OVERNIGHT)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=overnight_ts)
        self.assertTrue(all("session_not_tradeable" not in r for r in result.reasons))

    def test_weekend_allowed_by_this_gate(self):
        obs = make_obs(timestamp=WEEKEND_TS, session=SessionType.WEEKEND)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(all("session_not_tradeable" not in r for r in result.reasons))


class TestCorporateActionProtection(unittest.TestCase):
    def test_inside_configured_window_rejected(self):
        window = CorporateActionWindow(
            symbol="RNVDA",
            start=WEEKEND_TS - dt.timedelta(hours=1),
            end=WEEKEND_TS + dt.timedelta(hours=1),
            kind="earnings",
        )
        cfg = base_config(corporate_action_windows=(window,))
        obs = make_obs(timestamp=WEEKEND_TS, symbol="RNVDA")
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        self.assertIn("corporate_action_window:earnings", result.reasons)

    def test_window_start_boundary_is_inclusive(self):
        window = CorporateActionWindow(symbol="RNVDA", start=WEEKEND_TS, end=WEEKEND_TS + dt.timedelta(hours=1))
        cfg = base_config(corporate_action_windows=(window,))
        obs = make_obs(timestamp=WEEKEND_TS, symbol="RNVDA")
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)

    def test_window_end_boundary_is_exclusive(self):
        window = CorporateActionWindow(
            symbol="RNVDA", start=WEEKEND_TS - dt.timedelta(hours=1), end=WEEKEND_TS
        )
        cfg = base_config(corporate_action_windows=(window,))
        obs = make_obs(timestamp=WEEKEND_TS, symbol="RNVDA")
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(all("corporate_action_window" not in r for r in result.reasons))

    def test_different_symbol_unaffected(self):
        window = CorporateActionWindow(
            symbol="RTSLA", start=WEEKEND_TS - dt.timedelta(hours=1), end=WEEKEND_TS + dt.timedelta(hours=1)
        )
        cfg = base_config(corporate_action_windows=(window,))
        obs = make_obs(timestamp=WEEKEND_TS, symbol="RNVDA")
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(all("corporate_action_window" not in r for r in result.reasons))

    def test_no_configured_windows_never_rejects_on_this_gate(self):
        cfg = base_config(corporate_action_windows=())
        obs = make_obs(timestamp=WEEKEND_TS, symbol="RNVDA")
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(all("corporate_action_window" not in r for r in result.reasons))


class TestFailClosedOverall(unittest.TestCase):
    def test_all_gates_pass_together_for_a_clean_observation(self):
        obs = make_obs(timestamp=WEEKEND_TS)
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, ())

    def test_multiple_simultaneous_failures_all_reported(self):
        obs = make_obs(
            timestamp=et(2026, 9, 15, 12, 0),  # RTH -> session failure
            session=SessionType.RTH,
            spread_bps=1000.0,  # spread failure
            available_depth=1.0,  # depth failure
        )
        cfg = base_config()
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=et(2026, 9, 15, 12, 0))
        self.assertFalse(result.passed)
        self.assertIn("session_not_tradeable:RTH", result.reasons)
        self.assertIn("spread_too_wide", result.reasons)
        self.assertIn("depth_below_minimum", result.reasons)

    def test_default_config_construction_fails_closed_for_unconfigured_session(self):
        # MarkQualityConfig() with no explicit staleness config has an
        # empty max_age_by_session by default - staleness must fail
        # closed, not silently allow every session through.
        cfg = MarkQualityConfig()
        obs = make_obs(timestamp=WEEKEND_TS)
        result = evaluate_mark_quality(obs, config=cfg, reference_timestamp=WEEKEND_TS)
        self.assertFalse(result.passed)
        # depth also fails closed by default (default_min_depth=None)
        self.assertIn("depth_minimum_not_configured", result.reasons)
        self.assertIn("reference_staleness_threshold_not_configured_for_session", result.reasons)


if __name__ == "__main__":
    unittest.main(verbosity=2)
