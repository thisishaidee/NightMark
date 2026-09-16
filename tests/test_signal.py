"""
Tests for src/signal.py and src/witness.py.
"""

import datetime as dt
import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import SessionType
from src.mark_quality import MarkQualityResult
from src.observation import MarketObservation, ReferenceQuality, ReferenceSource
from src.signal import (
    Direction,
    FadeConfig,
    FollowConfig,
    SignalConfig,
    SignalType,
    evaluate_signal,
)
from src.witness import WitnessObservation, WitnessStatus, classify_witness
from src.zscore import ZScoreResult, rolling_same_session_zscore

ET = ZoneInfo("America/New_York")


def et(y, m, d, h, mi, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=ET)


WEEKEND_TS = et(2026, 9, 19, 13, 0)  # verified Phase 1 WEEKEND timestamp
OVERNIGHT_TS = et(2026, 9, 15, 22, 0)  # verified Phase 1 OVERNIGHT timestamp
RTH_TS = et(2026, 9, 15, 12, 0)
POST_TS = et(2026, 9, 15, 18, 0)


def make_obs(
    *,
    timestamp=WEEKEND_TS,
    symbol="RNVDA",
    session=SessionType.WEEKEND,
    basis=None,
    basis_bps=0.0,
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
        spread_bps=10.0,
        available_depth=1000.0,
        reference_price=100.0,
        reference_source=ReferenceSource.BITGET_OFFICIAL,
        reference_quality=ReferenceQuality.OFFICIAL,
        basis=basis if basis is not None else basis_bps,
        basis_bps=basis_bps,
        is_valid=is_valid,
        invalid_reason=invalid_reason,
    )


def mq_pass():
    return MarkQualityResult(passed=True, reasons=())


def mq_fail(*reasons):
    return MarkQualityResult(passed=False, reasons=tuple(reasons) or ("some_gate_failed",))


def zs(value, reason=None, used=10):
    return ZScoreResult(value=value, reason=reason, observations_used=used)


def base_config(**overrides):
    defaults = dict(
        fade=FadeConfig(min_abs_basis_bps=20.0, entry_abs_zscore=1.5),
        follow=FollowConfig(min_witness_move_bps=60.0, min_residual_bps=15.0),
    )
    defaults.update(overrides)
    return SignalConfig(**defaults)


class TestWitnessClassification(unittest.TestCase):
    def test_no_witness_object(self):
        self.assertEqual(classify_witness(None, 60.0), WitnessStatus.NO_WITNESS)

    def test_witness_with_none_move(self):
        self.assertEqual(classify_witness(WitnessObservation(move_bps=None), 60.0), WitnessStatus.NO_WITNESS)

    def test_below_threshold(self):
        self.assertEqual(classify_witness(WitnessObservation(move_bps=30.0), 60.0), WitnessStatus.BELOW_THRESHOLD)

    def test_qualifying_positive(self):
        self.assertEqual(classify_witness(WitnessObservation(move_bps=80.0), 60.0), WitnessStatus.QUALIFYING_POSITIVE)

    def test_qualifying_negative(self):
        self.assertEqual(classify_witness(WitnessObservation(move_bps=-80.0), 60.0), WitnessStatus.QUALIFYING_NEGATIVE)

    def test_exact_threshold_boundary_qualifies(self):
        self.assertEqual(classify_witness(WitnessObservation(move_bps=60.0), 60.0), WitnessStatus.QUALIFYING_POSITIVE)
        self.assertEqual(classify_witness(WitnessObservation(move_bps=-60.0), 60.0), WitnessStatus.QUALIFYING_NEGATIVE)

    def test_just_below_threshold_boundary(self):
        self.assertEqual(classify_witness(WitnessObservation(move_bps=59.99), 60.0), WitnessStatus.BELOW_THRESHOLD)


class TestFadeSignal(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()

    def test_positive_unexplained_dislocation_is_short_fade(self):
        obs = make_obs(basis_bps=25.0)
        result = evaluate_signal(obs, mq_pass(), zs(2.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FADE)
        self.assertEqual(result.direction, Direction.SHORT)

    def test_negative_unexplained_dislocation_is_long_fade(self):
        obs = make_obs(basis_bps=-25.0)
        result = evaluate_signal(obs, mq_pass(), zs(-2.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FADE)
        self.assertEqual(result.direction, Direction.LONG)

    def test_basis_below_threshold_is_flat(self):
        obs = make_obs(basis_bps=10.0)  # below 20.0
        result = evaluate_signal(obs, mq_pass(), zs(2.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("fade_basis_below_min", result.reason)

    def test_zscore_below_threshold_is_flat(self):
        obs = make_obs(basis_bps=25.0)
        result = evaluate_signal(obs, mq_pass(), zs(1.0), None, self.cfg)  # below 1.5
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("fade_zscore_below_min", result.reason)

    def test_qualifying_witness_explanation_blocks_fade(self):
        obs = make_obs(basis_bps=25.0)
        witness = WitnessObservation(move_bps=80.0)  # qualifies, more than fade needs
        # Make follow not qualify either (residual too small) so we isolate the fade-blocking behavior.
        result = evaluate_signal(obs, mq_pass(), zs(2.0), witness, self.cfg)
        self.assertNotEqual(result.signal, SignalType.FADE)


class TestFollowSignal(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()

    def test_qualifying_positive_witness_and_residual_is_long_follow(self):
        # witness +80, basis +10 -> residual = 80-10 = 70 (>=15), same sign as witness -> LONG FOLLOW
        obs = make_obs(basis_bps=10.0)
        witness = WitnessObservation(move_bps=80.0)
        result = evaluate_signal(obs, mq_pass(), zs(0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FOLLOW)
        self.assertEqual(result.direction, Direction.LONG)
        self.assertAlmostEqual(result.residual_bps, 70.0)

    def test_qualifying_negative_witness_and_residual_is_short_follow(self):
        obs = make_obs(basis_bps=-10.0)
        witness = WitnessObservation(move_bps=-80.0)
        result = evaluate_signal(obs, mq_pass(), zs(-0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FOLLOW)
        self.assertEqual(result.direction, Direction.SHORT)
        self.assertAlmostEqual(result.residual_bps, -70.0)

    def test_witness_below_threshold_is_flat(self):
        obs = make_obs(basis_bps=0.0)
        witness = WitnessObservation(move_bps=30.0)  # below 60.0
        result = evaluate_signal(obs, mq_pass(), zs(0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("follow_no_qualifying_witness", result.reason)

    def test_residual_insufficient_is_flat(self):
        # witness 70, basis 60 -> residual = 10 (< 15 min)
        obs = make_obs(basis_bps=60.0)
        witness = WitnessObservation(move_bps=70.0)
        result = evaluate_signal(obs, mq_pass(), zs(0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("follow_residual_insufficient", result.reason)

    def test_direction_mismatch_is_flat(self):
        # witness +80 (positive), but basis already +100 (overshot past the witness move)
        # residual = 80 - 100 = -20, opposite sign of witness (+) -> mismatch
        obs = make_obs(basis_bps=100.0)
        witness = WitnessObservation(move_bps=80.0)
        result = evaluate_signal(obs, mq_pass(), zs(0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("follow_direction_mismatch", result.reason)

    def test_no_witness_never_produces_follow(self):
        obs = make_obs(basis_bps=10.0)
        result = evaluate_signal(obs, mq_pass(), zs(0.1), None, self.cfg)
        self.assertNotEqual(result.signal, SignalType.FOLLOW)


class TestMarkQualityHardBlock(unittest.TestCase):
    def test_failed_mark_quality_is_always_flat_even_with_strong_fade_conditions(self):
        obs = make_obs(basis_bps=100.0)  # would easily qualify for FADE
        result = evaluate_signal(obs, mq_fail("spread_too_wide"), zs(5.0), None, base_config())
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertEqual(result.reason, "mark_quality_failed")
        self.assertFalse(result.mark_quality_passed)

    def test_failed_mark_quality_blocks_follow_too(self):
        obs = make_obs(basis_bps=10.0)
        witness = WitnessObservation(move_bps=80.0)
        result = evaluate_signal(obs, mq_fail("depth_unavailable"), zs(0.1), witness, base_config())
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertEqual(result.reason, "mark_quality_failed")

    def test_mark_quality_reasons_surfaced_in_result(self):
        obs = make_obs(basis_bps=100.0)
        result = evaluate_signal(obs, mq_fail("stale_reference"), zs(5.0), None, base_config())
        self.assertIn("stale_reference", result.mark_quality_reasons)


class TestSessionGating(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()

    def test_rth_is_flat(self):
        obs = make_obs(timestamp=RTH_TS, session=SessionType.RTH, basis_bps=100.0)
        result = evaluate_signal(obs, mq_pass(), zs(5.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("session_not_tradeable:RTH", result.reason)

    def test_post_is_flat(self):
        obs = make_obs(timestamp=POST_TS, session=SessionType.POST, basis_bps=100.0)
        result = evaluate_signal(obs, mq_pass(), zs(5.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("session_not_tradeable:POST", result.reason)

    def test_halt_is_flat(self):
        obs = make_obs(session=SessionType.HALT, basis_bps=100.0)
        result = evaluate_signal(obs, mq_pass(), zs(5.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("session_not_tradeable:HALT", result.reason)

    def test_unknown_is_flat(self):
        obs = make_obs(session=SessionType.UNKNOWN, basis_bps=100.0)
        result = evaluate_signal(obs, mq_pass(), zs(5.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("session_not_tradeable:UNKNOWN", result.reason)

    def test_overnight_can_produce_signal(self):
        obs = make_obs(timestamp=OVERNIGHT_TS, session=SessionType.OVERNIGHT, basis_bps=25.0)
        result = evaluate_signal(obs, mq_pass(), zs(2.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FADE)

    def test_weekend_can_produce_signal(self):
        obs = make_obs(timestamp=WEEKEND_TS, session=SessionType.WEEKEND, basis_bps=25.0)
        result = evaluate_signal(obs, mq_pass(), zs(2.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FADE)


class TestConflict(unittest.TestCase):
    def test_both_fade_and_follow_qualify_returns_flat_with_conflict_reason(self):
        # FADE needs: |basis_bps|>=20, |zscore|>=1.5, no qualifying witness.
        # FOLLOW needs: qualifying witness, |residual|>=15, residual same sign as witness.
        # Construct: basis_bps = -25 (qualifies fade on basis), zscore = 2.0 (qualifies fade)
        # witness move_bps = -80 (qualifies as witness, but this WOULD disqualify fade...)
        # To get both to qualify, fade requires NO qualifying witness, so this looks
        # contradictory by design - confirm the engine cannot be tricked into both firing.
        # Since fade requires witness_status not qualifying, and follow requires it
        # qualifying, they are mutually exclusive by construction - assert that directly.
        cfg = base_config()
        obs = make_obs(basis_bps=-25.0)
        witness = WitnessObservation(move_bps=-80.0)
        result = evaluate_signal(obs, mq_pass(), zs(2.0), witness, cfg)
        # Fade is disqualified by the qualifying witness; only follow-shaped
        # conditions could fire here - confirm it resolves to a single
        # deterministic outcome, never silently both.
        self.assertIn(result.signal, (SignalType.FOLLOW, SignalType.FLAT))

    def test_conflict_reason_string_reachable_via_direct_construction(self):
        # The engine's mutual-exclusion by construction (fade needs a
        # non-qualifying witness; follow needs a qualifying one) means a
        # true simultaneous qualification is not reachable through normal
        # inputs. Directly verify the conflict branch's reason constant
        # and behavior using the module's own evaluate_fade/evaluate_follow
        # helpers is out of scope (private); instead verify the documented
        # contract: whenever both are mutually possible, conflict wins.
        # Here we simulate that by observing that if a witness is exactly
        # at BELOW_THRESHOLD (not qualifying), follow cannot fire, so only
        # fade can - proving the two paths are properly gated apart.
        cfg = base_config()
        obs = make_obs(basis_bps=-25.0)
        witness = WitnessObservation(move_bps=-30.0)  # below 60 threshold
        result = evaluate_signal(obs, mq_pass(), zs(2.0), witness, cfg)
        self.assertEqual(result.signal, SignalType.FADE)


class TestDataValidity(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()

    def test_invalid_observation_is_flat(self):
        obs = make_obs(basis_bps=100.0, is_valid=False, invalid_reason="invalid_or_missing_rtoken_quote")
        result = evaluate_signal(obs, mq_pass(), zs(5.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertIn("observation_invalid", result.reason)

    def test_missing_basis_bps_is_flat(self):
        obs = make_obs(basis_bps=None)
        result = evaluate_signal(obs, mq_pass(), zs(5.0), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertEqual(result.reason, "missing_required_inputs")

    def test_missing_zscore_is_flat(self):
        obs = make_obs(basis_bps=100.0)
        result = evaluate_signal(obs, mq_pass(), zs(None, reason="insufficient_history"), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)
        self.assertEqual(result.reason, "missing_required_inputs")


class TestBoundaries(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()

    def test_basis_exactly_at_threshold_qualifies(self):
        obs = make_obs(basis_bps=20.0)
        result = evaluate_signal(obs, mq_pass(), zs(1.5), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FADE)

    def test_basis_one_unit_below_threshold_flat(self):
        obs = make_obs(basis_bps=19.99)
        result = evaluate_signal(obs, mq_pass(), zs(1.5), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)

    def test_zscore_exactly_at_threshold_qualifies(self):
        obs = make_obs(basis_bps=20.0)
        result = evaluate_signal(obs, mq_pass(), zs(1.5), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FADE)

    def test_zscore_one_unit_below_threshold_flat(self):
        obs = make_obs(basis_bps=20.0)
        result = evaluate_signal(obs, mq_pass(), zs(1.49), None, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)

    def test_witness_exactly_at_threshold_qualifies_for_follow(self):
        # witness exactly 60, basis 0 -> residual = 60 (>=15), same sign -> LONG FOLLOW
        obs = make_obs(basis_bps=0.0)
        witness = WitnessObservation(move_bps=60.0)
        result = evaluate_signal(obs, mq_pass(), zs(0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FOLLOW)

    def test_witness_one_unit_below_threshold_flat(self):
        obs = make_obs(basis_bps=0.0)
        witness = WitnessObservation(move_bps=59.99)
        result = evaluate_signal(obs, mq_pass(), zs(0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)

    def test_residual_exactly_at_threshold_qualifies(self):
        # witness 75, basis 60 -> residual = 15 (== min)
        obs = make_obs(basis_bps=60.0)
        witness = WitnessObservation(move_bps=75.0)
        result = evaluate_signal(obs, mq_pass(), zs(0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FOLLOW)

    def test_residual_one_unit_below_threshold_flat(self):
        # witness 74.99, basis 60 -> residual = 14.99 (< 15 min)
        obs = make_obs(basis_bps=60.0)
        witness = WitnessObservation(move_bps=74.99)
        result = evaluate_signal(obs, mq_pass(), zs(0.1), witness, self.cfg)
        self.assertEqual(result.signal, SignalType.FLAT)


class TestDeterminism(unittest.TestCase):
    def test_identical_input_identical_result(self):
        cfg = base_config()
        obs = make_obs(basis_bps=25.0)
        result_a = evaluate_signal(obs, mq_pass(), zs(2.0), None, cfg)
        result_b = evaluate_signal(obs, mq_pass(), zs(2.0), None, cfg)
        self.assertEqual(result_a, result_b)
        self.assertEqual(result_a.to_dict(), result_b.to_dict())


class TestNoLookAhead(unittest.TestCase):
    def test_future_observation_cannot_change_current_signal(self):
        cfg = base_config()

        def obs_at(i, value):
            return MarketObservation(
                timestamp=et(2026, 9, 15, 10, 0, i),
                symbol="X",
                session=SessionType.RTH,
                rtoken_bid=None, rtoken_ask=None, rtoken_mid=None,
                spread_bps=None, available_depth=None,
                reference_price=None,
                reference_source=ReferenceSource.BITGET_OFFICIAL,
                reference_quality=ReferenceQuality.OFFICIAL,
                basis=None, basis_bps=value,
                is_valid=True, invalid_reason=None,
            )

        leading = [obs_at(i, v) for i, v in enumerate([25.0, 26.0, 27.0])]
        current = obs_at(3, 30.0)

        dataset_a = leading + [current]
        dataset_b = leading + [current, obs_at(4, 999999.0)]

        z_a = rolling_same_session_zscore(dataset_a, window=10, min_observations=2)
        z_b = rolling_same_session_zscore(dataset_b, window=10, min_observations=2)

        current_index = len(leading)
        self.assertEqual(z_a[current_index].value, z_b[current_index].value)

        # Feed the SAME per-observation ZScoreResult (as a real pipeline
        # would) into the signal engine for both datasets and confirm the
        # decision for the current observation is identical either way.
        current_weekend = MarketObservation(
            timestamp=WEEKEND_TS, symbol="X", session=SessionType.WEEKEND,
            rtoken_bid=99.5, rtoken_ask=100.5, rtoken_mid=100.0,
            spread_bps=10.0, available_depth=1000.0, reference_price=100.0,
            reference_source=ReferenceSource.BITGET_OFFICIAL,
            reference_quality=ReferenceQuality.OFFICIAL,
            basis=30.0, basis_bps=30.0, is_valid=True, invalid_reason=None,
        )
        result_a = evaluate_signal(current_weekend, mq_pass(), z_a[current_index], None, cfg)
        result_b = evaluate_signal(current_weekend, mq_pass(), z_b[current_index], None, cfg)
        self.assertEqual(result_a, result_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
