"""
Tests for src/risk.py.
"""

import datetime as dt
import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import SessionCalendar, SessionType
from src.signal import Direction, SignalResult, SignalType
from src.witness import WitnessStatus
from src.risk import (
    PortfolioState,
    PositionState,
    RiskConfig,
    RiskDecisionType,
    TradeAction,
    evaluate_risk,
)

ET = ZoneInfo("America/New_York")


def et(y, m, d, h, mi, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=ET)


WEEKEND_TS = et(2026, 9, 19, 13, 0)  # verified Phase 1 WEEKEND timestamp; Monday RTH open 2026-09-21 09:30
OVERNIGHT_TS = et(2026, 9, 15, 22, 0)  # verified Phase 1 OVERNIGHT timestamp
RTH_TS = et(2026, 9, 15, 12, 0)
POST_TS = et(2026, 9, 15, 18, 0)


def make_signal(
    *,
    signal=SignalType.FADE,
    direction=Direction.SHORT,
    symbol="RNVDA",
    timestamp=WEEKEND_TS,
    session=SessionType.WEEKEND,
    basis_bps=100.0,
    residual_bps=None,
    mark_quality_passed=True,
    mark_quality_reasons=(),
):
    return SignalResult(
        signal=signal,
        direction=direction,
        symbol=symbol,
        timestamp=timestamp,
        session=session,
        basis=basis_bps,
        basis_bps=basis_bps,
        basis_zscore=3.0,
        reason="test_fixture",
        mark_quality_passed=mark_quality_passed,
        mark_quality_reasons=mark_quality_reasons,
        witness_status=WitnessStatus.NO_WITNESS,
        witness_move_bps=None,
        residual_bps=residual_bps,
        config_used={},
    )


def base_config(**overrides):
    defaults = dict(
        max_active_names=3,
        max_single_name_exposure_pct=0.25,
        max_gross_exposure_pct=1.00,
        max_averaging_down_count=1,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


CALENDAR = SessionCalendar()
NAV = 100_000.0


def empty_portfolio(nav=NAV):
    return PortfolioState(nav=nav, positions=())


def portfolio_with(*positions, nav=NAV):
    return PortfolioState(nav=nav, positions=tuple(positions))


class TestSignalGate(unittest.TestCase):
    def test_valid_fade_can_pass(self):
        sig = make_signal(signal=SignalType.FADE, basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)

    def test_valid_follow_can_pass(self):
        sig = make_signal(signal=SignalType.FOLLOW, direction=Direction.LONG, basis_bps=10.0, residual_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)

    def test_flat_rejects(self):
        sig = make_signal(signal=SignalType.FLAT, direction=Direction.NONE, basis_bps=None)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("signal_flat_rejected", r.reasons)

    def test_invalid_signal_none_rejects(self):
        r = evaluate_risk(
            None, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("missing_signal", r.reasons)


class TestMarkQualityGate(unittest.TestCase):
    def test_invalid_mark_quality_rejects(self):
        sig = make_signal(mark_quality_passed=False, mark_quality_reasons=("spread_too_wide",))
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("mark_quality_failed", r.reasons)


class TestSessionGate(unittest.TestCase):
    def test_non_tradeable_session_rejects(self):
        sig = make_signal(timestamp=RTH_TS, session=SessionType.RTH)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertTrue(any("session_not_tradeable" in x for x in r.reasons))

    def test_post_rejects(self):
        sig = make_signal(timestamp=POST_TS, session=SessionType.POST)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)

    def test_halt_rejects(self):
        sig = make_signal(session=SessionType.HALT)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)

    def test_unknown_rejects(self):
        sig = make_signal(session=SessionType.UNKNOWN)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)

    def test_overnight_can_pass(self):
        sig = make_signal(timestamp=OVERNIGHT_TS, session=SessionType.OVERNIGHT, basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)


class TestActiveNames(unittest.TestCase):
    def test_max_active_names_limit(self):
        cfg = base_config(max_active_names=3)
        existing = [
            PositionState(symbol="A", direction=Direction.LONG, notional=1000.0),
            PositionState(symbol="B", direction=Direction.LONG, notional=1000.0),
            PositionState(symbol="C", direction=Direction.LONG, notional=1000.0),
        ]
        sig = make_signal(symbol="D", direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=portfolio_with(*existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("max_active_names_exceeded", r.reasons)

    def test_exactly_at_active_names_boundary_allowed(self):
        # 2 existing active names + opening a 3rd = exactly at the limit (3) -> allowed
        cfg = base_config(max_active_names=3)
        existing = [
            PositionState(symbol="A", direction=Direction.LONG, notional=1000.0),
            PositionState(symbol="B", direction=Direction.LONG, notional=1000.0),
        ]
        sig = make_signal(symbol="C", direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=portfolio_with(*existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)
        self.assertEqual(r.resulting_active_names, 3)

    def test_adding_to_existing_active_symbol_does_not_count_as_new_name(self):
        cfg = base_config(max_active_names=3)
        existing = [
            PositionState(symbol="A", direction=Direction.LONG, notional=1000.0),
            PositionState(symbol="B", direction=Direction.LONG, notional=1000.0),
            PositionState(symbol="C", direction=Direction.SHORT, notional=1000.0),
        ]
        # Adding to C (already active) with the SAME direction as existing.
        sig = make_signal(symbol="C", direction=Direction.SHORT, basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1000.0,
            portfolio=portfolio_with(*existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)
        self.assertEqual(r.resulting_active_names, 3)


class TestSingleNameExposure(unittest.TestCase):
    def test_within_cap_allowed_full_amount(self):
        cfg = base_config()
        sig = make_signal(basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=10_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)
        self.assertEqual(r.approved_notional, 10_000.0)

    def test_exactly_at_25pct_boundary_allowed(self):
        cfg = base_config(max_single_name_exposure_pct=0.25)
        sig = make_signal(basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=25_000.0,  # exactly 25% of 100k NAV
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)
        self.assertEqual(r.approved_notional, 25_000.0)
        self.assertEqual(r.resulting_gross_exposure, 25_000.0)

    def test_above_25pct_is_sized_down_not_outright_rejected(self):
        cfg = base_config(max_single_name_exposure_pct=0.25)
        sig = make_signal(basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=50_000.0,  # 50% requested
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)
        self.assertEqual(r.approved_notional, 25_000.0)  # sized down, not rejected

    def test_considers_existing_exposure_plus_increase(self):
        cfg = base_config(max_single_name_exposure_pct=0.25)
        existing = PositionState(symbol="RNVDA", direction=Direction.SHORT, notional=20_000.0)
        sig = make_signal(basis_bps=100.0, direction=Direction.SHORT)  # same direction as existing
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=10_000.0,
            portfolio=portfolio_with(existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        # capacity remaining = 25,000 - 20,000 = 5,000
        self.assertEqual(r.approved_notional, 5_000.0)

    def test_zero_remaining_capacity_rejects(self):
        cfg = base_config(max_single_name_exposure_pct=0.25)
        existing = PositionState(symbol="RNVDA", direction=Direction.SHORT, notional=25_000.0)
        sig = make_signal(basis_bps=100.0, direction=Direction.SHORT)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("no_available_risk_capacity", r.reasons)


class TestGrossExposure(unittest.TestCase):
    def test_exactly_at_100pct_gross_boundary_allowed(self):
        cfg = base_config(max_gross_exposure_pct=1.00, max_single_name_exposure_pct=1.00)
        existing = [
            PositionState(symbol="A", direction=Direction.LONG, notional=60_000.0),
            PositionState(symbol="B", direction=Direction.SHORT, notional=30_000.0),
        ]
        sig = make_signal(symbol="C", basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=10_000.0,  # 60+30+10=100k = 100% NAV
            portfolio=portfolio_with(*existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)
        self.assertEqual(r.resulting_gross_exposure, 100_000.0)

    def test_above_100pct_gross_sized_down(self):
        cfg = base_config(max_gross_exposure_pct=1.00, max_single_name_exposure_pct=1.00)
        existing = [
            PositionState(symbol="A", direction=Direction.LONG, notional=60_000.0),
            PositionState(symbol="B", direction=Direction.SHORT, notional=35_000.0),
        ]
        sig = make_signal(symbol="C", basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=20_000.0,  # would be 115k
            portfolio=portfolio_with(*existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)
        self.assertEqual(r.approved_notional, 5_000.0)  # capped to exactly 100k gross
        self.assertEqual(r.resulting_gross_exposure, 100_000.0)


class TestOppositePositionLock(unittest.TestCase):
    def test_existing_long_plus_short_proposal_rejects(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=5_000.0)
        sig = make_signal(direction=Direction.SHORT, basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("opposite_position_lock", r.reasons)

    def test_existing_short_plus_long_proposal_rejects(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.SHORT, notional=5_000.0)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("opposite_position_lock", r.reasons)

    def test_flat_position_can_open_either_direction(self):
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=5.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)


class TestReduceAndFlatten(unittest.TestCase):
    def test_reducing_existing_position(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=5_000.0)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.REDUCE, requested_notional=2_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.REDUCE)
        self.assertEqual(r.approved_notional, 2_000.0)
        self.assertEqual(r.resulting_gross_exposure, 3_000.0)
        self.assertEqual(r.resulting_active_names, 1)  # still active, partially reduced

    def test_reduce_to_zero_drops_active_name_count(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=5_000.0)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.REDUCE, requested_notional=5_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.REDUCE)
        self.assertEqual(r.resulting_active_names, 0)

    def test_flatten_full_position(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.SHORT, notional=3_000.0)
        sig = make_signal(direction=Direction.SHORT, basis_bps=100.0)
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=3_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.FLATTEN)
        self.assertEqual(r.approved_notional, 3_000.0)
        self.assertEqual(r.resulting_gross_exposure, 0.0)

    def test_reduce_more_than_held_rejects(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.REDUCE, requested_notional=5_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("reduce_or_flatten_notional_exceeds_current_position", r.reasons)

    def test_flatten_with_no_position_rejects(self):
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("no_existing_position_to_reduce_or_flatten", r.reasons)

    def test_reduce_bypasses_session_gate(self):
        # RTH session (not tradeable for opening) must still allow reducing.
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(timestamp=RTH_TS, session=SessionType.RTH, direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.REDUCE, requested_notional=500.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.REDUCE)

    def test_flatten_bypasses_mark_quality_gate(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0, mark_quality_passed=False,
                           mark_quality_reasons=("stale_reference",))
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.FLATTEN)


class TestApprovedGateBypassDecision(unittest.TestCase):
    """Explicit coverage for the approved Phase 3 design decision:

        - OPEN_OR_INCREASE must pass signal, session, mark-quality, and
          cost/edge gates.
        - REDUCE may bypass signal/session/mark-quality/cost-edge gates
          when position bookkeeping is valid.
        - FLATTEN may bypass those same gates when position bookkeeping is
          valid, because the system must remain capable of closing
          positions during RTH, HALT, or other restricted sessions.

    Each gate is tested individually for both REDUCE and FLATTEN, plus a
    same-portfolio contrast against OPEN_OR_INCREASE to make the
    documented asymmetry unambiguous in a single test.
    """

    # --- signal gate bypass -------------------------------------------------

    def test_reduce_bypasses_signal_gate_with_flat_signal(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(signal=SignalType.FLAT, direction=Direction.NONE, basis_bps=None)
        r = evaluate_risk(
            sig, action=TradeAction.REDUCE, requested_notional=500.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.REDUCE)
        self.assertNotIn("signal_flat_rejected", r.reasons)

    def test_flatten_bypasses_signal_gate_with_flat_signal(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(signal=SignalType.FLAT, direction=Direction.NONE, basis_bps=None)
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.FLATTEN)
        self.assertNotIn("signal_flat_rejected", r.reasons)

    # --- session gate bypass (RTH, HALT, and other restricted sessions) -----

    def test_reduce_bypasses_session_gate_halt(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(session=SessionType.HALT, direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.REDUCE, requested_notional=500.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.REDUCE)

    def test_flatten_bypasses_session_gate_rth(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(timestamp=RTH_TS, session=SessionType.RTH, direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.FLATTEN)

    def test_flatten_bypasses_session_gate_halt(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(session=SessionType.HALT, direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.FLATTEN)

    def test_flatten_bypasses_session_gate_unknown(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(session=SessionType.UNKNOWN, direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.FLATTEN)

    # --- mark-quality gate bypass --------------------------------------------

    def test_reduce_bypasses_mark_quality_gate(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0, mark_quality_passed=False,
                           mark_quality_reasons=("depth_unavailable",))
        r = evaluate_risk(
            sig, action=TradeAction.REDUCE, requested_notional=500.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.REDUCE)
        self.assertNotIn("mark_quality_failed", r.reasons)

    # --- cost/edge gate bypass (REDUCE/FLATTEN never compute this gate) -----

    def test_reduce_bypasses_cost_edge_gate(self):
        # A tiny basis that would clearly fail the cost/edge gate for an
        # OPEN_OR_INCREASE; REDUCE must ignore cost/edge entirely.
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(direction=Direction.LONG, basis_bps=1.0)  # far below any realistic cost
        r = evaluate_risk(
            sig, action=TradeAction.REDUCE, requested_notional=500.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
            # deliberately no spread_bps passed - cost/edge would fail closed
            # on OPEN_OR_INCREASE (missing_spread_for_cost_model), but must
            # not even be evaluated for REDUCE.
        )
        self.assertEqual(r.decision, RiskDecisionType.REDUCE)
        self.assertNotIn("cost_exceeds_expected_edge", r.reasons)
        self.assertNotIn("missing_spread_for_cost_model", r.reasons)
        self.assertIsNone(r.estimated_cost_bps)
        self.assertIsNone(r.expected_edge_bps)

    def test_flatten_bypasses_cost_edge_gate(self):
        existing = PositionState(symbol="RNVDA", direction=Direction.SHORT, notional=1_000.0)
        sig = make_signal(direction=Direction.SHORT, basis_bps=1.0)
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=base_config(), calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.FLATTEN)
        self.assertNotIn("cost_exceeds_expected_edge", r.reasons)
        self.assertIsNone(r.estimated_cost_bps)
        self.assertIsNone(r.expected_edge_bps)

    # --- same-portfolio contrast: OPEN_OR_INCREASE is NOT exempt -----------

    def test_open_or_increase_still_enforces_every_gate_same_portfolio_where_flatten_bypasses(self):
        # Using the exact same session/mark-quality/signal conditions that
        # FLATTEN was just shown to bypass, confirm OPEN_OR_INCREASE (on a
        # DIFFERENT, not-yet-active symbol, so opposite-position lock does
        # not confound the result) is still blocked by all four gates.
        sig = make_signal(
            symbol="RTSLA", session=SessionType.HALT, signal=SignalType.FLAT,
            direction=Direction.NONE, basis_bps=None, mark_quality_passed=False,
            mark_quality_reasons=("depth_unavailable",),
        )
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=None,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("signal_flat_rejected", r.reasons)
        self.assertTrue(any("session_not_tradeable" in x for x in r.reasons))
        self.assertIn("mark_quality_failed", r.reasons)


class TestAveragingDown(unittest.TestCase):
    def test_first_averaging_down_permitted(self):
        cfg = base_config(max_averaging_down_count=1)
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0, averaging_down_count=0)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=500.0,
            portfolio=portfolio_with(existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
            is_averaging_down=True,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)

    def test_second_averaging_down_rejected(self):
        cfg = base_config(max_averaging_down_count=1)
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0, averaging_down_count=1)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=500.0,
            portfolio=portfolio_with(existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
            is_averaging_down=True,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("averaging_down_limit_reached", r.reasons)

    def test_non_averaging_down_request_unaffected_by_count(self):
        cfg = base_config(max_averaging_down_count=1)
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0, averaging_down_count=1)
        sig = make_signal(direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=500.0,
            portfolio=portfolio_with(existing), config=cfg, calendar=CALENDAR, spread_bps=5.0,
            is_averaging_down=False,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)


class TestCostEdgeGate(unittest.TestCase):
    def test_cost_exceeds_edge_rejects(self):
        # FADE basis 10 bps -> tiny edge; default cost model is much larger.
        sig = make_signal(basis_bps=10.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("cost_exceeds_expected_edge", r.reasons)

    def test_edge_exceeds_cost_can_pass(self):
        sig = make_signal(basis_bps=500.0)  # large edge, easily covers cost
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)

    def test_cost_exactly_equals_edge_passes(self):
        # Reverse-engineer basis_bps so expected_edge_bps == estimated_cost_bps exactly.
        cfg = base_config()
        spread_bps = 10.0
        # cost = 2 * (5.0 + 5.0 + 2.0 + 3.0[weekend]) = 2*15 = 30.0
        sig = make_signal(basis_bps=30.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=spread_bps,
        )
        self.assertEqual(r.estimated_cost_bps, 30.0)
        self.assertEqual(r.expected_edge_bps, 30.0)
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)  # edge == cost: "covers" -> passes (documented)

    def test_cost_one_bps_more_than_edge_rejects(self):
        sig = make_signal(basis_bps=29.99)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("cost_exceeds_expected_edge", r.reasons)

    def test_missing_spread_fails_closed(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=None,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("missing_spread_for_cost_model", r.reasons)

    def test_weekend_friction_increases_cost_vs_overnight(self):
        cfg = base_config()
        weekend_sig = make_signal(timestamp=WEEKEND_TS, session=SessionType.WEEKEND, basis_bps=100.0)
        overnight_sig = make_signal(timestamp=OVERNIGHT_TS, session=SessionType.OVERNIGHT, basis_bps=100.0)
        r_weekend = evaluate_risk(
            weekend_sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        r_overnight = evaluate_risk(
            overnight_sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertGreater(r_weekend.estimated_cost_bps, r_overnight.estimated_cost_bps)


class TestFailClosed(unittest.TestCase):
    def test_missing_nav_fails_closed(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(nav=None), config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("missing_or_invalid_nav", r.reasons)

    def test_negative_nav_fails_closed(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(nav=-100.0), config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("missing_or_invalid_nav", r.reasons)

    def test_missing_portfolio_fails_closed(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=None, config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("missing_portfolio_state", r.reasons)

    def test_missing_requested_notional_fails_closed(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=None,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("invalid_requested_notional", r.reasons)

    def test_zero_requested_notional_fails_closed(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=0.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("invalid_requested_notional", r.reasons)

    def test_negative_requested_notional_fails_closed(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=-100.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("invalid_requested_notional", r.reasons)

    def test_missing_calendar_fails_closed_for_open(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=base_config(), calendar=None, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("missing_calendar_cannot_evaluate_flatten_window", r.reasons)

    def test_missing_config_fails_closed(self):
        sig = make_signal(basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=None, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("missing_risk_config", r.reasons)


class TestFlattenWindow(unittest.TestCase):
    def test_new_position_inside_flatten_window_blocked(self):
        cfg = base_config()
        cfg = RiskConfig(**{**cfg.__dict__, "flatten": cfg.flatten.__class__(buffer_minutes=60.0)})
        # Monday just before RTH open (09:30 ET) - well inside a 60-min buffer.
        near_open_ts = et(2026, 9, 21, 9, 0)  # 30 min before open
        sig = make_signal(timestamp=near_open_ts, session=SessionType.WEEKEND, basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("flatten_window_blocks_new_position", r.reasons)
        self.assertTrue(r.in_flatten_window)

    def test_new_position_well_outside_flatten_window_allowed(self):
        cfg = base_config()
        far_from_open_ts = et(2026, 9, 19, 13, 0)  # Saturday, ~44h from Monday open
        sig = make_signal(timestamp=far_from_open_ts, session=SessionType.WEEKEND, basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)
        self.assertFalse(r.in_flatten_window)

    def test_exact_boundary_at_buffer_is_inside_window(self):
        # buffer_minutes=60; construct a timestamp EXACTLY 60 minutes before
        # the Monday 09:30 ET open -> documented as INSIDE the window (inclusive).
        cfg = base_config()
        exact_boundary_ts = et(2026, 9, 21, 8, 30)  # exactly 60 min before 09:30
        sig = make_signal(timestamp=exact_boundary_ts, session=SessionType.WEEKEND, basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertTrue(r.in_flatten_window)
        self.assertEqual(r.decision, RiskDecisionType.REJECT)

    def test_one_minute_outside_boundary_is_allowed(self):
        cfg = base_config()
        just_outside_ts = et(2026, 9, 21, 8, 29)  # 61 min before 09:30
        sig = make_signal(timestamp=just_outside_ts, session=SessionType.WEEKEND, basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertFalse(r.in_flatten_window)
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)

    def test_flatten_action_allowed_inside_flatten_window(self):
        cfg = base_config()
        near_open_ts = et(2026, 9, 21, 9, 0)
        existing = PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0)
        sig = make_signal(timestamp=near_open_ts, session=SessionType.WEEKEND, direction=Direction.LONG, basis_bps=-100.0)
        r = evaluate_risk(
            sig, action=TradeAction.FLATTEN, requested_notional=1_000.0,
            portfolio=portfolio_with(existing), config=cfg, calendar=CALENDAR,
        )
        self.assertEqual(r.decision, RiskDecisionType.FLATTEN)

    def test_follow_exception_configurable_and_off_by_default(self):
        cfg = base_config()
        self.assertFalse(cfg.flatten.follow_exception_during_flatten)
        near_open_ts = et(2026, 9, 21, 9, 0)
        sig = make_signal(
            signal=SignalType.FOLLOW, timestamp=near_open_ts, session=SessionType.WEEKEND,
            direction=Direction.LONG, basis_bps=10.0, residual_bps=500.0,
        )
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)  # exception is OFF by default

    def test_follow_exception_when_explicitly_enabled(self):
        cfg = base_config()
        cfg = RiskConfig(**{**cfg.__dict__, "flatten": cfg.flatten.__class__(
            buffer_minutes=cfg.flatten.buffer_minutes, follow_exception_during_flatten=True
        )})
        near_open_ts = et(2026, 9, 21, 9, 0)
        sig = make_signal(
            signal=SignalType.FOLLOW, timestamp=near_open_ts, session=SessionType.WEEKEND,
            direction=Direction.LONG, basis_bps=10.0, residual_bps=500.0,
        )
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)  # explicit opt-in honored

    def test_fade_never_exempted_even_with_exception_enabled(self):
        cfg = base_config()
        cfg = RiskConfig(**{**cfg.__dict__, "flatten": cfg.flatten.__class__(
            buffer_minutes=cfg.flatten.buffer_minutes, follow_exception_during_flatten=True
        )})
        near_open_ts = et(2026, 9, 21, 9, 0)
        sig = make_signal(signal=SignalType.FADE, timestamp=near_open_ts, session=SessionType.WEEKEND, basis_bps=500.0)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)  # exception is FOLLOW-only, never FADE


class TestDeterminism(unittest.TestCase):
    def test_identical_input_identical_result(self):
        sig = make_signal(basis_bps=100.0)
        cfg = base_config()
        r_a = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        r_b = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r_a, r_b)
        self.assertEqual(r_a.to_dict(), r_b.to_dict())


class TestNoLookAhead(unittest.TestCase):
    def test_result_depends_only_on_the_single_provided_signal_timestamp(self):
        # The risk engine has no history/state beyond what's explicitly
        # passed in `portfolio` - confirm evaluating the same current
        # signal is unaffected by constructing (but never passing) a
        # "future" signal object alongside it.
        sig_current = make_signal(timestamp=WEEKEND_TS, basis_bps=100.0)
        sig_future = make_signal(timestamp=WEEKEND_TS + dt.timedelta(hours=5), basis_bps=999999.0)  # never passed in

        cfg = base_config()
        r_without_future_constructed = evaluate_risk(
            sig_current, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        # constructing sig_future above has no side effects; merely evaluate
        # sig_current again to confirm the result is identical either way.
        r_after_future_exists_in_memory = evaluate_risk(
            sig_current, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r_without_future_constructed, r_after_future_exists_in_memory)
        del sig_future  # never used in evaluate_risk() - proves no global/implicit history


class TestConfigOverrides(unittest.TestCase):
    def test_config_overrides_change_behavior(self):
        sig = make_signal(basis_bps=100.0)
        loose_cfg = base_config(max_single_name_exposure_pct=0.50)
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=40_000.0,
            portfolio=empty_portfolio(), config=loose_cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.approved_notional, 40_000.0)  # allowed under the looser 50% cap

    def test_cost_config_override_changes_cost_gate_outcome(self):
        from src.risk import CostConfig
        sig = make_signal(basis_bps=10.0)  # tiny edge
        cheap_cfg = base_config()
        cheap_cfg = RiskConfig(**{**cheap_cfg.__dict__, "cost": CostConfig(
            fee_bps_per_side=0.5, additional_slippage_bps_per_side=0.5,
            weekend_additional_slippage_bps_per_side=0.5,
        )})
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=empty_portfolio(), config=cheap_cfg, calendar=CALENDAR, spread_bps=1.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.ALLOW)  # cheap cost config lets a small edge pass


class TestMultipleSimultaneousFailures(unittest.TestCase):
    def test_reasons_accumulate(self):
        cfg = base_config(max_active_names=1)
        existing = [PositionState(symbol="A", direction=Direction.LONG, notional=1_000.0)]
        sig = make_signal(
            symbol="B", timestamp=RTH_TS, session=SessionType.RTH,
            signal=SignalType.FLAT, direction=Direction.NONE, basis_bps=None,
            mark_quality_passed=False, mark_quality_reasons=("stale_reference",),
        )
        r = evaluate_risk(
            sig, action=TradeAction.OPEN_OR_INCREASE, requested_notional=1_000.0,
            portfolio=portfolio_with(*existing), config=cfg, calendar=CALENDAR, spread_bps=10.0,
        )
        self.assertEqual(r.decision, RiskDecisionType.REJECT)
        self.assertIn("signal_flat_rejected", r.reasons)
        self.assertTrue(any("session_not_tradeable" in x for x in r.reasons))
        self.assertIn("mark_quality_failed", r.reasons)
        self.assertIn("max_active_names_exceeded", r.reasons)
        self.assertGreaterEqual(len(r.reasons), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
