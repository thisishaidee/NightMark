"""
Tests for src/authorize.py — the Phase 3.5 OPEN_OR_INCREASE façade.

These are integration tests: they compose calendar, eligibility, mark
quality, signal, and risk through authorize_open_or_increase() and prove
that a gate cannot be skipped by going through the façade.
"""

import datetime as dt
import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.authorize import authorize_open_or_increase
from src.calendar import SessionCalendar, SessionType
from src.eligibility import SymbolEligibility, SymbolEligibilityConfig
from src.mark_quality import MarkQualityResult
from src.observation import MarketObservation, ReferenceQuality, ReferenceSource
from src.risk import PortfolioState, PositionState, RiskConfig
from src.signal import Direction, SignalConfig, SignalType
from src.zscore import ZScoreResult

ET = ZoneInfo("America/New_York")


def et(y, m, d, h, mi, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=ET)


WEEKEND_TS = et(2026, 9, 19, 13, 0)
OVERNIGHT_TS = et(2026, 9, 15, 22, 0)
RTH_TS = et(2026, 9, 15, 12, 0)
NEAR_OPEN_TS = et(2026, 9, 21, 9, 0)  # Monday, 30 min before RTH; WEEKEND session


def make_obs(
    *,
    timestamp=OVERNIGHT_TS,
    symbol="RNVDA",
    session=SessionType.OVERNIGHT,
    basis_bps=80.0,
    spread_bps=8.0,
    is_valid=True,
    invalid_reason=None,
):
    ref = 100.0
    mid = ref * (1.0 + basis_bps / 10_000.0)
    half = (spread_bps / 10_000.0) * mid / 2.0 if spread_bps is not None else 0.0
    return MarketObservation(
        timestamp=timestamp,
        symbol=symbol,
        session=session,
        rtoken_bid=mid - half if spread_bps is not None else None,
        rtoken_ask=mid + half if spread_bps is not None else None,
        rtoken_mid=mid,
        spread_bps=spread_bps,
        available_depth=1_000.0,
        reference_price=ref,
        reference_source=ReferenceSource.US_CASH_CLOSE,
        reference_quality=ReferenceQuality.OFFICIAL,
        basis=mid - ref,
        basis_bps=basis_bps,
        is_valid=is_valid,
        invalid_reason=invalid_reason,
    )


def mq_pass():
    return MarkQualityResult(passed=True, reasons=())


def mq_fail(*reasons):
    return MarkQualityResult(passed=False, reasons=tuple(reasons) or ("some_gate_failed",))


def zs(value=2.0):
    return ZScoreResult(value=value, reason=None, observations_used=8)


def empty_elig():
    return SymbolEligibility(config=SymbolEligibilityConfig())


def listed_elig(*symbols):
    return SymbolEligibility(config=SymbolEligibilityConfig(eligible_symbols=frozenset(symbols)))


CALENDAR = SessionCalendar()
SIGNAL_CFG = SignalConfig()
RISK_CFG = RiskConfig()
PORTFOLIO = PortfolioState(nav=100_000.0, positions=())


def authorize(**overrides):
    kwargs = dict(
        observation=make_obs(),
        mark_quality=mq_pass(),
        zscore=zs(),
        witness=None,
        signal_config=SIGNAL_CFG,
        requested_notional=1_000.0,
        portfolio=PORTFOLIO,
        risk_config=RISK_CFG,
        calendar=CALENDAR,
        eligibility=empty_elig(),
    )
    kwargs.update(overrides)
    return authorize_open_or_increase(**kwargs)


class TestFacadeHappyPath(unittest.TestCase):
    def test_overnight_fade_can_allow(self):
        result = authorize()
        self.assertTrue(result.allowed, result.reasons)
        self.assertEqual(result.signal.signal, SignalType.FADE)
        self.assertEqual(result.signal.direction, Direction.SHORT)
        self.assertGreater(result.approved_notional, 0.0)
        self.assertIsNotNone(result.risk)
        self.assertFalse(result.risk.authorizes_open)

    def test_weekend_eligible_symbol_can_allow(self):
        result = authorize(
            observation=make_obs(timestamp=WEEKEND_TS, session=SessionType.WEEKEND),
            eligibility=listed_elig("RNVDA"),
        )
        self.assertTrue(result.allowed, result.reasons)

    def test_deterministic(self):
        a = authorize()
        b = authorize()
        self.assertEqual(a, b)
        self.assertEqual(a.to_dict(), b.to_dict())


class TestFacadeCannotSkipMarkQuality(unittest.TestCase):
    def test_failed_mark_quality_blocks_even_with_strong_fade_inputs(self):
        result = authorize(mark_quality=mq_fail("spread_too_wide"))
        self.assertFalse(result.allowed)
        self.assertIn("mark_quality_failed", result.reasons)
        self.assertEqual(result.approved_notional, 0.0)
        self.assertEqual(result.signal.signal, SignalType.FLAT)

    def test_missing_mark_quality_fails_closed(self):
        result = authorize(mark_quality=None)
        self.assertFalse(result.allowed)
        self.assertIn("missing_mark_quality", result.reasons)
        self.assertIsNone(result.signal)


class TestFacadeCannotSkipSession(unittest.TestCase):
    def test_rth_timestamp_cannot_open(self):
        result = authorize(observation=make_obs(timestamp=RTH_TS, session=SessionType.RTH))
        self.assertFalse(result.allowed)
        self.assertTrue(
            any("session_not_tradeable:RTH" in x or "signal_not_actionable" in x for x in result.reasons),
            result.reasons,
        )

    def test_observation_claiming_overnight_during_rth_cannot_open(self):
        result = authorize(observation=make_obs(timestamp=RTH_TS, session=SessionType.OVERNIGHT))
        self.assertFalse(result.allowed)
        self.assertTrue(any("observation_session_timestamp_mismatch" in x for x in result.reasons), result.reasons)
        self.assertTrue(any("session_timestamp_mismatch" in x for x in result.reasons), result.reasons)
        self.assertEqual(result.approved_notional, 0.0)

    def test_missing_calendar_fails_closed(self):
        result = authorize(calendar=None)
        self.assertFalse(result.allowed)
        self.assertIn("missing_calendar", result.reasons)


class TestFacadeCannotSkipWeekendEligibility(unittest.TestCase):
    def test_unlisted_weekend_symbol_blocked(self):
        result = authorize(
            observation=make_obs(
                timestamp=WEEKEND_TS, session=SessionType.WEEKEND, symbol="RNOTLISTED"
            ),
            eligibility=listed_elig("RNVDA"),
        )
        self.assertFalse(result.allowed)
        self.assertIn("weekend_symbol_not_eligible", result.reasons)

    def test_empty_eligibility_fail_closed_on_weekend(self):
        result = authorize(
            observation=make_obs(timestamp=WEEKEND_TS, session=SessionType.WEEKEND),
            eligibility=empty_elig(),
        )
        self.assertFalse(result.allowed)
        self.assertIn("weekend_symbol_not_eligible", result.reasons)

    def test_default_shipped_eligibility_fail_closed_on_weekend(self):
        result = authorize(
            observation=make_obs(timestamp=WEEKEND_TS, session=SessionType.WEEKEND),
            eligibility=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("weekend_symbol_not_eligible", result.reasons)

    def test_overnight_does_not_require_weekend_list(self):
        result = authorize(eligibility=empty_elig())
        self.assertTrue(result.allowed, result.reasons)


class TestFacadeCannotSkipSignalOrRisk(unittest.TestCase):
    def test_insufficient_basis_is_flat_and_not_allowed(self):
        result = authorize(observation=make_obs(basis_bps=5.0), zscore=zs(0.1))
        self.assertFalse(result.allowed)
        self.assertEqual(result.signal.signal, SignalType.FLAT)
        self.assertTrue(any("signal_not_actionable" in x or "signal_flat_rejected" in x for x in result.reasons))

    def test_flatten_window_blocks_open(self):
        result = authorize(
            observation=make_obs(timestamp=NEAR_OPEN_TS, session=SessionType.WEEKEND),
            eligibility=listed_elig("RNVDA"),
        )
        self.assertFalse(result.allowed)
        self.assertIn("flatten_window_blocks_new_position", result.reasons)

    def test_max_active_names_blocks_open(self):
        portfolio = PortfolioState(
            nav=100_000.0,
            positions=(
                PositionState(symbol="A", direction=Direction.LONG, notional=1_000.0),
                PositionState(symbol="B", direction=Direction.LONG, notional=1_000.0),
                PositionState(symbol="C", direction=Direction.LONG, notional=1_000.0),
            ),
        )
        result = authorize(portfolio=portfolio)
        self.assertFalse(result.allowed)
        self.assertIn("max_active_names_exceeded", result.reasons)

    def test_nan_notional_blocked_through_facade(self):
        result = authorize(requested_notional=float("nan"))
        self.assertFalse(result.allowed)
        self.assertIn("non_finite_requested_notional", result.reasons)

    def test_nan_spread_on_observation_blocked_through_facade(self):
        result = authorize(observation=make_obs(spread_bps=float("nan")))
        self.assertFalse(result.allowed)
        self.assertIn("non_finite_spread", result.reasons)

    def test_missing_observation_fails_closed(self):
        result = authorize(observation=None)
        self.assertFalse(result.allowed)
        self.assertIn("missing_observation", result.reasons)

    def test_missing_zscore_fails_closed(self):
        result = authorize(zscore=None)
        self.assertFalse(result.allowed)
        self.assertIn("missing_zscore", result.reasons)

    def test_facade_does_not_accept_a_prebuilt_signal(self):
        # The public signature has no SignalResult parameter — a constructed
        # FADE cannot be injected to skip mark quality / session checks.
        import inspect
        params = inspect.signature(authorize_open_or_increase).parameters
        self.assertNotIn("signal", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
