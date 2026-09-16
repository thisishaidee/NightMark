"""
Deterministic unit tests for src/calendar.py.

Uses only the standard library (unittest) - no network calls, no live data,
no randomness, no third-party test framework dependency. Every test uses a
fixed, explicit datetime so results are reproducible.

Run with:  python3 -m unittest discover -s tests -v
"""

import datetime as dt
import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import (
    HaltWindow,
    MarketClosure,
    SessionCalendar,
    SessionConfig,
    SessionType,
    is_tradeable,
)

ET = ZoneInfo("America/New_York")


def et(y, m, d, h, mi, s=0, fold=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=ET, fold=fold)


# ---------------------------------------------------------------------------
# RTH
# ---------------------------------------------------------------------------

class TestRTH(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_mid_session(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 12, 0)), SessionType.RTH)

    def test_open_boundary_inclusive(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 9, 30, 0)), SessionType.RTH)

    def test_one_second_before_open_is_not_rth(self):
        self.assertNotEqual(self.cal.classify(et(2026, 9, 15, 9, 29, 59)), SessionType.RTH)

    def test_close_boundary_exclusive(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 16, 0, 0)), SessionType.POST)

    def test_one_second_before_close_is_rth(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 15, 59, 59)), SessionType.RTH)

    def test_weekend_day_never_rth(self):
        self.assertNotEqual(self.cal.classify(et(2026, 9, 19, 12, 0)), SessionType.RTH)


# ---------------------------------------------------------------------------
# POST
# ---------------------------------------------------------------------------

class TestPOST(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_mid_post(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 18, 0)), SessionType.POST)

    def test_open_boundary_inclusive(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 16, 0)), SessionType.POST)

    def test_close_boundary_exclusive_weekday(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 20, 0)), SessionType.OVERNIGHT)

    def test_friday_post_then_weekend_handoff(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 18, 19, 59, 59)), SessionType.POST)
        self.assertEqual(self.cal.classify(et(2026, 9, 18, 20, 0, 0)), SessionType.WEEKEND)


# ---------------------------------------------------------------------------
# OVERNIGHT
# ---------------------------------------------------------------------------

class TestOvernight(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_weeknight_post_close(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 22, 0)), SessionType.OVERNIGHT)

    def test_weeknight_premarket(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 16, 3, 0)), SessionType.OVERNIGHT)

    def test_monday_premarket_before_weekend_close_is_weekend_not_overnight(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 21, 3, 0)), SessionType.WEEKEND)

    def test_friday_early_morning_is_overnight_not_weekend(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 18, 3, 0)), SessionType.OVERNIGHT)


# ---------------------------------------------------------------------------
# WEEKEND
# ---------------------------------------------------------------------------

class TestWeekend(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_friday_night_open(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 18, 20, 0, 0)), SessionType.WEEKEND)

    def test_friday_evening_before_open_not_weekend(self):
        self.assertNotEqual(self.cal.classify(et(2026, 9, 18, 19, 59, 59)), SessionType.WEEKEND)

    def test_saturday_all_day(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 19, 0, 0, 1)), SessionType.WEEKEND)
        self.assertEqual(self.cal.classify(et(2026, 9, 19, 13, 0)), SessionType.WEEKEND)
        self.assertEqual(self.cal.classify(et(2026, 9, 19, 23, 59, 59)), SessionType.WEEKEND)

    def test_sunday_all_day(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 20, 0, 0, 1)), SessionType.WEEKEND)
        self.assertEqual(self.cal.classify(et(2026, 9, 20, 23, 59, 59)), SessionType.WEEKEND)

    def test_monday_before_open_is_weekend(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 21, 9, 29, 59)), SessionType.WEEKEND)

    def test_monday_at_open_is_rth(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 21, 9, 30, 0)), SessionType.RTH)

    def test_weekend_is_tradeable(self):
        self.assertTrue(is_tradeable(self.cal.classify(et(2026, 9, 19, 13, 0))))


# ---------------------------------------------------------------------------
# DST transitions
# ---------------------------------------------------------------------------

class TestDST(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_spring_forward_gap_is_unknown(self):
        self.assertEqual(self.cal.classify(et(2026, 3, 8, 2, 30)), SessionType.UNKNOWN)

    def test_spring_forward_just_before_gap_is_classified(self):
        result = self.cal.classify(et(2026, 3, 8, 1, 30))
        self.assertNotEqual(result, SessionType.UNKNOWN)

    def test_fall_back_ambiguous_hour_both_folds_classify(self):
        first = self.cal.classify(et(2026, 11, 1, 1, 30, fold=0))
        second = self.cal.classify(et(2026, 11, 1, 1, 30, fold=1))
        self.assertNotEqual(first, SessionType.UNKNOWN)
        self.assertNotEqual(second, SessionType.UNKNOWN)

    def test_weekend_boundary_correct_across_spring_forward(self):
        self.assertEqual(self.cal.classify(et(2026, 3, 6, 20, 0, 0)), SessionType.WEEKEND)

    def test_weekend_boundary_correct_across_fall_back(self):
        self.assertEqual(self.cal.classify(et(2026, 10, 30, 20, 0, 0)), SessionType.WEEKEND)
        self.assertEqual(self.cal.classify(et(2026, 11, 6, 20, 0, 0)), SessionType.WEEKEND)

    def test_naive_datetime_is_unknown(self):
        naive = dt.datetime(2026, 9, 15, 12, 0)
        self.assertEqual(self.cal.classify(naive), SessionType.UNKNOWN)


# ---------------------------------------------------------------------------
# HALT
# ---------------------------------------------------------------------------

class TestHalt(unittest.TestCase):
    def test_configured_halt_overrides_rth(self):
        halt = HaltWindow(start=et(2026, 9, 15, 10, 0), end=et(2026, 9, 15, 10, 15))
        cal = SessionCalendar(extra_halts=(halt,))
        self.assertEqual(cal.classify(et(2026, 9, 15, 10, 5)), SessionType.HALT)
        self.assertEqual(cal.classify(et(2026, 9, 15, 10, 15, 0)), SessionType.RTH)

    def test_configured_halt_overrides_weekend(self):
        halt = HaltWindow(start=et(2026, 9, 19, 8, 0), end=et(2026, 9, 19, 9, 0))
        cal = SessionCalendar(extra_halts=(halt,))
        self.assertEqual(cal.classify(et(2026, 9, 19, 8, 30)), SessionType.HALT)

    def test_halt_is_not_tradeable(self):
        halt = HaltWindow(start=et(2026, 9, 15, 10, 0), end=et(2026, 9, 15, 10, 15))
        cal = SessionCalendar(extra_halts=(halt,))
        self.assertFalse(is_tradeable(cal.classify(et(2026, 9, 15, 10, 5))))


# ---------------------------------------------------------------------------
# UNKNOWN
# ---------------------------------------------------------------------------

class TestUnknown(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_naive_datetime(self):
        self.assertEqual(self.cal.classify(dt.datetime(2026, 9, 15, 12, 0)), SessionType.UNKNOWN)

    def test_nonexistent_dst_time(self):
        self.assertEqual(self.cal.classify(et(2026, 3, 8, 2, 30)), SessionType.UNKNOWN)

    def test_configured_market_closure_is_unknown(self):
        base = SessionConfig.from_yaml()
        fields = dict(base.__dict__)
        fields["market_closures"] = (MarketClosure(date="2026-11-26"),)
        config = SessionConfig(**fields)
        cal = SessionCalendar(config=config)
        self.assertEqual(cal.classify(et(2026, 11, 26, 12, 0)), SessionType.UNKNOWN)

    def test_unknown_is_not_tradeable(self):
        self.assertFalse(is_tradeable(SessionType.UNKNOWN))


# ---------------------------------------------------------------------------
# Market closure / holiday-data precedence (Task 4 hardening)
# ---------------------------------------------------------------------------

class TestMarketClosurePrecedence(unittest.TestCase):
    def _calendar_with_closure(self, date_str, kind="full_closure"):
        base = SessionConfig.from_yaml()
        fields = dict(base.__dict__)
        fields["market_closures"] = (MarketClosure(date=date_str, kind=kind),)
        return SessionCalendar(config=SessionConfig(**fields))

    def test_configured_closure_overrides_what_would_otherwise_be_rth(self):
        # 2026-11-26 is a Thursday; at noon this would otherwise be RTH.
        cal = self._calendar_with_closure("2026-11-26")
        self.assertEqual(cal.classify(et(2026, 11, 26, 12, 0)), SessionType.UNKNOWN)

    def test_configured_closure_overrides_what_would_otherwise_be_weekend(self):
        # 2026-09-19 is a Saturday; would otherwise be WEEKEND.
        cal = self._calendar_with_closure("2026-09-19")
        self.assertEqual(cal.classify(et(2026, 9, 19, 12, 0)), SessionType.UNKNOWN)

    def test_unconfigured_weekday_is_unaffected_by_empty_closure_list(self):
        # Default calendar has an empty market_closures list. A normal
        # weekday must classify exactly as it did before this hardening
        # pass introduced the market_closures schema.
        cal = SessionCalendar()
        self.assertEqual(cal.classify(et(2026, 9, 15, 12, 0)), SessionType.RTH)

    def test_closure_on_unrelated_date_does_not_affect_other_days(self):
        cal = self._calendar_with_closure("2026-11-26")
        # A different Thursday, no closure configured for it.
        self.assertEqual(cal.classify(et(2026, 9, 17, 12, 0)), SessionType.RTH)

    def test_closure_kind_is_recorded_but_all_kinds_map_to_unknown_in_phase1(self):
        # Phase 1 does not implement modified/special sessions - even a
        # closure record with a different `kind` still yields UNKNOWN
        # rather than guessing at behavior that isn't implemented.
        cal = self._calendar_with_closure("2026-11-26", kind="early_close")
        self.assertEqual(cal.classify(et(2026, 11, 26, 12, 0)), SessionType.UNKNOWN)


# ---------------------------------------------------------------------------
# Explicit Bitget-normalized weekend boundary checks (Task 6.1)
# ---------------------------------------------------------------------------

class TestBitgetNormalizedWeekendBoundary(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_friday_19_59_59_is_not_weekend(self):
        self.assertNotEqual(self.cal.classify(et(2026, 9, 18, 19, 59, 59)), SessionType.WEEKEND)

    def test_friday_20_00_00_is_weekend(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 18, 20, 0, 0)), SessionType.WEEKEND)

    def test_monday_09_29_59_is_weekend(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 21, 9, 29, 59)), SessionType.WEEKEND)

    def test_monday_09_30_00_is_rth_and_closed_to_weekend_trading(self):
        session = self.cal.classify(et(2026, 9, 21, 9, 30, 0))
        self.assertEqual(session, SessionType.RTH)
        self.assertFalse(is_tradeable(session))


# ---------------------------------------------------------------------------
# Explicit POST boundary checks (Task 6.3)
# ---------------------------------------------------------------------------

class TestPostBoundaryExplicit(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_15_59_59_is_rth(self):
        # Ordinary Tuesday.
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 15, 59, 59)), SessionType.RTH)

    def test_16_00_00_is_post(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 16, 0, 0)), SessionType.POST)

    def test_19_59_59_is_post(self):
        self.assertEqual(self.cal.classify(et(2026, 9, 15, 19, 59, 59)), SessionType.POST)

    def test_20_00_00_on_friday_is_weekend(self):
        # The 20:00 ET boundary only becomes WEEKEND on Friday - the
        # Bitget-sourced weekend window open. On other weeknights 20:00 ET
        # is OVERNIGHT (see TestOvernight.test_weeknight_post_close).
        self.assertEqual(self.cal.classify(et(2026, 9, 18, 20, 0, 0)), SessionType.WEEKEND)


# ---------------------------------------------------------------------------
# Trading gate: tradeable vs non-tradeable
# ---------------------------------------------------------------------------

class TestTradeGate(unittest.TestCase):
    def setUp(self):
        self.cal = SessionCalendar()

    def test_gate_matches_spec(self):
        expected = {
            SessionType.RTH: False,
            SessionType.POST: False,
            SessionType.HALT: False,
            SessionType.UNKNOWN: False,
            SessionType.OVERNIGHT: True,
            SessionType.WEEKEND: True,
        }
        for session, expect in expected.items():
            with self.subTest(session=session):
                self.assertEqual(is_tradeable(session), expect)

    def test_end_to_end_rth_blocked(self):
        self.assertFalse(is_tradeable(self.cal.classify(et(2026, 9, 15, 12, 0))))

    def test_end_to_end_overnight_allowed(self):
        self.assertTrue(is_tradeable(self.cal.classify(et(2026, 9, 15, 22, 0))))

    def test_end_to_end_weekend_allowed(self):
        self.assertTrue(is_tradeable(self.cal.classify(et(2026, 9, 19, 12, 0))))

    def test_end_to_end_post_blocked(self):
        self.assertFalse(is_tradeable(self.cal.classify(et(2026, 9, 15, 18, 0))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
