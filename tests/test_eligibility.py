"""
Tests for src/eligibility.py.

Uses stdlib unittest, no network, no live Bitget data. Config is injected
explicitly in every test rather than relying on the (intentionally empty)
default config/symbols.yaml, so these tests are stable regardless of what
gets populated there later.
"""

import datetime as dt
import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calendar import SessionCalendar, SessionConfig
from src.eligibility import (
    SymbolEligibility,
    SymbolEligibilityConfig,
    is_weekend_eligible,
)

ET = ZoneInfo("America/New_York")


def et(y, m, d, h, mi, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=ET)


class TestSymbolEligibility(unittest.TestCase):
    def setUp(self):
        self.eligibility = SymbolEligibility(
            config=SymbolEligibilityConfig(eligible_symbols=frozenset({"RNVDA", "RAAPL"}))
        )

    def test_eligible_symbol_passes(self):
        self.assertTrue(self.eligibility.is_symbol_configured_eligible("RNVDA"))

    def test_eligible_symbol_case_insensitive(self):
        self.assertTrue(self.eligibility.is_symbol_configured_eligible("rnvda"))

    def test_ineligible_symbol_fails(self):
        self.assertFalse(self.eligibility.is_symbol_configured_eligible("RTSLA"))

    def test_unconfigured_symbol_fails_conservatively(self):
        # Nothing configured at all -> default-deny, not default-allow.
        empty = SymbolEligibility(config=SymbolEligibilityConfig())
        self.assertFalse(empty.is_symbol_configured_eligible("RNVDA"))

    def test_none_symbol_fails(self):
        self.assertFalse(self.eligibility.is_symbol_configured_eligible(None))

    def test_empty_string_symbol_fails(self):
        self.assertFalse(self.eligibility.is_symbol_configured_eligible(""))


class TestWeekendEligibilityComposition(unittest.TestCase):
    """Tests the composed is_weekend_eligible(symbol, timestamp) gate."""

    def setUp(self):
        self.calendar = SessionCalendar()
        self.eligibility = SymbolEligibility(
            config=SymbolEligibilityConfig(eligible_symbols=frozenset({"RNVDA"}))
        )

    def test_eligible_symbol_during_weekend_passes(self):
        # Saturday, ordinary weekend window.
        result = is_weekend_eligible(
            "RNVDA",
            et(2026, 9, 19, 12, 0),
            calendar=self.calendar,
            eligibility=self.eligibility,
        )
        self.assertTrue(result)

    def test_ineligible_symbol_during_weekend_fails(self):
        result = is_weekend_eligible(
            "RTSLA",
            et(2026, 9, 19, 12, 0),
            calendar=self.calendar,
            eligibility=self.eligibility,
        )
        self.assertFalse(result)

    def test_unconfigured_symbol_during_weekend_fails_conservatively(self):
        result = is_weekend_eligible(
            "RSOME_UNLISTED_SYMBOL",
            et(2026, 9, 19, 12, 0),
            calendar=self.calendar,
            eligibility=self.eligibility,
        )
        self.assertFalse(result)

    def test_weekend_timestamp_alone_does_not_imply_eligibility(self):
        # Even a symbol never configured anywhere must fail, purely because
        # the timestamp being WEEKEND is not sufficient by itself.
        empty_eligibility = SymbolEligibility(config=SymbolEligibilityConfig())
        result = is_weekend_eligible(
            "RNVDA",
            et(2026, 9, 19, 12, 0),
            calendar=self.calendar,
            eligibility=empty_eligibility,
        )
        self.assertFalse(result)

    def test_eligible_symbol_outside_weekend_window_fails(self):
        # Tuesday midday RTH - eligible symbol, wrong session entirely.
        result = is_weekend_eligible(
            "RNVDA",
            et(2026, 9, 15, 12, 0),
            calendar=self.calendar,
            eligibility=self.eligibility,
        )
        self.assertFalse(result)

    def test_eligible_symbol_during_overnight_fails(self):
        # OVERNIGHT is trade-eligible at the session level, but weekend
        # eligibility is specifically about the WEEKEND session, not any
        # session that happens to be tradeable.
        result = is_weekend_eligible(
            "RNVDA",
            et(2026, 9, 15, 22, 0),
            calendar=self.calendar,
            eligibility=self.eligibility,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
