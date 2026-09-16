"""
Deterministic synthetic fixtures for Phase 2A tests.

Every value here is a made-up, synthetic test number - NOT real Bitget
data and NOT a claim about real market performance anywhere in this
project. Timestamps reuse dates already verified against Phase 1's own
session-calendar tests (tests/test_calendar.py), so session classification
inside these tests is never ambiguous or newly-introduced.
"""

import datetime as dt
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def et(y, m, d, h, mi, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=ET)


# Previously-verified session timestamps (see tests/test_calendar.py):
TUE_RTH = et(2026, 9, 15, 12, 0)  # Tuesday, ordinary RTH
TUE_OVERNIGHT = et(2026, 9, 15, 22, 0)  # Tuesday, weeknight overnight
SAT_WEEKEND = et(2026, 9, 19, 13, 0)  # Saturday, inside Bitget weekend window
SUN_WEEKEND = et(2026, 9, 20, 13, 0)  # Sunday, inside Bitget weekend window
