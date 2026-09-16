"""
Nightmark session calendar.

Classifies a timestamp into exactly one SessionType:

    RTH | POST | OVERNIGHT | WEEKEND | HALT | UNKNOWN

RULES FIRST. This module contains no strategy logic, no signals, and no
sizing. It answers exactly one question: "what kind of session is this
timestamp in, and is trading permitted right now?"

All session boundaries are loaded from config/sessions.yaml. Do not hardcode
session times in this file - if a boundary needs to change (DST rollover
handling aside, which is automatic), it should change in the config, not here.

Safety principle: when a timestamp cannot be classified with confidence
(naive datetime, ambiguous/nonexistent local time across a DST transition,
a date landing in a configured market closure with no explicit session
override, or any other case this module isn't sure about), the classifier
returns UNKNOWN rather than guessing. UNKNOWN is NO TRADE.

This module classifies TIME only. It has no notion of which symbols are
eligible to trade during a WEEKEND session - that is a separate, explicit
gate (see src/eligibility.py). A WEEKEND classification here is necessary
but never sufficient for a symbol to actually be weekend-tradeable.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sessions.yaml"

_WEEKDAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


class SessionType(Enum):
    RTH = "RTH"
    POST = "POST"
    OVERNIGHT = "OVERNIGHT"
    WEEKEND = "WEEKEND"
    HALT = "HALT"
    UNKNOWN = "UNKNOWN"


# Trading gate. This is the ONLY place that decides eligibility from a
# SessionType. Nothing here reads model output, Qwen output, or any
# signal - it is a pure lookup table over SessionType.
_TRADE_GATE = {
    SessionType.RTH: False,
    SessionType.POST: False,
    SessionType.HALT: False,
    SessionType.UNKNOWN: False,
    SessionType.OVERNIGHT: True,
    SessionType.WEEKEND: True,
}


def is_tradeable(session: SessionType) -> bool:
    """Pure lookup: is this SessionType eligible for trading at all.

    This function does not know about signals, risk limits, position
    sizing, or mark quality. A True here means "the session-level gate is
    open" - it is necessary but not sufficient for an actual trade.
    """
    return _TRADE_GATE[session]


@dataclass(frozen=True)
class HaltWindow:
    start: dt.datetime  # tz-aware
    end: dt.datetime  # tz-aware, exclusive


@dataclass(frozen=True)
class MarketClosure:
    """An authoritative full-day (or, in future, special-session) closure.

    `kind` is recorded for forward compatibility (e.g. a future
    "early_close" type) but Phase 1 only implements one behavior: any
    configured closure, regardless of kind, causes classify() to return
    UNKNOWN for that date rather than guessing at modified-session rules
    it doesn't yet model.
    """

    date: str  # "YYYY-MM-DD"
    kind: str = "full_closure"


@dataclass(frozen=True)
class SessionConfig:
    timezone: str
    rth_start: dt.time
    rth_end: dt.time
    rth_weekdays: frozenset
    post_start: dt.time
    post_end: dt.time
    post_weekdays: frozenset
    weekend_open_weekday: int  # 0=MON ... 6=SUN
    weekend_open_time: dt.time
    weekend_close_weekday: int
    weekend_close_time: dt.time
    overnight_pre_start: dt.time
    overnight_pre_end: dt.time
    overnight_post_start: dt.time
    overnight_post_end_is_midnight: bool
    halts: tuple = field(default_factory=tuple)
    market_closures: tuple = field(default_factory=tuple)  # tuple[MarketClosure, ...]

    @staticmethod
    def _parse_time(s: str) -> dt.time:
        if s == "24:00":
            # dt.time cannot represent 24:00; callers treat this specially.
            return dt.time(23, 59, 59, 999999)
        h, m = s.split(":")
        return dt.time(int(h), int(m))

    @staticmethod
    def _weekday_index(name: str) -> int:
        return _WEEKDAY_NAMES.index(name)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "SessionConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        rth = raw["rth"]
        post = raw["post"]
        weekend = raw["weekend"]
        overnight = raw["overnight"]

        halts = tuple(
            HaltWindow(
                start=dt.datetime.fromisoformat(h["start"]).replace(
                    tzinfo=ZoneInfo(raw["timezone"])
                )
                if dt.datetime.fromisoformat(h["start"]).tzinfo is None
                else dt.datetime.fromisoformat(h["start"]),
                end=dt.datetime.fromisoformat(h["end"]).replace(
                    tzinfo=ZoneInfo(raw["timezone"])
                )
                if dt.datetime.fromisoformat(h["end"]).tzinfo is None
                else dt.datetime.fromisoformat(h["end"]),
            )
            for h in raw.get("halts", [])
        )

        market_closures = tuple(
            MarketClosure(date=c["date"], kind=c.get("type", "full_closure"))
            if isinstance(c, dict)
            else MarketClosure(date=c)  # back-compat: bare "YYYY-MM-DD" string
            for c in raw.get("market_closures", [])
        )

        return cls(
            timezone=raw["timezone"],
            rth_start=cls._parse_time(rth["start"]),
            rth_end=cls._parse_time(rth["end"]),
            rth_weekdays=frozenset(cls._weekday_index(d) for d in rth["weekdays"]),
            post_start=cls._parse_time(post["start"]),
            post_end=cls._parse_time(post["end"]),
            post_weekdays=frozenset(cls._weekday_index(d) for d in post["weekdays"]),
            weekend_open_weekday=cls._weekday_index(weekend["open_weekday"]),
            weekend_open_time=cls._parse_time(weekend["open_time"]),
            weekend_close_weekday=cls._weekday_index(weekend["close_weekday"]),
            weekend_close_time=cls._parse_time(weekend["close_time"]),
            overnight_pre_start=cls._parse_time(overnight["pre_market_start"]),
            overnight_pre_end=cls._parse_time(overnight["pre_market_end"]),
            overnight_post_start=cls._parse_time(overnight["post_close_start"]),
            overnight_post_end_is_midnight=overnight["post_close_end"] == "24:00",
            halts=halts,
            market_closures=market_closures,
        )


class SessionCalendar:
    """Classifies timestamps into SessionType using a SessionConfig.

    Usage:
        cal = SessionCalendar()  # loads config/sessions.yaml
        session = cal.classify(some_datetime)
        tradeable = is_tradeable(session)
    """

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
        extra_halts: tuple = (),
    ):
        self.config = config or SessionConfig.from_yaml(config_path)
        self._tz = ZoneInfo(self.config.timezone)
        self._extra_halts = extra_halts

    # -- public API ---------------------------------------------------

    def classify(self, timestamp: dt.datetime) -> SessionType:
        """Classify a timestamp. Returns UNKNOWN if it cannot be classified safely."""
        safe_local = self._to_safe_local(timestamp)
        if safe_local is None:
            return SessionType.UNKNOWN

        if self._is_halted(safe_local):
            return SessionType.HALT

        if self._is_market_closure(safe_local):
            # Phase 1 does not model modified/special sessions (e.g. an
            # early close) even though the config schema can describe them
            # (see MarketClosure.kind). Any configured closure is
            # intentionally UNKNOWN rather than guessing RTH/closed.
            return SessionType.UNKNOWN

        if self._in_weekend_window(safe_local):
            return SessionType.WEEKEND

        if self._in_rth(safe_local):
            return SessionType.RTH

        if self._in_post(safe_local):
            return SessionType.POST

        if self._in_overnight(safe_local):
            return SessionType.OVERNIGHT

        return SessionType.UNKNOWN

    # -- timestamp safety -----------------------------------------------

    def _to_safe_local(self, timestamp: dt.datetime) -> Optional[dt.datetime]:
        """Convert to local session tz, rejecting anything unsafe.

        Rejects (returns None -> caller maps to UNKNOWN):
          - naive datetimes (no tzinfo): ambiguous by construction
          - local times that don't exist (spring-forward gap), e.g. a
            timestamp directly constructed as 2026-03-08 02:30 in
            America/New_York, which is never a real instant

        Does NOT reject the fall-back ambiguous hour (e.g. 01:30 occurring
        twice on the November transition): a tz-aware Python datetime
        carries an explicit `fold` value that disambiguates which of the
        two real instants was meant, so both occurrences classify normally.

        Method: convert the input to a genuine UTC instant (always
        well-defined for any valid aware datetime), then convert that UTC
        instant back into the local session timezone. If the caller's
        input was already expressed directly in the session timezone, its
        wall-clock time must round-trip exactly; if it doesn't, the
        original wall-clock time never existed and we refuse rather than
        silently normalize it to a nearby real time.
        """
        if timestamp.tzinfo is None:
            return None

        utc_instant = timestamp.astimezone(dt.timezone.utc)
        local = utc_instant.astimezone(self._tz)

        same_zone = getattr(timestamp.tzinfo, "key", None) == self.config.timezone
        if same_zone and local.replace(tzinfo=None) != timestamp.replace(tzinfo=None):
            return None

        return local

    # -- session windows --------------------------------------------------

    def _is_halted(self, local: dt.datetime) -> bool:
        for halt in tuple(self.config.halts) + tuple(self._extra_halts):
            start = halt.start.astimezone(self._tz)
            end = halt.end.astimezone(self._tz)
            if start <= local < end:
                return True
        return False

    def _is_market_closure(self, local: dt.datetime) -> bool:
        target = local.date().isoformat()
        return any(c.date == target for c in self.config.market_closures)

    def _in_rth(self, local: dt.datetime) -> bool:
        if local.weekday() not in self.config.rth_weekdays:
            return False
        return self.config.rth_start <= local.time() < self.config.rth_end

    def _in_post(self, local: dt.datetime) -> bool:
        if local.weekday() not in self.config.post_weekdays:
            return False
        return self.config.post_start <= local.time() < self.config.post_end

    def _in_weekend_window(self, local: dt.datetime) -> bool:
        """Friday open_time -> Monday close_time, inclusive of the whole span."""
        weekday = local.weekday()
        t = local.time()

        open_wd = self.config.weekend_open_weekday
        open_t = self.config.weekend_open_time
        close_wd = self.config.weekend_close_weekday
        close_t = self.config.weekend_close_time

        if weekday == open_wd:
            return t >= open_t
        if weekday == close_wd:
            return t < close_t
        # Fully-enclosed days between open weekday and close weekday
        # (wrapping across the week, e.g. Sat, Sun between Fri and Mon).
        span = []
        d = (open_wd + 1) % 7
        while d != close_wd:
            span.append(d)
            d = (d + 1) % 7
        return weekday in span

    def _in_overnight(self, local: dt.datetime) -> bool:
        # Overnight never applies inside the weekend window (weekend takes
        # precedence and is checked earlier in classify()).
        if local.weekday() not in self.config.rth_weekdays:
            # Only weekdays can be "weeknight overnight" in this model;
            # anything on SAT/SUN outside RTH weekdays that isn't already
            # caught by the weekend window falls through to UNKNOWN.
            return False

        t = local.time()
        if self.config.overnight_pre_start <= t < self.config.overnight_pre_end:
            return True
        if self.config.overnight_post_end_is_midnight:
            if t >= self.config.overnight_post_start:
                return True
        return False
