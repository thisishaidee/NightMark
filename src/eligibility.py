"""
Nightmark symbol weekend-eligibility gate.

Conceptual pipeline (Phase 1 implements only the first two stages):

    timestamp
      |
      v
    session classification (src/calendar.py)   <- "what kind of session is this"
      |
      v
    symbol weekend eligibility (this module)    <- "may THIS symbol trade weekends"
      |
      v
    mark quality      (not implemented)
      |
      v
    signal            (not implemented - RULES FIRST, Qwen never controls this)
      |
      v
    risk              (not implemented)
      |
      v
    execution         (not implemented)

Why this is a separate module from calendar.py: src/calendar.py answers a
pure time question and knows nothing about symbols. Bitget's weekend
matching window is a real-time fact, but Bitget only allows an explicit,
Bitget-curated (and expanding) list of symbols to actually trade during
it. Folding symbol eligibility into the time-based calendar would let a
future bug accidentally treat "timestamp is WEEKEND" as "symbol X may
trade" - keeping them as two independently-testable gates makes that
mistake structurally harder to make.

Phase 1 note: config/symbols.yaml ships with an EMPTY eligible-symbol set.
Nightmark does not fabricate or guess which symbols Bitget currently
supports for weekend trading. Before any live/paper trading decision
depends on this module, config/symbols.yaml must be populated from an
authoritative, current Bitget source - see the comments in that file.

This module never imports Qwen, never calls the network, and never makes
a trading decision by itself - it only answers a yes/no eligibility
question.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional

import yaml

from src.calendar import SessionCalendar, SessionType

DEFAULT_SYMBOLS_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "symbols.yaml"
)


@dataclass(frozen=True)
class SymbolEligibilityConfig:
    eligible_symbols: FrozenSet[str] = field(default_factory=frozenset)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_SYMBOLS_CONFIG_PATH) -> "SymbolEligibilityConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        symbols = raw.get("weekend_eligible_symbols") or []
        return cls(eligible_symbols=frozenset(s.upper() for s in symbols))


class SymbolEligibility:
    """Answers: is this symbol configured as Bitget weekend-eligible?

    Conservative by construction: any symbol not explicitly present in the
    configured set is NOT eligible. There is no wildcard, no "default
    allow", and no inference from the symbol's name or format.
    """

    def __init__(self, config: Optional[SymbolEligibilityConfig] = None):
        self.config = config or SymbolEligibilityConfig.from_yaml()

    def is_symbol_configured_eligible(self, symbol: Optional[str]) -> bool:
        if not symbol:
            return False
        return symbol.strip().upper() in self.config.eligible_symbols


def is_weekend_eligible(
    symbol: Optional[str],
    timestamp: dt.datetime,
    *,
    calendar: Optional[SessionCalendar] = None,
    eligibility: Optional[SymbolEligibility] = None,
) -> bool:
    """True only if BOTH hold:
      1. the timestamp classifies as SessionType.WEEKEND, and
      2. the symbol is explicitly configured as weekend-eligible.

    A WEEKEND timestamp alone is never sufficient - this is the one
    function in Phase 1 that composes the time gate and the symbol gate,
    and it composes them with AND, never OR. `calendar` and `eligibility`
    are injectable (for tests / explicit config); if omitted, both are
    constructed from their default config files.
    """
    calendar = calendar or SessionCalendar()
    eligibility = eligibility or SymbolEligibility()

    session = calendar.classify(timestamp)
    if session != SessionType.WEEKEND:
        return False

    return eligibility.is_symbol_configured_eligible(symbol)
