"""
Nightmark Phase 2A data contract: the normalized market observation.

This module defines DATA ONLY - no strategy logic, no signals, no
mark-quality scoring, no sizing, no execution, no Qwen. It is the frozen
shape that every later Phase 2 component (and eventually signals) reads
from.

All market and reference data is INJECTED by the caller (tests, fixtures,
or a future adapter that is not part of this phase). Nothing in this
module makes a network call or invents a data source.

Reference-price priority (enforced in src/reference.py, recorded here):
    1. Bitget official reference/NAV, if exposed and usable
    2. Last official US cash print/close, while the cash market is closed
    3. A liquid proxy, ONLY when necessary - and ALWAYS labelled ESTIMATED,
       never presented as an official reference.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.calendar import SessionType


class ReferenceSource(Enum):
    """Which tier of the fixed priority order actually produced the reference price."""

    BITGET_OFFICIAL = "BITGET_OFFICIAL"
    US_CASH_CLOSE = "US_CASH_CLOSE"
    LIQUID_PROXY_ESTIMATED = "LIQUID_PROXY_ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"


class ReferenceQuality(Enum):
    """Coarse provenance label for the selected reference price.

    This is intentionally minimal - a direct readout of which priority
    tier was used, nothing more. It is NOT the mark-quality engine
    (staleness thresholds, depth-weighted confidence scoring, etc.) that
    Phase 2 explicitly defers to a later phase.
    """

    OFFICIAL = "OFFICIAL"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class RawQuote:
    """Raw rToken market-side data as injected by the caller.

    No validation happens here - this is a plain data carrier. Validation
    and safe derivation (mid, spread, depth) happen in src/normalize.py,
    which is where "missing/invalid data" is actually handled.
    """

    timestamp: dt.datetime  # tz-aware; interpretation is the caller's responsibility
    symbol: str
    bid: Optional[float]
    ask: Optional[float]
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None


@dataclass(frozen=True)
class ReferenceCandidate:
    """One tier of candidate reference price, as injected by the caller.

    `stale` and `available` are plain booleans the caller sets - this
    module does not compute or infer either of them. A candidate the
    caller marks stale or unavailable is skipped by the priority-order
    selection in src/reference.py, falling through to the next tier
    rather than being silently trusted.
    """

    source: ReferenceSource
    price: Optional[float]
    available: bool = True
    stale: bool = False


@dataclass(frozen=True)
class MarketObservation:
    """Nightmark's normalized market observation. Immutable, data-only.

    Fields, per the frozen Phase 2A data contract:
        timestamp, symbol, session,
        rtoken_bid, rtoken_ask, rtoken_mid, spread_bps, available_depth,
        reference_price, reference_source, reference_quality,
        basis, basis_bps

    Plus two explicit validity fields (`is_valid`, `invalid_reason`) so
    "missing/invalid data" is a visible, inspectable fact about the
    observation rather than a silently-produced None a caller might miss.
    """

    timestamp: dt.datetime
    symbol: str
    session: SessionType

    rtoken_bid: Optional[float]
    rtoken_ask: Optional[float]
    rtoken_mid: Optional[float]
    spread_bps: Optional[float]
    available_depth: Optional[float]

    reference_price: Optional[float]
    reference_source: ReferenceSource
    reference_quality: ReferenceQuality

    basis: Optional[float]
    basis_bps: Optional[float]

    is_valid: bool
    invalid_reason: Optional[str] = None
