"""
Nightmark Phase 2A deterministic market-data normalization.

Combines an injected raw rToken quote, injected reference-price
candidates, and Phase 1's session calendar into one normalized
MarketObservation. No network calls, no fabrication - every input is
supplied by the caller; this module only derives (deterministically) the
fields the data contract requires from what it's given.

basis      = rtoken_mid - reference_price
basis_bps  = basis / reference_price * 10,000

"Missing/invalid data" is handled explicitly rather than raising or
silently producing nonsense:
    - a missing bid or ask       -> mid, spread_bps are None
    - a crossed book (bid > ask) -> treated as invalid, mid is None
      (a crossed top-of-book is not a valid market to derive a mid from;
      this module does not try to guess which side is "right")
    - a non-positive bid/ask     -> treated as invalid, mid is None
    - no usable reference price  -> basis, basis_bps are None
    - reference_price == 0       -> basis_bps is None (division guarded)
In every case the returned MarketObservation still has `is_valid` and
`invalid_reason` set so the gap is a visible fact, not a silent None.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from src.calendar import SessionCalendar
from src.observation import MarketObservation, RawQuote, ReferenceCandidate
from src.reference import select_reference_price


def _safe_mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    if bid > ask:
        return None  # crossed/invalid book - never fabricate a mid from it
    return (bid + ask) / 2.0


def _safe_spread_bps(
    bid: Optional[float], ask: Optional[float], mid: Optional[float]
) -> Optional[float]:
    if bid is None or ask is None or mid is None or mid <= 0:
        return None
    return (ask - bid) / mid * 10_000.0


def _safe_depth(bid_size: Optional[float], ask_size: Optional[float]) -> Optional[float]:
    if bid_size is None or ask_size is None:
        return None
    if bid_size < 0 or ask_size < 0:
        return None
    # Conservative: the smaller of the two top-of-book sizes is what's
    # actually available to trade a full round-trip at the quoted spread.
    return min(bid_size, ask_size)


def _safe_basis(
    mid: Optional[float], reference_price: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    if mid is None or reference_price is None or reference_price == 0:
        return None, None
    basis = mid - reference_price
    basis_bps = basis / reference_price * 10_000.0
    return basis, basis_bps


def normalize_observation(
    raw: RawQuote,
    reference_candidates: Sequence[ReferenceCandidate],
    calendar: SessionCalendar,
) -> MarketObservation:
    """Build one normalized MarketObservation from injected raw inputs.

    `calendar` is injected (not constructed internally) so callers reuse
    the exact same Phase 1 SessionCalendar instance/config across a whole
    batch, and so tests can inject a calendar with custom config (e.g. a
    configured market closure) without touching this module.
    """
    session = calendar.classify(raw.timestamp)

    mid = _safe_mid(raw.bid, raw.ask)
    spread_bps = _safe_spread_bps(raw.bid, raw.ask, mid)
    depth = _safe_depth(raw.bid_size, raw.ask_size)

    ref = select_reference_price(reference_candidates)
    basis, basis_bps = _safe_basis(mid, ref.price)

    is_valid = True
    invalid_reason = None
    if mid is None:
        is_valid = False
        invalid_reason = "invalid_or_missing_rtoken_quote"
    elif ref.price is None:
        is_valid = False
        invalid_reason = "no_reference_price_available"

    return MarketObservation(
        timestamp=raw.timestamp,
        symbol=raw.symbol,
        session=session,
        rtoken_bid=raw.bid,
        rtoken_ask=raw.ask,
        rtoken_mid=mid,
        spread_bps=spread_bps,
        available_depth=depth,
        reference_price=ref.price,
        reference_source=ref.source,
        reference_quality=ref.quality,
        basis=basis,
        basis_bps=basis_bps,
        is_valid=is_valid,
        invalid_reason=invalid_reason,
    )
