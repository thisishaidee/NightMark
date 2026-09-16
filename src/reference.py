"""
Nightmark Phase 2A reference-price selection.

Fixed priority order, deterministic, no network calls, no fabrication:
    1. Bitget official reference/NAV, if exposed and usable
    2. Last official US cash print/close, while the cash market is closed
    3. A liquid proxy, ONLY when necessary, and ALWAYS labelled ESTIMATED -
       never presented as an official reference.

Candidates are injected by the caller (tests, fixtures, or a future data
adapter that is not part of this phase). This module does not fetch
anything and does not decide WHICH candidates exist - only, given the ones
it's handed, which one wins and how to label the result honestly.

A candidate the caller marks `stale` or `available=False` is skipped
entirely (not silently trusted, not demoted-and-still-used) - selection
falls through to the next tier. This is deliberately simple provenance/
availability handling, not the mark-quality engine (staleness thresholds,
confidence scoring) that later phases will add.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.observation import ReferenceCandidate, ReferenceQuality, ReferenceSource

# Fixed priority order. Position in this tuple is what "priority" means -
# the order candidates happen to be passed in is irrelevant.
_PRIORITY = (
    ReferenceSource.BITGET_OFFICIAL,
    ReferenceSource.US_CASH_CLOSE,
    ReferenceSource.LIQUID_PROXY_ESTIMATED,
)

_QUALITY_BY_SOURCE = {
    ReferenceSource.BITGET_OFFICIAL: ReferenceQuality.OFFICIAL,
    ReferenceSource.US_CASH_CLOSE: ReferenceQuality.OFFICIAL,
    ReferenceSource.LIQUID_PROXY_ESTIMATED: ReferenceQuality.ESTIMATED,
}


@dataclass(frozen=True)
class ReferenceSelection:
    price: Optional[float]
    source: ReferenceSource
    quality: ReferenceQuality


def select_reference_price(candidates: Sequence[ReferenceCandidate]) -> ReferenceSelection:
    """Pick the highest-priority usable candidate. Never fabricates a price.

    If the same ReferenceSource appears more than once in `candidates`,
    the first occurrence wins (callers should not do this; it's not a
    supported use case, just documented behavior rather than an implicit
    footgun).

    Returns ReferenceSelection(price=None, source=UNAVAILABLE,
    quality=UNAVAILABLE) if nothing usable was supplied - this is the
    explicit "no fabrication" fallback.
    """
    by_source = {}
    for c in candidates:
        by_source.setdefault(c.source, c)

    for source in _PRIORITY:
        candidate = by_source.get(source)
        if candidate is None:
            continue
        if not candidate.available:
            continue
        if candidate.price is None:
            continue
        if candidate.stale:
            continue
        return ReferenceSelection(
            price=candidate.price,
            source=source,
            quality=_QUALITY_BY_SOURCE[source],
        )

    return ReferenceSelection(
        price=None,
        source=ReferenceSource.UNAVAILABLE,
        quality=ReferenceQuality.UNAVAILABLE,
    )
