"""
Nightmark witness-market model.

A "witness" is a 24/7 market (e.g. a crypto-native proxy, futures, or any
other continuously-trading instrument) whose move can corroborate or
explain an rToken basis dislocation while the U.S. cash market is closed.

Witness data is ALWAYS injected by the caller - this module does not
fetch, infer, or fabricate a witness move, and makes no network calls. It
only classifies a supplied move against a configurable minimum threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class WitnessObservation:
    """A single injected witness-market data point.

    `move_bps` is a signed basis-point move of the witness market over
    whatever window the caller considers relevant (e.g. since the U.S.
    cash close). Positive = witness market up; negative = witness market
    down. `None` means no witness data is available at all - distinct
    from an actual witness move of exactly 0.

    `source` is a free-text provenance label (e.g. "24_7_PROXY_FUTURES")
    recorded for the signal output but never interpreted by this module.
    """

    move_bps: Optional[float] = None
    source: Optional[str] = None


class WitnessStatus(Enum):
    NO_WITNESS = "NO_WITNESS"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    QUALIFYING_POSITIVE = "QUALIFYING_POSITIVE"
    QUALIFYING_NEGATIVE = "QUALIFYING_NEGATIVE"


def classify_witness(
    witness: Optional[WitnessObservation], min_move_bps: float
) -> WitnessStatus:
    """Classify an injected witness observation against a configurable minimum.

    Boundary: |move_bps| == min_move_bps qualifies (inclusive).
    A witness with move_bps == 0 is BELOW_THRESHOLD (assuming a positive
    min_move_bps, as configured) - it is never NO_WITNESS, since NO_WITNESS
    specifically means no data was supplied at all.
    """
    if witness is None or witness.move_bps is None:
        return WitnessStatus.NO_WITNESS
    if abs(witness.move_bps) < min_move_bps:
        return WitnessStatus.BELOW_THRESHOLD
    return (
        WitnessStatus.QUALIFYING_POSITIVE
        if witness.move_bps > 0
        else WitnessStatus.QUALIFYING_NEGATIVE
    )
