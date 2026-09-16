"""
Nightmark Phase 2A rolling same-session z-score.

Computes a z-score of `basis_bps` (or another configured numeric field on
MarketObservation) for each observation in a chronologically-ordered,
single-symbol sequence, using ONLY strictly-prior observations from the
SAME session as the baseline.

This directly implements "never mix RTH and WEEKEND observations" as the
more general rule "never mix observations from two different sessions" -
RTH-vs-WEEKEND is the specific pairing the rule names, but POST and
OVERNIGHT baselines are kept equally separate from each other and from
RTH/WEEKEND, for the same underlying reason: each session has a distinct
liquidity/microstructure regime, so pooling their statistics would produce
a baseline that describes no real regime at all.

No look-ahead, by construction:
    - the baseline for observation i is built exclusively from
      observations at list positions < i;
    - a given observation's own value is appended to its session's history
      only AFTER that observation's z-score has been computed, so it can
      never influence its own baseline;
    - this function does not re-sort its input by timestamp. The input
      order IS the authoritative processing order. Callers must supply
      each symbol's observations in the exact chronological order they
      want history to accumulate in - an internal re-sort would be able to
      silently move a "future" observation earlier and leak it into an
      earlier z-score, which is exactly the class of bug this design
      avoids by never doing that re-sort in the first place.

Explicit handling (never a silent NaN/inf, never a raised exception for
these expected cases):
    - insufficient history        -> reason="insufficient_history"
    - zero variance in baseline   -> reason="zero_variance"
    - missing/None current value  -> reason="missing_value" (and it is not
      added to any baseline - a missing reading can't be historical data
      either)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.calendar import SessionType
from src.observation import MarketObservation


@dataclass(frozen=True)
class ZScoreResult:
    value: Optional[float]
    reason: Optional[str]  # None exactly when value is not None
    observations_used: int  # size of the baseline actually used (or available)


def rolling_same_session_zscore(
    observations: Sequence[MarketObservation],
    *,
    window: int,
    min_observations: int,
    field: str = "basis_bps",
) -> List[ZScoreResult]:
    """Compute one ZScoreResult per input observation, in input order.

    `window`: maximum number of prior same-session observations used as
      the baseline (a rolling window, most-recent-first truncated).
    `min_observations`: minimum number of prior same-session observations
      required before a z-score is attempted at all. Must be <= window.
      Note a z-score additionally requires at least 2 baseline points to
      have a defined sample standard deviation - if `min_observations` is
      set to 1, results still explicitly report "insufficient_history"
      until a second same-session point exists, rather than raising or
      returning an undefined stdev.
    `field`: which MarketObservation attribute to compute the z-score of;
      defaults to "basis_bps" (this module was built for basis_bps but is
      not hardcoded to it).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_observations < 1:
        raise ValueError("min_observations must be >= 1")
    if min_observations > window:
        raise ValueError("min_observations cannot exceed window")

    history: Dict[SessionType, List[float]] = {}
    results: List[ZScoreResult] = []

    for obs in observations:
        buf = history.setdefault(obs.session, [])
        current_value = getattr(obs, field, None)

        if current_value is None:
            # Missing readings are neither scoreable nor historical.
            results.append(
                ZScoreResult(value=None, reason="missing_value", observations_used=len(buf))
            )
            continue

        baseline = buf[-window:]

        if len(buf) < min_observations or len(baseline) < 2:
            results.append(
                ZScoreResult(
                    value=None, reason="insufficient_history", observations_used=len(baseline)
                )
            )
        else:
            mean = statistics.fmean(baseline)
            stdev = statistics.stdev(baseline)  # sample stdev (ddof=1); len(baseline) >= 2 here
            if stdev == 0:
                results.append(
                    ZScoreResult(
                        value=None, reason="zero_variance", observations_used=len(baseline)
                    )
                )
            else:
                z = (current_value - mean) / stdev
                results.append(
                    ZScoreResult(value=z, reason=None, observations_used=len(baseline))
                )

        # Only now (after this observation's own result is computed) does
        # its value enter the baseline for future observations.
        buf.append(current_value)

    return results
