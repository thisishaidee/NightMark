"""
Nightmark Phase 2B mark-quality gate.

Answers exactly one question: is a given MarketObservation reliable enough
to be considered by a future signal at all? Nothing in this module creates
a signal, sizes a position, or executes a trade - it only decides
PASS/REJECT, with reasons, and it is designed to FAIL CLOSED: every
ambiguous or unconfigured case rejects rather than silently passing.

No signal in a later phase may bypass this gate - evaluate_mark_quality()
is meant to be the single, mandatory checkpoint between "we have a
normalized observation" (Phase 2A) and "a signal is allowed to look at
it" (a later phase, not implemented here).

Six deterministic checks, corresponding to the six required gates:

    1. spread quality        - observation.spread_bps vs a configurable
                                threshold (historical same-ticker/session
                                90th percentile when enough history exists,
                                else a conservative static fallback)
    2. top-of-book depth      - observation.available_depth vs a
                                configurable, ticker-specific minimum
    3. stale reference        - age of the selected reference price vs a
                                configurable per-session maximum, with an
                                explicit, injected (never inferred)
                                weekend >48h corroboration exception
    4. stale/missing data     - observation.is_valid (already computed by
                                Phase 2A's normalize_observation) plus this
                                module's own depth-availability check
    5. session validity       - reuses Phase 1's is_tradeable(): only
                                OVERNIGHT/WEEKEND are eligible; RTH, POST,
                                HALT, and UNKNOWN are all rejected here too
    6. corporate-action/halt  - an explicit, configured [start, end)
                                window per symbol; empty and not fabricated
                                by default

No network calls, no live data, no Qwen, no signal creation, no sizing, no
execution.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import yaml

from src.calendar import SessionType, is_tradeable
from src.observation import MarketObservation

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "mark_quality.yaml"

_SESSION_NAME_TO_TYPE = {s.value: s for s in SessionType}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpreadQualityConfig:
    percentile: float = 0.90
    min_history_for_percentile: int = 20
    fallback_max_spread_bps: float = 150.0
    fallback_max_spread_bps_by_symbol: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DepthQualityConfig:
    default_min_depth: Optional[float] = None  # None -> unconfigured symbols fail closed
    min_depth_by_symbol: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceStalenessConfig:
    max_age_by_session: Dict[SessionType, dt.timedelta] = field(default_factory=dict)
    weekend_corroboration_required_beyond: dt.timedelta = dt.timedelta(hours=48)


@dataclass(frozen=True)
class CorporateActionWindow:
    symbol: str
    start: dt.datetime
    end: dt.datetime  # exclusive
    kind: str = "corporate_action"


@dataclass(frozen=True)
class MarkQualityConfig:
    spread: SpreadQualityConfig = field(default_factory=SpreadQualityConfig)
    depth: DepthQualityConfig = field(default_factory=DepthQualityConfig)
    staleness: ReferenceStalenessConfig = field(default_factory=ReferenceStalenessConfig)
    corporate_action_windows: Tuple[CorporateActionWindow, ...] = field(default_factory=tuple)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "MarkQualityConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        spread_raw = raw.get("spread", {}) or {}
        spread = SpreadQualityConfig(
            percentile=spread_raw.get("percentile", 0.90),
            min_history_for_percentile=spread_raw.get("min_history_for_percentile", 20),
            fallback_max_spread_bps=spread_raw.get("fallback_max_spread_bps", 150.0),
            fallback_max_spread_bps_by_symbol=dict(
                spread_raw.get("fallback_max_spread_bps_by_symbol", {}) or {}
            ),
        )

        depth_raw = raw.get("depth", {}) or {}
        depth = DepthQualityConfig(
            default_min_depth=depth_raw.get("default_min_depth", None),
            min_depth_by_symbol=dict(depth_raw.get("min_depth_by_symbol", {}) or {}),
        )

        staleness_raw = raw.get("staleness", {}) or {}
        max_age_hours = staleness_raw.get("max_age_hours_by_session", {}) or {}
        max_age_by_session = {
            _SESSION_NAME_TO_TYPE[name]: dt.timedelta(hours=hours)
            for name, hours in max_age_hours.items()
            if name in _SESSION_NAME_TO_TYPE
        }
        weekend_corrob_hours = staleness_raw.get(
            "weekend_corroboration_required_beyond_hours", 48
        )
        staleness = ReferenceStalenessConfig(
            max_age_by_session=max_age_by_session,
            weekend_corroboration_required_beyond=dt.timedelta(hours=weekend_corrob_hours),
        )

        windows = tuple(
            CorporateActionWindow(
                symbol=w["symbol"],
                start=dt.datetime.fromisoformat(w["start"]),
                end=dt.datetime.fromisoformat(w["end"]),
                kind=w.get("kind", "corporate_action"),
            )
            for w in raw.get("corporate_action_windows", []) or []
        )

        return cls(spread=spread, depth=depth, staleness=staleness, corporate_action_windows=windows)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkQualityResult:
    passed: bool
    reasons: Tuple[str, ...]  # empty iff passed
    spread_threshold_used: Optional[float] = None
    spread_threshold_source: Optional[str] = None  # "HISTORICAL_PERCENTILE" | "FALLBACK_STATIC"


# ---------------------------------------------------------------------------
# Percentile helper (no numpy dependency; deterministic linear-interpolation
# method, matching the common "linear" percentile definition)
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    lower = sorted_values[int(f)] * (c - k)
    upper = sorted_values[int(c)] * (k - f)
    return lower + upper


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def _compute_spread_threshold(
    symbol: str, cfg: SpreadQualityConfig, history_bps: Sequence[float]
) -> Tuple[float, str]:
    valid_history = sorted(v for v in history_bps if v is not None)
    if len(valid_history) >= cfg.min_history_for_percentile:
        return _percentile(valid_history, cfg.percentile), "HISTORICAL_PERCENTILE"
    fallback = cfg.fallback_max_spread_bps_by_symbol.get(symbol, cfg.fallback_max_spread_bps)
    return fallback, "FALLBACK_STATIC"


def _check_spread(observation: MarketObservation, threshold: float) -> Optional[str]:
    if observation.spread_bps is None:
        return "spread_unavailable"
    if observation.spread_bps > threshold:
        return "spread_too_wide"
    return None


def _check_depth(observation: MarketObservation, cfg: DepthQualityConfig) -> Optional[str]:
    if observation.available_depth is None:
        return "depth_unavailable"
    min_depth = cfg.min_depth_by_symbol.get(observation.symbol, cfg.default_min_depth)
    if min_depth is None:
        return "depth_minimum_not_configured"
    if observation.available_depth < min_depth:
        return "depth_below_minimum"
    return None


def _check_staleness(
    observation: MarketObservation,
    cfg: ReferenceStalenessConfig,
    reference_timestamp: Optional[dt.datetime],
    has_corroborating_witness_move: bool,
) -> Optional[str]:
    if observation.reference_price is None:
        return "reference_unavailable"
    if reference_timestamp is None:
        return "reference_timestamp_unavailable"
    if reference_timestamp > observation.timestamp:
        return "reference_timestamp_in_future"

    max_age = cfg.max_age_by_session.get(observation.session)
    if max_age is None:
        return "reference_staleness_threshold_not_configured_for_session"

    age = observation.timestamp - reference_timestamp
    if age <= max_age:
        return None

    if observation.session == SessionType.WEEKEND and age > cfg.weekend_corroboration_required_beyond:
        if has_corroborating_witness_move:
            return None
        return "stale_reference_weekend_uncorroborated"

    return "stale_reference"


def _check_corporate_action(
    observation: MarketObservation, windows: Sequence[CorporateActionWindow]
) -> Optional[str]:
    for w in windows:
        if w.symbol == observation.symbol and w.start <= observation.timestamp < w.end:
            return f"corporate_action_window:{w.kind}"
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_mark_quality(
    observation: MarketObservation,
    *,
    config: MarkQualityConfig,
    spread_history_bps: Sequence[float] = (),
    reference_timestamp: Optional[dt.datetime] = None,
    has_corroborating_witness_move: bool = False,
) -> MarkQualityResult:
    """Evaluate all six mark-quality gates for one observation.

    Fails closed: `passed` is True only when every gate explicitly passes.
    Any missing, unconfigured, or ambiguous input produces a rejection
    reason rather than being treated as acceptable.

    `spread_history_bps`: same-ticker/same-session historical spread_bps
      values (chronological order not required here - only used to derive
      a percentile threshold). Caller-injected; this module fetches
      nothing.
    `reference_timestamp`: when the selected reference price was actually
      printed/observed. Not part of the Phase 2A MarketObservation
      contract (which intentionally was not modified for this phase) -
      passed in separately so staleness can be assessed without changing
      Phase 1/2A's frozen data contract.
    `has_corroborating_witness_move`: an explicit, caller-supplied signal
      that a 24/7 witness market corroborates the current price
      relationship. This module never computes or infers this itself -
      that determination belongs to a future signal-layer phase.
    """
    reasons = []

    if not is_tradeable(observation.session):
        reasons.append(f"session_not_tradeable:{observation.session.value}")

    if not observation.is_valid:
        reasons.append(f"unusable_observation:{observation.invalid_reason}")

    depth_reason = _check_depth(observation, config.depth)
    if depth_reason:
        reasons.append(depth_reason)

    threshold, threshold_source = _compute_spread_threshold(
        observation.symbol, config.spread, spread_history_bps
    )
    spread_reason = _check_spread(observation, threshold)
    if spread_reason:
        reasons.append(spread_reason)

    staleness_reason = _check_staleness(
        observation, config.staleness, reference_timestamp, has_corroborating_witness_move
    )
    if staleness_reason:
        reasons.append(staleness_reason)

    corp_action_reason = _check_corporate_action(observation, config.corporate_action_windows)
    if corp_action_reason:
        reasons.append(corp_action_reason)

    return MarkQualityResult(
        passed=len(reasons) == 0,
        reasons=tuple(reasons),
        spread_threshold_used=threshold,
        spread_threshold_source=threshold_source,
    )
