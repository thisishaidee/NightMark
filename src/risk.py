"""
Nightmark Phase 3 risk engine.

Answers exactly one question: "given an already-valid Nightmark signal,
current portfolio state, trading costs, and time-to-cash-open, is this
trade allowed, and what is the maximum safe notional?"

This module does NOT decide whether the market is bullish or bearish, does
NOT generate signals, does NOT place real orders, and has no Qwen/UI/
backtesting/execution code. It is a pure, deterministic function of its
inputs plus a configured RiskConfig - no network calls, no fabricated
data.

REUSE, NOT DUPLICATION:
    - session tradeability reuses src.calendar.is_tradeable() and
      SessionCalendar's own configured RTH boundaries (via
      SessionCalendar.config) to locate the next cash open for the
      flatten-window calculation - this module does not hardcode or
      re-derive any session time.
    - mark-quality is read directly off the injected SignalResult's own
      mark_quality_passed/mark_quality_reasons fields (which the signal
      engine already populated from a MarkQualityResult) - this module
      never recreates mark-quality rules.
    - signal type/direction/basis/residual are read directly off the
      injected SignalResult - this module never re-derives a trading
      signal.

FAIL CLOSED: the default decision is REJECT. Missing or invalid NAV,
portfolio state, signal, spread/cost inputs, or timestamps all produce an
explicit rejection reason rather than an unsafe default. Reasons
accumulate - a single evaluation surfaces every applicable failure, not
just the first one encountered, wherever that is meaningful to compute.

DESIGN CHOICE - reduce/flatten actions bypass the signal/session/mark-
quality/cost-edge gates (APPROVED, see spec.md's "Phase 3 approved design
decisions" section):
    Nightmark's signal, session, and mark-quality gates exist to decide
    whether it is safe to take on NEW or INCREASED exposure. A REDUCE or
    FLATTEN request is the opposite: it is a risk-management action that
    lowers exposure. Blocking a de-risking action because, say, the
    session just became non-tradeable (RTH, HALT, or another restricted
    session) or mark quality just degraded would be actively unsafe - you
    would be forced to stay in a position you are trying to exit.
    REDUCE/FLATTEN therefore only require valid NAV/portfolio/notional
    bookkeeping (you cannot reduce or flatten more than you actually
    hold), never the position-opening gates - explicitly including cost/
    edge, since a cost/edge check answers "is this worth entering", not
    "is this worth exiting". OPEN_OR_INCREASE continues to require every
    gate (signal, session, mark-quality, cost/edge) with no exceptions.
    See tests/test_risk.py's TestApprovedGateBypassDecision for explicit
    per-gate coverage of both the REDUCE/FLATTEN bypass and the contrasting
    OPEN_OR_INCREASE enforcement on the same conditions.

CONCEPTUAL NOTE - "expected edge": the spec requires comparing expected
edge against estimated cost, but does not specify how to compute edge.
This module derives it deterministically from data the (already-approved)
signal already computed - abs(basis_bps) for FADE, abs(residual_bps) for
FOLLOW - rather than inventing a new market read. This is a magnitude
calculation on already-decided data, not a new trading decision, but it
is a simplification worth reviewing before Phase 4 depends on it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import yaml

from src.calendar import SessionCalendar, SessionType, is_tradeable
from src.signal import Direction, SignalResult, SignalType

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "risk.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConfig:
    """Round-trip cost model. See module docstring / config/risk.yaml for
    the full documented formula. All values are per-side basis points;
    evaluate_risk() doubles them to estimate a round-trip cost, since the
    expected edge is also a round-trip (enter now, exit later) quantity."""

    fee_bps_per_side: float = 5.0
    additional_slippage_bps_per_side: float = 2.0
    weekend_additional_slippage_bps_per_side: float = 3.0


@dataclass(frozen=True)
class FlattenConfig:
    buffer_minutes: float = 60.0
    # Explicit, configurable, OFF by default. Never silently invented.
    follow_exception_during_flatten: bool = False


@dataclass(frozen=True)
class SizingConfig:
    # Extra derate applied on top of the raw single-name/gross capacity
    # minimum. 1.0 = no extra derate (use full computed capacity).
    max_notional_scaling_factor: float = 1.0


@dataclass(frozen=True)
class RiskConfig:
    max_active_names: int = 3
    max_single_name_exposure_pct: float = 0.25
    max_gross_exposure_pct: float = 1.00
    max_averaging_down_count: int = 1
    cost: CostConfig = field(default_factory=CostConfig)
    flatten: FlattenConfig = field(default_factory=FlattenConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "RiskConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        cost_raw = raw.get("cost", {}) or {}
        cost = CostConfig(
            fee_bps_per_side=cost_raw.get("fee_bps_per_side", 5.0),
            additional_slippage_bps_per_side=cost_raw.get("additional_slippage_bps_per_side", 2.0),
            weekend_additional_slippage_bps_per_side=cost_raw.get(
                "weekend_additional_slippage_bps_per_side", 3.0
            ),
        )

        flatten_raw = raw.get("flatten", {}) or {}
        flatten = FlattenConfig(
            buffer_minutes=flatten_raw.get("buffer_minutes", 60.0),
            follow_exception_during_flatten=flatten_raw.get(
                "follow_exception_during_flatten", False
            ),
        )

        sizing_raw = raw.get("sizing", {}) or {}
        sizing = SizingConfig(
            max_notional_scaling_factor=sizing_raw.get("max_notional_scaling_factor", 1.0),
        )

        return cls(
            max_active_names=raw.get("max_active_names", 3),
            max_single_name_exposure_pct=raw.get("max_single_name_exposure_pct", 0.25),
            max_gross_exposure_pct=raw.get("max_gross_exposure_pct", 1.00),
            max_averaging_down_count=raw.get("max_averaging_down_count", 1),
            cost=cost,
            flatten=flatten,
            sizing=sizing,
        )


# ---------------------------------------------------------------------------
# Portfolio state (injected - this module never fetches or infers it)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionState:
    """One symbol's current position, as explicitly tracked by the caller.

    `notional` is a non-negative absolute exposure magnitude (not signed);
    `direction` carries the sign. A flat/no-position symbol is represented
    either by omission from PortfolioState.positions, or by an explicit
    PositionState with notional=0.0 and direction=Direction.NONE - both
    are treated identically by this module.

    `averaging_down_count` MUST be tracked explicitly by the caller's
    portfolio bookkeeping - this module never infers it from PnL.
    """

    symbol: str
    direction: Direction = Direction.NONE
    notional: float = 0.0
    averaging_down_count: int = 0


@dataclass(frozen=True)
class PortfolioState:
    """Current paper-portfolio state, entirely injected by the caller.

    `nav` is paper NAV. None is treated as "missing" and fails closed
    (see RiskConfig-driven gates in evaluate_risk()).
    """

    nav: Optional[float]
    positions: Tuple[PositionState, ...] = ()

    def position_for(self, symbol: str) -> Optional[PositionState]:
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    @property
    def active_positions(self) -> Tuple[PositionState, ...]:
        return tuple(p for p in self.positions if p.notional > 0 and p.direction != Direction.NONE)

    @property
    def gross_notional(self) -> float:
        return sum(p.notional for p in self.positions if p.notional > 0)


# ---------------------------------------------------------------------------
# Request / result types
# ---------------------------------------------------------------------------


class TradeAction(Enum):
    """What the caller is asking the risk engine to evaluate.

    Reduce/flatten must be represented EXPLICITLY by the caller (per the
    spec's requirement 7/8) - this module never infers "this must be a
    reduce" from comparing requested notional to current notional.
    """

    OPEN_OR_INCREASE = "OPEN_OR_INCREASE"
    REDUCE = "REDUCE"
    FLATTEN = "FLATTEN"


class RiskDecisionType(Enum):
    ALLOW = "ALLOW"
    REDUCE = "REDUCE"
    REJECT = "REJECT"
    FLATTEN = "FLATTEN"


@dataclass(frozen=True)
class RiskDecision:
    decision: RiskDecisionType
    symbol: str
    signal_type: Optional[SignalType]
    direction: Optional[Direction]

    reasons: Tuple[str, ...]

    requested_notional: Optional[float]
    approved_notional: float

    current_position_notional: float
    current_gross_exposure: float
    resulting_gross_exposure: float

    current_active_names: int
    resulting_active_names: int

    estimated_cost_bps: Optional[float]
    expected_edge_bps: Optional[float]

    time_to_cash_open: Optional[dt.timedelta]
    in_flatten_window: bool

    limits_applied: Dict[str, float]

    def to_dict(self) -> dict:
        """Deterministic, JSON-friendly representation for reporting/debugging."""
        return {
            "decision": self.decision.value,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value if self.signal_type else None,
            "direction": self.direction.value if self.direction else None,
            "reasons": list(self.reasons),
            "requested_notional": self.requested_notional,
            "approved_notional": self.approved_notional,
            "current_position_notional": self.current_position_notional,
            "current_gross_exposure": self.current_gross_exposure,
            "resulting_gross_exposure": self.resulting_gross_exposure,
            "current_active_names": self.current_active_names,
            "resulting_active_names": self.resulting_active_names,
            "estimated_cost_bps": self.estimated_cost_bps,
            "expected_edge_bps": self.expected_edge_bps,
            "time_to_cash_open_seconds": (
                self.time_to_cash_open.total_seconds() if self.time_to_cash_open is not None else None
            ),
            "in_flatten_window": self.in_flatten_window,
            "limits_applied": dict(self.limits_applied),
        }


# ---------------------------------------------------------------------------
# Reason string constants
# ---------------------------------------------------------------------------

REASON_MISSING_SIGNAL = "missing_signal"
REASON_MISSING_PORTFOLIO = "missing_portfolio_state"
REASON_MISSING_CONFIG = "missing_risk_config"
REASON_MISSING_OR_INVALID_NAV = "missing_or_invalid_nav"
REASON_INVALID_REQUESTED_NOTIONAL = "invalid_requested_notional"

REASON_SIGNAL_FLAT = "signal_flat_rejected"
REASON_SESSION_NOT_TRADEABLE = "session_not_tradeable"
REASON_MARK_QUALITY_FAILED = "mark_quality_failed"

REASON_MAX_ACTIVE_NAMES = "max_active_names_exceeded"
REASON_OPPOSITE_POSITION_LOCK = "opposite_position_lock"
REASON_AVERAGING_DOWN_LIMIT = "averaging_down_limit_reached"
REASON_NO_CAPACITY = "no_available_risk_capacity"

REASON_MISSING_SPREAD = "missing_spread_for_cost_model"
REASON_MISSING_EDGE = "missing_expected_edge_inputs"
REASON_COST_EXCEEDS_EDGE = "cost_exceeds_expected_edge"

REASON_MISSING_CALENDAR = "missing_calendar_cannot_evaluate_flatten_window"
REASON_FLATTEN_WINDOW = "flatten_window_blocks_new_position"

REASON_NO_POSITION_TO_REDUCE = "no_existing_position_to_reduce_or_flatten"
REASON_REDUCE_EXCEEDS_POSITION = "reduce_or_flatten_notional_exceeds_current_position"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _next_rth_open(now: dt.datetime, calendar: SessionCalendar) -> dt.datetime:
    """Next America/New_York RTH open at/after `now`, reusing the
    calendar's OWN configured rth_start/rth_weekdays - never re-derived
    or hardcoded here."""
    tz = ZoneInfo(calendar.config.timezone)
    local_now = now.astimezone(tz)
    candidate_date = local_now.date()
    for _ in range(14):  # generous bound; a weekend is at most ~3 days
        candidate = dt.datetime.combine(candidate_date, calendar.config.rth_start, tzinfo=tz)
        if candidate.weekday() in calendar.config.rth_weekdays and candidate >= local_now:
            return candidate
        candidate_date = candidate_date + dt.timedelta(days=1)
    raise RuntimeError("could not locate next RTH open within 14-day search bound")


def _compute_cost_bps(spread_bps: float, session: SessionType, cfg: CostConfig) -> float:
    """Round-trip cost estimate in bps.

    Documented model (per side, per the Phase 3 spec's wording):
        per_side_cost_bps = fee_bps_per_side
                           + half_spread_bps
                           + additional_slippage_bps_per_side
                           + (weekend_additional_slippage_bps_per_side if WEEKEND else 0)
    Round trip (enter + exit) doubles the per-side cost:
        total_cost_bps = 2 * per_side_cost_bps

    Never assumes a mid-price fill: half_spread_bps is derived from the
    observation's actual injected spread_bps, not a constant.
    """
    half_spread_bps = spread_bps / 2.0
    weekend_extra = cfg.weekend_additional_slippage_bps_per_side if session == SessionType.WEEKEND else 0.0
    per_side_cost_bps = cfg.fee_bps_per_side + half_spread_bps + cfg.additional_slippage_bps_per_side + weekend_extra
    return 2.0 * per_side_cost_bps


def _compute_expected_edge_bps(signal: SignalResult) -> Optional[float]:
    """Magnitude of the edge already implied by the approved signal.

    FADE: the dislocation itself is the expected reversion -> |basis_bps|.
    FOLLOW: the unclosed gap to the witness move is the expected catch-up
    -> |residual_bps|. See the module docstring's conceptual note - this
    is a simplification, flagged for review.
    """
    if signal.signal == SignalType.FADE:
        return abs(signal.basis_bps) if signal.basis_bps is not None else None
    if signal.signal == SignalType.FOLLOW:
        return abs(signal.residual_bps) if signal.residual_bps is not None else None
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_risk(
    signal: SignalResult,
    *,
    action: TradeAction,
    requested_notional: Optional[float],
    portfolio: PortfolioState,
    config: RiskConfig,
    calendar: Optional[SessionCalendar] = None,
    spread_bps: Optional[float] = None,
    is_averaging_down: bool = False,
) -> RiskDecision:
    """Evaluate one proposed trade against every configured risk rule.

    `spread_bps` should be the spread_bps from the SAME MarketObservation
    that produced `signal` - this module does not carry that field itself
    (SignalResult does not include spread), so it must be passed
    separately by the caller. Missing spread fails closed on the cost/
    edge gate for OPEN_OR_INCREASE (see REASON_MISSING_SPREAD); it is not
    required at all for REDUCE/FLATTEN (see module docstring).

    `calendar` is used only to compute the flatten-window / time-to-cash-
    open fields; if omitted, those fields are None and, for
    OPEN_OR_INCREASE, an explicit REASON_MISSING_CALENDAR failure is
    accumulated (fail closed) rather than silently skipping the check.
    """
    limits_applied = {
        "max_active_names": float(config.max_active_names) if config else None,
        "max_single_name_exposure_pct": config.max_single_name_exposure_pct if config else None,
        "max_gross_exposure_pct": config.max_gross_exposure_pct if config else None,
        "max_averaging_down_count": float(config.max_averaging_down_count) if config else None,
        "flatten_buffer_minutes": config.flatten.buffer_minutes if config else None,
    }

    # --- true prerequisites: without these, nothing else is computable ---
    if config is None:
        return RiskDecision(
            decision=RiskDecisionType.REJECT,
            symbol="UNKNOWN",
            signal_type=None,
            direction=None,
            reasons=(REASON_MISSING_CONFIG,),
            requested_notional=requested_notional,
            approved_notional=0.0,
            current_position_notional=0.0,
            current_gross_exposure=0.0,
            resulting_gross_exposure=0.0,
            current_active_names=0,
            resulting_active_names=0,
            estimated_cost_bps=None,
            expected_edge_bps=None,
            time_to_cash_open=None,
            in_flatten_window=False,
            limits_applied={},
        )

    if signal is None:
        return RiskDecision(
            decision=RiskDecisionType.REJECT,
            symbol="UNKNOWN",
            signal_type=None,
            direction=None,
            reasons=(REASON_MISSING_SIGNAL,),
            requested_notional=requested_notional,
            approved_notional=0.0,
            current_position_notional=0.0,
            current_gross_exposure=0.0,
            resulting_gross_exposure=0.0,
            current_active_names=0,
            resulting_active_names=0,
            estimated_cost_bps=None,
            expected_edge_bps=None,
            time_to_cash_open=None,
            in_flatten_window=False,
            limits_applied=limits_applied,
        )

    if portfolio is None:
        return RiskDecision(
            decision=RiskDecisionType.REJECT,
            symbol=signal.symbol,
            signal_type=signal.signal,
            direction=signal.direction,
            reasons=(REASON_MISSING_PORTFOLIO,),
            requested_notional=requested_notional,
            approved_notional=0.0,
            current_position_notional=0.0,
            current_gross_exposure=0.0,
            resulting_gross_exposure=0.0,
            current_active_names=0,
            resulting_active_names=0,
            estimated_cost_bps=None,
            expected_edge_bps=None,
            time_to_cash_open=None,
            in_flatten_window=False,
            limits_applied=limits_applied,
        )

    reasons: list = []

    existing = portfolio.position_for(signal.symbol)
    current_position_notional = existing.notional if existing else 0.0
    current_gross_exposure = portfolio.gross_notional
    current_active_names = len(portfolio.active_positions)

    is_existing_active = existing is not None and existing.notional > 0 and existing.direction != Direction.NONE

    # --- requested notional validity (applies to every action) ---
    if requested_notional is None or requested_notional <= 0:
        reasons.append(REASON_INVALID_REQUESTED_NOTIONAL)

    # --- NAV validity (applies to every action; graceful degrade, not a hard exit) ---
    nav_valid = portfolio.nav is not None and portfolio.nav > 0
    if not nav_valid:
        reasons.append(REASON_MISSING_OR_INVALID_NAV)

    # --- flatten-window / time-to-cash-open (informational for all actions;
    #     only blocks OPEN_OR_INCREASE) ---
    time_to_cash_open: Optional[dt.timedelta] = None
    in_flatten_window = False
    if calendar is not None:
        next_open = _next_rth_open(signal.timestamp, calendar)
        time_to_cash_open = next_open - signal.timestamp.astimezone(ZoneInfo(calendar.config.timezone))
        buffer = dt.timedelta(minutes=config.flatten.buffer_minutes)
        in_flatten_window = time_to_cash_open <= buffer
    elif action == TradeAction.OPEN_OR_INCREASE:
        reasons.append(REASON_MISSING_CALENDAR)

    estimated_cost_bps: Optional[float] = None
    expected_edge_bps: Optional[float] = None
    sized_notional = 0.0
    resulting_active_names = current_active_names
    resulting_gross_exposure = current_gross_exposure

    if action == TradeAction.OPEN_OR_INCREASE:
        # 1. Signal gate
        if signal.signal == SignalType.FLAT:
            reasons.append(REASON_SIGNAL_FLAT)

        # 2. Session gate (reuses Phase 1 is_tradeable() unmodified)
        if not is_tradeable(signal.session):
            reasons.append(f"{REASON_SESSION_NOT_TRADEABLE}:{signal.session.value}")

        # 3. Mark-quality gate (reads the SignalResult's own embedded
        #    MarkQualityResult data - never recreated here)
        if not signal.mark_quality_passed:
            reasons.append(REASON_MARK_QUALITY_FAILED)

        # 4. Max active names (only a truly new, not-yet-active symbol counts)
        is_new_name = not is_existing_active
        if is_new_name:
            resulting_active_names = current_active_names + 1
            if resulting_active_names > config.max_active_names:
                reasons.append(REASON_MAX_ACTIVE_NAMES)

        # 7. Opposite position lock
        if is_existing_active and signal.direction != existing.direction:
            reasons.append(REASON_OPPOSITE_POSITION_LOCK)

        # 8. Averaging-down limit
        if is_averaging_down:
            existing_avg_count = existing.averaging_down_count if existing else 0
            if existing_avg_count >= config.max_averaging_down_count:
                reasons.append(REASON_AVERAGING_DOWN_LIMIT)

        # 5/6. Sizing: single-name + gross capacity (a sizing constraint,
        # not an outright reject, unless capacity is exhausted)
        if nav_valid:
            single_name_limit_notional = portfolio.nav * config.max_single_name_exposure_pct
            gross_limit_notional = portfolio.nav * config.max_gross_exposure_pct
            available_single_name_capacity = max(0.0, single_name_limit_notional - current_position_notional)
            available_gross_capacity = max(0.0, gross_limit_notional - current_gross_exposure)
            raw_capacity = min(available_single_name_capacity, available_gross_capacity)
            capacity_after_scaling = raw_capacity * config.sizing.max_notional_scaling_factor
            if requested_notional is not None and requested_notional > 0:
                sized_notional = max(0.0, min(requested_notional, capacity_after_scaling))
                if sized_notional <= 0:
                    reasons.append(REASON_NO_CAPACITY)

        # 9. Cost / edge gate
        expected_edge_bps = _compute_expected_edge_bps(signal)
        if spread_bps is None:
            reasons.append(REASON_MISSING_SPREAD)
        else:
            estimated_cost_bps = _compute_cost_bps(spread_bps, signal.session, config.cost)

        if expected_edge_bps is None:
            reasons.append(REASON_MISSING_EDGE)
        elif estimated_cost_bps is not None and expected_edge_bps < estimated_cost_bps:
            reasons.append(REASON_COST_EXCEEDS_EDGE)

        # 10. Flatten window (already computed above; apply the gate here,
        # respecting the explicit, configurable FOLLOW exception)
        if in_flatten_window:
            follow_exempt = signal.signal == SignalType.FOLLOW and config.flatten.follow_exception_during_flatten
            if not follow_exempt:
                reasons.append(REASON_FLATTEN_WINDOW)

        resulting_gross_exposure = current_gross_exposure + sized_notional

        if reasons:
            sized_notional = 0.0
            resulting_gross_exposure = current_gross_exposure
            resulting_active_names = current_active_names
            decision_type = RiskDecisionType.REJECT
        else:
            decision_type = RiskDecisionType.ALLOW

    elif action in (TradeAction.REDUCE, TradeAction.FLATTEN):
        # Safety-valve actions: only NAV/portfolio bookkeeping validity is
        # enforced - see module docstring for the rationale.
        if not is_existing_active:
            reasons.append(REASON_NO_POSITION_TO_REDUCE)
        elif requested_notional is not None and requested_notional > 0:
            if requested_notional > existing.notional:
                reasons.append(REASON_REDUCE_EXCEEDS_POSITION)
            else:
                sized_notional = requested_notional

        if not reasons and is_existing_active:
            remaining = existing.notional - sized_notional
            resulting_gross_exposure = current_gross_exposure - sized_notional
            if remaining <= 0:
                resulting_active_names = max(0, current_active_names - 1)

        if reasons:
            sized_notional = 0.0
            resulting_gross_exposure = current_gross_exposure
            decision_type = RiskDecisionType.REJECT
        else:
            decision_type = RiskDecisionType.REDUCE if action == TradeAction.REDUCE else RiskDecisionType.FLATTEN

    else:
        reasons.append(f"unknown_trade_action:{action}")
        decision_type = RiskDecisionType.REJECT

    return RiskDecision(
        decision=decision_type,
        symbol=signal.symbol,
        signal_type=signal.signal,
        direction=signal.direction,
        reasons=tuple(reasons),
        requested_notional=requested_notional,
        approved_notional=sized_notional,
        current_position_notional=current_position_notional,
        current_gross_exposure=current_gross_exposure,
        resulting_gross_exposure=resulting_gross_exposure,
        current_active_names=current_active_names,
        resulting_active_names=resulting_active_names,
        estimated_cost_bps=estimated_cost_bps,
        expected_edge_bps=expected_edge_bps,
        time_to_cash_open=time_to_cash_open,
        in_flatten_window=in_flatten_window,
        limits_applied=limits_applied,
    )
