"""
Nightmark Phase 3.5 authorization façade.

`authorize_open_or_increase()` is the sanctioned OPEN authorization
path. Direct calls to `evaluate_signal()` and `evaluate_risk()`
remain public for tests, inspection, and REDUCE/FLATTEN exits, but they
are not OPEN authorization.

`evaluate_risk()` is the lower-level sizing/limit engine. Even when it
returns `RiskDecision.decision == ALLOW` for OPEN_OR_INCREASE,
`RiskDecision.authorizes_open` is always False. The only OPEN
authorization boolean is `AuthorizationResult.allowed`.

This module does not re-implement engine rules. It sequences existing
gates and fails closed if a required input is missing, so a caller cannot
skip session validation, weekend eligibility, mark quality, the signal
engine, or the risk engine by constructing a SignalResult by hand.

Pipeline composed here:

    timestamp
      -> SessionCalendar.classify()          [authoritative session]
      -> weekend eligibility when WEEKEND    [src.eligibility]
      -> mark-quality result must have passed
      -> evaluate_signal()                   [FADE/FOLLOW/FLAT]
      -> evaluate_risk(OPEN_OR_INCREASE)     [limits, sizing, cost/edge,
                                              flatten window, non-finite
                                              input rejection, session
                                              revalidation, eligibility]

Spread used for the cost/edge gate is always `observation.spread_bps`
(the observation that produced the signal). There is no spread override
on this façade.

REDUCE/FLATTEN are not handled here; they remain `evaluate_risk()` calls
because they are exits, not opens, and the approved Phase 3 bypass still
applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.calendar import SessionCalendar
from src.eligibility import SymbolEligibility
from src.mark_quality import MarkQualityResult
from src.observation import MarketObservation
from src.risk import (
    PortfolioState,
    RiskConfig,
    RiskDecision,
    RiskDecisionType,
    TradeAction,
    evaluate_risk,
)
from src.signal import SignalConfig, SignalResult, SignalType, evaluate_signal
from src.witness import WitnessObservation
from src.zscore import ZScoreResult

REASON_MISSING_OBSERVATION = "missing_observation"
REASON_MISSING_MARK_QUALITY = "missing_mark_quality"
REASON_MISSING_ZSCORE = "missing_zscore"
REASON_MISSING_SIGNAL_CONFIG = "missing_signal_config"
REASON_MISSING_RISK_CONFIG = "missing_risk_config"
REASON_MISSING_PORTFOLIO = "missing_portfolio_state"
REASON_MISSING_CALENDAR = "missing_calendar"
REASON_OBSERVATION_SESSION_MISMATCH = "observation_session_timestamp_mismatch"
REASON_MARK_QUALITY_FAILED = "mark_quality_failed"
REASON_SIGNAL_NOT_ACTIONABLE = "signal_not_actionable"


@dataclass(frozen=True)
class AuthorizationResult:
    """Outcome of authorize_open_or_increase().

    `allowed` is True only when every composed gate passed and the risk
    engine returned ALLOW with a positive approved notional. `reasons`
    concatenates façade-level failures with any risk-engine reasons.
    This is the only sanctioned OPEN authorization boolean.
    `risk.authorizes_open` remains False even when `allowed` is True:
    that field is never set by `evaluate_risk()`.
    """

    allowed: bool
    symbol: Optional[str]
    reasons: Tuple[str, ...]
    signal: Optional[SignalResult]
    risk: Optional[RiskDecision]
    approved_notional: float

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "symbol": self.symbol,
            "reasons": list(self.reasons),
            "approved_notional": self.approved_notional,
            "signal": self.signal.to_dict() if self.signal is not None else None,
            "risk": self.risk.to_dict() if self.risk is not None else None,
        }


def _reject(
    *,
    reasons: Tuple[str, ...],
    symbol: Optional[str] = None,
    signal: Optional[SignalResult] = None,
    risk: Optional[RiskDecision] = None,
) -> AuthorizationResult:
    return AuthorizationResult(
        allowed=False,
        symbol=symbol,
        reasons=reasons,
        signal=signal,
        risk=risk,
        approved_notional=0.0,
    )


def authorize_open_or_increase(
    observation: MarketObservation,
    *,
    mark_quality: MarkQualityResult,
    zscore: ZScoreResult,
    witness: Optional[WitnessObservation],
    signal_config: SignalConfig,
    requested_notional: Optional[float],
    portfolio: PortfolioState,
    risk_config: RiskConfig,
    calendar: SessionCalendar,
    eligibility: Optional[SymbolEligibility] = None,
    is_averaging_down: bool = False,
) -> AuthorizationResult:
    """Authorize an OPEN_OR_INCREASE. Fail closed on missing inputs.

    Does not accept a pre-built SignalResult: the signal is always
    produced here from the observation, mark-quality result, z-score, and
    witness. Does not accept a spread override: cost/edge uses
    observation.spread_bps.

    `eligibility` defaults to the shipped (empty) symbol list, so weekend
    opens fail closed until config/symbols.yaml is populated from an
    authoritative Bitget source.
    """
    reasons = []
    symbol = getattr(observation, "symbol", None) if observation is not None else None

    if observation is None:
        reasons.append(REASON_MISSING_OBSERVATION)
    if mark_quality is None:
        reasons.append(REASON_MISSING_MARK_QUALITY)
    if zscore is None:
        reasons.append(REASON_MISSING_ZSCORE)
    if signal_config is None:
        reasons.append(REASON_MISSING_SIGNAL_CONFIG)
    if risk_config is None:
        reasons.append(REASON_MISSING_RISK_CONFIG)
    if portfolio is None:
        reasons.append(REASON_MISSING_PORTFOLIO)
    if calendar is None:
        reasons.append(REASON_MISSING_CALENDAR)

    if reasons:
        return _reject(reasons=tuple(reasons), symbol=symbol)

    elig = eligibility if eligibility is not None else SymbolEligibility()

    authoritative_session = calendar.classify(observation.timestamp)
    if observation.session != authoritative_session:
        reasons.append(
            f"{REASON_OBSERVATION_SESSION_MISMATCH}:"
            f"{observation.session.value}!={authoritative_session.value}"
        )

    if not mark_quality.passed:
        reasons.append(REASON_MARK_QUALITY_FAILED)

    signal = evaluate_signal(observation, mark_quality, zscore, witness, signal_config)

    if signal.signal == SignalType.FLAT:
        reasons.append(f"{REASON_SIGNAL_NOT_ACTIONABLE}:{signal.reason}")

    risk = evaluate_risk(
        signal,
        action=TradeAction.OPEN_OR_INCREASE,
        requested_notional=requested_notional,
        portfolio=portfolio,
        config=risk_config,
        calendar=calendar,
        spread_bps=observation.spread_bps,
        is_averaging_down=is_averaging_down,
        eligibility=elig,
    )

    combined_reasons = []
    seen = set()
    for r in tuple(reasons) + risk.reasons:
        if r not in seen:
            seen.add(r)
            combined_reasons.append(r)

    if (
        not combined_reasons
        and risk.decision == RiskDecisionType.ALLOW
        and risk.approved_notional > 0
        and signal.signal in (SignalType.FADE, SignalType.FOLLOW)
    ):
        return AuthorizationResult(
            allowed=True,
            symbol=signal.symbol,
            reasons=(),
            signal=signal,
            risk=risk,
            approved_notional=risk.approved_notional,
        )

    return AuthorizationResult(
        allowed=False,
        symbol=signal.symbol,
        reasons=tuple(combined_reasons) if combined_reasons else (REASON_SIGNAL_NOT_ACTIONABLE,),
        signal=signal,
        risk=risk,
        approved_notional=0.0,
    )
