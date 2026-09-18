#!/usr/bin/env python3
"""Independent Phase 3.5 follow-up verifier.

Not collected by unittest (lives outside tests/). Hand-built inputs;
does not reuse tests/fixtures.py.

Run:  python3 audit/phase35_verifier.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.authorize import authorize_open_or_increase
from src.calendar import HaltWindow, SessionCalendar, SessionType
from src.eligibility import SymbolEligibility, SymbolEligibilityConfig
from src.mark_quality import MarkQualityResult
from src.observation import MarketObservation, ReferenceQuality, ReferenceSource
from src.risk import (
    PortfolioState,
    PositionState,
    RiskConfig,
    RiskDecisionType,
    TradeAction,
    evaluate_risk,
)
from src.signal import Direction, SignalConfig, SignalResult, SignalType
from src.witness import WitnessStatus
from src.zscore import ZScoreResult

ET = ZoneInfo("America/New_York")

PASS = 0
FAIL = 0
FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def et(y, m, d, h, mi, s=0):
    return dt.datetime(y, m, d, h, mi, s, tzinfo=ET)


# Dates off the repo's usual 2026-09-15 WEEKEND cluster.
TUE_RTH = et(2026, 10, 6, 12, 0)
TUE_OVERNIGHT = et(2026, 10, 6, 22, 0)
SAT_WEEKEND = et(2026, 10, 10, 14, 0)
HALT_TS = et(2026, 10, 6, 10, 5)
UNKNOWN_TS = et(2026, 3, 8, 2, 30)  # DST spring-forward gap

CAL = SessionCalendar()
HALT_CAL = SessionCalendar(
    extra_halts=(HaltWindow(start=et(2026, 10, 6, 10, 0), end=et(2026, 10, 6, 10, 15)),)
)
RISK_CFG = RiskConfig()
SIG_CFG = SignalConfig()


def sig(
    *,
    timestamp,
    session,
    symbol="RXYZ",
    signal=SignalType.FADE,
    direction=Direction.SHORT,
    basis_bps=80.0,
    residual_bps=None,
    mark_quality_passed=True,
):
    return SignalResult(
        signal=signal,
        direction=direction,
        symbol=symbol,
        timestamp=timestamp,
        session=session,
        basis=basis_bps,
        basis_bps=basis_bps,
        basis_zscore=3.0,
        reason="verifier",
        mark_quality_passed=mark_quality_passed,
        mark_quality_reasons=(),
        witness_status=WitnessStatus.NO_WITNESS,
        witness_move_bps=None,
        residual_bps=residual_bps,
        config_used={},
    )


def obs(*, timestamp, session, symbol="RXYZ", basis_bps=80.0, spread_bps=8.0):
    ref = 100.0
    mid = ref * (1.0 + basis_bps / 10_000.0)
    half = (spread_bps / 10_000.0) * mid / 2.0
    return MarketObservation(
        timestamp=timestamp,
        symbol=symbol,
        session=session,
        rtoken_bid=mid - half,
        rtoken_ask=mid + half,
        rtoken_mid=mid,
        spread_bps=spread_bps,
        available_depth=500.0,
        reference_price=ref,
        reference_source=ReferenceSource.US_CASH_CLOSE,
        reference_quality=ReferenceQuality.OFFICIAL,
        basis=mid - ref,
        basis_bps=basis_bps,
        is_valid=True,
    )


def mq_pass():
    return MarkQualityResult(passed=True, reasons=())


def zs(v=2.0):
    return ZScoreResult(value=v, reason=None, observations_used=6)


EMPTY = SymbolEligibility(config=SymbolEligibilityConfig())
LISTED = SymbolEligibility(config=SymbolEligibilityConfig(eligible_symbols=frozenset({"RXYZ", "RNVDA"})))
PORT = PortfolioState(nav=80_000.0, positions=())


def open_risk(signal, **kw):
    kwargs = dict(
        action=TradeAction.OPEN_OR_INCREASE,
        requested_notional=1_000.0,
        portfolio=PORT,
        config=RISK_CFG,
        calendar=CAL,
        spread_bps=8.0,
        eligibility=LISTED,
    )
    kwargs.update(kw)
    return evaluate_risk(signal, **kwargs)


def facade(**kw):
    kwargs = dict(
        observation=obs(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT),
        mark_quality=mq_pass(),
        zscore=zs(),
        witness=None,
        signal_config=SIG_CFG,
        requested_notional=1_000.0,
        portfolio=PORT,
        risk_config=RISK_CFG,
        calendar=CAL,
        eligibility=EMPTY,
    )
    kwargs.update(kw)
    return authorize_open_or_increase(**kwargs)


def main() -> int:
    print("=== authorizes_open boundary ===")
    r = open_risk(sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT))
    check("direct evaluate_risk OPEN can ALLOW", r.decision == RiskDecisionType.ALLOW, str(r.reasons))
    check("direct ALLOW has authorizes_open False", r.authorizes_open is False, str(r.authorizes_open))
    ok = facade()
    check("façade can allow OPEN", ok.allowed is True and ok.approved_notional > 0, str(ok.reasons))
    check(
        "façade allowed still has risk.authorizes_open False",
        ok.risk is not None and ok.risk.authorizes_open is False,
        str(getattr(ok.risk, "authorizes_open", None)),
    )

    print("=== malformed sibling vs clean exit ===")
    mixed_nan = PortfolioState(
        nav=80_000.0,
        positions=(
            PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0),
            PositionState(symbol="RTSLA", direction=Direction.SHORT, notional=float("nan")),
        ),
    )
    r = evaluate_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RNVDA", direction=Direction.LONG),
        action=TradeAction.REDUCE,
        requested_notional=500.0,
        portfolio=mixed_nan,
        config=RISK_CFG,
        calendar=CAL,
    )
    check("REDUCE clean RNVDA despite NaN RTSLA", r.decision == RiskDecisionType.REDUCE, str(r.reasons))
    r = evaluate_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RNVDA", direction=Direction.LONG),
        action=TradeAction.FLATTEN,
        requested_notional=1_000.0,
        portfolio=mixed_nan,
        config=RISK_CFG,
        calendar=CAL,
    )
    check("FLATTEN clean RNVDA despite NaN RTSLA", r.decision == RiskDecisionType.FLATTEN, str(r.reasons))
    r = open_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RXYZ"),
        portfolio=mixed_nan,
    )
    check("OPEN still rejected when sibling is NaN", r.decision == RiskDecisionType.REJECT and "non_finite_position_notional" in r.reasons, str(r.reasons))
    r = evaluate_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RTSLA", direction=Direction.SHORT),
        action=TradeAction.REDUCE,
        requested_notional=500.0,
        portfolio=mixed_nan,
        config=RISK_CFG,
        calendar=CAL,
    )
    check("REDUCE of NaN target still rejected", r.decision == RiskDecisionType.REJECT and "non_finite_position_notional" in r.reasons, str(r.reasons))

    mixed_neg = PortfolioState(
        nav=80_000.0,
        positions=(
            PositionState(symbol="RNVDA", direction=Direction.LONG, notional=1_000.0),
            PositionState(symbol="RTSLA", direction=Direction.SHORT, notional=-5_000.0),
        ),
    )
    r = evaluate_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RNVDA", direction=Direction.LONG),
        action=TradeAction.FLATTEN,
        requested_notional=1_000.0,
        portfolio=mixed_neg,
        config=RISK_CFG,
        calendar=CAL,
    )
    check("FLATTEN clean RNVDA despite negative RTSLA", r.decision == RiskDecisionType.FLATTEN, str(r.reasons))
    r = open_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RXYZ"),
        portfolio=mixed_neg,
    )
    check("OPEN still rejected when sibling is negative", r.decision == RiskDecisionType.REJECT and "non_finite_position_notional" in r.reasons, str(r.reasons))
    r = evaluate_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RTSLA", direction=Direction.SHORT),
        action=TradeAction.REDUCE,
        requested_notional=500.0,
        portfolio=mixed_neg,
        config=RISK_CFG,
        calendar=CAL,
    )
    check("REDUCE of negative target still rejected", r.decision == RiskDecisionType.REJECT and "non_finite_position_notional" in r.reasons, str(r.reasons))

    print("=== NAV is OPEN-only ===")
    held = PortfolioState(
        nav=None,
        positions=(PositionState(symbol="RXYZ", direction=Direction.LONG, notional=1_000.0),),
    )
    r = evaluate_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RXYZ", direction=Direction.LONG),
        action=TradeAction.REDUCE,
        requested_notional=500.0,
        portfolio=held,
        config=RISK_CFG,
        calendar=CAL,
    )
    check("REDUCE allowed with nav=None", r.decision == RiskDecisionType.REDUCE, str(r.reasons))
    r = evaluate_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RXYZ", direction=Direction.LONG),
        action=TradeAction.FLATTEN,
        requested_notional=1_000.0,
        portfolio=PortfolioState(
            nav=float("nan"),
            positions=(PositionState(symbol="RXYZ", direction=Direction.LONG, notional=1_000.0),),
        ),
        config=RISK_CFG,
        calendar=CAL,
    )
    check("FLATTEN allowed with nav=NaN", r.decision == RiskDecisionType.FLATTEN, str(r.reasons))
    r = evaluate_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RXYZ", direction=Direction.LONG),
        action=TradeAction.REDUCE,
        requested_notional=500.0,
        portfolio=PortfolioState(
            nav=float("inf"),
            positions=(PositionState(symbol="RXYZ", direction=Direction.LONG, notional=1_000.0),),
        ),
        config=RISK_CFG,
        calendar=CAL,
    )
    check("REDUCE allowed with nav=+inf", r.decision == RiskDecisionType.REDUCE, str(r.reasons))
    r = open_risk(sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT), portfolio=PortfolioState(nav=None, positions=()))
    check("OPEN rejected with nav=None", r.decision == RiskDecisionType.REJECT and "missing_or_invalid_nav" in r.reasons, str(r.reasons))
    r = open_risk(sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT), portfolio=PortfolioState(nav=float("nan"), positions=()))
    check("OPEN rejected with nav=NaN", r.decision == RiskDecisionType.REJECT and "non_finite_nav" in r.reasons, str(r.reasons))
    r = open_risk(sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT), portfolio=PortfolioState(nav=float("inf"), positions=()))
    check("OPEN rejected with nav=+inf", r.decision == RiskDecisionType.REJECT and "non_finite_nav" in r.reasons, str(r.reasons))

    print("=== retained OPEN hardening ===")
    r = open_risk(sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, direction=Direction.NONE, basis_bps=80.0))
    check("Direction.NONE FADE cannot OPEN", r.decision == RiskDecisionType.REJECT and "invalid_signal_direction_for_open" in r.reasons, str(r.reasons))
    r = open_risk(sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT), spread_bps=-1.0)
    check("negative spread cannot OPEN", r.decision == RiskDecisionType.REJECT and "invalid_spread" in r.reasons, str(r.reasons))
    r = open_risk(
        sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT, symbol="RXYZ"),
        portfolio=PortfolioState(
            nav=80_000.0,
            positions=(PositionState(symbol="RXYZ", direction=Direction.SHORT, notional=-1_000.0),),
        ),
    )
    check("negative target notional cannot OPEN", r.decision == RiskDecisionType.REJECT and "non_finite_position_notional" in r.reasons, str(r.reasons))

    print("=== actual HALT / UNKNOWN ===")
    check("injected halt window classifies HALT", HALT_CAL.classify(HALT_TS) == SessionType.HALT, str(HALT_CAL.classify(HALT_TS)))
    check("DST-gap classifies UNKNOWN", CAL.classify(UNKNOWN_TS) == SessionType.UNKNOWN, str(CAL.classify(UNKNOWN_TS)))
    r = open_risk(
        sig(timestamp=HALT_TS, session=SessionType.HALT),
        calendar=HALT_CAL,
    )
    check(
        "HALT OPEN rejects as session_not_tradeable:HALT",
        r.decision == RiskDecisionType.REJECT
        and any("session_not_tradeable:HALT" in x for x in r.reasons)
        and not any("session_timestamp_mismatch" in x for x in r.reasons),
        str(r.reasons),
    )
    r = open_risk(sig(timestamp=UNKNOWN_TS, session=SessionType.UNKNOWN))
    check(
        "UNKNOWN OPEN rejects as session_not_tradeable:UNKNOWN",
        r.decision == RiskDecisionType.REJECT
        and any("session_not_tradeable:UNKNOWN" in x for x in r.reasons)
        and not any("session_timestamp_mismatch" in x for x in r.reasons),
        str(r.reasons),
    )
    held_halt = PortfolioState(
        nav=80_000.0,
        positions=(PositionState(symbol="RXYZ", direction=Direction.LONG, notional=1_000.0),),
    )
    r = evaluate_risk(
        sig(timestamp=HALT_TS, session=SessionType.HALT, symbol="RXYZ", direction=Direction.LONG),
        action=TradeAction.REDUCE,
        requested_notional=500.0,
        portfolio=held_halt,
        config=RISK_CFG,
        calendar=HALT_CAL,
    )
    check("REDUCE allowed during actual HALT", r.decision == RiskDecisionType.REDUCE, str(r.reasons))
    r = evaluate_risk(
        sig(timestamp=HALT_TS, session=SessionType.HALT, symbol="RXYZ", direction=Direction.LONG),
        action=TradeAction.FLATTEN,
        requested_notional=1_000.0,
        portfolio=held_halt,
        config=RISK_CFG,
        calendar=HALT_CAL,
    )
    check("FLATTEN allowed during actual HALT", r.decision == RiskDecisionType.FLATTEN, str(r.reasons))
    r = evaluate_risk(
        sig(timestamp=UNKNOWN_TS, session=SessionType.UNKNOWN, symbol="RXYZ", direction=Direction.LONG),
        action=TradeAction.FLATTEN,
        requested_notional=1_000.0,
        portfolio=held_halt,
        config=RISK_CFG,
        calendar=CAL,
    )
    check("FLATTEN allowed during actual UNKNOWN", r.decision == RiskDecisionType.FLATTEN, str(r.reasons))

    print("=== retained 3.5 fail-closed (sanity) ===")
    r = open_risk(sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT), spread_bps=float("nan"))
    check("NaN spread still rejects OPEN", r.decision == RiskDecisionType.REJECT and "non_finite_spread" in r.reasons)
    r = open_risk(sig(timestamp=TUE_RTH, session=SessionType.OVERNIGHT))
    check(
        "claimed OVERNIGHT during RTH still cannot OPEN",
        r.decision == RiskDecisionType.REJECT and any("session_timestamp_mismatch" in x for x in r.reasons),
        str(r.reasons),
    )
    r = open_risk(sig(timestamp=SAT_WEEKEND, session=SessionType.WEEKEND, symbol="RXYZ"), eligibility=EMPTY)
    check("empty weekend eligibility still fail-closed", r.decision == RiskDecisionType.REJECT and "weekend_symbol_not_eligible" in r.reasons)

    print("=== Determinism ===")
    a = facade()
    b = facade()
    check("façade identical inputs identical result", a == b)
    s = sig(timestamp=TUE_OVERNIGHT, session=SessionType.OVERNIGHT)
    check("evaluate_risk deterministic", open_risk(s) == open_risk(s))
    a2 = facade()
    b2 = facade()
    check("second façade pair still identical", a2 == b2 and a2 == a)

    print()
    print(f"INDEPENDENT VERIFIER: {PASS}/{PASS + FAIL} passed ({FAIL} failed)")
    if FAILURES:
        print("Failed:")
        for n, d in FAILURES:
            print(f"  - {n}: {d}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
