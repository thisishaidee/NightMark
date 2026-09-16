"""
Nightmark Phase 2C signal engine.

Answers exactly one question per observation: given everything known at
this single timestamp, should Nightmark FADE, FOLLOW, or stay FLAT? This
module makes the deterministic signal DECISION only - it does not size a
position, place an order, or touch a broker/exchange. Nothing here calls
the network or fabricates data.

RULES FIRST. Every threshold this module compares against comes from an
injected SignalConfig (see config/signal.yaml) - none are hardcoded here.

FAIL-CLOSED DEFAULT: the default signal is always FLAT. A signal can only
become FADE or FOLLOW when every required condition is explicitly met.
Mark-quality failure hard-blocks both FADE and FOLLOW unconditionally.

Two signals:

    SIGNAL A - FADE (unexplained dislocation):
        Fade an rToken whose basis has moved materially away from its
        reference without a qualifying witness-market move to explain it.
        Direction: positive/rich basis -> SHORT; negative/cheap -> LONG.

    SIGNAL B - FOLLOW (explained residual):
        Follow an rToken that is lagging a meaningful, already-qualifying
        24/7 witness-market move. The signal depends on an actual
        measurable residual - the portion of the witness's move NOT yet
        reflected in the rToken's basis - not merely "the witness moved."
        Direction: matches the witness move's sign (the direction the
        rToken still needs to move to catch up).

If both FADE and FOLLOW would independently qualify on the same
observation, this is treated as an ambiguous/conflicting read and the
result is FLAT with an explicit conflict reason - never an arbitrary
tie-break.

LOOK-AHEAD: this module is purely a function of its inputs for a single
timestamp - it holds no history and reaches into no other observation.
The z-score it consumes (a `ZScoreResult` from src/zscore.py) was already
computed with Phase 2A's no-look-ahead guarantee; as long as callers pass
the ZScoreResult that belongs to the observation being evaluated (not a
future one), the composition is look-ahead-free by construction.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from src.calendar import SessionType, is_tradeable
from src.mark_quality import MarkQualityResult
from src.observation import MarketObservation
from src.witness import WitnessObservation, WitnessStatus, classify_witness
from src.zscore import ZScoreResult

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "signal.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FadeConfig:
    min_abs_basis_bps: float = 20.0
    entry_abs_zscore: float = 1.5


@dataclass(frozen=True)
class FollowConfig:
    min_witness_move_bps: float = 60.0
    min_residual_bps: float = 15.0


@dataclass(frozen=True)
class SignalConfig:
    fade: FadeConfig = field(default_factory=FadeConfig)
    follow: FollowConfig = field(default_factory=FollowConfig)

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH) -> "SignalConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        fade_raw = raw.get("fade", {}) or {}
        follow_raw = raw.get("follow", {}) or {}
        return cls(
            fade=FadeConfig(
                min_abs_basis_bps=fade_raw.get("min_abs_basis_bps", 20.0),
                entry_abs_zscore=fade_raw.get("entry_abs_zscore", 1.5),
            ),
            follow=FollowConfig(
                min_witness_move_bps=follow_raw.get("min_witness_move_bps", 60.0),
                min_residual_bps=follow_raw.get("min_residual_bps", 15.0),
            ),
        )


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class SignalType(Enum):
    FADE = "FADE"
    FOLLOW = "FOLLOW"
    FLAT = "FLAT"


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass(frozen=True)
class SignalResult:
    signal: SignalType
    direction: Direction
    symbol: str
    timestamp: dt.datetime
    session: SessionType

    basis: Optional[float]
    basis_bps: Optional[float]
    basis_zscore: Optional[float]

    reason: str

    mark_quality_passed: bool
    mark_quality_reasons: Tuple[str, ...]

    witness_status: WitnessStatus
    witness_move_bps: Optional[float]

    residual_bps: Optional[float]

    config_used: Dict[str, float]

    def to_dict(self) -> dict:
        """Deterministic, JSON-friendly representation for later risk/execution layers."""
        return {
            "signal": self.signal.value,
            "direction": self.direction.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "session": self.session.value,
            "basis": self.basis,
            "basis_bps": self.basis_bps,
            "basis_zscore": self.basis_zscore,
            "reason": self.reason,
            "mark_quality_passed": self.mark_quality_passed,
            "mark_quality_reasons": list(self.mark_quality_reasons),
            "witness_status": self.witness_status.value,
            "witness_move_bps": self.witness_move_bps,
            "residual_bps": self.residual_bps,
            "config_used": dict(self.config_used),
        }


# ---------------------------------------------------------------------------
# Reason string constants (kept as constants so tests can assert on them
# without hardcoding literals in two places)
# ---------------------------------------------------------------------------

REASON_OBSERVATION_INVALID = "observation_invalid"
REASON_SESSION_NOT_TRADEABLE = "session_not_tradeable"
REASON_MARK_QUALITY_FAILED = "mark_quality_failed"
REASON_MISSING_REQUIRED_INPUTS = "missing_required_inputs"
REASON_CONFLICT = "conflict_fade_and_follow_both_qualify"
REASON_FADE_QUALIFIES = "fade_qualifies"
REASON_FOLLOW_QUALIFIES = "follow_qualifies"

REASON_FADE_BASIS_BELOW_MIN = "fade_basis_below_min"
REASON_FADE_ZSCORE_BELOW_MIN = "fade_zscore_below_min"
REASON_FADE_DISQUALIFIED_BY_WITNESS = "fade_disqualified_by_qualifying_witness"

REASON_FOLLOW_NO_QUALIFYING_WITNESS = "follow_no_qualifying_witness"
REASON_FOLLOW_RESIDUAL_INSUFFICIENT = "follow_residual_insufficient"
REASON_FOLLOW_DIRECTION_MISMATCH = "follow_direction_mismatch"


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Individual signal evaluations
# ---------------------------------------------------------------------------


def _fade_core_conditions(
    basis_bps: float,
    zscore: float,
    cfg: FadeConfig,
) -> Tuple[bool, Optional[Direction], list]:
    """Evaluate FADE's magnitude conditions (basis + z-score) ONLY.

    Deliberately does NOT factor in the witness here - whether a
    qualifying witness exists is evaluated separately by the caller, so
    that "fade-shaped" (magnitude-qualifying) and "follow-shaped"
    (witness-qualifying) conditions can be compared against each other to
    detect a genuine conflict, rather than one silently pre-empting the
    other inside this helper.
    """
    unmet = []

    if abs(basis_bps) < cfg.min_abs_basis_bps:
        unmet.append(REASON_FADE_BASIS_BELOW_MIN)

    if abs(zscore) < cfg.entry_abs_zscore:
        unmet.append(REASON_FADE_ZSCORE_BELOW_MIN)

    if unmet:
        return False, None, unmet

    direction = Direction.SHORT if basis_bps > 0 else Direction.LONG
    return True, direction, []


def _evaluate_fade(
    basis_bps: float,
    zscore: float,
    witness_status: WitnessStatus,
    cfg: FadeConfig,
) -> Tuple[bool, Optional[Direction], list]:
    """Evaluate FADE's full condition set: the magnitude conditions from
    _fade_core_conditions() PLUS condition 4 - there must be no qualifying
    witness move explaining the dislocation. A qualifying witness (either
    direction) disqualifies FADE outright, regardless of how extended the
    basis/z-score are.

    This is what makes FADE and FOLLOW mutually exclusive by construction:
    FADE requires witness_status to NOT be qualifying, FOLLOW requires it
    TO BE qualifying. A true simultaneous qualification is therefore not
    reachable through normal inputs - the conflict branch in
    evaluate_signal() remains as an explicit defensive check per the spec
    ("if both qualify, never arbitrarily choose one"), not because it is
    expected to trigger in practice.
    """
    _, core_direction, unmet = _fade_core_conditions(basis_bps, zscore, cfg)
    unmet = list(unmet)

    if witness_status in (WitnessStatus.QUALIFYING_POSITIVE, WitnessStatus.QUALIFYING_NEGATIVE):
        unmet.append(REASON_FADE_DISQUALIFIED_BY_WITNESS)

    if unmet:
        return False, None, unmet

    return True, core_direction, []


def _evaluate_follow(
    basis_bps: float,
    witness: Optional[WitnessObservation],
    witness_status: WitnessStatus,
    cfg: FollowConfig,
) -> Tuple[bool, Optional[Direction], Optional[float], list]:
    unmet = []

    if witness_status not in (WitnessStatus.QUALIFYING_POSITIVE, WitnessStatus.QUALIFYING_NEGATIVE):
        unmet.append(REASON_FOLLOW_NO_QUALIFYING_WITNESS)
        return False, None, None, unmet

    witness_move_bps = witness.move_bps  # not None here, guaranteed by classify_witness
    residual_bps = witness_move_bps - basis_bps

    if abs(residual_bps) < cfg.min_residual_bps:
        unmet.append(REASON_FOLLOW_RESIDUAL_INSUFFICIENT)

    if _sign(residual_bps) != _sign(witness_move_bps):
        unmet.append(REASON_FOLLOW_DIRECTION_MISMATCH)

    if unmet:
        return False, None, residual_bps, unmet

    direction = Direction.LONG if witness_move_bps > 0 else Direction.SHORT
    return True, direction, residual_bps, []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_signal(
    observation: MarketObservation,
    mark_quality: MarkQualityResult,
    zscore: ZScoreResult,
    witness: Optional[WitnessObservation],
    config: SignalConfig,
) -> SignalResult:
    """Evaluate FADE/FOLLOW/FLAT for one observation. Defaults to FLAT.

    Required inputs, in the order checked (any failure short-circuits
    straight to FLAT/NONE, before FADE or FOLLOW are considered at all):
        1. observation.is_valid
        2. is_tradeable(observation.session)
        3. mark_quality.passed
        4. observation.basis_bps is not None and zscore.value is not None

    `witness` may legitimately be None (no witness data) - that is not a
    missing-required-input failure; it simply means FOLLOW cannot qualify
    and FADE is not disqualified by a witness explanation.
    """
    config_used = {
        "fade.min_abs_basis_bps": config.fade.min_abs_basis_bps,
        "fade.entry_abs_zscore": config.fade.entry_abs_zscore,
        "follow.min_witness_move_bps": config.follow.min_witness_move_bps,
        "follow.min_residual_bps": config.follow.min_residual_bps,
    }

    witness_status = classify_witness(witness, config.follow.min_witness_move_bps)
    witness_move_bps = witness.move_bps if witness is not None else None

    def flat(reason: str, residual_bps: Optional[float] = None) -> SignalResult:
        return SignalResult(
            signal=SignalType.FLAT,
            direction=Direction.NONE,
            symbol=observation.symbol,
            timestamp=observation.timestamp,
            session=observation.session,
            basis=observation.basis,
            basis_bps=observation.basis_bps,
            basis_zscore=zscore.value,
            reason=reason,
            mark_quality_passed=mark_quality.passed,
            mark_quality_reasons=mark_quality.reasons,
            witness_status=witness_status,
            witness_move_bps=witness_move_bps,
            residual_bps=residual_bps,
            config_used=config_used,
        )

    # 1. Observation validity
    if not observation.is_valid:
        return flat(f"{REASON_OBSERVATION_INVALID}:{observation.invalid_reason}")

    # 2. Session validity (reuses Phase 1's is_tradeable(), unmodified)
    if not is_tradeable(observation.session):
        return flat(f"{REASON_SESSION_NOT_TRADEABLE}:{observation.session.value}")

    # 3. Mark quality - hard blocks both FADE and FOLLOW unconditionally
    if not mark_quality.passed:
        return flat(REASON_MARK_QUALITY_FAILED)

    # 4. Required signal inputs
    if observation.basis_bps is None or zscore.value is None:
        return flat(REASON_MISSING_REQUIRED_INPUTS)

    basis_bps = observation.basis_bps
    zscore_value = zscore.value

    fade_qualifies, fade_direction, fade_unmet = _evaluate_fade(
        basis_bps, zscore_value, witness_status, config.fade
    )
    follow_qualifies, follow_direction, residual_bps, follow_unmet = _evaluate_follow(
        basis_bps, witness, witness_status, config.follow
    )

    if fade_qualifies and follow_qualifies:
        return flat(REASON_CONFLICT, residual_bps=residual_bps)

    if fade_qualifies:
        return SignalResult(
            signal=SignalType.FADE,
            direction=fade_direction,
            symbol=observation.symbol,
            timestamp=observation.timestamp,
            session=observation.session,
            basis=observation.basis,
            basis_bps=basis_bps,
            basis_zscore=zscore_value,
            reason=REASON_FADE_QUALIFIES,
            mark_quality_passed=mark_quality.passed,
            mark_quality_reasons=mark_quality.reasons,
            witness_status=witness_status,
            witness_move_bps=witness_move_bps,
            residual_bps=residual_bps,
            config_used=config_used,
        )

    if follow_qualifies:
        return SignalResult(
            signal=SignalType.FOLLOW,
            direction=follow_direction,
            symbol=observation.symbol,
            timestamp=observation.timestamp,
            session=observation.session,
            basis=observation.basis,
            basis_bps=basis_bps,
            basis_zscore=zscore_value,
            reason=REASON_FOLLOW_QUALIFIES,
            mark_quality_passed=mark_quality.passed,
            mark_quality_reasons=mark_quality.reasons,
            witness_status=witness_status,
            witness_move_bps=witness_move_bps,
            residual_bps=residual_bps,
            config_used=config_used,
        )

    combined_reason = (
        "flat:"
        + ",".join(f"fade[{u}]" for u in fade_unmet)
        + ","
        + ",".join(f"follow[{u}]" for u in follow_unmet)
    )
    return flat(combined_reason, residual_bps=residual_bps)
