# Nightmark — Spec

## Hackathon context

- Event: Bitget AI Hackathon S2
- Track: Alpha Factory
- Sub-theme: After-Hours Information Pricing
- Deadline: September 21, 2026, 23:59 UTC+8

## Core thesis

Nightmark is a session-aware quantitative strategy for Bitget rTokens. It
trades only during Bitget-supported overnight/weekend periods when the U.S.
cash market is closed. It distinguishes:

- **(A)** unexplained rToken basis dislocations that may mean-revert
- **(B)** explained residual moves where a 24/7 witness market has moved but
  the rToken has lagged
- **(C)** otherwise flat

## Governing principle: rules first, LLM second

Qwen (optional, added later) is restricted to:
- event relevance classification
- rationale generation

Qwen must **never** control:
- session locks
- mark-quality locks
- position sizing
- risk limits
- signal creation
- execution permission

Every one of the above is deterministic, rule-based code. If a future PR
routes any of these six through a model call, that PR is out of spec.

## Phase 1 scope (calendar + eligibility hardening)

`src/calendar.py`: a pure, time-only session classifier and trading gate.
`src/eligibility.py`: a pure, symbol-only weekend-eligibility gate,
composed with (never merged into) the time classifier. Nothing else.

Conceptual pipeline (only the first two stages exist in this repository):

```
timestamp
  |
  v
session classification (src/calendar.py)
  |
  v
symbol weekend eligibility (src/eligibility.py)
  |
  v
mark quality        <- not implemented
  |
  v
signal               <- not implemented (rules first, Qwen never controls this)
  |
  v
risk                 <- not implemented
  |
  v
execution             <- not implemented
```

See README.md's "Phase 1 Calendar Assumptions" section for exactly which
boundaries are Bitget-sourced, which are Nightmark conventions, and what
must be populated from an authoritative source before production use.

## Phase 2A scope (data foundation)

`src/observation.py`, `src/reference.py`, `src/normalize.py`,
`src/zscore.py`: the normalized market-observation data contract,
deterministic normalization, reference-price selection, and a
same-session rolling z-score. All market and reference data is INJECTED
by the caller (tests, fixtures, or a future adapter that is not part of
this phase) - nothing in Phase 2A makes a network call.

Updated conceptual pipeline (stages implemented so far are marked):

```
timestamp
  |
  v
session classification (src/calendar.py)                 [Phase 1]
  |
  v
symbol weekend eligibility (src/eligibility.py)           [Phase 1]
  |
  v
market observation normalization (src/normalize.py)       [Phase 2A]
  + reference-price selection (src/reference.py)          [Phase 2A]
  + basis / basis_bps                                     [Phase 2A]
  + rolling same-session z-score (src/zscore.py)           [Phase 2A]
  |
  v
mark quality         <- not implemented (deliberately deferred - see below)
  |
  v
signal                <- not implemented (rules first, Qwen never controls this)
  |
  v
risk                  <- not implemented
  |
  v
execution              <- not implemented
```

Reference-price priority (fixed, enforced in `src/reference.py`):
1. Bitget official reference/NAV, if exposed and usable
2. Last official US cash print/close, while the cash market is closed
3. A liquid proxy, only when necessary, always labelled `ESTIMATED` -
   never presented as an official reference

`reference_quality` in the data contract is a direct readout of which
priority tier was used (`OFFICIAL` / `ESTIMATED` / `UNAVAILABLE`) - it is
explicitly NOT the mark-quality engine (staleness thresholds, depth-
weighted confidence scoring) that a later phase will add. Phase 2A treats
"mark quality" as still out of scope.

The rolling z-score never mixes observations from different sessions
(generalizing the explicit "never mix RTH and WEEKEND" rule to all session
types), has no look-ahead by construction, and makes `window` and
`min_observations` caller-configurable.

## Explicit non-goals for Phase 2A

Not implemented, and not to be implemented without a separate approval:
- mark-quality logic (staleness scoring, confidence, depth-weighted checks)
- signal generation
- Qwen integration
- UI
- trading execution
- live API keys / live data feeds
- backtesting
- any live market-data fetching

## Phase 2B scope (mark-quality gate)

`src/mark_quality.py`: the mandatory checkpoint between a normalized
Phase 2A `MarketObservation` and any future signal. `evaluate_mark_quality()`
runs six deterministic, fail-closed gates and returns
`MarkQualityResult(passed, reasons, ...)`. No signal in a later phase may
bypass this gate.

Updated pipeline:

```
timestamp
  |
  v
session classification (src/calendar.py)                 [Phase 1]
  |
  v
symbol weekend eligibility (src/eligibility.py)           [Phase 1]
  |
  v
market observation normalization (src/normalize.py)       [Phase 2A]
  + reference-price selection (src/reference.py)          [Phase 2A]
  + basis / basis_bps                                     [Phase 2A]
  + rolling same-session z-score (src/zscore.py)           [Phase 2A]
  |
  v
mark quality (src/mark_quality.py)                        [Phase 2B]
  1. spread quality (historical 90th pctile / fallback)
  2. top-of-book depth (per-ticker minimum)
  3. stale reference (per-session max age; weekend >48h
     corroboration exception, injected not inferred)
  4. stale/missing market data (reuses observation.is_valid)
  5. session validity (reuses Phase 1 is_tradeable())
  6. corporate-action/halt protection (configured windows)
  |
  v
signal                <- not implemented (rules first, Qwen never controls this)
  |
  v
risk                  <- not implemented
  |
  v
execution              <- not implemented
```

Every threshold in `config/mark_quality.yaml` is configurable; every
unconfigured or ambiguous case rejects rather than silently passing
(fail closed). Corporate-action windows and per-symbol depth/spread
overrides ship empty - no data is fabricated.

## Explicit non-goals for Phase 2B

Not implemented, and not to be implemented without a separate approval:
- mark-quality logic (staleness scoring, confidence, depth-weighted checks)
- signal generation
- Qwen integration
- UI
- trading execution
- live API keys / live data feeds
- backtesting
- any live market-data fetching

## Phase 2C scope (signal engine)

`src/witness.py`, `src/signal.py`: a 24/7 witness-market classification
(`classify_witness()`, injected data only, no fetching) and the
deterministic `evaluate_signal()` decision function. Three signal states —
`FADE`, `FOLLOW`, `FLAT` — default to `FLAT`. Mark-quality failure hard-
blocks both `FADE` and `FOLLOW` unconditionally. `FADE` and `FOLLOW` are
mutually exclusive by construction (`FADE` requires a non-qualifying
witness; `FOLLOW` requires a qualifying one), so the explicit conflict-
detection branch exists as a defensive safeguard per the original spec
("never arbitrarily choose one") even though it is not reachable through
the current public API. See `config/signal.yaml` for the configured
`fade`/`follow` thresholds. No look-ahead: the engine is a pure function
of the single observation/z-score/witness passed to it.

## Explicit non-goals for Phase 2C

Not implemented, and not to be implemented without a separate approval:
- risk sizing / position limits
- execution
- Qwen integration
- UI
- backtesting
- PnL reporting
- broker/API integration

## Phase 3 scope (risk engine)

`src/risk.py`: `evaluate_risk()` answers "given an already-approved
Phase 2C `SignalResult`, current portfolio state, trading costs, and
time-to-cash-open, is this trade allowed, and at what maximum safe
notional?" It does not decide direction, generate signals, or place real
orders.

Updated pipeline:

```
timestamp
  |
  v
session classification (src/calendar.py)                 [Phase 1]
  |
  v
symbol weekend eligibility (src/eligibility.py)           [Phase 1]
  |
  v
market observation normalization (src/normalize.py)       [Phase 2A]
  + reference-price selection (src/reference.py)          [Phase 2A]
  + basis / basis_bps                                     [Phase 2A]
  + rolling same-session z-score (src/zscore.py)           [Phase 2A]
  |
  v
mark quality (src/mark_quality.py)                        [Phase 2B]
  |
  v
signal: FADE / FOLLOW / FLAT (src/signal.py, src/witness.py) [Phase 2C]
  |
  v
risk engine (src/risk.py)                                 [Phase 3]
  1. signal gate (FLAT/invalid rejected)
  2. session gate (reuses is_tradeable())
  3. mark-quality gate (reads SignalResult's own field)
  4. max active names (3)
  5. max single-name exposure (25% of NAV, sized down not rejected)
  6. max gross exposure (100% of NAV, sized down not rejected)
  7. opposite-position lock
  8. averaging-down limit (explicit count, never inferred from PnL)
  9. cost/edge gate (round-trip cost vs. signal-derived edge magnitude)
  10. flatten-before-cash-open window (configurable buffer, FOLLOW-only
      opt-in exception, off by default)
  11. fail closed on any missing/invalid required input
  |
  v
execution / position simulation   <- not implemented
  |
  v
reporting                          <- not implemented
```

See README.md's "Phase 3 Risk Engine Assumptions" section for the exact
cost-model formula, sizing behavior, and the REDUCE/FLATTEN gate-bypass
design choice.

## Phase 3 approved design decisions

**Gate scope by trade action (approved):**
- `OPEN_OR_INCREASE` must pass the signal gate, session gate, mark-quality
  gate, and cost/edge gate. No exceptions.
- `REDUCE` may bypass the signal, session, mark-quality, and cost/edge
  gates when position bookkeeping is valid (i.e. the requested notional
  does not exceed the currently held position).
- `FLATTEN` may bypass those same four gates under the same bookkeeping
  condition, because the system must remain capable of closing positions
  during RTH, HALT, or other restricted sessions - refusing to flatten
  because, say, the session just became non-tradeable would force the
  system to hold a position it is actively trying to exit.

This asymmetry is intentional and approved, not an oversight: gates exist
to decide whether *opening or increasing* exposure is safe; a `REDUCE`/
`FLATTEN` request is the opposite action (reducing exposure), so applying
the same opening-safety gates to it would be actively unsafe rather than
conservative. See `src/risk.py`'s module docstring and
`tests/test_risk.py`'s `TestApprovedGateBypassDecision` class for the
per-gate test coverage of this decision (signal, session — including RTH/
HALT/UNKNOWN specifically, mark-quality, and cost/edge, for both `REDUCE`
and `FLATTEN`), plus a same-portfolio contrast test proving
`OPEN_OR_INCREASE` is never exempt under the same conditions.

## Phase 3.5 scope (authorization hardening)

`src/authorize.py` plus fail-closed additions in `src/risk.py`. No
execution, backtesting, UI, Qwen, or live Bitget trading.

Updated tail of the pipeline:

```
risk engine (src/risk.py)                                 [Phase 3]
  + non-finite input rejection                            [Phase 3.5]
  + timestamp session revalidation on OPEN                [Phase 3.5]
  + weekend eligibility on WEEKEND OPEN                   [Phase 3.5]
  |
  v
authorize_open_or_increase (src/authorize.py)             [Phase 3.5]
  sanctioned OPEN authorization; sequences
  session classification, weekend eligibility, mark
  quality, evaluate_signal(), evaluate_risk().
  AuthorizationResult.allowed is the only OPEN
  authorization boolean. evaluate_risk() ALLOW is a
  sizing verdict (authorizes_open is always False).
  |
  v
execution / position simulation   <- not implemented
```

**OPEN vs sizing:** `evaluate_risk()` remains public as the lower-level
risk/sizing engine and still runs OPEN_OR_INCREASE. It never sets
`authorizes_open=True`. Callers must not treat `RiskDecision.decision ==
ALLOW` as permission to send an opening order.

**NAV:** required for OPEN_OR_INCREASE only (missing / non-finite /
`<= 0` reject). REDUCE/FLATTEN ignore NAV.

**Sibling notionals:** any non-finite or negative position notional
fail-closes OPEN. REDUCE/FLATTEN validate only the target position.

**Session revalidation (chosen behavior):** reject OPEN_OR_INCREASE if
`SignalResult.session` disagrees with `SessionCalendar.classify(timestamp)`,
and use the calendar classification for `is_tradeable()` and weekend
cost extra. Do not silently rewrite the signal. REDUCE/FLATTEN do not
revalidate session.

**Weekend eligibility:** WEEKEND OPEN_OR_INCREASE requires
`SymbolEligibility.is_symbol_configured_eligible(symbol)`. Default
`config/symbols.yaml` is empty (fail closed). No symbol list is
fabricated. REDUCE/FLATTEN skip this gate.

**Non-finite inputs:** explicit `math.isfinite` checks on NAV, requested
notional, spread, expected edge, cost, and position notionals for
OPEN_OR_INCREASE. Do not rely on Python's `nan` comparison behavior.

Out of scope for Phase 3.5 (deferred, do not silently "fix"):
- FOLLOW residual vs reference provenance
- symbol-keyed z-score
- averaging-down inference
- holiday calendar population
- shipped depth-minimum calibration

## Explicit non-goals for Phase 3

Not implemented, and not to be implemented without a separate approval:
- real Bitget trading, API keys, wallet integration, withdrawals
- live order execution
- Qwen integration
- UI/dashboard
- backtesting
- performance optimization
- new trading signals or indicators
- machine learning / portfolio optimization

## Session gate (initial)

| Session   | Trading permitted |
|-----------|--------------------|
| RTH       | NO                 |
| POST      | NO                 |
| HALT      | NO                 |
| UNKNOWN   | NO                 |
| OVERNIGHT | YES (eligible)     |
| WEEKEND   | YES (eligible)     |

"Eligible" means the session-level gate is open — it is necessary, not
sufficient. Actual trade permission in later phases will still require
signal confirmation, mark-quality checks, and risk-limit checks, none of
which exist yet.

## Explicit non-goals for Phase 1

Not implemented, and not to be implemented without a separate approval:
- basis calculation
- signal generation
- Qwen integration
- UI
- trading execution
- live API keys / live data feeds
- backtesting
- any indicator beyond session classification

## Data integrity rule

No fabricated data, no fabricated performance figures, at any phase.
