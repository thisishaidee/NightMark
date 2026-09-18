# Nightmark

Session-aware quantitative strategy for Bitget rTokens — Bitget AI Hackathon S2,
Track: Alpha Factory, Sub-theme: After-Hours Information Pricing.

**Status: Phases 1, 2A, 2B, 2C, 3, and 3.5 implemented.** Session calendar,
weekend eligibility, the market-observation data contract/basis/z-score,
the mark-quality gate, the deterministic FADE/FOLLOW/FLAT signal engine,
the risk engine (position sizing + trade authorization), and the Phase 3.5
authorization façade (non-finite rejection, session revalidation, weekend
eligibility on OPEN, single legal path for OPEN_OR_INCREASE) all exist
and are tested. No Qwen integration, execution, backtesting, or UI exist
yet. See `spec.md` for the governing architecture and rules.

## Core thesis

Nightmark trades only during Bitget-supported overnight/weekend sessions
(when the U.S. cash market is closed), and distinguishes:

- **(A)** unexplained rToken basis dislocations that may mean-revert
- **(B)** explained residual moves where a 24/7 witness market has moved but
  the rToken has lagged
- **(C)** otherwise flat

Rules first, LLM second. An optional Qwen component (added later) is
restricted to event-relevance classification and rationale generation. It
never controls session locks, mark-quality locks, position sizing, risk
limits, signal creation, or execution permission.

## Repository layout

```
nightmark/
├── README.md
├── spec.md
├── config/
│   ├── sessions.yaml     # session boundaries (RTH/POST/WEEKEND/OVERNIGHT/HALT/closures)
│   ├── symbols.yaml      # per-symbol weekend eligibility (empty by default)
│   ├── mark_quality.yaml # spread/depth/staleness thresholds, corporate-action windows
│   ├── signal.yaml       # FADE/FOLLOW decision thresholds
│   └── risk.yaml         # position limits, cost model, flatten-window buffer
├── src/
│   ├── calendar.py     # pure time-based session classification + trade gate      [Phase 1]
│   ├── eligibility.py  # symbol weekend-eligibility gate                          [Phase 1]
│   ├── observation.py  # normalized market observation data contract              [Phase 2A]
│   ├── reference.py    # fixed-priority reference-price selection                 [Phase 2A]
│   ├── normalize.py    # deterministic market-data normalization (mid/spread/basis) [Phase 2A]
│   ├── zscore.py         # rolling same-session z-score, no look-ahead              [Phase 2A]
│   ├── mark_quality.py  # fail-closed mark-quality gate (6 checks)                 [Phase 2B]
│   ├── witness.py       # 24/7 witness-market classification (injected data only)  [Phase 2C]
│   ├── signal.py        # deterministic FADE/FOLLOW/FLAT signal engine             [Phase 2C]
│   ├── risk.py          # position sizing + trade authorization, fail-closed       [Phase 3]
│   └── authorize.py     # OPEN_OR_INCREASE façade (legal authorization path)      [Phase 3.5]
├── tests/              # deterministic unit tests
├── notebooks/          # (empty — reserved for later phases)
├── playbook/           # (empty — reserved for later phases)
├── demo/               # (empty — reserved for later phases)
├── submission/         # (empty — reserved for hackathon submission materials)
└── data/               # (empty — no fabricated or live data)
```

## Phase 1 Calendar Assumptions

This section is the single place to check what the calendar does and does
not claim, before anything downstream (signals, sizing, execution) is
built on top of it.

- **Bitget's published weekend schedule is the authoritative source** for
  the WEEKEND window. See the citation and full quote in
  `config/sessions.yaml`.
- **Nightmark normalizes Bitget's UTC+8, DST-dependent schedule into
  America/New_York wall-clock time.** Bitget publishes the weekend open in
  UTC+8 with a clock time that shifts by DST regime; the close is stated
  qualitatively as "until the regular US stock market opens on Monday."
  Nightmark encodes both sides directly in ET so the config never needs
  editing across US DST changes.
- **20:00 ET is Nightmark's normalized Friday transition boundary** into
  the weekend window — this is Nightmark's derived interpretation, not a
  claim that Bitget natively publishes "20:00 ET" as a session label.
- **16:00–20:00 ET POST is a US-equity convention Nightmark uses
  internally** as a session-bookkeeping boundary. This is not a claim that
  Bitget publishes an rToken session under that exact ET label — Bitget's
  own documentation does not.
- **OVERNIGHT is Nightmark's residual weekday closed-cash session** — the
  weekday time outside RTH/POST/WEEKEND/HALT. It is not a named Bitget
  session with its own published boundaries.
- **Market-closure (holiday) data is intentionally not fabricated.**
  `config/sessions.yaml`'s `market_closures` list ships empty and must be
  populated from an authoritative source (e.g. the official NYSE/NASDAQ
  holiday calendar) before production trading. The schema supports future
  closure types beyond a simple full-day closure without another breaking
  change; Phase 1 treats every configured closure as UNKNOWN regardless of
  type, since modified-session handling isn't implemented yet.
- **Weekend eligibility is symbol-specific and must never be assumed from
  the calendar alone.** `config/symbols.yaml` ships with an empty
  eligible-symbol set; `src/eligibility.py` fails closed for any
  unconfigured symbol, and `is_weekend_eligible(symbol, timestamp)`
  requires both a WEEKEND session classification AND explicit symbol
  configuration — never one without the other.
- **Phase 1 does not make trading decisions.** It answers exactly two
  yes/no questions ("what session is this" and "is this symbol
  weekend-eligible") and nothing else. No basis, no signal, no sizing, no
  risk check, no execution, no Qwen involvement exists in this repository
  yet.

## Phase 2A Data Foundation Assumptions

- **Every market/reference input is injected by the caller.** Nothing in
  `src/observation.py`, `src/reference.py`, `src/normalize.py`, or
  `src/zscore.py` makes a network call or invents a data source. Tests use
  synthetic, clearly-labelled fixture numbers only (see `tests/fixtures.py`)
  — never real market data, and no claim about real market performance is
  made anywhere in this repository.
- **Reference-price selection is a strict, fixed priority order**: Bitget
  official reference/NAV, then the last official US cash close, then a
  liquid proxy — the proxy tier is always labelled `ESTIMATED` and never
  presented as official. A candidate the caller marks stale or unavailable
  is skipped, falling through to the next tier; if nothing usable is
  supplied, the result is `UNAVAILABLE` with a `None` price — never a
  fabricated number.
- **`reference_quality` is provenance labeling, not mark-quality logic.**
  It's a direct readout of which priority tier was used. The staleness-
  threshold / confidence-scoring / depth-weighted mark-quality engine
  implied by the project's later phases is intentionally NOT built yet.
- **Missing or invalid rToken quotes never produce a fabricated mid.** A
  missing bid/ask, a non-positive price, or a crossed book (bid > ask) all
  result in `rtoken_mid = None` and the observation is marked
  `is_valid=False` with an explicit `invalid_reason` — nothing is guessed.
- **`basis` and `basis_bps` are undefined (`None`), not zero or an error,**
  whenever either the rToken mid or the reference price is unavailable, or
  the reference price is exactly zero (guarded to avoid a division error).
- **The rolling z-score never mixes sessions.** This generalizes the
  explicit "never mix RTH and WEEKEND" rule to every session pairing
  (OVERNIGHT and POST are also kept separate from each other and from
  RTH/WEEKEND), since each session represents a different liquidity
  regime. `window` and `min_observations` are caller-configurable.
- **No look-ahead is structural, not just tested.** A z-score's baseline
  only ever contains observations from earlier positions in the input
  sequence, and an observation's own value is appended to its session's
  history only after that observation's own z-score has been computed.
  The function does not re-sort its input — callers must supply each
  symbol's observations in true chronological order.
- **Insufficient history, zero variance, and missing values are all
  explicit, inspectable outcomes** (`ZScoreResult.reason`), never a
  silently-wrong number, an exception, or a `NaN`/`inf`.
- **Phase 2A still makes no trading decisions.** It only normalizes data
  and computes a statistic. No signal, no mark-quality gate, no sizing, no
  risk check, no execution, and no Qwen involvement exists yet.

## Phase 3 Risk Engine Assumptions

- **The risk engine does not decide direction or generate signals.** It
  only answers "given an already-approved `SignalResult`, portfolio state,
  costs, and time-to-cash-open, is this trade allowed, and at what size?"
  `src/risk.py`'s `evaluate_risk()` reuses Phase 1's `is_tradeable()` and
  reads Phase 2C's `SignalResult.mark_quality_passed` directly — it never
  recreates session or mark-quality rules.
- **Fail closed by default.** Missing/invalid NAV, portfolio state,
  signal, requested notional, spread, or calendar all produce an explicit
  rejection reason; nothing silently defaults to an unsafe value.
  Rejection reasons accumulate — a single evaluation surfaces every
  applicable failure, not just the first.
- **Single-name (25%) and gross (100%) exposure limits are sizing
  constraints, not outright rejections**, unless available capacity is
  already zero: a request above the cap is sized down to fit rather than
  refused outright. Reaching a limit exactly is allowed; only exceeding it
  is capped.
- **The opposite-position lock and averaging-down limit apply only to
  `OPEN_OR_INCREASE`.** `REDUCE`/`FLATTEN` are treated as risk-reducing
  safety actions and only require valid NAV/position bookkeeping — this
  is an **approved** design decision (see `spec.md`'s "Phase 3 approved
  design decisions" section): the signal, session, mark-quality, and
  cost/edge gates all exist to decide whether *opening or increasing*
  exposure is safe, and applying them to an exit action would force the
  system to hold a position during RTH, HALT, or another restricted
  session precisely when it's trying to close it. `OPEN_OR_INCREASE`
  itself is never exempt from any gate.
- **The cost model never assumes a mid-price fill.** Round-trip cost is
  `2 * (fee_bps_per_side + half_spread_bps + additional_slippage_bps_per_side
  + weekend_extra_if_applicable)`, where `half_spread_bps` comes from the
  real injected observation spread, not a constant. See `config/risk.yaml`
  for the exact configured defaults (5 bps fee, 2 bps generic slippage,
  3 bps extra weekend slippage, all independently configurable).
- **"Expected edge" is a simplification, not a new market read.** It is
  derived from data the already-approved signal computed:
  `abs(basis_bps)` for FADE, `abs(residual_bps)` for FOLLOW. Flagged in
  `src/risk.py`'s module docstring for review before a later phase
  depends on it.
- **The flatten-before-cash-open buffer reuses the existing
  `SessionCalendar`**, not a new calendar — `evaluate_risk()` locates the
  next RTH open using the calendar's own configured `rth_start`/
  `rth_weekdays`. The default 60-minute buffer sits inside the spec's
  30–90 minute guidance; the boundary is inclusive (exactly at the buffer
  counts as inside the flatten window).
- **The FOLLOW-during-flatten exception is off by default and never
  silently assumed.** `config/risk.yaml`'s `flatten.follow_exception_during_flatten`
  must be explicitly set to `true`; even then it never exempts FADE.
- **Phase 3 places no real orders.** It returns a `RiskDecision`
  (`ALLOW`/`REDUCE`/`REJECT`/`FLATTEN`) with an approved/max notional —
  execution, PnL tracking, and portfolio-state persistence remain
  unimplemented.

## Phase 3.5 Authorization Hardening

Closes confirmed fail-open paths on `OPEN_OR_INCREASE` before any
execution/backtest work. `REDUCE`/`FLATTEN` keep the approved Phase 3
exit bypass (session, mark-quality, signal, cost/edge, and weekend
eligibility are not applied to exits).

- **Non-finite inputs fail closed.** `evaluate_risk()` rejects NaN,
  `+inf`, and `-inf` for NAV, requested notional, spread, expected edge,
  computed cost, and position notionals. Python comparison quirks
  (`x < nan` is always False) are not relied on. Negative spreads and
  negative position notionals are also rejected on OPEN.
- **Session revalidation.** `SignalResult.session` is not the session
  authority for an open. When a calendar is supplied, `evaluate_risk()`
  classifies `signal.timestamp` and (1) rejects a mismatch against the
  claimed session, (2) uses the calendar result for `is_tradeable()` and
  weekend extra-slippage. A signal that claims OVERNIGHT at an RTH
  timestamp cannot authorize an opening trade. Missing calendar still
  fails closed rather than trusting the self-reported session.
- **Weekend eligibility on OPEN.** A WEEKEND `OPEN_OR_INCREASE` requires
  the symbol to be in the configured weekend-eligible set
  (`src.eligibility.SymbolEligibility`). `config/symbols.yaml` still
  ships empty, so the default is fail-closed. OVERNIGHT opens do not use
  this list. No Bitget symbol list is fabricated here.
- **Sanctioned OPEN path vs sizing engine.** `evaluate_risk()` is the
  public lower-level sizing/limit primitive and still evaluates
  `OPEN_OR_INCREASE` (so tests can inspect capacity). `ALLOW` from
  `evaluate_risk()` is not permission to open: `RiskDecision.authorizes_open`
  is always `False`. The only OPEN authorization boolean is
  `AuthorizationResult.allowed` from `src.authorize.authorize_open_or_increase()`.
  Direct `evaluate_risk()` remains the REDUCE/FLATTEN path.
- **NAV is an OPEN gate.** Missing, non-finite, and `<= 0` NAV reject
  `OPEN_OR_INCREASE`. REDUCE/FLATTEN do not use NAV and must not be
  blocked by it.
- **Sibling malformed notionals.** Any non-finite or negative position
  notional fail-closes OPEN (gross/capacity cannot be trusted). REDUCE/
  FLATTEN of a clean target are not blocked by an unrelated sibling.

## Running the tests

```
cd nightmark
python3 -m unittest discover -s tests -v
```

No third-party test framework is required (stdlib `unittest` only). If
`pytest` is available in your environment, `pytest tests/` also works against
the same test file.
