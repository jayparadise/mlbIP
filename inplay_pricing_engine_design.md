# MLB In-Play Player-Prop Pricing Engine — Design

Markets covered: **Hitters** — Hits (O/U + milestones), Total Bases (O/U + milestones), Home Runs (1+, 2+). **Pitchers** — Strikeouts (O/U + milestones), Hits Allowed (O/U + milestones).

This document specifies how each pre-game line/price should move during a live game. A runnable reference implementation of the core math accompanies it (`inplay_engine.py`).

---

## 1. The one idea everything rests on

At any instant, a player's final stat splits into a part that is already settled and a part that is still random:

```
S_final  =  S_locked  +  S_remaining
```

- **`S_locked`** — what has already happened (an integer we read off the feed: hits so far, TB so far, K so far…). Once it happens it never changes.
- **`S_remaining`** — a random variable equal to the sum of per-opportunity outcomes over the player's *remaining* opportunities.

Every market is then just a readout of the distribution of `S_final`:

- **Milestone `k+`**: `P(S_final ≥ k) = P(S_remaining ≥ k − S_locked)`
- **Over/Under line `L.5`**: `P(S_final > L) = P(S_remaining > L − S_locked)`

So the entire live engine reduces to one job: **maintain the distribution of `S_remaining` and re-read the prices.** `S_remaining` is a *compound* random variable:

```
S_remaining = Σ_{i=1..N} X_i
```

where **`N`** = remaining opportunities (plate appearances for hitters, batters-faced for pitchers) — itself uncertain — and **`X_i`** = the per-opportunity outcome (a hit, a total-base count, a strikeout…). Almost all the intelligence of the engine lives in modeling `N` well.

### Why this decomposition is the right one
- It guarantees **t=0 consistency**: if we calibrate per-opportunity rates so that `rate × pre-game_expected_opportunities = pre-game projection`, then at first pitch the live engine reproduces your existing pre-game book exactly. No discontinuity when markets flip from pre-game to in-play.
- It cleanly separates **"what happened" (deterministic)** from **"what's left" (a forecasting problem)**. Only the second part needs a model; the first is just bookkeeping off the feed.
- Milestones and O/U all come from the *same* distribution, so they stay mutually coherent (e.g. `1+ ≥ 2+ ≥ 3+` always holds, O/U two-sided probs sum correctly before margin).

---

## 2. Tracking the distribution: mean + variance, then fit

We don't need the full pmf analytically. We carry the **mean and variance** of `S_remaining` in closed form (law of total expectation / variance over `N`), then fit a discrete count distribution to that mean and variance and read the CDF.

For an opportunity outcome with per-trial mean `μ_X` and variance `σ²_X`, and a random opportunity count `N` with mean `m_N`, variance `v_N`:

```
E[S_remaining] = m_N · μ_X
Var[S_remaining] = m_N · σ²_X  +  v_N · μ_X²
```

- The `m_N · σ²_X` term is **outcome randomness** (even with a fixed number of PA, hits vary).
- The `v_N · μ_X²` term is **opportunity randomness** (you might get 4 PA or 5; a pitcher might face 20 batters or 28). Late in games this second term dominates — *the biggest live driver is how many more chances the player gets, not the per-chance rate.*

**Fitting.** For count stats (hits, HR, K, hits-allowed) we fit:
- **Poisson** when `Var ≈ Mean`,
- **Negative Binomial** when `Var > Mean` (the usual case once `N` is random — this is where your pre-game **Std Deviation** column is doing its job; we reproduce that overdispersion live).

For **Total Bases** the per-PA outcome is a `{0,1,2,3,4}` multinomial (out / 1B / 2B / 3B / HR), so `X_i` has a richer shape; we still summarize with mean/variance and fit an NB on the count grid, which is accurate enough for the milestone ladder. (If you want exact TB tails, swap the NB fit for a small convolution or Monte Carlo — noted in §9.)

This whole path is a few microseconds per player per update — fast enough to re-price the full board on every event.

---

## 3. Per-opportunity outcome rates (calibrated from *your* book market)

We never invent projections — we **derive the full-game mean from the book's O/U line and price**, which is the real read on expected performance (a hand-entered projection is only an internal reference). The steps:

1. **De-vig the book O/U.** Turn the two-way over/under prices into a fair P(over line) (one-sided prices are shaved by an assumed hold).
2. **Invert to a mean.** Solve for the full-game mean whose distribution — with spread set by the book's std dev — reproduces that fair P(over). This is monotonic in the mean, so a bisection nails it. Home runs, which have no O/U, are inverted from the 1+ price instead.

This inversion round-trips exactly: at first pitch the model reproduces the book's fair over to the third decimal, and it captures the key intuition — on a fixed 1.5 hits line, a cheaper over price implies a higher mean (over @ 2.40 → ~1.31 hits; over @ 1.30 → ~2.01 hits).

**The book has no pre-game expected-PA of its own, so we compute it too.** `expected opportunities` is derived from the same live machinery evaluated at first pitch:
- **Hitters:** `pregame_xPA(slot)` = run the team-PA + order-walk model (§4) from a fresh game state. ~4.9 for the top of the order down to ~3.8 for the 9-hole, internally consistent with in-play pricing.
- **Pitchers:** `pregame_BF` = integrate the hook survival curve (§5) from pitch 0.

The per-opportunity rate is then the market-derived full-game mean divided by expected opportunities:
```
p_hit = implied_mean(hits O/U line, over, under, std) / pregame_xPA(slot)
p_K   = implied_mean(K O/U line, over, under, std)    / pregame_BF
```
Non-HR hits are split into 1B/2B/3B so the Total Bases mean (also market-derived) is matched exactly.

### Std dev shapes the spread (widen-only)
The book std dev sets the dispersion used both in the mean inversion and in live pricing. It can only **widen** the distribution beyond the natural sampling variance, never tighten it below — a count built from per-PA coin flips has a hard variance floor, and allowing sub-floor variance produces degenerate distributions.

### Pre-game price is served verbatim until the first event
The engine serves the book price unchanged until the player records his first in-game event (first completed PA for a hitter, first batter faced for a pitcher), then switches to the live model. Hard-gated in both the Python (`first_event_seen`) and the terminal (a toggle).

> Design note: rates can be made **matchup- and state-aware** — e.g. bump `p_K` for a hitter's slot when a high-K reliever enters. The framework supports live rate updates; the reference code keeps rates static per player for clarity.

---

## 4. Hitters — live expected remaining plate appearances (xRPA)

This is the heart of hitter pricing. `xRPA` is a function of lineup slot, inning, outs, score, and times-through-the-order.

**Step 1 — remaining *team* PA.** Each remaining half-inning is 3 outs plus however many batters reach. If the batting team's effective reach-base rate is `q`, then expected PA per inning is `3 / (1 − q)` (≈ 4.4 at `q = 0.32`), with variance `≈ 3q/(1−q)²`. Sum over the innings this team still bats:
- Count whole future innings this side bats, plus the fraction of the current inning left (`(3 − outs)/3`).
- **Shave the bottom of the 9th** if the home team already leads (they may not bat).
- **Add an extra-innings tail** when the game is tied late (a probability-weighted couple of frames).

`q` should be the *live* reach-base rate against the current pitcher/bullpen, and it drifts as pitchers change and the order turns over.

**Step 2 — walk the order.** Given the batting-order pointer ("who's due up"), a specific slot bats on upcoming queue positions `offset+1, offset+10, offset+19, …`. Expected remaining PA for that slot = how many of those positions fall within the team's expected remaining PA, with **fractional credit** for the final partially-reached turn. That fractional turn is also the main source of a single hitter's PA *variance* (the "one more trip to the plate or not" swing), which feeds the `v_N` term.

**Step 3 — removal risk.** Late-inning pinch-hits, defensive subs, and platoon swaps reduce a hitter's remaining chances. A `pull_prob` scales `xRPA` down (higher when losing late, for weak hitters, or against a same-handed specialist).

**Behavioral sanity check** (from the reference run): Goldschmidt batting 3rd in a fresh game → xRPA ≈ 5.0; by top of the 6th with the order having turned over → xRPA ≈ 2.0. Prices move accordingly.

---

## 5. Pitchers — live expected remaining batters faced (via a "hook" survival model)

For pitchers, the dominant uncertainty is **whether he's still in the game**, not his per-batter rate. We model removal as a survival process over upcoming batters.

For each future batter *i*, the manager removes the pitcher with **hazard `h_i`** before that batter. The survival probability `s_i = Π_{j≤i}(1 − h_j)`, and:

```
E[remaining BF] = Σ_i s_i
Var[remaining BF] = Σ_i s_i(1 − s_i) + 2 Σ_{i<j}(s_j − s_i s_j)   (nested indicators)
```

The hazard rises with:
- **Pitch count** — logistic centered on the pitcher's soft limit (starter ~95–105, tunable per pitcher). This is the single biggest factor.
- **Times through the order** — a penalty that kicks in crossing the 3rd time through (`tto > 2`), matching modern bullpen behavior.
- **Role** — relievers get a short, steep leash (often 1 inning); the hazard ramps fast after ~4 batters.
- **Game state** — pull faster when losing badly; extend in a blowout win to save the bullpen.

From `remaining BF` and per-BF rates we get strikeouts-remaining and hits-allowed-remaining via the same compound formula (§2). The soft pitch limit is the knob that enforces **t=0 consistency for pitchers**: set it so pre-game `E[remaining BF] ≈ pre-game_BF`.

> A pitcher's milestone that's already achieved locks to 1.0 (e.g. once he has 6 K, `6+` is settled and only `7+`, `8+`… stay live). The engine handles this automatically because `S_locked` shifts the whole distribution.

---

## 6. Suspension logic

Markets must go **SUSPENDED** (no bets) whenever the outcome is momentarily indeterminate or a discrete event is being resolved:

| Trigger | Suspend | Reason |
|---|---|---|
| **Hitter is at bat** | that hitter's Hits / TB / HR markets | the in-progress PA is unresolved |
| **Pitcher facing a batter** | that pitcher's K / Hits-Allowed markets | the in-progress PA is unresolved |
| A ball is in play / pitch pending | relevant player(s) | outcome resolving |
| HR / hit just occurred, awaiting official scoring or **replay review** | affected markets | result not yet final (e.g. hit vs. error) |
| **Pitching change** in progress | incoming/outgoing pitcher markets + affected hitters | new matchup → rates change |
| Injury delay, rain delay, lineup change | affected players | state uncertain |
| Feed gap / stale data > threshold | everything affected | never price on stale state |

**Resume** as soon as the PA/event settles: update `S_locked`, decrement `xRPA` / `xRemBF`, recompute, re-open. Between-batter windows are when in-play prices are live.

**Settlement/void edge cases:** if a player is removed (pinch-hit, injury) their still-open `k+` markets that can no longer be reached should settle per your existing void/settle rules; the engine flags "no remaining opportunities" (`xRPA → 0`) so downstream settlement can act.

---

## 7. Comps anchoring, then margin

Two steps convert fair probabilities into the price you publish.

### 7a. Comps anchor (keep us close to the market)
You don't want to drift far from where DraftKings/FanDuel/etc. sit. The engine anchors each market to comps:

1. **De-vig the comp.** A competitor's two-way price (e.g. hits 1.5 over 2.20 / under 1.67) carries their hold. Normalize the two implied probabilities so they sum to 1 → the comp's *fair* probability. (One-sided comps are shaved by an assumed hold.)
2. **Blend toward comp.** `p_anchored = (1 − w)·p_model + w·p_comp`, where `w` is the anchor weight (0 = trust our model fully, 1 = match the market).
3. **Clamp the deviation.** Regardless of `w`, force `p_anchored` to sit within `max_dev` (in probability points) of the comp. This is the hard guarantee that we're *never too far* from comps — set `max_dev` = 0.06 and our fair prob can never be more than 6 points off the market, even if our model disagrees strongly.

`w` and `max_dev` are both live-tunable. Typical use: a moderate `w` (0.4–0.6) so you track the market but let your live game-state read move you off it, plus a tight `max_dev` as a safety rail. When a market has no comp, the model price stands alone.

This ordering matters: **anchor on fair probabilities, then add margin** — so the anchoring reasons about true value, not vig-contaminated prices.

### 7b. Margin
Apply your overround to the anchored fair probability. The engine uses a **7% margin** by default (`decimal = 1 / (p·1.07)`), applied per side. This is a pluggable hook — swap in your two-way overround / bias-balancing if you want the exact pre-game methodology, and layer risk-manager overrides on top. Keep margin separate from probability so you can widen it in high-uncertainty states (right after a lead change, or when a pitching change looms) without touching the model.

---

## 8. Update loop (per pitch — you get pitch-by-pitch data)

With pitch-by-pitch you can drive the tightest possible suspension windows and the freshest state:

```
on_pitch(pitch):
    update game_state (inning, outs, score, base state, batting-order pointer)
    update player_state (pitch count++, and on PA-complete: locked hits/TB/HR/K/HA, BF++, TTO)
    SUSPEND the batter's and pitcher's markets for the duration of the live PA
    on PA-complete / between pitches when no PA is live -> re-open
    for each player with an open market:
        if pre-game (no first event yet): serve book price; continue
        recompute xRPA (hitters) / xRemBF (pitchers)     # §4 / §5
        recompute S_remaining mean+var per stat          # §2
        fit distribution, shift by S_locked, read CDF
        price milestones + O/U (fair probs)              # §1
        anchor to comps, then apply 7% margin + overrides # §7
        emit updated prices
```

Recompute cost is trivial, so the whole board can refresh on every pitch. The events that *move prices most*: PA completions (locked totals + xRPA/xRemBF step down), pitching changes (hitter rates change; pitcher markets close/open), pitch count crossing the hook zone, and lead changes (extra-innings tail + hook game-state term).

---

## 9. Assumptions to calibrate against your data (and where they live)

These are set to reasonable baseball defaults in the reference code and should be tuned to your own historical feed:

1. **Pre-game xPA / xBF per lineup slot & role** — ideally taken directly from your pre-game model so calibration is exact (§3).
2. **Reach-base rate `q`** driving team PA/inning — league default 0.320; make it live and matchup-aware (§4).
3. **Extra-innings probability & length** when tied late (§4).
4. **Pull/pinch-hit probability curve** for hitters by inning/score/handedness (§4).
5. **Pitcher hook curve** — soft pitch limit per pitcher, TTO penalty slope, reliever leash, game-state adjustments (§5). This is the highest-leverage thing to fit well.
6. **XBH shape** (1B/2B/3B/HR split) per hitter for Total Bases (§3) — better with player-specific ISO.
7. **NB-vs-exact for Total-Bases tails** — the NB fit is fine for the common milestones; use convolution/Monte-Carlo if you price deep TB milestones (7+, 8+) heavily.
8. **Margin model** — 7% flat by default; your existing overround plugs into §7b.
9. **Comps anchor settings** — anchor weight `w` and max-deviation `max_dev` (§7a). Tune per market type: milestones and deep tails move more, main O/U lines stay tight to comps.

Every one of these is isolated behind a function so you can A/B and backtest them independently.

---

## 10. What ships with this

**`inplay_engine.py`** (pure Python, no deps) implements:
- Poisson / Negative-Binomial fitting from mean+variance
- Model-derived pre-game xPA-by-slot and pre-game BF (no external assumption needed)
- Rate calibration from your pre-game projections (hitter & pitcher)
- Pre-game-book gating until the first in-game event
- Live `xRPA` (team-PA model + order-walk) and live `xRemBF` (hook survival model)
- Compound mean/variance for hits, total bases, HR, K, hits-allowed
- Milestone and O/U fair-prob pricing → comps anchoring (de-vig + blend + clamp) → 7% margin
- A worked demo (`python inplay_engine.py`) showing pre-game vs live vs comp-anchored prices

**`inplay_terminal.jsx`** — an interactive trading terminal (a JS port of the same math) to play with in-game situations and compare against comps before you commit to the Python build. Set the game state, lineup, pitch count, and accumulated stats; type competitor decimal prices next to each market; and watch fair value, the comp, and your anchored 7%-loaded price move together. Anchor weight, max-deviation, reach-base rate, and pull probability are all live sliders so you can feel out the behavior and give feedback.
