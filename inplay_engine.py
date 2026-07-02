"""
MLB In-Play Player-Prop Pricing Engine — reference implementation.

Design principle: at any instant a player's final stat S decomposes as

        S_final = S_locked  +  S_remaining

  * S_locked      = what has already happened (a known integer)
  * S_remaining   = compound random variable  sum_{i=1..N} X_i
                       N   = remaining opportunities (PA for hitters,
                             batters-faced for pitchers) -- itself random
                       X_i = per-opportunity outcome (hit / TB / HR / K ...)

We track the MEAN and VARIANCE of S_remaining analytically (law of total
expectation / variance over N), fit a discrete distribution to that
mean+variance, shift by S_locked, and read every market off the resulting
CDF.  This keeps live prices consistent with the pre-game book at t=0.

The module is deliberately dependency-light (pure Python + math) so it can
run anywhere and be ported.  Swap in numpy/scipy for speed in production.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from math import exp, lgamma, log, sqrt
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------------
# 1.  Discrete distributions:  fit a count distribution to (mean, var)
# ----------------------------------------------------------------------------
# For "count" stats (hits, HR, K, hits-allowed) we fit a Negative Binomial
# when var > mean (overdispersion, the usual case once N is random) and fall
# back to Poisson when var ~= mean.  For total bases (a compound sum of a
# {0,1,2,3,4} outcome) we carry the distribution on a support grid directly.

def _log_nbinom_pmf(k: int, r: float, p: float) -> float:
    # NB parameterised by (r>0 "successes", p in (0,1)); mean=r(1-p)/p
    return (lgamma(k + r) - lgamma(k + 1) - lgamma(r)
            + r * log(p) + k * log(1.0 - p))


def nbinom_from_mean_var(mean: float, var: float) -> Tuple[float, float]:
    """Return (r, p) matching a target mean & variance. Requires var>mean."""
    # mean = r(1-p)/p ; var = r(1-p)/p^2  =>  p = mean/var,  r = mean^2/(var-mean)
    var = max(var, mean * 1.0000001)          # guard: NB needs var>mean
    p = mean / var
    r = mean * mean / (var - mean)
    return r, p


def count_cdf(mean: float, var: float, kmax: int = 60) -> List[float]:
    """CDF list where cdf[k] = P(S <= k). Chooses NB or Poisson automatically."""
    pmf = [0.0] * (kmax + 1)
    if mean <= 1e-9:
        pmf[0] = 1.0
    elif var <= mean * 1.05:                  # ~Poisson regime
        # Poisson pmf via recurrence
        p0 = exp(-mean)
        pmf[0] = p0
        for k in range(1, kmax + 1):
            pmf[k] = pmf[k - 1] * mean / k
    else:                                     # Negative Binomial
        r, p = nbinom_from_mean_var(mean, var)
        for k in range(kmax + 1):
            pmf[k] = exp(_log_nbinom_pmf(k, r, p))
    # normalise & accumulate
    tot = sum(pmf)
    cdf, run = [], 0.0
    for k in range(kmax + 1):
        run += pmf[k] / tot
        cdf.append(min(run, 1.0))
    return cdf


# ----------------------------------------------------------------------------
# 2.  Per-opportunity outcome profiles  (calibrated from the pre-game book)
# ----------------------------------------------------------------------------
@dataclass
class HitterRates:
    """Per-PA outcome probabilities. Calibrated so that
       rate * pregame_xPA == the pre-game projection (t=0 consistency)."""
    p_hit: float          # P(hit | PA)
    p_1b: float           # P(single | PA)
    p_2b: float
    p_3b: float
    p_hr: float
    p_bb_hbp: float       # walk/hbp (a PA that is not an AB, no hit/TB)

    @property
    def tb_mean(self) -> float:
        return self.p_1b + 2 * self.p_2b + 3 * self.p_3b + 4 * self.p_hr

    @property
    def tb_var(self) -> float:
        m = self.tb_mean
        e2 = self.p_1b + 4 * self.p_2b + 9 * self.p_3b + 16 * self.p_hr
        return e2 - m * m


@dataclass
class PitcherRates:
    """Per-batter-faced outcome probabilities."""
    p_k: float            # P(strikeout | BF)
    p_hit_allowed: float  # P(hit allowed | BF)
    p_out_nonk: float     # P(out that is not a K | BF)  (for outs bookkeeping)


def calibrate_hitter(proj_hits: float, proj_tb: float, proj_hr: float,
                     pregame_xpa: float,
                     xbh_split=(0.62, 0.26, 0.04, 0.08)) -> HitterRates:
    """Back out per-PA rates from the pre-game projections.

    xbh_split = (share of hits that are 1B, 2B, 3B, HR) as a starting shape;
    it is then rescaled so hit-rate, TB and HR projections are ALL honoured.

    NOTE: pregame_xpa is now MODEL-DERIVED from lineup slot (see
    pregame_xpa_by_slot) rather than supplied, since the book has no
    pre-game xPA of its own.
    """
    p_hit = proj_hits / pregame_xpa
    p_hr = proj_hr / pregame_xpa
    # distribute remaining (non-HR) hits across 1B/2B/3B using the split shape
    non_hr_hits = max(p_hit - p_hr, 1e-9)
    s1, s2, s3, _ = xbh_split
    denom = s1 + s2 + s3
    p_1b = non_hr_hits * s1 / denom
    p_2b = non_hr_hits * s2 / denom
    p_3b = non_hr_hits * s3 / denom
    # nudge doubles/triples so total-bases projection is matched exactly
    tb_from_hits = p_1b + 2 * p_2b + 3 * p_3b + 4 * p_hr
    target_tb_rate = proj_tb / pregame_xpa
    if tb_from_hits > 0:
        scale_xb = (target_tb_rate - p_1b - 4 * p_hr) / max(2 * p_2b + 3 * p_3b, 1e-9)
        scale_xb = max(0.2, min(3.0, scale_xb))
        p_2b *= scale_xb
        p_3b *= scale_xb
    p_bb_hbp = 0.085                      # league-ish; affects PA-not-AB, not hits/TB
    return HitterRates(p_hit, p_1b, p_2b, p_3b, p_hr, p_bb_hbp)


def calibrate_pitcher(proj_k: float, proj_hits_allowed: float,
                      pregame_bf: float) -> PitcherRates:
    p_k = proj_k / pregame_bf
    p_h = proj_hits_allowed / pregame_bf
    # remaining PA end in a non-K out or a walk; assume ~9% walks
    p_out_nonk = max(1.0 - p_k - p_h - 0.09, 0.05)
    return PitcherRates(p_k, p_h, p_out_nonk)


# ----------------------------------------------------------------------------
# 2b.  Derive the full-game mean (projection) from the book market
# ----------------------------------------------------------------------------
# The O/U line + de-vigged over price is the real read on expected performance,
# not a hand-entered projection. Solve for the mean whose distribution (spread
# set by std dev) reproduces the market's fair P(over line). Monotonic -> bisect.
def implied_mean_from_ou(line: float, over: Optional[float],
                         under: Optional[float], std: float = 0.0) -> Optional[float]:
    po = 1.0 / over if over and over > 1 else None
    pu = 1.0 / under if under and under > 1 else None
    if po and pu:
        fair = po / (po + pu)
    elif po:
        fair = po / 1.03
    else:
        return None
    fair = min(0.995, max(0.005, fair))
    lo, hi = 0.001, 30.0
    for _ in range(46):
        mid = (lo + hi) / 2
        var = std * std if std and std > 0 else mid
        cdf = count_cdf(mid, var, 90)
        k = int(math.floor(line))
        p_over = 1.0 - (cdf[k] if 0 <= k < len(cdf) else 1.0)
        if p_over < fair:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def implied_mean_from_1plus(price: Optional[float], std: float = 0.0) -> Optional[float]:
    p = 1.0 / price if price and price > 1 else None
    if not p:
        return None
    fair = min(0.97, max(0.003, p / 1.03))
    lo, hi = 0.001, 6.0
    for _ in range(42):
        mid = (lo + hi) / 2
        var = std * std if std and std > 0 else mid
        cdf = count_cdf(mid, var, 40)
        p1 = 1.0 - cdf[0]
        if p1 < fair:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------------------
# 3.  Expected REMAINING opportunities  (the live core)
# ----------------------------------------------------------------------------
@dataclass
class GameState:
    inning: int                 # 1..9(+)
    top: bool                   # True = top half (away batting)
    outs: int                   # 0..2
    away_score: int
    home_score: int
    # batting-order pointer for each side: index 0..8 of who's due up NEXT
    away_due_up: int = 0
    home_due_up: int = 0


def expected_remaining_team_pa(gs: GameState, batting_home: bool,
                               reach_base_rate: float = 0.320) -> Tuple[float, float]:
    """Mean & variance of remaining PA for the *batting* team.

    Model: each remaining half-inning contributes 3 outs; PA per half-inning
    ~ 3 + Geometric baserunners with reach-base prob q  =>  mean 3/(1-q).
    We sum over the half-innings this team still bats, discount the bottom-9th
    if the home team is already ahead, and add a small extra-innings mass.
    """
    q = reach_base_rate
    pa_per_full_inning_mean = 3.0 / (1.0 - q)
    # variance of PA in one inning (sum of a fixed 3 outs among Geometric trials):
    # Var ~ 3 * q / (1-q)^2   (negative-binomial number of successes before 3 failures)
    pa_per_inning_var = 3.0 * q / (1.0 - q) ** 2

    # how many *full* innings does this team have left to bat?
    innings_left = _batting_innings_left(gs, batting_home)

    # partial current inning: outs already recorded shrink the current frame
    cur_frac = 0.0
    team_is_batting_now = (batting_home and not gs.top) or ((not batting_home) and gs.top)
    if team_is_batting_now and gs.inning <= 9:
        cur_frac = max(0.0, (3 - gs.outs) / 3.0)

    mean_pa = pa_per_inning_var * 0  # placeholder to keep var name in scope
    mean_pa = innings_left * pa_per_full_inning_mean + cur_frac * pa_per_full_inning_mean
    var_pa = (innings_left + cur_frac) * pa_per_inning_var

    # walk-off / bottom-9 truncation: if home leads entering its last at-bat,
    # it may not bat.  Approximate by shaving ~50% of one inning's PA when close.
    if batting_home and gs.inning >= 9 and gs.home_score > gs.away_score:
        mean_pa = max(0.0, mean_pa - 0.5 * pa_per_full_inning_mean)

    # extra-innings tail: only once we're in/through the 9th and still tied.
    # before that, the 9th is already counted as a normal inning.
    if gs.inning >= 9 and gs.away_score == gs.home_score:
        p_extra = 0.33
        mean_pa += p_extra * pa_per_full_inning_mean
        var_pa += p_extra * (1 - p_extra) * (pa_per_full_inning_mean) ** 2
    return mean_pa, var_pa


def _batting_innings_left(gs: GameState, batting_home: bool) -> float:
    """Whole future innings (excluding current partial) this team still bats."""
    reg = 9
    if gs.inning > reg:
        return 0.0
    # innings strictly after the current one in which this team bats
    future = reg - gs.inning
    # does this team still bat *later in the current inning*? handled by cur_frac
    return float(max(future, 0))


def pregame_xpa_by_slot(lineup_slot: int, batting_home: bool = False,
                        reach_base_rate: float = 0.320) -> float:
    """MODEL-DERIVED full-game expected PA for a lineup slot (0..8).

    Uses the *same* live machinery evaluated from a fresh game so pre-game
    calibration and in-play pricing are internally consistent. No external
    pre-game xPA is required (the book doesn't provide one)."""
    fresh = GameState(inning=1, top=(not batting_home), outs=0,
                      away_score=0, home_score=0,
                      away_due_up=0, home_due_up=0)
    mean_pa, _ = hitter_remaining_pa(fresh, lineup_slot, batting_home,
                                     reach_base_rate, pull_prob=0.0)
    return mean_pa


def pregame_bf(hook_pitch_limit: int = 95, is_starter: bool = True) -> float:
    """MODEL-DERIVED expected batters faced for a starter, from the hook
    survival model evaluated at pitch 0."""
    fresh = PitcherState(pitches_thrown=0, outs_recorded=0, batters_faced=0,
                         hits_allowed_so_far=0, strikeouts_so_far=0,
                         times_through_order=0.0, is_starter=is_starter,
                         hook_pitch_limit=hook_pitch_limit)
    # rates don't affect BF survival, pass a dummy
    mean_bf, _ = pitcher_remaining_bf(fresh, PitcherRates(0.24, 0.22, 0.45))
    return mean_bf


def hitter_remaining_pa(gs: GameState, lineup_slot: int, batting_home: bool,
                        reach_base_rate: float = 0.320,
                        pull_prob: float = 0.0) -> Tuple[float, float]:
    """Distribute the team's remaining PA across the 9 slots to get THIS
    hitter's expected remaining PA (mean & variance).

    We walk the order forward from 'due_up' and count how many times the
    pointer laps back to `lineup_slot` given the team's expected remaining PA.
    """
    team_mean, team_var = expected_remaining_team_pa(gs, batting_home, reach_base_rate)
    due_up = gs.home_due_up if batting_home else gs.away_due_up

    # position of this slot in the upcoming sequence (0 = up next)
    offset = (lineup_slot - due_up) % 9
    # expected number of times this slot bats out of team_mean upcoming PA:
    #   slot bats on PA numbers offset+1, offset+10, offset+19, ...
    mean_pa = 0.0
    for j in range(0, 15):
        pa_index = offset + 1 + 9 * j       # 1-based position in the queue
        if pa_index <= team_mean:
            mean_pa += 1.0
        else:
            # fractional credit for the final, partially-reached turn
            frac = max(0.0, min(1.0, team_mean - (pa_index - 1)))
            mean_pa += frac
            break
    # variance: dominated by the "one more / one fewer turn" uncertainty.
    # approximate as Bernoulli on the marginal turn plus team-level dispersion.
    marginal = mean_pa - int(mean_pa)
    var_pa = marginal * (1 - marginal) + team_var / 81.0
    # possibility of being pulled (pinch-hit / substitution) scales opportunities down
    mean_pa *= (1.0 - pull_prob)
    var_pa = var_pa * (1 - pull_prob) + (pull_prob) * mean_pa ** 2 * 0.0
    return mean_pa, var_pa


# ----------------------------------------------------------------------------
# 4.  Pitcher survival: expected remaining batters faced
# ----------------------------------------------------------------------------
@dataclass
class PitcherState:
    pitches_thrown: int
    outs_recorded: int          # this pitcher, this game
    batters_faced: int
    hits_allowed_so_far: int
    strikeouts_so_far: int
    times_through_order: float  # e.g. 2.3
    is_starter: bool = True
    hook_pitch_limit: int = 100 # soft cap
    score_diff_for_team: int = 0  # +ahead / -behind (their team)


def pitcher_remaining_bf(ps: PitcherState,
                         rates: PitcherRates) -> Tuple[float, float]:
    """Survival model for batters faced remaining.

    Each upcoming batter, the pitcher is removed with hazard h that rises with
    pitch count, times-through-order, and adverse game state.  Expected
    remaining BF = sum over future batters of the survival probability.
    """
    # baseline pitches per upcoming batter
    ppb = 3.9
    survive = 1.0
    mean_bf = 0.0
    e_x2 = 0.0                       # for variance of a sum of survival indicators
    proj_pitch = ps.pitches_thrown
    tto = ps.times_through_order

    for i in range(1, 40):          # cap the horizon
        proj_pitch += ppb
        tto_i = tto + i / 9.0
        # hazard of being pulled BEFORE facing this batter
        h = _hook_hazard(proj_pitch, tto_i, ps)
        survive *= (1.0 - h)
        mean_bf += survive
        e_x2 += survive             # E[(sum I)^2] built below via pairwise later
        if survive < 0.01:
            break
    # variance of sum of (positively correlated) survival indicators.
    # Indicators are nested (I_{i+1} <= I_i), so Var = sum_i s_i(1-s_i)
    # + 2*sum_{i<j}(s_j - s_i s_j).  For nested events s_j = P(reach j) and
    # cov(I_i,I_j)=s_j - s_i s_j (j>i).  We approximate with the diagonal +
    # a correlation inflation factor.
    var_bf = 0.0
    survive = 1.0
    proj_pitch = ps.pitches_thrown
    svs = []
    for i in range(1, 40):
        proj_pitch += ppb
        tto_i = tto + i / 9.0
        h = _hook_hazard(proj_pitch, tto_i, ps)
        survive *= (1.0 - h)
        svs.append(survive)
        if survive < 0.01:
            break
    for i, si in enumerate(svs):
        var_bf += si * (1 - si)
        for j in range(i + 1, len(svs)):
            sj = svs[j]
            var_bf += 2.0 * (sj - si * sj)   # cov for nested indicators
    return mean_bf, max(var_bf, 0.0)


def _hook_hazard(proj_pitch: float, tto: float, ps: PitcherState) -> float:
    """Probability the manager removes the pitcher before the next batter."""
    if not ps.is_starter:
        # relievers: short leash by role, hazard rises fast after ~1 inning
        base = 0.12
        over = max(0.0, ps.batters_faced - 4) * 0.06
        return min(0.9, base + over)
    # starter: logistic in pitch count centred near the soft limit.
    # Centre at the limit itself and use a wider scale so the survival curve
    # integrates to ~ (limit / pitches-per-batter) batters faced pre-game.
    limit = ps.hook_pitch_limit
    z = (proj_pitch - limit) / 11.0
    h_pitch = 1.0 / (1.0 + exp(-z))
    # times-through-order penalty: sharp rise crossing the 3rd time (tto>2)
    tto_pen = max(0.0, (tto - 2.0)) * 0.06
    # game state: pull faster if losing badly; extend if blowout win (save pen)
    state = 0.0
    if ps.score_diff_for_team <= -4:
        state += 0.08
    if ps.score_diff_for_team >= 6:
        state -= 0.05
    return max(0.0, min(0.95, 0.005 + 0.9 * h_pitch + tto_pen + state))


# ----------------------------------------------------------------------------
# 5.  Assemble final-stat distribution and price the markets
# ----------------------------------------------------------------------------
def remaining_count_moments(mean_n: float, var_n: float,
                            p: float) -> Tuple[float, float]:
    """Compound (random N, Bernoulli(p) per opportunity) mean & variance."""
    mean = mean_n * p
    var = mean_n * p * (1 - p) + var_n * p * p
    return mean, var


def remaining_tb_moments(mean_n: float, var_n: float,
                         rates: HitterRates) -> Tuple[float, float]:
    mu_x, s2_x = rates.tb_mean, rates.tb_var
    mean = mean_n * mu_x
    var = mean_n * s2_x + var_n * mu_x * mu_x
    return mean, var


def price_over_under(cdf: List[float], locked: int, line: float) -> Tuple[float, float]:
    """Return fair (p_over, p_under) for a total-final vs a .5 line."""
    # need final = locked + remaining > line  => remaining > line - locked
    thresh = line - locked
    # P(remaining <= k) from cdf; for a .5 line, over means remaining >= ceil(line-locked)
    import math
    need = math.floor(thresh) if abs(thresh - round(thresh)) > 1e-9 else int(round(thresh))
    # over: remaining >= need+1 when line is X.5
    k = int(math.floor(thresh))
    p_under = cdf[k] if 0 <= k < len(cdf) else (1.0 if k >= len(cdf) else 0.0)
    p_over = 1.0 - p_under
    return max(0.0, min(1.0, p_over)), max(0.0, min(1.0, p_under))


def price_milestone(cdf: List[float], locked: int, k_plus: int) -> float:
    """Fair P(final >= k_plus)  =  P(remaining >= k_plus - locked)."""
    need = k_plus - locked
    if need <= 0:
        return 1.0
    idx = need - 1
    if idx >= len(cdf):
        return 0.0
    return max(0.0, min(1.0, 1.0 - cdf[idx]))


def fair_prob_to_decimal(p: float, margin: float = 0.07) -> Optional[float]:
    """Convert a fair prob to a decimal price, optionally loading `margin`
    (default 7% overround applied to this side)."""
    if p <= 0.0:
        return None
    eff = min(0.999, p * (1.0 + margin))
    return round(1.0 / eff, 2)


def decimal_to_prob(price: Optional[float]) -> Optional[float]:
    if not price or price <= 1.0:
        return None
    return 1.0 / price


def devig_two_way(over_price: Optional[float],
                  under_price: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """Remove the book's hold from a two-way comp market -> fair probs."""
    po, pu = decimal_to_prob(over_price), decimal_to_prob(under_price)
    if po and pu:
        s = po + pu
        return po / s, pu / s
    # one-sided: assume a typical 6% hold to estimate the fair side
    if po:
        return min(po / 1.03, 0.999), None
    if pu:
        return None, min(pu / 1.03, 0.999)
    return None, None


def anchor_to_comp(our_fair_prob: float,
                   comp_fair_prob: Optional[float],
                   weight: float = 0.5,
                   max_dev: float = 0.06) -> float:
    """Pull our fair probability toward a competitor's fair probability.

    weight   : 0 = ignore comp, 1 = match comp exactly. Blends the two probs.
    max_dev  : hard cap on how far (in probability) the FINAL number may sit
               from the comp. Guarantees we're never "too far" from comps.

    Returns the anchored fair probability (pre-margin).
    """
    if comp_fair_prob is None:
        return our_fair_prob
    blended = (1.0 - weight) * our_fair_prob + weight * comp_fair_prob
    # clamp to within max_dev of the comp
    lo, hi = comp_fair_prob - max_dev, comp_fair_prob + max_dev
    return max(lo, min(hi, blended))


# ----------------------------------------------------------------------------
# 6.  Top-level convenience: price one hitter, all markets, live
# ----------------------------------------------------------------------------
def price_hitter_live(rates: HitterRates, gs: GameState, lineup_slot: int,
                      batting_home: bool, locked_hits: int, locked_tb: int,
                      locked_hr: int, reach_base_rate: float = 0.320,
                      pull_prob: float = 0.0, margin: float = 0.07,
                      suspended: bool = False,
                      first_event_seen: bool = True,
                      pregame_prices: Optional[Dict] = None) -> Dict:
    if suspended:
        return {"status": "SUSPENDED"}
    # RULE: use the pre-game book price until the player's first in-game event.
    if not first_event_seen and pregame_prices is not None:
        return {"status": "PREGAME", "source": "book", "markets": pregame_prices}
    mean_n, var_n = hitter_remaining_pa(gs, lineup_slot, batting_home,
                                        reach_base_rate, pull_prob)
    out: Dict = {"status": "OPEN", "xRPA": round(mean_n, 3),
                 "markets": {}}

    # --- Hits ---
    m, v = remaining_count_moments(mean_n, var_n, rates.p_hit)
    cdf = count_cdf(m, v)
    hits = {"line_1.5": _ou(cdf, locked_hits, 1.5, margin),
            "line_0.5": _ou(cdf, locked_hits, 0.5, margin),
            "milestones": {f"{k}+": fair_prob_to_decimal(
                price_milestone(cdf, locked_hits, k), margin) for k in (1, 2, 3, 4)},
            "fair_prob_milestones": {f"{k}+": round(
                price_milestone(cdf, locked_hits, k), 4) for k in (1, 2, 3, 4)}}
    out["markets"]["hits"] = hits

    # --- Total bases ---
    m, v = remaining_tb_moments(mean_n, var_n, rates)
    cdf = count_cdf(m, v, kmax=80)
    tb = {"line_1.5": _ou(cdf, locked_tb, 1.5, margin),
          "line_2.5": _ou(cdf, locked_tb, 2.5, margin),
          "milestones": {f"{k}+": fair_prob_to_decimal(
              price_milestone(cdf, locked_tb, k), margin) for k in range(1, 9)}}
    out["markets"]["total_bases"] = tb

    # --- Home runs ---
    m, v = remaining_count_moments(mean_n, var_n, rates.p_hr)
    cdf = count_cdf(m, v)
    hr = {"milestones": {f"{k}+": fair_prob_to_decimal(
        price_milestone(cdf, locked_hr, k), margin) for k in (1, 2)},
        "fair_prob": {f"{k}+": round(price_milestone(cdf, locked_hr, k), 4)
                      for k in (1, 2)}}
    out["markets"]["home_runs"] = hr
    return out


def price_pitcher_live(rates: PitcherRates, ps: PitcherState,
                       locked_k: int, locked_h: int,
                       margin: float = 0.07, suspended: bool = False,
                       first_event_seen: bool = True,
                       pregame_prices: Optional[Dict] = None) -> Dict:
    if suspended:
        return {"status": "SUSPENDED"}
    if not first_event_seen and pregame_prices is not None:
        return {"status": "PREGAME", "source": "book", "markets": pregame_prices}
    mean_bf, var_bf = pitcher_remaining_bf(ps, rates)
    out: Dict = {"status": "OPEN", "xRemBF": round(mean_bf, 3), "markets": {}}

    m, v = remaining_count_moments(mean_bf, var_bf, rates.p_k)
    cdf = count_cdf(m, v, kmax=30)
    out["markets"]["strikeouts"] = {
        "line_5.5": _ou(cdf, locked_k, 5.5, margin),
        "line_6.5": _ou(cdf, locked_k, 6.5, margin),
        "milestones": {f"{k}+": fair_prob_to_decimal(
            price_milestone(cdf, locked_k, k), margin) for k in (4, 6, 8, 10)}}

    m, v = remaining_count_moments(mean_bf, var_bf, rates.p_hit_allowed)
    cdf = count_cdf(m, v, kmax=30)
    out["markets"]["hits_allowed"] = {
        "line_5.5": _ou(cdf, locked_h, 5.5, margin),
        "milestones": {f"{k}+": fair_prob_to_decimal(
            price_milestone(cdf, locked_h, k), margin) for k in (4, 6, 8)}}
    return out


def _ou(cdf, locked, line, margin):
    po, pu = price_over_under(cdf, locked, line)
    return {"over": fair_prob_to_decimal(po, margin),
            "under": fair_prob_to_decimal(pu, margin),
            "fair_over": round(po, 4), "fair_under": round(pu, 4)}


# ----------------------------------------------------------------------------
# 7.  Demo / self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    slot = 2                       # Goldschmidt batting 3rd (index 2)
    xpa = pregame_xpa_by_slot(slot, batting_home=False)
    print(f"Model-derived pre-game xPA for slot {slot+1}: {xpa:.3f}")

    # MEANS come from the book market (line + de-vigged over price), NOT a
    # hand-entered projection. Hits O/U 0.5 @ 1.43/2.74, TB O/U 1.5 @ 1.92/1.82.
    hits_mean = implied_mean_from_ou(0.5, 1.43, 2.74, std=0.9)
    tb_mean = implied_mean_from_ou(1.5, 1.92, 1.82, std=1.67)
    hr_mean = implied_mean_from_1plus(5.00, std=0.5)
    print(f"Market-derived means -> hits {hits_mean:.2f}, TB {tb_mean:.2f}, HR {hr_mean:.2f}")
    gold = calibrate_hitter(hits_mean, tb_mean, hr_mean, pregame_xpa=xpa)

    print("\n--- LIVE, mid-game (top 6, 1 out, 1 hit=double=2TB) ---")
    gs1 = GameState(inning=6, top=True, outs=1, away_score=2, home_score=3,
                    away_due_up=1)
    r1 = price_hitter_live(gold, gs1, slot, False, 1, 2, 0,
                           first_event_seen=True, margin=0.07)
    print("xRPA:", r1["xRPA"], "| hits 1.5 fair over:",
          r1["markets"]["hits"]["line_1.5"]["fair_over"])

    print("\n--- COMPS ANCHORING (DraftKings hits 1.5: over 2.20 / under 1.67) ---")
    our_over = r1["markets"]["hits"]["line_1.5"]["fair_over"]
    comp_over, _ = devig_two_way(2.20, 1.67)
    anchored = anchor_to_comp(our_over, comp_over, weight=0.5, max_dev=0.06)
    print("Ours:", fair_prob_to_decimal(our_over, 0.07),
          "| comp: 2.20 | anchored @7%:", fair_prob_to_decimal(anchored, 0.07))

    print("\n--- Pitcher: K O/U 5.5 @ 1.80/2.00, HA O/U 5.5 @ 1.90/1.90 ---")
    bf = pregame_bf(hook_pitch_limit=95)
    k_mean = implied_mean_from_ou(5.5, 1.80, 2.00, std=2.4)
    ha_mean = implied_mean_from_ou(5.5, 1.90, 1.90, std=2.0)
    print(f"Model-derived pre-game BF: {bf:.2f} | market means K {k_mean:.2f}, HA {ha_mean:.2f}")
    warren = calibrate_pitcher(k_mean, ha_mean, bf)
    ps1 = PitcherState(pitches_thrown=80, outs_recorded=15, batters_faced=20,
                       hits_allowed_so_far=4, strikeouts_so_far=6,
                       times_through_order=2.2, hook_pitch_limit=95,
                       score_diff_for_team=1)
    pr1 = price_pitcher_live(warren, ps1, locked_k=6, locked_h=4, margin=0.07)
    print("xRemBF:", pr1["xRemBF"],
          "| K 6.5 over priced:", pr1["markets"]["strikeouts"]["line_6.5"]["over"])
