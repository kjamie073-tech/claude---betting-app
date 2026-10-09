"""Monte Carlo match simulator.

One simulated match is built in this order, so every number is consistent
with every other number in the same simulation:

1. Final score from the Dixon-Coles score matrix; half-time score by
   thinning each team's goals.
2. Corners, yellow and red cards, fouls and non-goal shots for each team,
   drawn from their score-conditional distributions and tied together with
   the fitted Gaussian copula (plus a random referee effect when the referee
   is unknown).
3. Players: minutes on the pitch, then the team's goals, assists, shots and
   cards are shared out among the players who are on, in proportion to each
   player's rates.

Because of that ordering, a bet builder such as "Arsenal win + Saka to score
+ over 4.5 Arsenal corners" is priced from the same simulated matches, so the
links between the legs are handled automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .counts import COPULA_DIMS, STATS, MatchStatsModel, nb_ppf
from .features import SCORE_COVS, score_covariates
from .goals import score_matrix

OWN_GOAL_SHARE = 0.03     # share of goals that are own goals
ASSIST_RATE = 0.76        # share of open-play goals with an assist (Understat)


@dataclass
class SquadSim:
    """Simulated per-player outputs for one team (arrays are n_sims x n_players)."""
    ids: list
    names: list[str]
    started: np.ndarray
    minutes: np.ndarray
    goals: np.ndarray
    assists: np.ndarray
    shots: np.ndarray
    sot: np.ndarray
    yellow: np.ndarray
    red: np.ndarray
    first_scorer: np.ndarray | None = None   # bool n x P: scored the match's first goal

    def index(self, name: str) -> int:
        from ..data.names import name_score
        scores = [name_score(name, n) for n in self.names]
        best = int(np.argmax(scores))
        if scores[best] < 0.75:
            raise KeyError(f"player '{name}' not in squad: {', '.join(self.names)}")
        return best


@dataclass
class SimResult:
    n: int
    home: str
    away: str
    hg: np.ndarray
    ag: np.ndarray
    hht: np.ndarray
    aht: np.ndarray
    stats: dict[str, np.ndarray]
    players: dict[str, SquadSim] = field(default_factory=dict)
    first_goal: np.ndarray | None = None     # 1 home, -1 away, 0 no goal
    card_weights: tuple[int, int] = (1, 2)   # how bookmakers count (yellow, red)


def simulate_scores(lh, la, rho, ht_frac, n, rng):
    mat = score_matrix(lh, la, rho)
    k = mat.shape[0]
    idx = rng.choice(k * k, size=n, p=mat.ravel())
    hg, ag = np.divmod(idx, k)
    hht = rng.binomial(hg, ht_frac[0])
    aht = rng.binomial(ag, ht_frac[1])
    return hg.astype(np.int16), ag.astype(np.int16), hht.astype(np.int16), aht.astype(np.int16)


def simulate_counts(sm: MatchStatsModel, fixture: pd.DataFrame, hg, ag, rng,
                    referee_known: bool, mean_scale: dict[str, tuple[float, float]] | None = None):
    """Corners, cards, fouls, non-goal shots for both teams, given simulated scores."""
    n = len(hg)
    base = sm.base_eta(fixture)
    ch = score_covariates(hg, ag)[list(SCORE_COVS)].to_numpy()
    ca = score_covariates(ag, hg)[list(SCORE_COVS)].to_numpy()
    ref_z = rng.standard_normal(n) if not referee_known else np.zeros(n)
    z = rng.multivariate_normal(np.zeros(len(COPULA_DIMS)), sm.corr_, size=n, method="cholesky")
    u = stats.norm.cdf(z)
    out = {}
    for name in STATS:
        b = sm.cov_beta(name)
        mod = sm.models_[name]
        ref = ref_z * sm.ref_sd_.get(name, 0.0)
        for side, cov, j in (("h", ch, 0), ("a", ca, 1)):
            eta = base[name][j] + cov @ b + ref
            if mean_scale and name in mean_scale:
                eta = eta + np.log(mean_scale[name][j])
            col = COPULA_DIMS.index(f"{side}_{name}")
            out[f"{side}_{name}"] = nb_ppf(u[:, col], np.exp(eta), mod.theta_)
    return out


# ---------------------------------------------------------------- players


def _minutes(squad: pd.DataFrame, n: int, rng) -> np.ndarray:
    """Simulated minutes for each squad player (n x P)."""
    P = len(squad)
    mins = np.zeros((n, P), dtype=np.float32)
    for j, (_, r) in enumerate(squad.iterrows()):
        if r["start"]:
            full = float(np.clip(r["full"], 0.05, 0.98))
            off_mean = float(np.clip((r["min_st"] - 90 * full) / (1 - full), 45, 85))
            off = np.clip(rng.normal(off_mean, 12, n), 20, 89)
            mins[:, j] = np.where(rng.random(n) < full, 90, off)
        else:
            on = rng.random(n) < r["p_sub"]
            m = np.clip(rng.normal(r["min_sb"], 8, n), 1, 60)
            mins[:, j] = np.where(on, m, 0)
    return mins


def _norm(w: np.ndarray) -> np.ndarray:
    s = w.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return w / s


def _categorical(p: np.ndarray, rng) -> np.ndarray:
    """One draw per row from row-wise probabilities."""
    c = np.cumsum(p, axis=1)
    u = rng.random((p.shape[0], 1)) * c[:, -1:]
    return np.minimum((u > c).sum(axis=1), p.shape[1] - 1)


def simulate_squad(squad: pd.DataFrame, goals: np.ndarray, ngon: np.ndarray, ngoff: np.ndarray,
                   yellow: np.ndarray, red: np.ndarray, pen_goal_share: float,
                   pen_order: list[int], rng) -> SquadSim:
    n, P = len(goals), len(squad)
    mins = _minutes(squad, n, rng)
    frac = mins / 90.0
    w_np = _norm(squad["share_npxg"].to_numpy()[None, :] * frac)
    w_xa = squad["share_xa"].to_numpy()[None, :] * frac
    w_on = _norm(squad["share_ngon"].to_numpy()[None, :] * frac)
    w_off = _norm(squad["share_ngoff"].to_numpy()[None, :] * frac)
    w_y = _norm(squad["yellow90"].to_numpy()[None, :] * frac)
    w_r = _norm(squad["red90"].to_numpy()[None, :] * frac)

    og = rng.binomial(goals, OWN_GOAL_SHARE)
    pen = rng.binomial(goals - og, np.clip(pen_goal_share, 0, 0.35))
    npg = goals - og - pen

    g = np.zeros((n, P), dtype=np.int16)
    a = np.zeros((n, P), dtype=np.int16)
    first = np.zeros((n, P), dtype=bool)
    rows = np.arange(n)
    # Type of the team's first goal (goals are in random order):
    # 0 open play, 1 penalty, 2 own goal.
    u = rng.random(n) * np.maximum(goals, 1)
    first_type = np.where(u < npg, 0, np.where(u < npg + pen, 1, 2))
    # Open-play goals: scorer by npxG share, assister by xA share (not the scorer).
    for k in range(int(npg.max()) if n else 0):
        live = npg > k
        if not live.any():
            break
        idx = rows[live]
        sc = _categorical(w_np[live], rng)
        g[idx, sc] += 1
        if k == 0:
            f = first_type[idx] == 0
            first[idx[f], sc[f]] = True
        assisted = rng.random(len(idx)) < ASSIST_RATE
        wa = w_xa[live].copy()
        wa[np.arange(len(idx)), sc] = 0.0
        ok = assisted & (wa.sum(axis=1) > 0)
        if ok.any():
            asr = _categorical(_norm(wa[ok]), rng)
            a[idx[ok], asr] += 1
    # Penalties: first-choice taker if on the pitch, then the next, then by share.
    if pen.any():
        order = [squad.index.get_loc(pid) for pid in pen_order if pid in squad.index]
        for k in range(int(pen.max())):
            live = pen > k
            idx = rows[live]
            taker = np.full(len(idx), -1)
            for j in order:
                avail = (taker < 0) & (rng.random(len(idx)) < frac[idx, j])
                taker[avail] = j
            rest = taker < 0
            if rest.any():
                taker[rest] = _categorical(w_np[idx[rest]], rng)
            g[idx, taker] += 1
            if k == 0:
                f = first_type[idx] == 1
                first[idx[f], taker[f]] = True
    ngon_p = rng.multinomial(ngon.astype(np.int64), w_on)
    ngoff_p = rng.multinomial(ngoff.astype(np.int64), w_off)
    yel = rng.multinomial(yellow.astype(np.int64), w_y)
    rd = rng.multinomial(red.astype(np.int64), w_r)
    shots = g + ngon_p + ngoff_p
    sot = g + ngon_p
    return SquadSim(
        ids=list(squad.index), names=list(squad["player"]),
        started=squad["start"].to_numpy(bool), minutes=mins, goals=g, assists=a,
        shots=shots.astype(np.int16), sot=sot.astype(np.int16),
        yellow=yel.astype(np.int16), red=rd.astype(np.int16), first_scorer=first,
    )


def simulate_match(
    *,
    home: str,
    away: str,
    lh: float,
    la: float,
    rho: float,
    ht_frac: tuple[float, float],
    stats_model: MatchStatsModel,
    fixture: pd.DataFrame,
    referee_known: bool,
    squads: dict[str, pd.DataFrame] | None = None,
    pen_info: dict[str, tuple[float, list[int]]] | None = None,
    mean_scale: dict[str, tuple[float, float]] | None = None,
    n: int = 100_000,
    seed: int = 7,
    card_weights: tuple[int, int] = (1, 2),
) -> SimResult:
    rng = np.random.default_rng(seed)
    hg, ag, hht, aht = simulate_scores(lh, la, rho, ht_frac, n, rng)
    st = simulate_counts(stats_model, fixture, hg, ag, rng, referee_known, mean_scale)
    res = SimResult(n=n, home=home, away=away, hg=hg, ag=ag, hht=hht, aht=aht, stats=st,
                    card_weights=card_weights)
    # Which side scored first: each goal equally likely to be the first.
    tot = hg + ag
    res.first_goal = np.where(tot == 0, 0,
                              np.where(rng.random(n) * np.maximum(tot, 1) < hg, 1, -1))
    if squads:
        for side, goals, own in (("h", hg, "h"), ("a", ag, "a")):
            sq = squads.get(side)
            if sq is None or sq.empty:
                continue
            share, order = (pen_info or {}).get(side, (0.08, []))
            res.players[side] = simulate_squad(
                sq, goals, st[f"{own}_ngon"], st[f"{own}_ngoff"], st[f"{own}_yellow"],
                st[f"{own}_red"], share, order, rng)
            # Only the scorer of that team's first goal can be the match's first
            # scorer, and only if that team scored first.
            fs = res.players[side].first_scorer
            team_first = res.first_goal == (1 if side == "h" else -1)
            res.players[side].first_scorer = fs & team_first[:, None]
    return res
