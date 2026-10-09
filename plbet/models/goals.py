"""Goals model: team ratings from goals and xG, Dixon-Coles score matrix.

Ratings
-------
Each team gets an attacking and a defensive rating from a Poisson model fitted
to a blend of actual goals and non-penalty xG (plus the league-average
penalty xG), weighted so recent matches count more. xG carries most of the
weight because it is a less noisy measure of chance quality than goals.

Score distribution
------------------
Home and away goals are Poisson with means lambda_h and lambda_a, with the
Dixon-Coles adjustment (rho) that corrects the frequencies of 0-0, 1-0, 0-1
and 1-1. Half-time goals are a binomial thinning of full-time goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson

from .glm import TeamCountModel, decay_weights

MAX_GOALS = 10


@dataclass
class GoalsParams:
    xi: float = 0.004         # time decay per day (half-life = ln2 / xi days)
    w_goals: float = 0.3      # weight on actual goals vs xG in the rating target
    tau: float = 2.0          # ridge precision on team effects
    rho_xi: float = 0.001     # slower decay for the (very stable) Dixon-Coles rho
    # Match totals: the ratings spread predicted total goals too widely (an
    # "open game" tendency is less persistent than strength), so each match's
    # predicted total is shrunk towards the league level:
    #     total = level x (raw total / typical raw total) ** total_beta
    # with the level set to actual goals per game over the last
    # ``level_days`` (Understat xG has run 5-13% above goals). total_beta was
    # fitted walk-forward on 2021-22 to 2024-25 (docs/BACKTEST.md).
    total_beta: float = 0.73
    level_days: int = 365


@dataclass
class GoalsModel:
    params: GoalsParams = field(default_factory=GoalsParams)

    model_: TeamCountModel | None = None
    ref_pred_: float = 2.8    # typical raw predicted total goals per match
    ref_act_: float = 2.8     # actual goals per match over the last level_days
    rho_: float = -0.05
    ht_frac_: tuple[float, float] = (0.45, 0.45)
    pen_xg_: float = 0.09     # league-average penalty xG per team per match
    pen_rate_: float = 0.12   # penalties awarded per team per match
    pen_conv_: float = 0.79
    as_of_: pd.Timestamp | None = None

    def fit(self, tr: pd.DataFrame, as_of: pd.Timestamp) -> "GoalsModel":
        """Fit on team rows (features.team_table) strictly before ``as_of``."""
        p = self.params
        hist = tr[tr["kickoff"] < as_of]
        w = decay_weights(hist["kickoff"], as_of, p.xi)
        recent = hist["kickoff"] > as_of - pd.Timedelta(days=3 * 365)
        has_xg = hist["npxg_for"].notna()
        pen_xg = (hist["xg_for"] - hist["npxg_for"])[recent & has_xg]
        if len(pen_xg) > 100:
            self.pen_xg_ = float(pen_xg.mean())
            self.pen_rate_ = float((pen_xg / 0.7612).mean())
        xg_part = (hist["npxg_for"] + self.pen_xg_).to_numpy()
        goals = hist["goals_for"].to_numpy(float)
        y = np.where(has_xg, p.w_goals * goals + (1 - p.w_goals) * xg_part, goals)
        self.model_ = TeamCountModel(family="poisson", tau_team=p.tau).fit(hist, y, w)

        # Level and spread of match totals (see GoalsParams.total_beta).
        lam = self.model_.predict(hist)
        last = (hist["kickoff"] > as_of - pd.Timedelta(days=p.level_days)).to_numpy()
        if last.sum() >= 100:
            self.ref_pred_ = 2 * float(lam[last].mean())
            self.ref_act_ = 2 * float(goals[last].mean())
        lam = self._calibrate(lam, hist["match_key"].to_numpy())

        # Dixon-Coles rho on actual scores, given the fitted means.
        home = hist["is_home"].to_numpy() == 1
        key = hist["match_key"].to_numpy()
        h = pd.DataFrame({"k": key[home], "lh": lam[home], "hg": goals[home],
                          "kick": hist["kickoff"].to_numpy()[home]})
        a = pd.DataFrame({"k": key[~home], "la": lam[~home], "ag": goals[~home]})
        mm = h.merge(a, on="k")
        wr = decay_weights(mm["kick"], as_of, p.rho_xi)
        self.rho_ = fit_rho(mm["lh"].to_numpy(), mm["la"].to_numpy(), mm["hg"].to_numpy(),
                            mm["ag"].to_numpy(), wr)

        # Share of goals scored before half-time, home and away.
        recent_rows = hist[recent]
        fr = []
        for is_home in (1, 0):
            r = recent_rows[recent_rows["is_home"] == is_home]
            fr.append(float(r["ht_goals_for"].sum() / max(r["goals_for"].sum(), 1)))
        self.ht_frac_ = (fr[0], fr[1])
        self.as_of_ = as_of
        return self

    def lambdas(self, rows: pd.DataFrame) -> np.ndarray:
        """Expected goals for each row (team view).

        Rows must come in matches: grouped by ``match_key`` if present,
        otherwise consecutive (home, away) pairs as fixture_rows makes them.
        """
        keys = rows["match_key"].to_numpy() if "match_key" in rows else np.repeat(
            np.arange((len(rows) + 1) // 2), 2)[: len(rows)]
        return self._calibrate(self.model_.predict(rows), keys)

    def _calibrate(self, lam: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Shrink each match's total towards the league level, keeping the split."""
        tot = pd.Series(lam).groupby(keys).transform("sum").to_numpy()
        beta = self.params.total_beta
        target = self.ref_act_ * (tot / self.ref_pred_) ** beta
        return lam * target / tot

    def ratings(self) -> pd.DataFrame:
        """Attack/defence ratings on the log scale (higher defence = better)."""
        t = self.model_.team_table()
        c = self.model_.coef()
        return t.assign(net=t["for"] + t["against"]).sort_values("net", ascending=False) \
            .assign(promoted_attack=c["promoted_team"], promoted_defence=-c["promoted_opp"])


def dc_tau(hg: np.ndarray, ag: np.ndarray, lh, la, rho) -> np.ndarray:
    t = np.ones(np.broadcast(hg, ag, lh, la).shape)
    t = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1 + lh * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1 + la * rho, t)
    t = np.where((hg == 1) & (ag == 1), 1 - rho, t)
    return t


def fit_rho(lh, la, hg, ag, w) -> float:
    def nll(r):
        t = dc_tau(hg, ag, lh, la, r)
        return -np.sum(w * np.log(np.clip(t, 1e-9, None)))
    res = optimize.minimize_scalar(nll, bounds=(-0.25, 0.15), method="bounded")
    return float(res.x)


def score_matrix(lh: float, la: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(home = i, away = j) for i, j in 0..max_goals (renormalised)."""
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g, lh)
    pa = poisson.pmf(g, la)
    mat = np.outer(ph, pa)
    mat[:2, :2] *= dc_tau(g[:2, None], g[None, :2], lh, la, rho)
    mat = np.clip(mat, 0, None)
    return mat / mat.sum()


def outcome_probs(mat: np.ndarray) -> tuple[float, float, float]:
    return float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum())


def total_over(mat: np.ndarray, line: float) -> float:
    g = np.arange(mat.shape[0])
    tot = g[:, None] + g[None, :]
    return float(mat[tot > line].sum())


def implied_lambdas(
    p_home: float, p_draw: float, p_away: float,
    p_over25: float | None = None,
    rho: float = -0.05,
    start: tuple[float, float] = (1.4, 1.1),
) -> tuple[float, float]:
    """Goal means that reproduce market probabilities (1X2 and optionally O/U 2.5)."""
    target = np.array([p_home, p_draw, p_away] + ([p_over25] if p_over25 is not None else []))

    def loss(z):
        lh, la = np.exp(z)
        mat = score_matrix(lh, la, rho)
        pr = list(outcome_probs(mat))
        if p_over25 is not None:
            pr.append(total_over(mat, 2.5))
        pr = np.clip(np.array(pr), 1e-9, 1)
        # Cross-entropy between target and model probabilities for each market.
        ce = -(target[:3] * np.log(pr[:3])).sum()
        if p_over25 is not None:
            ce += -(target[3] * np.log(pr[3]) + (1 - target[3]) * np.log(1 - pr[3]))
        return ce

    res = optimize.minimize(loss, np.log(start), method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
    lh, la = np.exp(res.x)
    return float(lh), float(la)
