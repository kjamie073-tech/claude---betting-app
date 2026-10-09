"""Corners, cards, fouls and shots, conditional on the final score.

Each team-level count (corners, yellow cards, red cards, fouls, shots on
target that were not goals, shots off target) gets its own negative binomial
(or Poisson) model with team for/against ratings, home advantage, referee
effects for the disciplinary stats, and score-state covariates.

The counts within a match are not independent: a team that dominates takes
more corners and more shots while its opponent takes fewer, and an ill-
tempered game raises both teams' cards. Those leftover correlations are
measured from data and reproduced with a Gaussian copula when simulating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .features import SCORE_COVS
from .glm import TeamCountModel, decay_weights

# stat name -> (team-table column, family, uses referee effect)
STATS: dict[str, tuple[str, str, bool]] = {
    "corners": ("corners_for", "negbin", False),
    "yellow": ("yellow_for", "negbin", True),
    "red": ("red_for", "poisson", True),
    "fouls": ("fouls_for", "negbin", True),
    "ngon": ("ngon_for", "negbin", False),
    "ngoff": ("ngoff_for", "negbin", False),
}
COPULA_DIMS = [f"{side}_{s}" for s in STATS for side in ("h", "a")]


# Per-stat (time decay per day, ridge on team effects, ridge on referee
# effects), chosen by walk-forward log-likelihood on 2022-23 to 2024-25.
DEFAULT_STAT_PARAMS: dict[str, tuple[float, float, float]] = {
    "corners": (0.003, 25.0, 20.0),
    "yellow": (0.004, 25.0, 150.0),
    "red": (0.001, 60.0, 150.0),
    "fouls": (0.003, 60.0, 150.0),
    "ngon": (0.003, 25.0, 20.0),
    "ngoff": (0.005, 25.0, 20.0),
}


# Match totals of each stat are shrunk towards the league's typical total:
#     total = typical x (raw total / typical) ** beta
# because team ratings spread totals too widely (a team's tendency to play in
# high-corner games persists less than its strength). Fitted walk-forward on
# 2021-22 to 2024-25 (docs/BACKTEST.md). Red cards show no reliable
# team-to-team signal, so every match gets the league's typical total.
DEFAULT_TOTAL_BETA: dict[str, float] = {
    "corners": 0.59, "yellow": 1.0, "red": 0.0, "fouls": 1.0, "ngon": 0.58, "ngoff": 0.65,
}


@dataclass
class CountParams:
    stat_params: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: dict(DEFAULT_STAT_PARAMS))
    total_beta: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOTAL_BETA))
    copula_years: float = 3.0
    level_days: int = 365


@dataclass
class MatchStatsModel:
    params: CountParams = field(default_factory=CountParams)
    models_: dict[str, TeamCountModel] = field(default_factory=dict)
    corr_: np.ndarray | None = None
    ref_sd_: dict[str, float] = field(default_factory=dict)
    typical_: dict[str, float] = field(default_factory=dict)
    as_of_: pd.Timestamp | None = None

    def fit(self, tr: pd.DataFrame, as_of: pd.Timestamp) -> "MatchStatsModel":
        p = self.params
        hist = tr[tr["kickoff"] < as_of]
        for name, (col, fam, use_ref) in STATS.items():
            xi, tau, tau_ref = p.stat_params[name]
            w = decay_weights(hist["kickoff"], as_of, xi)
            y = hist[col].to_numpy(float)
            self.models_[name] = TeamCountModel(
                family=fam, tau_team=tau, tau_ref=tau_ref, use_ref=use_ref,
                covariates=SCORE_COVS,
            ).fit(hist, y, w)
            if use_ref:
                eff = self.models_[name].ref_effects()
                counts = hist["referee"].value_counts()
                wts = counts.reindex(eff.index).fillna(0).to_numpy()
                self.ref_sd_[name] = float(np.sqrt(np.average(eff.to_numpy() ** 2, weights=wts))) \
                    if wts.sum() else 0.0
        # Typical match total of each stat (score covariates at zero), recent games.
        last = hist[hist["kickoff"] > as_of - pd.Timedelta(days=p.level_days)]
        if len(last) >= 200:
            base = self._raw_eta(last)
            for name in STATS:
                self.typical_[name] = 2 * float(np.exp(base[name]).mean())
        self.corr_ = self._copula(hist[hist["kickoff"] > as_of - pd.Timedelta(
            days=365 * p.copula_years)])
        self.as_of_ = as_of
        return self

    def _copula(self, hist: pd.DataFrame) -> np.ndarray:
        """Correlation of normal scores of the residuals (randomised PIT)."""
        rng = np.random.default_rng(12345)
        hist = hist.sort_values(["match_key", "is_home"], ascending=[True, False])
        # keep matches that have both rows
        keep = hist.groupby("match_key")["is_home"].transform("size") == 2
        hist = hist[keep]
        cols = {}
        for name, (col, fam, _) in STATS.items():
            mod = self.models_[name]
            mu = mod.predict(hist)
            y = hist[col].to_numpy(float)
            dist = _dist(mu, mod.theta_)
            u = dist.cdf(y - 1) + rng.uniform(size=len(y)) * dist.pmf(y)
            z = stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9)).reshape(-1, 2)
            cols[f"h_{name}"], cols[f"a_{name}"] = z[:, 0], z[:, 1]
        z = pd.DataFrame(cols)[COPULA_DIMS].to_numpy()
        z = z[np.isfinite(z).all(axis=1)]
        c = np.corrcoef(z, rowvar=False)
        # nearest positive-definite (clip eigenvalues) to be safe
        vals, vecs = np.linalg.eigh(c)
        c = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
        d = np.sqrt(np.diag(c))
        return c / np.outer(d, d)

    # ------------------------------------------------------------------
    def _raw_eta(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        rows = rows.copy()
        for c in SCORE_COVS:
            rows[c] = 0.0
        return {name: mod.linear_predictor(rows) for name, mod in self.models_.items()}

    def base_eta(self, fixture: pd.DataFrame) -> dict[str, np.ndarray]:
        """Linear predictor for each stat with score covariates set to zero.

        ``fixture`` is the (home, away) pair of rows for one match; each stat's
        match total is shrunk towards the typical total (DEFAULT_TOTAL_BETA).
        """
        out = self._raw_eta(fixture)
        for name, eta in out.items():
            typ = self.typical_.get(name)
            beta = self.params.total_beta.get(name, 1.0)
            if typ is None or beta == 1.0 or len(eta) != 2:
                continue
            tot = float(np.exp(eta).sum())
            out[name] = eta + np.log(typ * (tot / typ) ** beta / tot)
        return out

    def cov_beta(self, name: str) -> np.ndarray:
        c = self.models_[name].coef()
        return np.array([c[s] for s in SCORE_COVS])

    def expected(self, fixture: pd.DataFrame, gf: float, ga: float) -> dict[str, tuple[float, float]]:
        """Mean of each stat (home, away) for a given final score."""
        from .features import score_covariates
        base = self.base_eta(fixture)
        ch = score_covariates([gf], [ga]).to_numpy()[0]
        ca = score_covariates([ga], [gf]).to_numpy()[0]
        out = {}
        for name in self.models_:
            b = self.cov_beta(name)
            out[name] = (float(np.exp(base[name][0] + ch @ b)),
                         float(np.exp(base[name][1] + ca @ b)))
        return out


def _dist(mu: np.ndarray, theta: float):
    if np.isfinite(theta):
        return stats.nbinom(theta, theta / (theta + mu))
    return stats.poisson(mu)


def nb_ppf(u: np.ndarray, mu: np.ndarray, theta: float) -> np.ndarray:
    """Vectorised quantile of NB(mean mu, size theta), or Poisson if theta is inf."""
    if np.isfinite(theta):
        return stats.nbinom.ppf(u, theta, theta / (theta + mu)).astype(np.int16)
    return stats.poisson.ppf(u, mu).astype(np.int16)
