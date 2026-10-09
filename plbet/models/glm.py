"""Penalised Poisson and negative binomial regression with team effects.

Every team-level model in this package (goals, corners, cards, shots, fouls)
is a log-linear count model of the form

    log E[y] = intercept + home + for[team] - against[opponent]
               (+ referee[ref]) + covariates . beta

fitted with time-decay weights and an L2 (ridge) penalty that shrinks team and
referee effects towards their group means. The penalty is what keeps a team
with few recent matches (a promoted side, early season) from getting an
extreme rating off a handful of games.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, sparse
from scipy.special import digamma, gammaln


def fit_glm(
    X: sparse.csr_matrix,
    y: np.ndarray,
    w: np.ndarray,
    penalty: np.ndarray,
    *,
    family: str = "poisson",
    offset: np.ndarray | None = None,
    beta0: np.ndarray | None = None,
    log_theta0: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Minimise weighted negative log-likelihood + 0.5 * sum(penalty * beta^2).

    Returns (beta, theta). theta is inf for the Poisson family. For Poisson,
    y may be non-integer (quasi-likelihood), which is how xG is used as a
    response.
    """
    n, p = X.shape
    off = np.zeros(n) if offset is None else offset
    XT = X.T.tocsr()
    b0 = np.zeros(p) if beta0 is None else beta0.copy()
    if beta0 is None:
        # Start the intercept (assumed to be column 0) at the weighted log mean.
        b0[0] = np.log(max(np.average(y, weights=w), 1e-6))

    if family == "poisson":

        def fun(b):
            eta = X @ b + off
            mu = np.exp(eta)
            nll = -np.sum(w * (y * eta - mu))
            grad = -(XT @ (w * (y - mu)))
            return nll + 0.5 * np.sum(penalty * b * b), grad + penalty * b

        res = optimize.minimize(fun, b0, jac=True, method="L-BFGS-B",
                                options={"maxiter": 5000, "gtol": 1e-7})
        return res.x, np.inf

    if family != "negbin":
        raise ValueError(family)
    lgy1 = gammaln(y + 1)

    def fun_nb(z):
        b, lt = z[:-1], z[-1]
        th = np.exp(lt)
        eta = X @ b + off
        mu = np.exp(eta)
        ll = (gammaln(y + th) - gammaln(th) - lgy1 + th * (lt - np.log(th + mu))
              + y * (eta - np.log(th + mu)))
        nll = -np.sum(w * ll)
        dl_deta = th * (y - mu) / (th + mu)
        g_b = -(XT @ (w * dl_deta)) + penalty * b
        dl_dth = digamma(y + th) - digamma(th) + lt + 1 - np.log(th + mu) - (th + y) / (th + mu)
        g_lt = -np.sum(w * dl_dth) * th
        return nll + 0.5 * np.sum(penalty * b * b), np.append(g_b, g_lt)

    z0 = np.append(b0, log_theta0)
    bounds = [(None, None)] * p + [(-3.0, 9.0)]
    res = optimize.minimize(fun_nb, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 5000, "gtol": 1e-7})
    return res.x[:-1], float(np.exp(res.x[-1]))


@dataclass
class TeamCountModel:
    """Count model with team for/against effects (and optional referee effects).

    Parameters
    ----------
    family:   "poisson" or "negbin"
    tau_team: ridge precision on team effects (bigger = more shrinkage)
    tau_ref:  ridge precision on referee effects (only used if use_ref)
    covariates: extra numeric columns of the row table, unpenalised
    """

    family: str = "poisson"
    tau_team: float = 5.0
    tau_ref: float = 20.0
    use_ref: bool = False
    covariates: tuple[str, ...] = ()
    home_cols: tuple[str, ...] = ("is_home", "home_nocrowd")
    promoted: bool = True
    tau_cov: float = 1e-4

    teams_: list[str] = field(default_factory=list)
    refs_: list[str] = field(default_factory=list)
    beta_: np.ndarray | None = None
    theta_: float = np.inf
    names_: list[str] = field(default_factory=list)

    # ----------------------------------------------------------------- design
    def _fixed_cols(self) -> list[str]:
        cols = list(self.home_cols)
        if self.promoted:
            cols += ["promoted_team", "promoted_opp"]
        return cols + list(self.covariates)

    def _design(self, rows: pd.DataFrame) -> sparse.csr_matrix:
        n = len(rows)
        fixed = self._fixed_cols()
        nt, nr = len(self.teams_), len(self.refs_)
        p = 1 + len(fixed) + 2 * nt + (nr if self.use_ref else 0)
        tix = {t: i for i, t in enumerate(self.teams_)}
        data, ri, ci = [], [], []

        def add(col_vals, col_idx):
            nz = np.nonzero(col_vals)[0]
            data.extend(col_vals[nz])
            ri.extend(nz)
            ci.extend(np.full(len(nz), col_idx) if np.ndim(col_idx) == 0 else col_idx[nz])

        add(np.ones(n), 0)
        for j, c in enumerate(fixed):
            add(rows[c].to_numpy(float), 1 + j)
        base = 1 + len(fixed)
        t_idx = rows["team"].map(tix)
        o_idx = rows["opp"].map(tix)
        ok_t = t_idx.notna().to_numpy()
        ok_o = o_idx.notna().to_numpy()
        add(ok_t.astype(float), base + np.nan_to_num(t_idx.to_numpy(float)).astype(int))
        add(-ok_o.astype(float), base + nt + np.nan_to_num(o_idx.to_numpy(float)).astype(int))
        if self.use_ref:
            rix = {r: i for i, r in enumerate(self.refs_)}
            r_idx = rows["referee"].map(rix)
            ok_r = r_idx.notna().to_numpy()
            add(ok_r.astype(float), base + 2 * nt + np.nan_to_num(r_idx.to_numpy(float)).astype(int))
        return sparse.csr_matrix((data, (ri, ci)), shape=(n, p))

    def _penalty(self) -> np.ndarray:
        fixed = self._fixed_cols()
        nt, nr = len(self.teams_), len(self.refs_)
        pen = [0.0] + [self.tau_cov] * len(fixed) + [self.tau_team] * (2 * nt)
        if self.use_ref:
            pen += [self.tau_ref] * nr
        return np.array(pen)

    # -------------------------------------------------------------- fit/pred
    def fit(self, rows: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> "TeamCountModel":
        rows = _with_defaults(rows)
        self.teams_ = sorted(set(rows["team"]) | set(rows["opp"]))
        if self.use_ref:
            self.refs_ = sorted(rows["referee"].dropna().unique())
        X = self._design(rows)
        keep = np.isfinite(y) & (w > 0)
        X, yy, ww = X[keep], np.asarray(y, float)[keep], np.asarray(w, float)[keep]
        self.beta_, self.theta_ = fit_glm(X, yy, ww, self._penalty(), family=self.family)
        fixed = self._fixed_cols()
        self.names_ = (["intercept"] + fixed + [f"for:{t}" for t in self.teams_]
                       + [f"against:{t}" for t in self.teams_]
                       + ([f"ref:{r}" for r in self.refs_] if self.use_ref else []))
        return self

    def linear_predictor(self, rows: pd.DataFrame) -> np.ndarray:
        rows = _with_defaults(rows)
        return self._design(rows) @ self.beta_

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.exp(self.linear_predictor(rows))

    # ------------------------------------------------------------ inspection
    def coef(self) -> pd.Series:
        return pd.Series(self.beta_, index=self.names_)

    def team_table(self) -> pd.DataFrame:
        c = self.coef()
        return pd.DataFrame({
            "for": [c[f"for:{t}"] for t in self.teams_],
            "against": [c[f"against:{t}"] for t in self.teams_],
        }, index=self.teams_)

    def ref_effects(self) -> pd.Series:
        c = self.coef()
        return pd.Series({r: c[f"ref:{r}"] for r in self.refs_})


def _with_defaults(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for c, v in (("nocrowd", 0), ("promoted_team", 0), ("promoted_opp", 0), ("referee", None)):
        if c not in rows:
            rows[c] = v
    rows["home_nocrowd"] = rows["is_home"] * rows["nocrowd"]
    return rows


def decay_weights(dates: pd.Series, as_of: pd.Timestamp, xi: float) -> np.ndarray:
    """exp(-xi * days ago); zero for anything on or after as_of."""
    days = (as_of - pd.to_datetime(dates)).dt.total_seconds().to_numpy() / 86400.0
    w = np.exp(-xi * np.clip(days, 0, None))
    w[days <= 0] = 0.0
    return w
