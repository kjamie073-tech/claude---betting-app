"""Model-ready tables built from the raw match data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import load

# Promoted clubs in the first season of the data (2019-20), whose previous
# season is not downloaded.
PROMOTED_FALLBACK = {2019: {"Norwich", "Sheffield United", "Aston Villa"}}

# Score-state covariates (from the team's own point of view) used by the
# corners, cards, fouls and shots models, which are fitted conditional on the
# final score. That is what lets the simulator tie "team wins" to "fewer
# corners for that team" and so on.
SCORE_COVS = ("gd-2", "gd-1", "gd+1", "gd+2", "tot")


def promoted_sets(m: pd.DataFrame) -> dict[int, set[str]]:
    teams = {s: set(g["home"]) | set(g["away"]) for s, g in m.groupby("season")}
    out: dict[int, set[str]] = {}
    for s, ts in teams.items():
        prev = teams.get(s - 1)
        out[s] = (ts - prev) if prev else PROMOTED_FALLBACK.get(s, set())
    return out


def promoted_flags(teams: pd.Series, seasons: pd.Series, prom: dict[int, set[str]]) -> np.ndarray:
    return np.array([int(t in prom.get(s, set())) for t, s in zip(teams, seasons)])


def score_covariates(goals_for, goals_against) -> pd.DataFrame:
    gf = np.asarray(goals_for, float)
    ga = np.asarray(goals_against, float)
    gd = np.clip(gf - ga, -2, 2)
    out = {f"gd{k:+d}": (gd == k).astype(float) for k in (-2, -1, 1, 2)}
    out["tot"] = np.clip(gf + ga, 0, 6)
    return pd.DataFrame(out)


def team_table(m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Two rows per match (one per team) with everything the models need."""
    m = load.matches() if m is None else m
    prom = promoted_sets(m)
    tr = load.team_rows(m)
    tr = tr.sort_values(["kickoff", "match_key", "is_home"], ascending=[True, True, False])
    tr = tr.reset_index(drop=True)
    tr["promoted_team"] = promoted_flags(tr["team"], tr["season"], prom)
    tr["promoted_opp"] = promoted_flags(tr["opp"], tr["season"], prom)
    covs = score_covariates(tr["goals_for"], tr["goals_against"])
    for c in covs:
        tr[c] = covs[c].to_numpy()
    # Shots that were not goals, split by on/off target. Own goals are not
    # shots, so this slightly understates on-target misses in those matches.
    tr["ngon_for"] = (tr["sot_for"] - tr["goals_for"]).clip(lower=0)
    tr["ngoff_for"] = (tr["shots_for"] - tr["sot_for"]).clip(lower=0)
    tr["ngon_against"] = (tr["sot_against"] - tr["goals_against"]).clip(lower=0)
    tr["ngoff_against"] = (tr["shots_against"] - tr["sot_against"]).clip(lower=0)
    return tr


def fixture_rows(home: str, away: str, season: int, prom: dict[int, set[str]],
                 referee: str | None = None, nocrowd: int = 0) -> pd.DataFrame:
    """Prediction rows (home view, away view) for one fixture."""
    p = prom.get(season, set())
    return pd.DataFrame({
        "team": [home, away], "opp": [away, home], "is_home": [1, 0],
        "nocrowd": [nocrowd, nocrowd],
        "promoted_team": [int(home in p), int(away in p)],
        "promoted_opp": [int(away in p), int(home in p)],
        "referee": [referee, referee],
    })
