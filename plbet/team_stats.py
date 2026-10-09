"""Descriptive numbers for match reports: form, splits, head-to-head, referee."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load
from .models import features

STAT_COLS = {
    "goals_for": "GF", "goals_against": "GA", "xg_for": "xG", "xg_against": "xGA",
    "shots_for": "Shots", "shots_against": "Shots ag.", "sot_for": "SoT",
    "sot_against": "SoT ag.", "corners_for": "Corners", "corners_against": "Corners ag.",
    "yellow_for": "Yellows", "red_for": "Reds", "fouls_for": "Fouls",
    "fouls_against": "Fouled",
}


def _team_rows(as_of: pd.Timestamp) -> pd.DataFrame:
    tr = features.team_table()
    return tr[tr["kickoff"] < as_of]


def form(team: str, as_of: pd.Timestamp, n: int = 6) -> pd.DataFrame:
    """Last n league matches with result and key stats."""
    tr = _team_rows(as_of)
    d = tr[tr["team"] == team].sort_values("kickoff").tail(n)
    res = np.select([d["goals_for"] > d["goals_against"], d["goals_for"] == d["goals_against"]],
                    ["W", "D"], "L")
    return pd.DataFrame({
        "Date": d["date"].dt.strftime("%d %b %y"),
        "Venue": np.where(d["is_home"] == 1, "H", "A"),
        "Opponent": d["opp"],
        "Result": [f"{r} {int(f)}-{int(a)}" for r, f, a in zip(res, d["goals_for"], d["goals_against"])],
        "xG": d["xg_for"].round(2), "xGA": d["xg_against"].round(2),
        "Shots": d["shots_for"].astype("Int64"), "Shots ag.": d["shots_against"].astype("Int64"),
        "Corners": d["corners_for"].astype("Int64"),
        "Corners ag.": d["corners_against"].astype("Int64"),
        "Cards": (d["yellow_for"] + d["red_for"]).astype("Int64"),
        "Referee": d["referee"].fillna(""),
    })


def averages(team: str, as_of: pd.Timestamp, season: int | None = None,
             venue: str | None = None, last: int | None = None) -> pd.Series:
    """Per-match averages for a team (optionally one season / venue / last n)."""
    tr = _team_rows(as_of)
    d = tr[tr["team"] == team]
    if season is not None:
        d = d[d["season"] == season]
    if venue == "home":
        d = d[d["is_home"] == 1]
    elif venue == "away":
        d = d[d["is_home"] == 0]
    d = d.sort_values("kickoff")
    if last:
        d = d.tail(last)
    out = d[list(STAT_COLS)].mean().rename(STAT_COLS)
    pts = np.select([d["goals_for"] > d["goals_against"], d["goals_for"] == d["goals_against"]],
                    [3, 1], 0)
    out = out.astype(object)
    out["Played"] = str(len(d))
    out["PPG"] = pts.mean() if len(d) else np.nan
    out["W-D-L"] = f"{(pts == 3).sum()}-{(pts == 1).sum()}-{(pts == 0).sum()}"
    pct = lambda x: f"{x:.0%}" if len(d) else ""  # noqa: E731
    out["BTTS"] = pct(((d["goals_for"] > 0) & (d["goals_against"] > 0)).mean())
    out["Over 2.5 goals"] = pct(((d["goals_for"] + d["goals_against"]) > 2.5).mean())
    return out


def comparison(home: str, away: str, as_of: pd.Timestamp) -> pd.DataFrame:
    season = load.season_of(as_of)
    cols = {
        f"{home} season": averages(home, as_of, season),
        f"{home} at home (season)": averages(home, as_of, season, "home"),
        f"{home} last 10": averages(home, as_of, last=10),
        f"{away} season": averages(away, as_of, season),
        f"{away} away (season)": averages(away, as_of, season, "away"),
        f"{away} last 10": averages(away, as_of, last=10),
    }
    return pd.DataFrame(cols)


def last_season_splits(team: str, as_of: pd.Timestamp) -> pd.DataFrame:
    season = load.season_of(as_of) - 1
    return pd.DataFrame({
        "home": averages(team, as_of, season, "home"),
        "away": averages(team, as_of, season, "away"),
    })


def head_to_head(home: str, away: str, as_of: pd.Timestamp, n: int = 8) -> pd.DataFrame:
    m = load.matches()
    m = m[m["kickoff"] < as_of]
    d = m[((m["home"] == home) & (m["away"] == away)) | ((m["home"] == away) & (m["away"] == home))]
    d = d.sort_values("kickoff").tail(n)
    return pd.DataFrame({
        "date": d["date"].dt.strftime("%d %b %Y"),
        "match": d["home"] + " v " + d["away"],
        "score": d["hg"].astype(int).astype(str) + "-" + d["ag"].astype(int).astype(str),
        "xG": d["h_xg"].round(2).astype(str) + "-" + d["a_xg"].round(2).astype(str),
        "corners": d["h_corners"].astype("Int64").astype(str) + "-" + d["a_corners"].astype("Int64").astype(str),
        "cards": (d["h_yellow"] + d["h_red"]).astype("Int64").astype(str) + "-"
                 + (d["a_yellow"] + d["a_red"]).astype("Int64").astype(str),
        "referee": d["referee"],
    })


def referee_profile(referee: str | None, as_of: pd.Timestamp, years: float = 3) -> dict:
    m = load.matches()
    m = m[(m["kickoff"] < as_of) & (m["kickoff"] > as_of - pd.Timedelta(days=365 * years))]
    league = {
        "matches": len(m),
        "yellows": (m["h_yellow"] + m["a_yellow"]).mean(),
        "reds": (m["h_red"] + m["a_red"]).mean(),
        "fouls": (m["h_fouls"] + m["a_fouls"]).mean(),
        "home_yellows": m["h_yellow"].mean(), "away_yellows": m["a_yellow"].mean(),
    }
    if not referee:
        return {"referee": None, "league": league}
    d = m[m["referee"] == referee]
    ref = {
        "matches": len(d),
        "yellows": (d["h_yellow"] + d["a_yellow"]).mean(),
        "reds": (d["h_red"] + d["a_red"]).mean(),
        "fouls": (d["h_fouls"] + d["a_fouls"]).mean(),
        "home_yellows": d["h_yellow"].mean(), "away_yellows": d["a_yellow"].mean(),
        "over_3.5_cards_%": ((d["h_yellow"] + d["a_yellow"] + d["h_red"] + d["a_red"]) > 3.5).mean() * 100,
    }
    return {"referee": referee, "ref": ref, "league": league}


def league_table(as_of: pd.Timestamp) -> pd.DataFrame:
    season = load.season_of(as_of)
    tr = _team_rows(as_of)
    d = tr[tr["season"] == season]
    pts = np.select([d["goals_for"] > d["goals_against"], d["goals_for"] == d["goals_against"]],
                    [3, 1], 0)
    t = d.assign(pts=pts).groupby("team").agg(
        P=("pts", "size"), Pts=("pts", "sum"), GF=("goals_for", "sum"),
        GA=("goals_against", "sum"), xG=("xg_for", "sum"), xGA=("xg_against", "sum"))
    t["GD"] = t["GF"] - t["GA"]
    t["xGD"] = (t["xG"] - t["xGA"]).round(1)
    t = t.sort_values(["Pts", "GD", "GF"], ascending=False)
    t.insert(0, "Pos", range(1, len(t) + 1))
    return t
