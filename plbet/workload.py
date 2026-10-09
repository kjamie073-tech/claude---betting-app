"""Workload across every competition: rest days, recent minutes, international duty.

The model's ratings and player rates come from Premier League games only.
This module adds what happens in between, from ESPN's cup, European and
international matches (plbet/data/fetch.py, ``espn``):

* each team's games in all competitions and its rest before a fixture;
* each player's minutes over the last few days in all competitions, and his
  latest international games (with long-haul trips flagged).

ESPN players are linked to Understat players by name: within the same club
for cup and European games, and by an exact name match for players seen only
in international games.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .data import load, names

# Exact-name threshold for players seen only with their country.
INTERNATIONAL_MATCH = 0.97


@functools.lru_cache(maxsize=1)
def _links() -> pd.Series:
    """ESPN athlete id -> Understat player_id."""
    ep = load.espn_players()
    pm = load.player_matches()
    if ep.empty or pm.empty:
        return pd.Series(dtype=object)
    out: dict[str, object] = {}
    club = ep[ep["club"]]
    pm_last = pm.groupby(["team", "player_id"]).agg(player=("player", "last"),
                                                    first=("date", "min"), last=("date", "max"))
    for (team, eid), d in club.groupby(["team", "espn_id"]):
        if eid in out or team not in pm_last.index.get_level_values(0):
            continue
        cand = pm_last.loc[team]
        lo, hi = d["date"].min() - pd.Timedelta(days=400), d["date"].max() + pd.Timedelta(days=400)
        cand = cand[(cand["last"] >= lo) & (cand["first"] <= hi)]
        j, _ = names.best_match(d["player"].iat[0], cand["player"], surname_fallback=False)
        if j is not None:
            out[eid] = cand.index[j]
    # International-only players: exact names among recent Premier League players.
    rest = ep[~ep["espn_id"].isin(out.keys())].drop_duplicates("espn_id")
    pl = pm.sort_values("date").drop_duplicates("player_id", keep="last")
    by_key: dict[str, list] = {}
    for pid, nm in zip(pl["player_id"], pl["player"]):
        by_key.setdefault(names.player_key(nm), []).append(pid)
    for eid, nm in zip(rest["espn_id"], rest["player"]):
        ids = by_key.get(names.player_key(nm), [])
        if len(ids) == 1:
            out[eid] = ids[0]
    return pd.Series(out, dtype=object)


def appearances(as_of: pd.Timestamp, days: int = 45) -> pd.DataFrame:
    """Every appearance in the ``days`` before ``as_of``, all competitions.

    Columns: player_id, date, competition, team, minutes, starter, club.
    """
    lo = as_of - pd.Timedelta(days=days)
    pm = load.player_matches()
    pl = pm[(pm["date"] >= lo) & (pm["date"] < as_of)]
    rows = [pd.DataFrame({"player_id": pl["player_id"], "player": pl["player"],
                          "date": pl["date"], "competition": "Premier League",
                          "league": "eng.1", "team": pl["team"], "minutes": pl["minutes"],
                          "starter": pl["started"], "club": True, "far": False})]
    ep = load.espn_players()
    if not ep.empty:
        e = ep[(ep["date"] >= lo) & (ep["date"] < as_of)].copy()
        e["player_id"] = e["espn_id"].map(_links())
        e = e.dropna(subset=["player_id"])
        far = load.espn_far_teams()
        e["far"] = ~e["club"] & e["team_id"].astype(str).isin(far)
        rows.append(e[["player_id", "player", "date", "competition", "league", "team",
                       "minutes", "starter", "club", "far"]])
    out = pd.concat(rows, ignore_index=True)
    # Understat dates carry no kick-off time; ESPN's do. Compare by day.
    out["day"] = out["date"].dt.normalize()
    return out.sort_values("date").reset_index(drop=True)


def team_schedule(team: str, as_of: pd.Timestamp, days: int = 45) -> pd.DataFrame:
    """A team's games in all competitions in the ``days`` before ``as_of``."""
    lo = as_of - pd.Timedelta(days=days)
    m = load.matches()
    m = m[((m["home"] == team) | (m["away"] == team)) & (m["date"] >= lo) & (m["date"] < as_of)]
    rows = [{"date": r.date, "competition": "Premier League",
             "opponent": r.away if r.home == team else r.home,
             "venue": "H" if r.home == team else "A",
             "score": f"{int(r.hg)}-{int(r.ag)}" if r.home == team else f"{int(r.ag)}-{int(r.hg)}"}
            for r in m.itertuples() if pd.notna(r.hg)]
    et = load.espn_teams()
    if not et.empty:
        e = et[(et["team"] == team) & et["club"] & (et["date"] >= lo) & (et["date"] < as_of)]
        for r in e.itertuples():
            gf, ga = r.gf, r.ga
            rows.append({"date": r.date, "competition": r.competition, "opponent": r.opp,
                         "venue": "N" if r.neutral else ("H" if r.side == "home" else "A"),
                         "score": f"{gf:.0f}-{ga:.0f}" if pd.notna(gf) and pd.notna(ga) else ""})
    df = pd.DataFrame(rows, columns=["date", "competition", "opponent", "venue", "score"])
    return df.sort_values("date").reset_index(drop=True)


def rest_days(team: str, kickoff: pd.Timestamp) -> float | None:
    """Days since the team's last game in any competition."""
    s = team_schedule(team, kickoff, days=60)
    if s.empty:
        return None
    return float((kickoff.normalize() - s["date"].iat[-1].normalize()).days)


def player_load(player_ids, kickoff: pd.Timestamp) -> pd.DataFrame:
    """Recent load for the given players before ``kickoff``.

    min_4d / min_8d: minutes in all competitions in the last 4 / 8 days;
    last_date, last_comp, last_min: their latest appearance; intl_*: their
    latest international game in the last 12 days (long_haul if it was in
    a competition played mostly outside Europe).
    """
    ap = appearances(kickoff, days=21)
    ap = ap[ap["player_id"].isin(set(player_ids))]
    day = kickoff.normalize()
    out = []
    for pid, d in ap.groupby("player_id"):
        age = (day - d["day"]).dt.days
        last = d.iloc[-1]
        intl = d[~d["club"].astype(bool) & (age <= 12)]
        out.append({
            "player_id": pid,
            "min_4d": float(d.loc[age <= 4, "minutes"].sum()),
            "min_8d": float(d.loc[age <= 8, "minutes"].sum()),
            "games_8d": int((age <= 8).sum()),
            "last_date": last["date"], "last_comp": last["competition"],
            "last_min": float(last["minutes"]),
            "intl_games": int(len(intl)),
            "intl_min": float(intl["minutes"].sum()),
            "intl_last": intl["date"].max() if len(intl) else pd.NaT,
            "long_haul": bool(len(intl) and intl["far"].astype(bool).any()),
        })
    cols = ["player_id", "min_4d", "min_8d", "games_8d", "last_date", "last_comp", "last_min",
            "intl_games", "intl_min", "intl_last", "long_haul"]
    return pd.DataFrame(out, columns=cols).set_index("player_id")


def describe(team: str, squad: pd.DataFrame, kickoff: pd.Timestamp) -> tuple[list[str], list[str]]:
    """Report lines: the team's recent games, and load notes for its XI."""
    sched = team_schedule(team, kickoff, days=45)
    games = [f"{r.date:%a %d %b} {r.competition} {r.venue} v {r.opponent} {r.score}".rstrip()
             for r in sched.itertuples()][-4:]
    notes = []
    rest = rest_days(team, kickoff)
    if rest is not None and rest <= 3:
        notes.append(f"{team} last played {rest:.0f} days before kick-off "
                     f"({sched['competition'].iat[-1]}); expect some rotation.")
    up = load.espn_upcoming()
    up = up[(up["team"] == team) & (up["date"] > kickoff)
            & (up["date"] <= kickoff + pd.Timedelta(days=5))]
    for r in up.head(1).itertuples():
        gap = (r.date.normalize() - kickoff.normalize()).days
        notes.append(f"{team} play {r.opp} in the {r.competition} {gap} days later "
                     f"({r.date:%a %d %b}); key players may be rested or taken off early.")
    xi = squad[squad["start"]] if "start" in squad else squad
    ld = player_load(xi.index, kickoff)
    for pid, r in ld.iterrows():
        nm = squad.at[pid, "player"] if pid in squad.index else pid
        if r["intl_games"]:
            trip = " (long-haul trip back)" if r["long_haul"] else ""
            notes.append(f"{nm}: {r['intl_games']} international game(s), {r['intl_min']:.0f} "
                         f"minutes, last on {r['intl_last']:%a %d %b}{trip}.")
        elif r["min_4d"] >= 75 and r["last_comp"] != "Premier League":
            notes.append(f"{nm}: {r['last_min']:.0f} minutes in the {r['last_comp']} on "
                         f"{r['last_date']:%a %d %b}.")
    return games, notes
