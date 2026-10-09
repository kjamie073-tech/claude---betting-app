"""Load the downloaded data into tidy pandas tables.

The raw files live on the repository's ``data`` branch (written by the
``Update data`` GitHub Action). ``sync_data()`` checks that branch out into
``./data``; set ``PLBET_DATA`` to read from somewhere else.

Main tables
-----------
matches()          one row per Premier League match: result, half-time score,
                   shots, corners, fouls, cards, referee, odds, team xG
player_matches()   one row per player per match (Understat): minutes, goals,
                   shots, shots on target, xG, npxG, assists, xA, cards
shots()            every Understat shot
fpl_players()      current FPL squad list with availability news
fpl_fixtures()     current-season fixture list with kickoff times
espn_players()     one row per player per cup, European or international match
espn_teams()       one row per team per such match (score, box score)
"""

from __future__ import annotations

import functools
import html
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import names

REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches played without crowds (Covid): home advantage collapsed.
NO_CROWD_START = pd.Timestamp("2020-06-17")
NO_CROWD_END = pd.Timestamp("2021-05-19")


def data_dir() -> Path:
    env = os.environ.get("PLBET_DATA")
    return Path(env) if env else REPO_ROOT / "data"


def sync_data(dest: Path | None = None) -> Path:
    """Fetch the latest ``data`` branch and check it out at ``dest``."""
    dest = Path(dest or data_dir())
    run = functools.partial(subprocess.run, check=True, capture_output=True, text=True)
    run(["git", "-C", str(REPO_ROOT), "fetch", "--depth", "1", "origin",
         "+refs/heads/data:refs/remotes/origin/data"])
    if (dest / ".git").exists():
        run(["git", "-C", str(dest), "checkout", "--detach", "-f", "origin/data"])
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", "-f", str(dest),
             "origin/data"])
    for fn in (matches, player_matches, shots, fpl_bootstrap, understat_league, _espn_records):
        fn.cache_clear()
    return dest


def status() -> dict:
    path = data_dir() / "status.json"
    return json.loads(path.read_text()) if path.exists() else {}


def season_of(date: pd.Timestamp) -> int:
    return date.year if date.month >= 7 else date.year - 1


# --------------------------------------------------------------------------
# football-data.co.uk + Understat team xG
# --------------------------------------------------------------------------

FD_RENAME = {
    "HomeTeam": "home", "AwayTeam": "away",
    "FTHG": "hg", "FTAG": "ag", "HTHG": "hthg", "HTAG": "htag",
    "Referee": "referee",
    "HS": "h_shots", "AS": "a_shots", "HST": "h_sot", "AST": "a_sot",
    "HF": "h_fouls", "AF": "a_fouls", "HC": "h_corners", "AC": "a_corners",
    "HY": "h_yellow", "AY": "a_yellow", "HR": "h_red", "AR": "a_red",
}


def _read_fd_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def football_data(league: str = "E0") -> pd.DataFrame:
    folder = data_dir() / "football-data"
    frames = []
    for path in sorted(folder.glob(f"{league}_*.csv")):
        df = _read_fd_csv(path)
        df = df.dropna(subset=["HomeTeam", "AwayTeam"])
        code = path.stem.split("_")[1]
        df["season"] = 2000 + int(code[:2]) if int(code[:2]) < 90 else 1900 + int(code[:2])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no {league} files in {folder}; run sync_data() first")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    if "Time" in df:
        t = pd.to_datetime(df["Time"].fillna("15:00"), format="%H:%M", errors="coerce")
        df["kickoff"] = df["date"] + pd.to_timedelta(t.dt.hour.fillna(15), unit="h") \
            + pd.to_timedelta(t.dt.minute.fillna(0), unit="m")
    else:
        df["kickoff"] = df["date"] + pd.Timedelta(hours=15)
    df = df.rename(columns=FD_RENAME)
    df["home"] = df["home"].map(names.team)
    df["away"] = df["away"].map(names.team)
    if "referee" in df:
        df["referee"] = df["referee"].astype("string").str.strip()
    df["league"] = league
    return df


@functools.lru_cache(maxsize=None)
def understat_league(year: int) -> dict:
    path = data_dir() / "understat" / f"league_EPL_{year}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def understat_team_matches() -> pd.DataFrame:
    """Team-level Understat stats per match: xG, npxG, deep completions, PPDA."""
    rows = []
    folder = data_dir() / "understat"
    for path in sorted(folder.glob("league_EPL_*.json")):
        year = int(path.stem.rsplit("_", 1)[1])
        lg = understat_league(year)
        ids = {}
        for m in lg.get("dates", []):
            ids[(m["datetime"], m["h"]["title"])] = int(m["id"])
            ids[(m["datetime"], m["a"]["title"])] = int(m["id"])
        teams = lg.get("teams", {})
        for t in (teams.values() if isinstance(teams, dict) else teams):
            for h in t.get("history", []):
                ppda = h.get("ppda") or {}
                rows.append({
                    "season": year,
                    "us_match_id": ids.get((h["date"], t["title"])),
                    "datetime": h["date"],
                    "team": names.team(t["title"]),
                    "h_a": h["h_a"],
                    "xg": float(h["xG"]),
                    "xga": float(h["xGA"]),
                    "npxg": float(h["npxG"]),
                    "npxga": float(h["npxGA"]),
                    "deep": float(h.get("deep", np.nan)),
                    "ppda": (float(ppda["att"]) / float(ppda["def"])) if ppda.get("def") else np.nan,
                    "goals": int(h["scored"]),
                    "conceded": int(h["missed"]),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    return df


@functools.lru_cache(maxsize=None)
def matches() -> pd.DataFrame:
    """All Premier League matches with stats, odds and Understat xG."""
    df = football_data("E0")
    df["nocrowd"] = ((df["date"] >= NO_CROWD_START) & (df["date"] <= NO_CROWD_END)).astype(int)
    us = understat_team_matches()
    if not us.empty:
        cols = ["date", "team", "xg", "npxg", "deep", "ppda", "us_match_id"]
        h = us[us.h_a == "h"][cols].rename(columns={
            "team": "home", "xg": "h_xg", "npxg": "h_npxg", "deep": "h_deep", "ppda": "h_ppda"})
        a = us[us.h_a == "a"][cols].drop(columns="us_match_id").rename(columns={
            "team": "away", "xg": "a_xg", "npxg": "a_npxg", "deep": "a_deep", "ppda": "a_ppda"})
        df = df.merge(h, on=["date", "home"], how="left").merge(a, on=["date", "away"], how="left")
        # Late kick-offs can sit on different calendar days in the two sources.
        miss = df["h_xg"].isna()
        if miss.any():
            for shift in (-1, 1):
                hh = h.assign(date=h["date"] + pd.Timedelta(days=shift))
                aa = a.assign(date=a["date"] + pd.Timedelta(days=shift))
                fill = df.loc[miss, ["date", "home", "away"]].reset_index() \
                    .merge(hh, on=["date", "home"], how="left") \
                    .merge(aa, on=["date", "away"], how="left").set_index("index")
                for c in ["h_xg", "h_npxg", "h_deep", "h_ppda", "us_match_id",
                          "a_xg", "a_npxg", "a_deep", "a_ppda"]:
                    df.loc[miss, c] = df.loc[miss, c].fillna(fill[c])
                miss = df["h_xg"].isna()
    df = df.sort_values(["kickoff", "home"]).reset_index(drop=True)
    df["match_key"] = df["date"].dt.strftime("%Y-%m-%d") + " " + df["home"] + " v " + df["away"]
    return df


def team_rows(m: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match, one from each team's point of view."""
    pairs = [
        ("hg", "ag"), ("hthg", "htag"), ("h_shots", "a_shots"), ("h_sot", "a_sot"),
        ("h_fouls", "a_fouls"), ("h_corners", "a_corners"), ("h_yellow", "a_yellow"),
        ("h_red", "a_red"), ("h_xg", "a_xg"), ("h_npxg", "a_npxg"),
    ]
    base = ["season", "date", "kickoff", "referee", "nocrowd", "match_key"]
    base = [c for c in base if c in m]
    out = []
    for is_home in (1, 0):
        d = m[base].copy()
        d["team"] = m["home"] if is_home else m["away"]
        d["opp"] = m["away"] if is_home else m["home"]
        d["is_home"] = is_home
        for hcol, acol in pairs:
            if hcol not in m:
                continue
            stem = hcol[2:] if hcol.startswith("h_") else {"hg": "goals", "hthg": "ht_goals"}[hcol]
            d[f"{stem}_for"] = m[hcol] if is_home else m[acol]
            d[f"{stem}_against"] = m[acol] if is_home else m[hcol]
        out.append(d)
    return pd.concat(out, ignore_index=True).sort_values(["kickoff", "is_home"],
                                                          ascending=[True, False])


# --------------------------------------------------------------------------
# Understat player data
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def shots() -> pd.DataFrame:
    folder = data_dir() / "understat"
    frames = [pd.read_csv(p) for p in sorted(folder.glob("shots_*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    # Understat escapes names for HTML ("O&#039;Reilly").
    for c in ("player", "player_assisted"):
        df[c] = df[c].map(lambda v: html.unescape(v) if isinstance(v, str) else v)
    df["team"] = np.where(df["h_a"] == "h", df["h_team"], df["a_team"])
    df["team"] = df["team"].map(names.team)
    df["on_target"] = df["result"].isin(["Goal", "SavedShot"])
    df["is_pen"] = df["situation"].eq("Penalty")
    return df


@functools.lru_cache(maxsize=None)
def player_matches() -> pd.DataFrame:
    folder = data_dir() / "understat"
    frames = [pd.read_csv(p) for p in sorted(folder.glob("player_matches_*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["player"] = df["player"].map(lambda v: html.unescape(v) if isinstance(v, str) else v)
    home = df["h_a"] == "h"
    df["team"] = np.where(home, df["h_team"], df["a_team"])
    df["opp"] = np.where(home, df["a_team"], df["h_team"])
    df["team"] = df["team"].map(names.team)
    df["opp"] = df["opp"].map(names.team)
    df["is_home"] = home.astype(int)
    df = df.rename(columns={"time": "minutes", "xG": "xg", "xA": "xa",
                            "yellow_card": "yellow", "red_card": "red"})
    df["started"] = ~df["position"].eq("Sub")
    sh = shots()
    if not sh.empty:
        sh = sh.assign(pen_goal=sh["is_pen"] & sh["result"].eq("Goal"),
                       pen_xg=np.where(sh["is_pen"], sh["xG"], 0.0))
        agg = sh.groupby(["match_id", "player_id"]).agg(
            sot=("on_target", "sum"), pens=("is_pen", "sum"),
            pen_goals=("pen_goal", "sum"), pen_xg=("pen_xg", "sum")).reset_index()
        df = df.merge(agg, on=["match_id", "player_id"], how="left")
        for c in ("sot", "pens", "pen_goals", "pen_xg"):
            df[c] = df[c].fillna(0)
    else:
        for c in ("sot", "pens", "pen_goals", "pen_xg"):
            df[c] = np.nan
    df["npxg"] = df["xg"] - df["pen_xg"]
    df["np_goals"] = df["goals"] - df["pen_goals"]
    return df.sort_values(["date", "match_id", "h_a"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# ESPN: cup, European and international games
# --------------------------------------------------------------------------

ESPN_COMP_NAMES = {
    "uefa.champions": "Champions League", "uefa.europa": "Europa League",
    "uefa.europa.conf": "Conference League", "eng.fa": "FA Cup",
    "eng.league_cup": "EFL Cup", "eng.charity": "Community Shield",
    "fifa.cwc": "Club World Cup",
}
# International competitions played mostly outside Europe: long trips back.
LONG_HAUL = ("conmebol", "caf.", "afc.", "concacaf", "fifa.worldq.caf", "fifa.worldq.afc",
             "fifa.worldq.conmebol", "fifa.worldq.concacaf")


def espn_is_club(league: str) -> bool:
    return league in ESPN_COMP_NAMES


@functools.lru_cache(maxsize=None)
def _espn_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    import gzip
    folder = data_dir() / "espn"
    prow, trow = [], []
    for path in sorted(folder.glob("*/*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "players" not in rec:      # early raw format: skip
                    continue
                base = {"league": rec["league"], "event": rec["event"], "date": rec["date"]}
                teams = {t["id"]: t for t in rec.get("teams", [])}
                for t in teams.values():
                    opp = next((o for o in teams.values() if o["id"] != t["id"]), {})
                    trow.append({**base, "team_id": t["id"], "team_espn": t.get("name"),
                                 "opp_espn": opp.get("name"), "side": t.get("side"),
                                 "neutral": rec.get("neutral"), "gf": t.get("score"),
                                 "ga": opp.get("score"),
                                 **{k: t.get(k) for k in ("corners", "shots", "sot", "fouls",
                                                         "yellow", "red", "possession")}})
                for pl in rec["players"]:
                    t = teams.get(pl["team"], {})
                    prow.append({**base, "espn_id": pl["id"], "player": pl["name"],
                                 "team_id": pl["team"], "team_espn": t.get("name"),
                                 **{k: pl.get(k) for k in ("starter", "minutes", "pos", "goals",
                                                          "assists", "shots", "sot", "yellow",
                                                          "red", "fouls")}})
    cols_p = ["league", "event", "date", "espn_id", "player", "team_id", "team_espn", "starter",
              "minutes"]
    pdf = pd.DataFrame(prow) if prow else pd.DataFrame(columns=cols_p)
    tdf = pd.DataFrame(trow) if trow else pd.DataFrame(columns=["league", "event", "date",
                                                               "team_id", "team_espn"])
    for df in (pdf, tdf):
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/London") \
            .dt.tz_localize(None)
        df["club"] = df["league"].map(espn_is_club).astype(bool)
        df["competition"] = df["league"].map(ESPN_COMP_NAMES).fillna("International")
        # Club names in the canonical form where they are English clubs.
        df["team"] = [names.team(t) if c and isinstance(t, str) else t
                      for t, c in zip(df["team_espn"], df["club"])]
    tdf["opp"] = [names.team(t) if c and isinstance(t, str) else t
                  for t, c in zip(tdf["opp_espn"], tdf["club"])]
    return pdf, tdf


def espn_players() -> pd.DataFrame:
    return _espn_records()[0].copy()


def espn_teams() -> pd.DataFrame:
    return _espn_records()[1].copy()


# --------------------------------------------------------------------------
# Fantasy Premier League
# --------------------------------------------------------------------------

FPL_POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@functools.lru_cache(maxsize=None)
def fpl_bootstrap() -> dict:
    path = data_dir() / "fpl" / "bootstrap-static.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fpl_teams() -> dict[int, str]:
    return {t["id"]: names.team(t["name"]) for t in fpl_bootstrap().get("teams", [])}


def fpl_players() -> pd.DataFrame:
    bs = fpl_bootstrap()
    if not bs:
        return pd.DataFrame()
    teams = fpl_teams()
    df = pd.DataFrame(bs["elements"])
    df["team_name"] = df["team"].map(teams)
    df["position"] = df["element_type"].map(FPL_POS)
    df["full_name"] = (df["first_name"].str.strip() + " " + df["second_name"].str.strip())
    keep = [
        "id", "web_name", "full_name", "team_name", "position", "status", "news",
        "news_added", "chance_of_playing_next_round", "chance_of_playing_this_round",
        "minutes", "starts", "goals_scored", "assists", "expected_goals",
        "expected_assists", "yellow_cards", "red_cards", "penalties_order",
        "corners_and_indirect_freekicks_order", "direct_freekicks_order", "now_cost",
    ]
    return df[[c for c in keep if c in df.columns]].copy()


def fpl_fixtures() -> pd.DataFrame:
    path = data_dir() / "fpl" / "fixtures.json"
    if not path.exists():
        return pd.DataFrame()
    fx = pd.DataFrame(json.loads(path.read_text()))
    teams = fpl_teams()
    fx["home"] = fx["team_h"].map(teams)
    fx["away"] = fx["team_a"].map(teams)
    fx["kickoff"] = pd.to_datetime(fx["kickoff_time"], utc=True).dt.tz_convert("Europe/London") \
        .dt.tz_localize(None)
    return fx[["id", "event", "kickoff", "home", "away", "finished",
               "team_h_score", "team_a_score"]].sort_values("kickoff").reset_index(drop=True)


def fpl_chance_column(home: str, away: str) -> str:
    """Which FPL "chance of playing" field applies to this fixture.

    FPL gives a chance for the current gameweek (once its deadline has
    passed) and for the next one. Use "this round" only when the fixture
    belongs to the gameweek already under way.
    """
    events = fpl_bootstrap().get("events", [])
    current = next((e["id"] for e in events if e.get("is_current")), None)
    fx = fpl_fixtures()
    row = fx[(fx["home"] == home) & (fx["away"] == away) & (fx["finished"] == False)]  # noqa: E712
    if current is not None and len(row) and row["event"].iloc[0] == current:
        return "chance_of_playing_this_round"
    return "chance_of_playing_next_round"


def fpl_history() -> pd.DataFrame:
    folder = data_dir() / "fpl"
    frames = [pd.read_csv(p) for p in sorted(folder.glob("player_history_*.csv"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
