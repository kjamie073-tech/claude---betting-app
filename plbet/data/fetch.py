"""Download raw Premier League data into a local data directory.

This runs inside GitHub Actions (see .github/workflows/update-data.yml), which
has open internet access, and writes into a checkout of the repository's
``data`` branch. Analysis sessions then read that branch instead of hitting the
sites directly.

Sources
-------
football-data.co.uk  results, shots, corners, fouls, cards, referee and odds
Understat            team and player xG, player match stats, every shot
Fantasy PL API       injury/availability news, minutes, set-piece takers
ESPN                 other competitions: Champions/Europa/Conference League,
                     FA Cup, EFL Cup and internationals (line-ups, minutes,
                     player stats), for fixture congestion and player load

Only ``requests`` and ``pandas`` are needed so the job stays light.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
FD_BASE = "https://www.football-data.co.uk"
US_BASE = "https://understat.com"
FPL_BASE = "https://fantasy.premierleague.com/api"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def current_season(today: dt.date | None = None) -> int:
    """Start year of the season in progress (2026 means 2026-27)."""
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def fd_code(start_year: int) -> str:
    """football-data.co.uk season folder, e.g. 2026 -> '2627'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_is_live(start_year: int, today: dt.date | None = None) -> bool:
    """True while a season's files can still change (in progress or just ended)."""
    today = today or dt.date.today()
    return today <= dt.date(start_year + 1, 7, 31)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def get(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    tries: int = 4,
    timeout: int = 60,
) -> requests.Response:
    """GET with retries on network errors, 429 and 5xx."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(tries):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last = exc
            if attempt < tries - 1:
                time.sleep(delay)
                delay *= 2
    assert last is not None
    raise last


# --------------------------------------------------------------------------
# football-data.co.uk
# --------------------------------------------------------------------------


def fetch_football_data(
    out: Path, seasons: list[int], refresh_all: bool, session: requests.Session
) -> dict[str, Any]:
    folder = out / "football-data"
    folder.mkdir(parents=True, exist_ok=True)
    written, skipped, missing = [], [], []
    # E0 = Premier League, E1 = Championship (used for promoted-team priors).
    for league in ("E0", "E1"):
        for year in seasons:
            path = folder / f"{league}_{fd_code(year)}.csv"
            if path.exists() and not refresh_all and not season_is_live(year):
                skipped.append(path.name)
                continue
            url = f"{FD_BASE}/mmz4281/{fd_code(year)}/{league}.csv"
            r = get(session, url)
            if r.status_code == 404:
                missing.append(path.name)
                continue
            r.raise_for_status()
            body = r.content
            if not body.lstrip(b"\xef\xbb\xbf").startswith(b"Div"):
                raise ValueError(f"{url} did not return a football-data CSV")
            path.write_bytes(body)
            written.append(path.name)
            time.sleep(0.5)
    # Upcoming fixtures for every league, with pre-match odds.
    r = get(session, f"{FD_BASE}/fixtures.csv")
    if r.ok and r.content.lstrip(b"\xef\xbb\xbf").startswith(b"Div"):
        (folder / "fixtures.csv").write_bytes(r.content)
        written.append("fixtures.csv")
    log(f"football-data: wrote {len(written)}, kept {len(skipped)}, missing {missing}")
    return {"written": written, "missing": missing}


# --------------------------------------------------------------------------
# Understat
# --------------------------------------------------------------------------

US_XHR = {"X-Requested-With": "XMLHttpRequest"}

ROSTER_FIELDS = [
    "player_id", "player", "team_id", "h_a", "position", "positionOrder",
    "time", "goals", "own_goals", "shots", "xG", "assists", "xA",
    "key_passes", "yellow_card", "red_card", "roster_in", "roster_out",
    "xGChain", "xGBuildup",
]
SHOT_FIELDS = [
    "id", "minute", "result", "X", "Y", "xG", "player", "player_id", "h_a",
    "situation", "shotType", "player_assisted", "lastAction",
]


META_FIELDS = ["season", "match_id", "date", "h_team", "a_team", "h_goals", "a_goals"]
SHOT_COLUMNS = [*META_FIELDS, "shot_id", *[f for f in SHOT_FIELDS if f != "id"]]
PM_NUMERIC = [
    "season", "match_id", "h_goals", "a_goals", "player_id", "team_id", "positionOrder",
    "time", "goals", "own_goals", "shots", "xG", "assists", "xA", "key_passes",
    "yellow_card", "red_card", "roster_in", "roster_out", "xGChain", "xGBuildup",
]
SHOT_NUMERIC = ["season", "match_id", "h_goals", "a_goals", "shot_id", "minute", "X", "Y",
                "xG", "player_id"]


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def understat_session() -> requests.Session:
    s = make_session()
    # The JSON endpoints expect the cookies the home page sets.
    get(s, US_BASE + "/")
    return s


def _decode_embedded(html: str, var: str) -> Any:
    """Fallback for the older page format: var X = JSON.parse('\\x7B...')."""
    m = re.search(rf"var {var}\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        raise ValueError(f"{var} not found in page")
    raw = m.group(1).encode("utf-8").decode("unicode_escape")
    try:
        raw = raw.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return json.loads(raw)


def understat_league(s: requests.Session, year: int) -> dict[str, Any]:
    url = f"{US_BASE}/getLeagueData/EPL/{year}"
    r = get(s, url, headers={**US_XHR, "Referer": f"{US_BASE}/league/EPL/{year}"})
    if r.ok and r.headers.get("content-type", "").startswith(("application/json", "text/json")):
        data = r.json()
        return {"dates": data["dates"], "teams": data["teams"], "players": data["players"]}
    if r.ok:
        try:
            data = r.json()
            return {"dates": data["dates"], "teams": data["teams"], "players": data["players"]}
        except ValueError:
            pass
    log(f"understat league {year}: JSON endpoint gave {r.status_code}, trying page")
    page = get(s, f"{US_BASE}/league/EPL/{year}")
    page.raise_for_status()
    html = page.text
    return {
        "dates": _decode_embedded(html, "datesData"),
        "teams": _decode_embedded(html, "teamsData"),
        "players": _decode_embedded(html, "playersData"),
    }


def understat_match(s: requests.Session, match_id: int) -> dict[str, Any]:
    url = f"{US_BASE}/getMatchData/{match_id}"
    r = get(s, url, headers={**US_XHR, "Referer": f"{US_BASE}/match/{match_id}"})
    if r.ok:
        try:
            data = r.json()
            return {"rosters": data["rosters"], "shots": data["shots"]}
        except (ValueError, KeyError):
            pass
    page = get(s, f"{US_BASE}/match/{match_id}")
    page.raise_for_status()
    return {
        "rosters": _decode_embedded(page.text, "rostersData"),
        "shots": _decode_embedded(page.text, "shotsData"),
    }


def match_rows(
    match: dict[str, Any], data: dict[str, Any], season: int
) -> tuple[list[dict], list[dict]]:
    """Flatten one Understat match into player rows and shot rows."""
    mid = int(match["id"])
    meta = {
        "season": season,
        "match_id": mid,
        "date": match["datetime"],
        "h_team": match["h"]["title"],
        "a_team": match["a"]["title"],
        "h_goals": match["goals"]["h"],
        "a_goals": match["goals"]["a"],
    }
    players = []
    for side in ("h", "a"):
        for entry in (data["rosters"].get(side) or {}).values():
            row = dict(meta)
            row.update({k: entry.get(k) for k in ROSTER_FIELDS})
            row["h_a"] = side
            players.append(row)
    shots = []
    for side in ("h", "a"):
        for shot in data["shots"].get(side) or []:
            row = dict(meta)
            row.update({k: shot.get(k) for k in SHOT_FIELDS})
            row["shot_id"] = row.pop("id")
            row["h_a"] = side
            shots.append(row)
    return players, shots


def fetch_understat(
    out: Path,
    seasons: list[int],
    backfill_from: int,
    max_matches: int,
    refresh_all: bool,
) -> dict[str, Any]:
    folder = out / "understat"
    folder.mkdir(parents=True, exist_ok=True)
    s = understat_session()
    fetched = 0
    failures: list[str] = []
    for year in seasons:
        league_path = folder / f"league_EPL_{year}.json"
        if league_path.exists() and not refresh_all and not season_is_live(year):
            league = json.loads(league_path.read_text())
        else:
            league = understat_league(s, year)
            league_path.write_text(json.dumps(league, separators=(",", ":"), ensure_ascii=False))
            log(f"understat league {year}: {len(league['dates'])} matches")
            time.sleep(1.0)
        if year < backfill_from:
            continue

        pm_path = folder / f"player_matches_{year}.csv"
        sh_path = folder / f"shots_{year}.csv"
        have: set[int] = set()
        old_pm = old_sh = None
        if pm_path.exists():
            old_pm = pd.read_csv(pm_path)
            have = set(old_pm["match_id"].astype(int))
        if sh_path.exists():
            old_sh = pd.read_csv(sh_path)

        todo = [
            m for m in league["dates"]
            if m.get("isResult") and int(m["id"]) not in have
        ]
        if not todo:
            continue
        new_pm: list[dict] = []
        new_sh: list[dict] = []
        for m in todo:
            if fetched >= max_matches:
                break
            try:
                data = understat_match(s, int(m["id"]))
                p_rows, s_rows = match_rows(m, data, year)
                if not p_rows:
                    raise ValueError("empty rosters")
                new_pm.extend(p_rows)
                new_sh.extend(s_rows)
                fetched += 1
            except Exception as exc:  # keep going; one bad match should not stop the run
                failures.append(f"{m['id']}: {exc}")
            time.sleep(0.4)
            if fetched and fetched % 100 == 0:
                log(f"understat: {fetched} matches fetched so far")
        if new_pm:
            pm = _numeric(pd.DataFrame(new_pm), PM_NUMERIC)
            sh = _numeric(pd.DataFrame(new_sh, columns=SHOT_COLUMNS), SHOT_NUMERIC)
            if old_pm is not None:
                pm = pd.concat([old_pm, pm], ignore_index=True)
            if old_sh is not None:
                sh = pd.concat([old_sh, sh], ignore_index=True)
            pm.sort_values(["date", "match_id", "h_a", "positionOrder"]).to_csv(pm_path, index=False)
            sh.sort_values(["date", "match_id", "minute"]).to_csv(sh_path, index=False)
            log(f"understat {year}: +{len(set(r['match_id'] for r in new_pm))} matches")
        if fetched >= max_matches:
            log(f"understat: hit max_matches={max_matches}, stopping early")
            break
    if failures:
        log(f"understat: {len(failures)} match failures, first: {failures[:3]}")
    return {"matches_fetched": fetched, "failures": failures[:50]}


# --------------------------------------------------------------------------
# Fantasy Premier League API
# --------------------------------------------------------------------------


def fetch_fpl(out: Path) -> dict[str, Any]:
    folder = out / "fpl"
    folder.mkdir(parents=True, exist_ok=True)
    s = make_session()
    boot = get(s, f"{FPL_BASE}/bootstrap-static/")
    boot.raise_for_status()
    bs = boot.json()
    (folder / "bootstrap-static.json").write_text(
        json.dumps(bs, separators=(",", ":"), ensure_ascii=False)
    )
    fx = get(s, f"{FPL_BASE}/fixtures/")
    fx.raise_for_status()
    (folder / "fixtures.json").write_text(
        json.dumps(fx.json(), separators=(",", ":"), ensure_ascii=False)
    )
    first_deadline = bs["events"][0]["deadline_time"] if bs.get("events") else None
    season = int(first_deadline[:4]) if first_deadline else current_season()

    rows: list[dict] = []
    failures: list[str] = []
    played = [e for e in bs["elements"] if (e.get("minutes") or 0) > 0]
    for el in played:
        try:
            r = get(s, f"{FPL_BASE}/element-summary/{el['id']}/")
            r.raise_for_status()
            for h in r.json().get("history", []):
                h["element"] = el["id"]
                rows.append(h)
        except Exception as exc:
            failures.append(f"{el['id']}: {exc}")
        time.sleep(0.15)
    if rows:
        pd.DataFrame(rows).to_csv(folder / f"player_history_{season}.csv", index=False)
    log(f"fpl: {len(bs['elements'])} players, history for {len(played)}, {len(failures)} failures")
    return {"season": season, "players": len(bs["elements"]), "history_rows": len(rows),
            "failures": failures[:20]}


# --------------------------------------------------------------------------
# ESPN: other competitions
# --------------------------------------------------------------------------

# Competitions Premier League clubs and their players play besides the league.
# An unknown slug just logs a failure; the rest carry on.
ESPN_CLUB = ["uefa.champions", "uefa.europa", "uefa.europa.conf", "eng.fa",
             "eng.league_cup", "eng.charity", "fifa.cwc"]
ESPN_INTERNATIONAL = ["fifa.friendly", "uefa.nations", "uefa.euroq", "uefa.euro",
                      "fifa.world", "fifa.worldq.uefa", "fifa.worldq.conmebol",
                      "fifa.worldq.caf", "fifa.worldq.afc", "fifa.worldq.concacaf",
                      "conmebol.america", "caf.nations", "concacaf.gold",
                      "concacaf.nations.league", "afc.asian.cup"]
ESPN_PLAYER_STATS = {"totalGoals": "goals", "goalAssists": "assists", "totalShots": "shots",
                     "shotsOnTarget": "sot", "yellowCards": "yellow", "redCards": "red",
                     "foulsCommitted": "fouls"}
ESPN_TEAM_STATS = {"wonCorners": "corners", "totalShots": "shots", "shotsOnTarget": "sot",
                   "foulsCommitted": "fouls", "yellowCards": "yellow", "redCards": "red",
                   "possessionPct": "possession"}


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def espn_compact(league: str, ev: dict, sm: dict) -> dict[str, Any]:
    """The parts of an ESPN match summary the model uses, about 2 KB a match.

    Players: starter, minutes (from substitutions, red cards and extra time),
    goals, assists, shots, shots on target, cards, fouls. Teams: score and the
    box score's corners, shots, cards, fouls and possession.
    """
    comp = (sm.get("header", {}).get("competitions") or [{}])[0]
    teams = {}
    for c in comp.get("competitors", []):
        t = c.get("team", {})
        teams[str(t.get("id"))] = {"id": str(t.get("id")), "name": t.get("displayName"),
                                   "side": c.get("homeAway"), "score": _num(c.get("score"))}
    keys = sm.get("keyEvents", []) or []
    periods = max([(k.get("period") or {}).get("number", 0) for k in keys] + [2])
    end = 120.0 if periods >= 3 else 90.0
    on, off = {}, {}
    for k in keys:
        kind = (k.get("type") or {}).get("type", "")
        minute = min(((k.get("clock") or {}).get("value") or 0.0) / 60.0, end)
        ps = [str((p.get("athlete") or {}).get("id")) for p in k.get("participants", [])]
        if kind == "substitution" and len(ps) >= 2:
            on[ps[0]] = minute
            off[ps[1]] = minute
        elif "red" in kind and ps:
            off.setdefault(ps[0], minute)
    players = []
    for side in sm.get("rosters", []) or []:
        tid = str((side.get("team") or {}).get("id"))
        for p in side.get("roster", []) or []:
            a = p.get("athlete") or {}
            pid = str(a.get("id"))
            st = {s.get("name"): s.get("value") for s in p.get("stats", []) or []}
            starter = bool(p.get("starter"))
            if not starter and not p.get("subbedIn") and pid not in on:
                continue
            start = 0.0 if starter else on.get(pid, end)
            stop = off.get(pid, end)
            players.append({
                "id": pid, "name": a.get("displayName"), "team": tid, "starter": starter,
                "minutes": round(max(stop - start, 0.0), 1),
                "pos": (p.get("position") or {}).get("abbreviation"),
                **{v: st.get(k) for k, v in ESPN_PLAYER_STATS.items()},
            })
    for t in (sm.get("boxscore") or {}).get("teams", []) or []:
        tid = str((t.get("team") or {}).get("id"))
        st = {s.get("name"): _num(s.get("displayValue")) for s in t.get("statistics", []) or []}
        if tid in teams:
            teams[tid].update({v: st.get(k) for k, v in ESPN_TEAM_STATS.items()})
    return {"league": league, "event": str(ev.get("id")),
            "date": comp.get("date") or ev.get("date"), "name": ev.get("name"),
            "season": (sm.get("header", {}).get("season") or {}).get("name"),
            "neutral": comp.get("neutralSite"), "periods": periods,
            "teams": list(teams.values()), "players": players}


def _espn_dates(s: requests.Session, slug: str, season: int, today: dt.date) -> list[dt.date]:
    """Days with matches in a competition's season (July to June), up to today.

    The scoreboard rejects date ranges, but its ``calendar`` lists the days
    that have matches. Falls back to every day if there is no calendar.
    """
    first, last = dt.date(season, 7, 1), min(today, dt.date(season + 1, 6, 30))
    days: set[dt.date] = set()
    # Early July can still return the previous season's calendar; mid-October
    # is safely inside the season.
    probes = [min(dt.date(season, 10, 15), today), first]
    probe = probes[0]
    for _ in range(6):
        r = get(s, f"{ESPN_BASE}/{slug}/scoreboard?dates={probe:%Y%m%d}")
        if r.status_code != 200:
            raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]!r}")
        cal = []
        for lg in r.json().get("leagues", []) or []:
            for c in lg.get("calendar", []) or []:
                # Either match days ("2026-09-16T07:00Z") or rounds with a
                # start and end date (possibly nested as entries).
                if isinstance(c, str):
                    cal.append((c, c))
                elif isinstance(c, dict):
                    ents = [e for e in c.get("entries", []) or [] if isinstance(e, dict)] or [c]
                    cal += [(e.get("startDate") or e.get("value"),
                             e.get("endDate") or e.get("startDate") or e.get("value"))
                            for e in ents]
        for a, b in cal:
            try:
                d0 = dt.date.fromisoformat(str(a)[:10])
                d1 = dt.date.fromisoformat(str(b)[:10])
            except ValueError:
                continue
            for i in range(min((d1 - d0).days, 400) + 1):
                d = d0 + dt.timedelta(days=i)
                if first <= d <= last:
                    days.add(d)
        if len(probes) > 1:
            probe = probes.pop()
            continue
        later = [d for d in days if d > probe]
        # A calendar may cover one stage only; ask again after its last day.
        nxt = max(later) + dt.timedelta(days=1) if later else None
        if nxt is None or nxt > last:
            break
        probe = nxt
    # Calendars can lag behind: always look at the last ten days too.
    for i in range(11):
        d = today - dt.timedelta(days=i)
        if first <= d <= last:
            days.add(d)
    if not days and not cal:
        log(f"espn {slug} {season}: no calendar, checking every day")
        days = {first + dt.timedelta(days=i) for i in range((last - first).days + 1)}
    return sorted(days)


def fetch_espn(out: Path, seasons: list[int], refresh_all: bool, max_matches: int,
               minutes: float = 80.0) -> dict[str, Any]:
    """Line-ups, minutes and stats from cup, European and international games.

    One gzipped JSON-lines file per competition and month
    (``espn/<slug>/<YYYY-MM>.jsonl.gz``), one compact match per line
    (``espn_compact``). Days already stored are not fetched again unless
    they are within the last four days (late corrections).
    """
    import gzip

    folder = out / "espn"
    s = make_session()
    today = dt.date.today()
    fetched, failures, comps_failed = 0, [], {}
    # Stop in time to commit what was fetched (the job has a time limit);
    # the next run carries on from the days already stored.
    deadline = time.monotonic() + minutes * 60
    for slug in ESPN_CLUB + ESPN_INTERNATIONAL:
        for season in seasons:
            if fetched >= max_matches or time.monotonic() > deadline:
                break
            done_path = folder / slug / f"days_{season}.json"
            if done_path.exists() and not refresh_all and not season_is_live(season) \
                    and json.loads(done_path.read_text()):
                continue                     # finished season, already stored
            done_days = set(json.loads(done_path.read_text())) \
                if done_path.exists() and not refresh_all else set()
            try:
                days = _espn_dates(s, slug, season, today)
            except Exception as exc:
                comps_failed[slug] = f"{type(exc).__name__}: {exc}"[:200]
                break
            by_month: dict[str, dict[str, dict]] = {}
            for day in days:
                if fetched >= max_matches or time.monotonic() > deadline:
                    break
                settled = day < today - dt.timedelta(days=4)
                if day.isoformat() in done_days and settled:
                    continue
                month = f"{day:%Y-%m}"
                if month not in by_month:
                    path = folder / slug / f"{month}.jsonl.gz"
                    by_month[month] = {}
                    if path.exists() and not refresh_all:
                        with gzip.open(path, "rt", encoding="utf-8") as f:
                            for line in f:
                                rec = json.loads(line)
                                by_month[month][rec["event"]] = rec
                try:
                    r = get(s, f"{ESPN_BASE}/{slug}/scoreboard?dates={day:%Y%m%d}")
                    r.raise_for_status()
                    events = r.json().get("events", [])
                except Exception as exc:
                    failures.append(f"{slug} {day}: {exc}")
                    continue
                ok = True
                for ev in events:
                    eid = str(ev.get("id"))
                    completed = ((ev.get("status") or {}).get("type") or {}).get("completed")
                    if not completed:
                        ok = False
                        continue
                    if eid in by_month[month] and settled:
                        continue
                    try:
                        sr = get(s, f"{ESPN_BASE}/{slug}/summary?event={eid}")
                        sr.raise_for_status()
                        by_month[month][eid] = espn_compact(slug, ev, sr.json())
                        fetched += 1
                    except Exception as exc:
                        failures.append(f"{slug} {eid}: {exc}")
                        ok = False
                    time.sleep(0.05)
                if ok and settled:
                    done_days.add(day.isoformat())
            for month, recs in by_month.items():
                path = folder / slug / f"{month}.jsonl.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    for rec in sorted(recs.values(), key=lambda r: r.get("date") or ""):
                        f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
                        f.write("\n")
            if days:
                done_path.parent.mkdir(parents=True, exist_ok=True)
                done_path.write_text(json.dumps(sorted(done_days)))
            n = sum(len(v) for v in by_month.values())
            if by_month:
                log(f"espn {slug} {season}: {len(days)} match days, {n} matches in "
                    f"{len(by_month)} months touched")
    # Cup and European fixtures in the next ten days: a big game midweek
    # means rotation at the weekend.
    upcoming = []
    for slug in ESPN_CLUB:
        if slug in comps_failed:
            continue
        for i in range(11):
            day = today + dt.timedelta(days=i)
            try:
                r = get(s, f"{ESPN_BASE}/{slug}/scoreboard?dates={day:%Y%m%d}")
                r.raise_for_status()
            except Exception as exc:
                failures.append(f"{slug} upcoming {day}: {exc}")
                break
            for ev in r.json().get("events", []):
                comp = (ev.get("competitions") or [{}])[0]
                upcoming.append({
                    "league": slug, "event": str(ev.get("id")), "date": ev.get("date"),
                    "teams": [{"name": (c.get("team") or {}).get("displayName"),
                               "side": c.get("homeAway")} for c in comp.get("competitors", [])],
                })
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "upcoming.json").write_text(json.dumps(upcoming, ensure_ascii=False))
    if time.monotonic() > deadline:
        log("espn: stopped at the time limit; the next run continues")
    log(f"espn: {fetched} matches fetched, {len(failures)} failures, "
        f"competitions failed: {comps_failed}")
    if fetched == 0 and len(comps_failed) == len(ESPN_CLUB + ESPN_INTERNATIONAL):
        raise RuntimeError(f"every ESPN competition failed: {comps_failed}")
    return {"matches_fetched": fetched, "failures": failures[:50],
            "competitions_failed": comps_failed}


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path, help="data directory to write into")
    p.add_argument("--first-season", type=int, default=2019,
                   help="earliest season start year to download")
    p.add_argument("--backfill-from", type=int, default=2019,
                   help="earliest season to fetch Understat match detail for")
    p.add_argument("--max-matches", type=int, default=4000,
                   help="cap on Understat match pages per run")
    p.add_argument("--refresh-all", action="store_true",
                   help="re-download finished seasons too")
    p.add_argument("--espn-minutes", type=float, default=80.0,
                   help="time budget for the ESPN download")
    p.add_argument("--espn-from", type=int, default=2021,
                   help="earliest season to fetch ESPN other-competition games for")
    p.add_argument("--only", nargs="*", choices=["football-data", "understat", "fpl", "espn"],
                   help="limit to these sources")
    args = p.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    seasons = list(range(args.first_season, current_season() + 1))
    sources = args.only or ["football-data", "understat", "fpl", "espn"]

    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.setdefault("sources", {})
    any_ok = False
    for name in sources:
        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        try:
            if name == "football-data":
                info = fetch_football_data(out, seasons, args.refresh_all, make_session())
            elif name == "understat":
                info = fetch_understat(out, seasons, args.backfill_from, args.max_matches,
                                       args.refresh_all)
            elif name == "espn":
                info = fetch_espn(out, [y for y in seasons if y >= args.espn_from],
                                  args.refresh_all, args.max_matches, args.espn_minutes)
            else:
                info = fetch_fpl(out)
            status["sources"][name] = {"ok": True, "updated_at": started, **info}
            any_ok = True
        except Exception as exc:
            log(f"{name} FAILED: {exc}")
            traceback.print_exc()
            prev = status["sources"].get(name, {})
            status["sources"][name] = {
                "ok": False,
                "failed_at": started,
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "last_success": prev.get("updated_at") or prev.get("last_success"),
            }
    status["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    status_path.write_text(json.dumps(status, indent=2))
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
