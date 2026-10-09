"""Player profiles: how much of their team's output each player accounts for.

For every player we estimate, from Understat match data with recent matches
weighted more heavily:

* share of the team's non-penalty xG while he is on the pitch (goal threat)
* share of the team's xA (assist threat)
* share of the team's on-target and off-target non-goal shots
* yellow and red card rates per 90 minutes
* how often he starts, how long he plays when he starts, and how often he
  comes off the bench

Small samples are shrunk towards the average for the player's position, so a
centre-back with one lucky goal in 200 minutes is not priced like a striker.
Shares (rather than raw per-90 rates) are used so a player's output scales
with how good his team's attack is expected to be in a given match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import load, names
from .glm import decay_weights

POS_GROUP = {
    "GK": "GK", "DC": "CB", "DR": "FB", "DL": "FB", "DMR": "WB", "DML": "WB",
    "DMC": "DM", "MC": "CM", "MR": "WM", "ML": "WM", "AMC": "AM", "AMR": "W",
    "AML": "W", "FW": "FW",
}
FPL_TO_GROUP = {"GK": "GK", "DEF": "CB", "MID": "CM", "FWD": "FW"}


@dataclass
class PlayerParams:
    xi: float = 0.0025              # decay for rates and shares (per day)
    years: float = 3.0              # history window
    prior_minutes: float = 600.0    # shrinkage strength for shares
    prior_minutes_cards: float = 1200.0
    prior_minutes_red: float = 6000.0
    start_halflife_days: float = 45.0


def team_match_totals(pm: pd.DataFrame) -> pd.DataFrame:
    """Per match and team: npxG, xA, non-goal shots on/off target, cards."""
    t = pm.assign(
        ngon=(pm["sot"] - pm["goals"]).clip(lower=0),
        ngoff=(pm["shots"] - pm["sot"]).clip(lower=0),
    ).groupby(["match_id", "team"]).agg(
        t_npxg=("npxg", "sum"), t_xa=("xa", "sum"), t_ngon=("ngon", "sum"),
        t_ngoff=("ngoff", "sum"), t_yellow=("yellow", "sum"), t_red=("red", "sum"),
        t_npgoals=("np_goals", "sum"),
    )
    return t.reset_index()


def enrich(pm: pd.DataFrame) -> pd.DataFrame:
    tot = team_match_totals(pm)
    out = pm.merge(tot, on=["match_id", "team"], how="left")
    out["ngon"] = (out["sot"] - out["goals"]).clip(lower=0)
    out["ngoff"] = (out["shots"] - out["sot"]).clip(lower=0)
    out["frac"] = out["minutes"] / 90.0
    out["group"] = out["position"].map(POS_GROUP)
    return out


SHARE_STATS = {
    "npxg": ("npxg", "t_npxg"),
    "xa": ("xa", "t_xa"),
    "ngon": ("ngon", "t_ngon"),
    "ngoff": ("ngoff", "t_ngoff"),
}


def position_priors(pe: pd.DataFrame) -> pd.DataFrame:
    """League-wide shares and card rates by position group."""
    pe = pe.copy()
    grp = pe.groupby("player_id")["group"].agg(lambda s: s.dropna().mode().iat[0]
                                               if s.dropna().size else "CM")
    pe["pgroup"] = pe["player_id"].map(grp)
    rows = {}
    for g, d in pe.groupby("pgroup"):
        r = {}
        for k, (num, den) in SHARE_STATS.items():
            r[f"share_{k}"] = d[num].sum() / max((d[den] * d["frac"]).sum(), 1e-9)
        r["yellow90"] = d["yellow"].sum() / max(d["frac"].sum(), 1e-9)
        r["red90"] = d["red"].sum() / max(d["frac"].sum(), 1e-9)
        st = d[d["started"]]
        r["min_start"] = st["minutes"].mean()
        r["p_full"] = (st["minutes"] >= 90).mean()
        r["min_sub"] = d.loc[~d["started"], "minutes"].mean()
        rows[g] = r
    pri = pd.DataFrame(rows).T
    return pri


@dataclass
class TeamPlayers:
    team: str
    table: pd.DataFrame          # one row per squad player
    pen_order: list[int]         # player_ids in penalty-taking order
    notes: list[str]


def build_profiles(
    as_of: pd.Timestamp,
    pm: pd.DataFrame | None = None,
    params: PlayerParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profiles for every player with matches in the window, plus position priors."""
    params = params or PlayerParams()
    pm_all = load.player_matches() if pm is None else pm
    pm = pm_all[(pm_all["date"] < as_of)
                & (pm_all["date"] > as_of - pd.Timedelta(days=365 * params.years))]
    pe = enrich(pm)
    # Position norms are very stable; with too little history before as_of
    # (start of the data), take them from everything available.
    pri = position_priors(pe if len(pe) >= 3000 else enrich(pm_all))
    pe["w"] = decay_weights(pe["date"], as_of, params.xi)

    agg = {}
    for k, (num, den) in SHARE_STATS.items():
        pe[f"num_{k}"] = pe["w"] * pe[num]
        pe[f"den_{k}"] = pe["w"] * pe[den] * pe["frac"]
        agg[f"num_{k}"] = (f"num_{k}", "sum")
        agg[f"den_{k}"] = (f"den_{k}", "sum")
    pe["w_frac"] = pe["w"] * pe["frac"]
    pe["w_yellow"] = pe["w"] * pe["yellow"]
    pe["w_red"] = pe["w"] * pe["red"]
    prof = pe.groupby("player_id").agg(
        player=("player", "last"), last_team=("team", "last"), last_date=("date", "max"),
        apps=("match_id", "size"), starts=("started", "sum"), minutes=("minutes", "sum"),
        goals=("goals", "sum"), np_goals=("np_goals", "sum"), assists=("assists", "sum"),
        shots=("shots", "sum"), sot=("sot", "sum"), xg=("xg", "sum"), npxg=("npxg", "sum"),
        xa=("xa", "sum"), yellow=("yellow", "sum"), red=("red", "sum"),
        pens=("pens", "sum"), pen_goals=("pen_goals", "sum"),
        w_frac=("w_frac", "sum"), w_yellow=("w_yellow", "sum"), w_red=("w_red", "sum"),
        **agg,
    )
    grp = pe.dropna(subset=["group"]).groupby("player_id")["group"].agg(
        lambda s: s.value_counts().index[0])
    prof["group"] = grp.reindex(prof.index).fillna("CM")

    # Shrink shares towards the position prior. The prior is worth
    # prior_minutes of play at a typical team's level of that stat.
    league_team_avg = {k: pe.drop_duplicates(["match_id", "team"])[den].mean()
                       for k, (_, den) in SHARE_STATS.items()}
    for k in SHARE_STATS:
        kp = params.prior_minutes / 90.0 * league_team_avg[k]
        prior = prof["group"].map(pri[f"share_{k}"])
        prof[f"share_{k}"] = (prof[f"num_{k}"] + prior * kp) / (prof[f"den_{k}"] + kp)
        prof[f"raw_share_{k}"] = prof[f"num_{k}"] / prof[f"den_{k}"].replace(0, np.nan)
    kc = params.prior_minutes_cards / 90.0
    prof["yellow90"] = (prof["w_yellow"] + prof["group"].map(pri["yellow90"]) * kc) \
        / (prof["w_frac"] + kc)
    kr = params.prior_minutes_red / 90.0
    prof["red90"] = (prof["w_red"] + prof["group"].map(pri["red90"]) * kr) / (prof["w_frac"] + kr)

    # Minutes when starting / from the bench, shrunk to the position norm.
    st = pe[pe["started"]].groupby("player_id").agg(
        n_st=("minutes", "size"), min_st=("minutes", "mean"),
        full=("minutes", lambda x: (x >= 90).mean()))
    sb = pe[~pe["started"]].groupby("player_id").agg(n_sb=("minutes", "size"),
                                                     min_sb=("minutes", "mean"))
    prof = prof.join(st).join(sb)
    k = 5.0
    for col, pcol, ncol in (("min_st", "min_start", "n_st"), ("full", "p_full", "n_st"),
                            ("min_sb", "min_sub", "n_sb")):
        n = prof[ncol].fillna(0)
        prof[col] = (prof[col].fillna(0) * n + prof["group"].map(pri[pcol]) * k) / (n + k)
    prof = prof.drop(columns=[c for c in prof.columns if c.startswith(("num_", "den_"))])
    return prof.reset_index(), pri


def selection_probs(team: str, as_of: pd.Timestamp, pm: pd.DataFrame,
                    halflife_days: float = 45.0) -> pd.DataFrame:
    """Recent start and bench-appearance rates for a team's players."""
    tm = pm[(pm["team"] == team) & (pm["date"] < as_of)]
    if tm.empty:
        return pd.DataFrame(columns=["player_id", "p_start_hist", "p_sub_hist", "team_apps"])
    matches = tm.drop_duplicates("match_id")[["match_id", "date"]].sort_values("date")
    matches = matches[matches["date"] > as_of - pd.Timedelta(days=400)]
    xi = np.log(2) / halflife_days
    matches["w"] = decay_weights(matches["date"], as_of, xi)
    first = tm.groupby("player_id")["date"].min()
    out = []
    for pid, d in tm[tm["match_id"].isin(matches["match_id"])].groupby("player_id"):
        # only count team matches since the player's first appearance for the team
        mw = matches[matches["date"] >= first[pid]]
        if mw["w"].sum() <= 0:
            continue
        started = d.set_index("match_id")["started"].reindex(mw["match_id"]).fillna(False)
        played = d.set_index("match_id")["minutes"].reindex(mw["match_id"]).notna()
        wv = mw["w"].to_numpy()
        p_start = float(np.sum(wv * started.to_numpy()) / wv.sum())
        not_started = ~started.to_numpy()
        sub = played.to_numpy() & not_started
        p_sub = float(np.sum(wv * sub) / max(np.sum(wv * not_started), 1e-9)) \
            if not_started.any() else 0.0
        out.append({"player_id": pid, "p_start_hist": p_start, "p_sub_hist": p_sub,
                    "team_apps": int(played.sum())})
    return pd.DataFrame(out, columns=["player_id", "p_start_hist", "p_sub_hist", "team_apps"])


def fpl_match(name: str, fpl: pd.DataFrame) -> tuple[pd.Series | None, float]:
    """Best FPL entry for an Understat name (full name, or the FPL short name)."""
    best, best_s = None, 0.0
    for _, c in fpl.iterrows():
        s = max(names.name_score(name, c["full_name"]),
                names.name_score(name, c["web_name"]) - 0.02)
        if s > best_s:
            best, best_s = c, s
    return best, best_s


def link_fpl(squad: pd.DataFrame, fpl: pd.DataFrame, team: str,
             chance_col: str = "chance_of_playing_next_round") -> pd.DataFrame:
    """Attach FPL availability (status, chance of playing, news) by name.

    ``status`` is "?" for a player FPL does not list for this team.
    """
    if fpl is None or fpl.empty:
        return squad.assign(fpl_id=np.nan, status="a", chance=np.nan, news="",
                            fpl_pen_order=np.nan)
    f = fpl[fpl["team_name"] == team]
    ids, status, chance, news, pen = [], [], [], [], []
    for _, r in squad.iterrows():
        best, best_s = fpl_match(r["player"], f)
        if best is not None and best_s >= 0.8:
            ids.append(best["id"])
            status.append(best["status"])
            chance.append(best.get(chance_col, best.get("chance_of_playing_next_round")))
            news.append(best["news"] or "")
            pen.append(best.get("penalties_order"))
        else:
            ids.append(np.nan)
            status.append("?")
            chance.append(np.nan)
            news.append("")
            pen.append(np.nan)
    return squad.assign(fpl_id=ids, status=status, chance=chance, news=news, fpl_pen_order=pen)


def availability(status: str, chance) -> float:
    """Probability the player is available, from FPL status/chance."""
    if chance is not None and not (isinstance(chance, float) and np.isnan(chance)):
        return float(chance) / 100.0
    return {"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}.get(status, 0.9)


def pen_takers(team: str, as_of: pd.Timestamp, squad: pd.DataFrame,
               shots: pd.DataFrame | None) -> list:
    """Penalty order: FPL's listed order first, then recent penalty takers."""
    order: list = []
    if "fpl_pen_order" in squad:
        fp = squad.dropna(subset=["fpl_pen_order"]).sort_values("fpl_pen_order")
        order += list(fp.index)
    if shots is not None and not shots.empty:
        pens = shots[(shots["team"] == team) & shots["is_pen"] & (shots["date"] < as_of)
                     & (shots["date"] > as_of - pd.Timedelta(days=500))]
        w = decay_weights(pens["date"], as_of, 0.004)
        ranked = pd.Series(w, index=pens["player_id"].to_numpy()).groupby(level=0).sum() \
            .sort_values(ascending=False)
        order += [pid for pid in ranked.index if pid in squad.index and pid not in order]
    return order


def build_squad(
    team: str,
    as_of: pd.Timestamp,
    profiles: pd.DataFrame,
    priors: pd.DataFrame,
    pm: pd.DataFrame,
    fpl: pd.DataFrame | None = None,
    lineup: list[str] | None = None,
    absent: list[str] | None = None,
    bench: list[str] | None = None,
    chance_col: str = "chance_of_playing_next_round",
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Squad table for the simulator, indexed by player_id.

    ``lineup``: the starting XI if known (names). Without it the XI is projected
    from recent selections and FPL availability. ``absent``: players ruled out.
    Returns (squad, notes, regulars): ``regulars`` lists the team's recent
    regular starters with whether they start in this XI, for the absence
    adjustment.
    """
    notes: list[str] = []
    recent = profiles[(profiles["last_team"] == team)
                      & (profiles["last_date"] > as_of - pd.Timedelta(days=300))].copy()
    sel = selection_probs(team, as_of, pm)
    sq = recent.merge(sel, on="player_id", how="left")
    sq[["p_start_hist", "p_sub_hist"]] = sq[["p_start_hist", "p_sub_hist"]].fillna(0.0)

    def lookup(name: str) -> pd.Series | None:
        """Find a player by name: this squad first, then anyone in the data."""
        for pool, fallback in ((sq, True), (profiles, False)):
            if pool.empty:
                continue
            j, _ = names.best_match(name, pool["player"], surname_fallback=fallback)
            if j is not None:
                return pool.iloc[j]
        return None

    extra, named, ids_by_name = [], set(), {}
    for nm in (lineup or []) + (bench or []):
        row = lookup(nm)
        if row is None:
            # No Understat history (e.g. new signing from abroad): position prior.
            g = "CM"
            pr = priors.loc[g]
            extra.append({"player_id": f"new:{nm}", "player": nm, "group": g,
                          "share_npxg": pr["share_npxg"], "share_xa": pr["share_xa"],
                          "share_ngon": pr["share_ngon"], "share_ngoff": pr["share_ngoff"],
                          "yellow90": pr["yellow90"], "red90": pr["red90"],
                          "min_st": pr["min_start"], "full": pr["p_full"],
                          "min_sb": pr["min_sub"], "p_start_hist": 0.0, "p_sub_hist": 0.3,
                          "minutes": 0, "apps": 0})
            named.add(f"new:{nm}")
            ids_by_name[nm] = f"new:{nm}"
            notes.append(f"{nm}: no Premier League data, using average {g} rates")
            continue
        named.add(row["player_id"])
        ids_by_name[nm] = row["player_id"]
        if row["player_id"] not in set(sq["player_id"]):
            r = row.to_dict()
            r.setdefault("p_start_hist", 0.0)
            r.setdefault("p_sub_hist", 0.3)
            extra.append(r)
            if row.get("last_team") != team:
                notes.append(f"{row['player']}: rates from previous club ({row.get('last_team')})")
    if extra:
        sq = pd.concat([sq, pd.DataFrame(extra)], ignore_index=True)
    sq = sq.drop_duplicates("player_id").set_index("player_id")
    for c in ("p_start_hist", "p_sub_hist"):
        sq[c] = sq[c].fillna(0.0)

    have_fpl = fpl is not None and not fpl.empty and bool((fpl["team_name"] == team).any())
    sq = link_fpl(sq.reset_index(), fpl if have_fpl else None, team, chance_col) \
        .set_index("player_id")
    sq["avail"] = [availability(s, c) for s, c in zip(sq["status"], sq["chance"])]

    # FPL lists every registered player, so anyone it does not list for this
    # team has left (sold, released or moved within the league). Leave them
    # out unless the match file names them.
    if have_fpl:
        gone = sq.index[(sq["status"] == "?") & ~sq.index.isin(named)]
        for pid in gone:
            if sq.at[pid, "p_start_hist"] + sq.at[pid, "p_sub_hist"] < 0.25:
                continue
            nm = sq.at[pid, "player"]
            other, sc = fpl_match(nm, fpl[fpl["team_name"] != team])
            where = f"now at {other['team_name']}" if other is not None and sc >= 0.9 \
                else "no longer in FPL's Premier League player list"
            notes.append(f"{nm}: not in FPL's {team} squad ({where}), left out")
        sq.loc[gone, "avail"] = 0.0
        for pid in sq.index[(sq["status"] == "?") & sq.index.isin(named)]:
            notes.append(f"{sq.at[pid, 'player']}: named in the match file but not in FPL's "
                         f"{team} squad list; check the name")

    absent_ids = set()
    for nm in absent or []:
        j, _ = names.best_match(nm, sq["player"])
        if j is not None:
            absent_ids.add(sq.index[j])
        elif not have_fpl or fpl_match(nm, fpl[fpl["team_name"] == team])[1] < 0.8:
            # (a player FPL lists but with no recent league minutes needs no note)
            notes.append(f"Absent player '{nm}' not recognised in {team}'s squad; check the "
                         f"spelling")
    sq.loc[list(absent_ids), "avail"] = 0.0

    if lineup:
        starters = [ids_by_name[nm] for nm in lineup if nm in ids_by_name]
        sq["start"] = sq.index.isin(starters)
        if len(set(starters)) != 11:
            notes.append(f"{team} lineup has {len(set(starters))} recognised starters, not 11")
        clash = absent_ids & set(starters)
        if clash:
            notes.append(f"{team}: {', '.join(sq.loc[list(clash), 'player'])} named in the XI "
                         f"and as absent; kept in the XI")
            absent_ids -= clash
    else:
        if sq.empty:
            notes.append(f"{team}: no recent Premier League line-ups to project from; put the "
                         f"XI in the match file to price player markets")
        sq["p_start"] = sq["p_start_hist"] * sq["avail"]
        gk = sq[sq["group"] == "GK"].sort_values("p_start", ascending=False).head(1).index
        out = sq[sq["group"] != "GK"].sort_values("p_start", ascending=False).head(10).index
        sq["start"] = sq.index.isin(gk.append(out))
        notes.append(f"{team} XI projected from recent team selections and availability")
    sq["p_sub"] = np.where(sq["start"], 0.0, sq["p_sub_hist"].clip(0, 0.95) * sq["avail"])
    if bench:
        bench_ids = [ids_by_name[nm] for nm in bench if nm in ids_by_name]
        sq.loc[~sq.index.isin(bench_ids) & ~sq["start"], "p_sub"] = 0.0
    sq.loc[list(absent_ids), "start"] = False
    sq.loc[list(absent_ids), "p_sub"] = 0.0

    regulars = sq[sq["p_start_hist"] >= REGULAR_START_RATE][
        ["player", "group", "p_start_hist", "share_npxg", "min_st", "start", "avail",
         "status", "news"]].copy()
    sq = sq[(sq["start"]) | (sq["p_sub"] > 0.02)]
    return sq, notes, regulars


# A player who started at least this share of recent league games (45-day
# half-life) counts as a regular for the absence adjustment.
REGULAR_START_RATE = 0.5


def presence(team: str, as_of: pd.Timestamp, pm: pd.DataFrame, xi: float,
             days: int = 365) -> tuple[pd.Series, float]:
    """Decay-weighted share of the team's recent league games each player started.

    Uses the same time decay as the team ratings and counts games before a
    player joined (or after he left) as games he did not start, so it says how
    much of the team's current rating was earned with him in the side.
    Returns (shares by player_id, total weight of the games); a small total
    weight means too few recent league games to say what is usual.
    """
    tm = pm[(pm["team"] == team) & (pm["date"] < as_of)
            & (pm["date"] > as_of - pd.Timedelta(days=days))]
    if tm.empty:
        return pd.Series(dtype=float, name="presence"), 0.0
    games = tm.drop_duplicates("match_id")[["match_id", "date"]]
    w = pd.Series(decay_weights(games["date"], as_of, xi), index=games["match_id"].to_numpy())
    st = tm[tm["started"]]
    pres = st["match_id"].map(w).groupby(st["player_id"].to_numpy()).sum() / w.sum()
    return pres.rename("presence"), float(w.sum())


def attack_shift(starter_ids: list, profiles: pd.DataFrame, pres: pd.Series,
                 priors: pd.DataFrame | None = None, groups: dict | None = None) -> float:
    """How much more (+) or less (-) attacking threat this XI has than usual.

    A player's contribution is his share of team non-penalty xG times his
    usual minutes as a starter. The shift is the XI's total contribution minus
    the presence-weighted total of everyone who has started for the team, so
    an XI like the team's usual one scores about zero, a missing regular
    counts against it and a returning one counts for it. ``profiles`` is
    indexed by player_id; starters without a profile (no league data) get the
    average for their position group (``groups``: player_id -> group).
    """
    groups = groups or {}
    ids = set(starter_ids) | set(pres.index[pres > 0.005])
    starters = set(starter_ids)
    total = 0.0
    for pid in ids:
        if pid in profiles.index:
            r = profiles.loc[pid]
            c = float(r["share_npxg"]) * float(r["min_st"]) / 90.0
        elif pid in starters and priors is not None:
            g = groups.get(pid, "CM")
            c = float(priors.loc[g, "share_npxg"]) * float(priors.loc[g, "min_start"]) / 90.0
        else:
            continue
        total += c * ((pid in starters) - float(pres.get(pid, 0.0)))
    return total


def missing_attack(regulars: pd.DataFrame) -> float:
    """Recent regular starters' attacking contribution that this XI is missing.

    Each missing regular counts his share of team non-penalty xG, scaled by
    his usual minutes and how often he has been starting (the team's ratings
    already reflect the games he missed).
    """
    if regulars is None or regulars.empty:
        return 0.0
    out = regulars[~regulars["start"]]
    return float((out["share_npxg"] * (out["min_st"] / 90.0) * out["p_start_hist"]).sum())


def missing_defence(regulars: pd.DataFrame) -> float:
    """Number of regular goalkeepers/defenders missing, weighted by start rate."""
    if regulars is None or regulars.empty:
        return 0.0
    out = regulars[~regulars["start"] & regulars["group"].isin(DEFENSIVE_GROUPS)]
    return float(out["p_start_hist"].sum())


DEFENSIVE_GROUPS = ("GK", "CB", "FB", "WB", "DM")


def team_news(team: str, fpl: pd.DataFrame | None, pm: pd.DataFrame, as_of: pd.Timestamp,
              chance_col: str = "chance_of_playing_next_round") -> pd.DataFrame:
    """FPL injury, suspension and availability news for one team's players.

    Adds how often each player started the team's last 10 league games, so the
    news can be read for how much it matters. Players who left the club are
    only listed if they had been starting recently.
    """
    cols = ["player", "status", "chance", "news", "news_added", "starts_last10",
            "season_minutes"]
    if fpl is None or fpl.empty:
        return pd.DataFrame(columns=cols)
    f = fpl[fpl["team_name"] == team].copy()
    f["chance"] = f.get(chance_col, f["chance_of_playing_next_round"])
    f = f[(f["status"] != "a") | (f["chance"].notna() & (f["chance"] < 100))]
    tm = pm[(pm["team"] == team) & (pm["date"] < as_of)]
    last10 = tm.drop_duplicates("match_id").sort_values("date").tail(10)["match_id"]
    recent = tm[tm["match_id"].isin(last10)]
    starts = recent[recent["started"]].groupby("player")["match_id"].nunique()
    rows = []
    for _, r in f.iterrows():
        n = 0
        if len(starts):
            sc = starts.index.map(lambda x: max(names.name_score(x, r["full_name"]),
                                                names.name_score(x, r["web_name"]) - 0.02))
            best = int(np.argmax(sc))
            if sc[best] >= 0.8:
                n = int(starts.iloc[best])
        if r["status"] == "u" and n == 0:
            continue
        rows.append({"player": r["full_name"], "status": r["status"], "chance": r["chance"],
                     "news": r["news"], "news_added": r.get("news_added"),
                     "starts_last10": n, "season_minutes": r.get("minutes")})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["starts_last10", "season_minutes"], ascending=False) \
        .reset_index(drop=True)
