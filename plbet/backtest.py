"""Walk-forward backtests.

Every prediction uses only matches that kicked off before that match's date,
exactly as the model would have been used live. Results are compared with the
bookmakers' own prices from football-data.co.uk (Bet365 pre-match and
Pinnacle closing odds).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import odds as odds_mod
from .data import load
from .models import features
from .models.goals import GoalsModel, GoalsParams, implied_lambdas, outcome_probs, \
    score_matrix, total_over


def market_probs(m: pd.DataFrame, prefix: str = "B365", closing: bool = False) -> pd.DataFrame:
    """De-margined 1X2 and over 2.5 probabilities from football-data odds columns."""
    c = "C" if closing else ""
    h, d, a = f"{prefix}{c}H", f"{prefix}{c}D", f"{prefix}{c}A"
    ou_prefix = {"B365": "B365", "PS": "P", "Avg": "Avg", "Max": "Max"}.get(prefix, prefix)
    o, u = f"{ou_prefix}{c}>2.5", f"{ou_prefix}{c}<2.5"
    out = pd.DataFrame(index=m.index, columns=["mk_h", "mk_d", "mk_a", "mk_o25"], dtype=float)
    for i, row in m.iterrows():
        try:
            out.loc[i, ["mk_h", "mk_d", "mk_a"]] = odds_mod.demargin(
                [row[h], row[d], row[a]], "power")
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            pass
        try:
            out.loc[i, "mk_o25"] = odds_mod.demargin([row[o], row[u]], "power")[0]
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            pass
    return out


def goals_walk_forward(
    tr: pd.DataFrame,
    m: pd.DataFrame,
    params: GoalsParams,
    seasons: list[int],
) -> pd.DataFrame:
    """Refit before every match date in ``seasons`` and predict that day's games."""
    prom = features.promoted_sets(m)
    test = m[m["season"].isin(seasons)].copy()
    rows = []
    for day, games in test.groupby(test["date"]):
        gm = GoalsModel(replace(params)).fit(tr, as_of=pd.Timestamp(day))
        for _, g in games.iterrows():
            fr = features.fixture_rows(g["home"], g["away"], g["season"], prom,
                                       nocrowd=int(g["nocrowd"]))
            lh, la = gm.lambdas(fr)
            mat = score_matrix(lh, la, gm.rho_)
            ph, pd_, pa = outcome_probs(mat)
            hg, ag = int(g["hg"]), int(g["ag"])
            rows.append({
                "match_key": g["match_key"], "season": g["season"], "date": g["date"],
                "home": g["home"], "away": g["away"], "hg": hg, "ag": ag,
                "lh": lh, "la": la, "rho": gm.rho_,
                "p_h": ph, "p_d": pd_, "p_a": pa, "p_o25": total_over(mat, 2.5),
                "p_btts": float(mat[1:, 1:].sum()),
                "p_score": float(mat[min(hg, 10), min(ag, 10)]),
            })
    return pd.DataFrame(rows)


def score_1x2(df: pd.DataFrame, cols=("p_h", "p_d", "p_a")) -> dict[str, float]:
    p = df[list(cols)].to_numpy(float)
    p = p / p.sum(axis=1, keepdims=True)
    res = np.select([df["hg"] > df["ag"], df["hg"] == df["ag"]], [0, 1], 2)
    onehot = np.eye(3)[res]
    logloss = -np.mean(np.log(np.clip(p[np.arange(len(p)), res], 1e-12, 1)))
    brier = np.mean(np.sum((p - onehot) ** 2, axis=1))
    cum_p, cum_o = np.cumsum(p, axis=1), np.cumsum(onehot, axis=1)
    rps = np.mean(np.sum((cum_p[:, :2] - cum_o[:, :2]) ** 2, axis=1) / 2)
    return {"logloss": logloss, "brier": brier, "rps": rps}


def score_binary(p: pd.Series, outcome: pd.Series) -> dict[str, float]:
    p = np.clip(p.to_numpy(float), 1e-9, 1 - 1e-9)
    y = outcome.to_numpy(float)
    return {
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def calibration_table(p: pd.Series, outcome: pd.Series, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1)) -> pd.DataFrame:
    b = pd.cut(p, bins, include_lowest=True)
    return pd.DataFrame({"p": p, "y": outcome}).groupby(b, observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))


def blend_lambdas(lh_model, la_model, lh_mkt, la_mkt, w_market: float):
    """Geometric blend of model and market goal means."""
    lh = np.exp(w_market * np.log(lh_mkt) + (1 - w_market) * np.log(lh_model))
    la = np.exp(w_market * np.log(la_mkt) + (1 - w_market) * np.log(la_model))
    return lh, la


def add_market_lambdas(pred: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    """Solve for the goal means implied by market 1X2 + over/under 2.5 probabilities."""
    out = pred.copy()
    lhs, las = [], []
    for _, r in out.join(mk).iterrows():
        if np.isnan(r["mk_h"]):
            lhs.append(np.nan)
            las.append(np.nan)
            continue
        lh, la = implied_lambdas(r["mk_h"], r["mk_d"], r["mk_a"],
                                 None if np.isnan(r["mk_o25"]) else r["mk_o25"], r["rho"])
        lhs.append(lh)
        las.append(la)
    out["lh_mkt"], out["la_mkt"] = lhs, las
    return out


# --------------------------------------------------------------------------
# Full-simulation backtest: corners, cards, shots, players and builders
# --------------------------------------------------------------------------

TEAM_MARKETS = [
    "result:home", "result:draw", "result:away", "goals:over:2.5", "btts:yes",
    "corners:over:8.5", "corners:over:9.5", "corners:over:10.5", "corners:over:11.5",
    "team_corners:home:over:4.5", "team_corners:away:over:3.5", "corners_1x2:home",
    "cards:over:2.5", "cards:over:3.5", "cards:over:4.5", "cards:over:5.5",
    "team_cards:home:over:1.5", "team_cards:away:over:1.5",
    "shots:over:23.5", "shots:over:26.5", "sot:over:7.5", "sot:over:9.5",
    "team_shots:home:over:12.5", "team_shots:away:over:10.5",
    "team_sot:home:over:4.5", "team_sot:away:over:3.5",
    "fouls:over:20.5", "fouls:over:23.5",
    "ht:home", "ht_goals:over:0.5", "ht_goals:over:1.5",
]
# Per-team stats whose simulated means are compared with the real values.
MEAN_STATS = {"goals": ("hg", "ag"), "corners": ("h_corners", "a_corners"),
              "yellow": ("h_yellow", "a_yellow"), "red": ("h_red", "a_red"),
              "fouls": ("h_fouls", "a_fouls"), "shots": ("h_shots", "a_shots"),
              "sot": ("h_sot", "a_sot")}
PLAYER_MARKETS = ["goal", "assist", "goal_or_assist", "shots:1+", "shots:2+", "shots:3+",
                  "sot:1+", "sot:2+", "card"]


def _actual_team(spec: str, g: pd.Series, card_weights=(1, 2)) -> float:
    """Did a team market win in the real match? (from football-data columns)"""
    p = spec.split(":")
    hg, ag = g["hg"], g["ag"]
    stat = {
        "corners": (g["h_corners"], g["a_corners"]),
        "cards": (card_weights[0] * g["h_yellow"] + card_weights[1] * g["h_red"],
                  card_weights[0] * g["a_yellow"] + card_weights[1] * g["a_red"]),
        "shots": (g["h_shots"], g["a_shots"]), "sot": (g["h_sot"], g["a_sot"]),
        "fouls": (g["h_fouls"], g["a_fouls"]), "goals": (hg, ag),
    }
    k = p[0]
    if k == "result":
        return float({"home": hg > ag, "draw": hg == ag, "away": ag > hg}[p[1]])
    if k == "btts":
        return float(hg > 0 and ag > 0)
    if k == "ht":
        return float({"home": g["hthg"] > g["htag"], "draw": g["hthg"] == g["htag"],
                      "away": g["htag"] > g["hthg"]}[p[1]])
    if k == "ht_goals":
        return float(g["hthg"] + g["htag"] > float(p[2]))
    if k in stat:
        return float(sum(stat[k]) > float(p[2]))
    if k.startswith("team_"):
        h, a = stat[k[5:]]
        v = h if p[1] == "home" else a
        return float(v > float(p[3]))
    if k == "corners_1x2":
        return float(g["h_corners"] > g["a_corners"])
    raise ValueError(spec)


def _actual_player(row: pd.Series, what: str) -> float:
    if what == "goal":
        return float(row["goals"] >= 1)
    if what == "assist":
        return float(row["assists"] >= 1)
    if what == "goal_or_assist":
        return float(row["goals"] + row["assists"] >= 1)
    if what == "card":
        return float(row["yellow"] + row["red"] >= 1)
    stat, thr = what.split(":")
    return float(row[stat] >= int(thr.rstrip("+")))


def full_walk_forward(seasons: list[int], sims: int = 20000, refit_days: int = 7,
                      workers: int = 4, limit: int | None = None) -> dict[str, pd.DataFrame]:
    """Simulate every match in ``seasons`` with models fitted beforehand.

    Line-ups are the real starting XIs (as if you bet after team news), the
    referee is the real one, and bookmaker odds are not used. Returns team,
    player and builder prediction tables with outcomes.
    """
    from concurrent.futures import ProcessPoolExecutor

    m = load.matches()
    test = m[m["season"].isin(seasons)].sort_values("kickoff")
    if limit:
        test = test.head(limit)
    start = test["kickoff"].min().normalize()
    blocks = []
    t0 = start
    while t0 <= test["kickoff"].max():
        t1 = t0 + pd.Timedelta(days=refit_days)
        keys = test.loc[(test["kickoff"] >= t0) & (test["kickoff"] < t1), "match_key"].tolist()
        if keys:
            blocks.append((t0, keys, sims))
        t0 = t1
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_run_block, blocks))
    out = {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}
    return out


def _run_block(args) -> dict[str, pd.DataFrame]:
    import warnings
    warnings.filterwarnings("ignore")
    from .analysis import Bundle, MatchSpec, analyse
    from . import markets as mk
    as_of, keys, sims = args
    bundle = Bundle.fit(as_of)
    m = bundle.matches.set_index("match_key")
    pm = bundle.pm
    team_rows, player_rows, builder_rows, mean_rows = [], [], [], []
    for key in keys:
        g = m.loc[key]
        roster = pm[pm["match_id"] == g["us_match_id"]]
        if roster.empty:
            roster = pm[(pm["date"].dt.normalize() == g["date"]) & (pm["team"].isin([g["home"], g["away"]]))]
        lineups = {
            "home": roster.loc[(roster["team"] == g["home"]) & roster["started"], "player"].tolist(),
            "away": roster.loc[(roster["team"] == g["away"]) & roster["started"], "player"].tolist(),
        }
        if len(lineups["home"]) != 11 or len(lineups["away"]) != 11:
            lineups = {}
        spec = MatchSpec(home=g["home"], away=g["away"], kickoff=g["kickoff"],
                         referee=g["referee"] if isinstance(g["referee"], str) else None,
                         lineups=lineups, sims=sims, use_fpl=False)
        try:
            res = analyse(spec, as_of=as_of, bundle=bundle, catalogue=False)
        except Exception as exc:  # keep the backtest going; report failures
            team_rows.append({"match_key": key, "error": str(exc)})
            continue
        sim = res.sim
        sim_stat = {"goals": (sim.hg, sim.ag),
                    "shots": (sim.hg + sim.stats["h_ngon"] + sim.stats["h_ngoff"],
                              sim.ag + sim.stats["a_ngon"] + sim.stats["a_ngoff"]),
                    "sot": (sim.hg + sim.stats["h_ngon"], sim.ag + sim.stats["a_ngon"])}
        for st in ("corners", "yellow", "red", "fouls"):
            sim_stat[st] = (sim.stats[f"h_{st}"], sim.stats[f"a_{st}"])
        for st, (hc, ac) in MEAN_STATS.items():
            for side, col, arr in (("home", hc, sim_stat[st][0]), ("away", ac, sim_stat[st][1])):
                mean_rows.append({"match_key": key, "season": g["season"], "stat": st,
                                  "side": side, "team": g[side], "pred": float(np.mean(arr)),
                                  "actual": float(g[col]), "referee_known": res.referee is not None})
        for spec_s in TEAM_MARKETS:
            team_rows.append({"match_key": key, "season": g["season"], "market": spec_s,
                              "p": mk.evaluate(spec_s, sim).prob(),
                              "y": _actual_team(spec_s, g),
                              "lh": res.lambdas[0], "la": res.lambdas[1]})
        # players: starters only, outcome from Understat
        best = {}
        for side, team in (("h", g["home"]), ("a", g["away"])):
            if side not in sim.players:
                continue
            sq = sim.players[side]
            for j, nm in enumerate(sq.names):
                if not sq.started[j]:
                    continue
                act = roster[(roster["team"] == team) & (roster["player"] == nm)]
                if act.empty:
                    continue
                a = act.iloc[0]
                pos = res.squads[side].loc[sq.ids[j], "group"]
                for what in PLAYER_MARKETS:
                    o = mk.evaluate(f"player:{nm}:{what}", sim)
                    player_rows.append({"match_key": key, "season": g["season"], "team": team,
                                        "player": nm, "pos": pos, "market": what,
                                        "p": o.prob(), "y": _actual_player(a, what)})
                pg = mk.evaluate(f"player:{nm}:goal", sim).prob()
                if pos != "GK" and pg > best.get(side, ("", 0))[1]:
                    best[side] = (nm, pg)
        # builders: joint (simulated) vs product of singles, against outcomes
        combos = [["goals:over:2.5", "btts:yes"],
                  ["result:home", "team_corners:home:over:4.5"],
                  ["result:away", "corners:over:9.5"],
                  ["btts:yes", "cards:over:3.5"],
                  ["goals:over:2.5", "corners:over:9.5", "cards:over:3.5"]]
        for side, res_key in (("h", "result:home"), ("a", "result:away")):
            if side in best:
                nm = best[side][0]
                combos.append([res_key, f"player:{nm}:goal"])
                combos.append([res_key, f"player:{nm}:sot:1+", "goals:over:1.5"])
        for legs in combos:
            outs = [mk.evaluate(l, sim) for l in legs]
            valid = np.ones(sim.n, dtype=bool)
            for o in outs:
                if o.valid is not None:
                    valid &= o.valid
            W = np.column_stack([o.win[valid] for o in outs])
            joint = float(W.all(axis=1).mean())
            prod = float(np.prod(W.mean(axis=0)))
            ys = []
            for l in legs:
                if l.startswith("player:"):
                    _, nm, what = l.split(":", 2)
                    team = g["home"] if nm in sim.players.get("h", SimpleNS()).names else g["away"]
                    act = roster[(roster["team"] == team) & (roster["player"] == nm)]
                    ys.append(_actual_player(act.iloc[0], what) if len(act) else np.nan)
                else:
                    ys.append(_actual_team(l, g))
            builder_rows.append({"match_key": key, "season": g["season"],
                                 "legs": " + ".join(legs) if not any(l.startswith("player") for l in legs)
                                 else " + ".join("player" if l.startswith("player") else l for l in legs),
                                 "p_joint": joint, "p_product": prod,
                                 "y": float(np.all(ys)) if not np.isnan(ys).any() else np.nan})
    return {"team": pd.DataFrame(team_rows), "players": pd.DataFrame(player_rows),
            "builders": pd.DataFrame(builder_rows), "means": pd.DataFrame(mean_rows)}


class SimpleNS:
    names: list = []
