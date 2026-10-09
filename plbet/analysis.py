"""Full analysis of one match: fit models, simulate, price every market.

Typical use (see CLAUDE.md for the whole workflow):

    from plbet.analysis import MatchSpec, analyse
    spec = MatchSpec.from_yaml("matches/2026-10-11-liverpool-man-city.yaml")
    result = analyse(spec)
    print(result.catalogue.head())

The spec carries everything known before kick-off: teams, kick-off time,
referee, confirmed or predicted line-ups, absentees and any bookmaker odds.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import builder, markets, odds as odds_mod
from .data import load, names
from .models import features, players
from .models.counts import MatchStatsModel
from .models.goals import GoalsModel, implied_lambdas, outcome_probs, score_matrix, total_over
from .models.simulate import SimResult, simulate_match

# Weight on the bookmaker's own 1X2 / over-under prices when they are given.
# Backtests (docs/BACKTEST.md) show the market is the better single forecast
# for those main markets, so it gets most of the weight.
MARKET_WEIGHT_GOALS = 0.75
# Corners/cards lines: no historical odds to test against, so an even split.
MARKET_WEIGHT_COUNTS = 0.5
# How strongly a line-up that differs from a team's usual one moves its
# expected goals: log-multiplier per unit of players.attack_shift. Fitted on
# real line-ups, 2021-22 to 2024-25 (docs/BACKTEST.md).
LINEUP_COEF = 0.3
# Need this much recent league history (decay-weighted games) to know what a
# team's usual XI is.
MIN_PRESENCE_WEIGHT = 3.0


@dataclass
class MatchSpec:
    home: str
    away: str
    kickoff: pd.Timestamp | None = None
    referee: str | None = None
    lineups: dict[str, list[str]] = field(default_factory=dict)
    bench: dict[str, list[str]] = field(default_factory=dict)
    absent: dict[str, list[str]] = field(default_factory=dict)
    odds: dict[str, float] = field(default_factory=dict)
    builders: list[dict] = field(default_factory=list)
    sims: int = 100_000
    card_weights: tuple[int, int] = (1, 2)
    market_weight: float | None = None
    notes: list[str] = field(default_factory=list)
    use_fpl: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "MatchSpec":
        s = d.get("settings", {}) or {}
        ko = d.get("kickoff")
        return cls(
            home=d["home"], away=d["away"],
            kickoff=pd.Timestamp(ko) if ko else None,
            referee=d.get("referee") or None,
            lineups={k: v for k, v in (d.get("lineups") or {}).items() if v},
            bench={k: v for k, v in (d.get("bench") or {}).items() if v},
            absent={k: v for k, v in (d.get("absent") or {}).items() if v},
            odds={str(k): odds_mod.to_decimal(v) for k, v in (d.get("odds") or {}).items()},
            builders=d.get("builders") or [],
            sims=int(s.get("sims", 100_000)),
            card_weights=tuple(s.get("card_weights", (1, 2))),
            market_weight=s.get("market_weight"),
            notes=d.get("notes") or [],
            use_fpl=bool(s.get("use_fpl", True)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MatchSpec":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


@dataclass
class AnalysisResult:
    spec: MatchSpec
    home: str
    away: str
    as_of: pd.Timestamp
    lambdas_model: tuple[float, float]
    lambdas_market: tuple[float, float] | None
    lambdas: tuple[float, float]
    rho: float
    sim: SimResult
    catalogue: pd.DataFrame
    player_table: pd.DataFrame
    squads: dict[str, pd.DataFrame]
    news: dict[str, pd.DataFrame]
    lineup_shift: dict[str, float]
    builders: list[builder.BuilderReport]
    strong_legs: pd.DataFrame
    suggestions: pd.DataFrame
    expected: dict[str, tuple[float, float]]
    referee: str | None
    notes: list[str]
    goals_model: GoalsModel
    stats_model: MatchStatsModel


# ------------------------------------------------------------------ helpers


def resolve_referee(name: str | None, known: list[str]) -> str | None:
    """Match 'Michael Oliver' or 'Oliver' to football-data's 'M Oliver'."""
    if not name:
        return None
    n = names.norm(name)
    for k in known:
        if names.norm(k) == n:
            return k
    parts = n.split()
    surname = parts[-1]
    cands = [k for k in known if names.norm(k).split()[-1] == surname]
    if len(parts) > 1:
        cands = [k for k in cands if names.norm(k)[0] == parts[0][0]] or cands
    return cands[0] if len(cands) == 1 else None


def market_family(spec: str) -> str:
    p = spec.split(":")
    k = p[0]
    if k == "player":
        return f"player:{p[1]}:{p[2] if len(p) > 2 else 'goal'}"
    if k in ("team_goals", "team_corners", "team_cards", "team_shots", "team_sot", "team_fouls"):
        return f"{k}:{p[1]}"
    if k in ("handicap", "corner_handicap", "win_to_nil", "clean_sheet", "score_both_halves"):
        return f"{k}:{p[1]}"
    return k


GROUPS = [
    ("Result", ("result", "dc", "dnb", "handicap", "win_to_nil", "clean_sheet")),
    ("Goals", ("goals", "team_goals", "btts", "cs")),
    ("Halves", ("ht", "htft", "ht_goals", "2h_goals", "score_both_halves")),
    ("Corners", ("corners", "team_corners", "corners_1x2", "corner_handicap")),
    ("Cards", ("cards", "team_cards", "cards_1x2", "red_card", "booking_points")),
    ("Shots", ("shots", "team_shots", "sot", "team_sot")),
    ("Fouls", ("fouls", "team_fouls")),
    ("Players", ("player",)),
]


def group_of(spec: str) -> str:
    k = spec.split(":")[0]
    for g, ks in GROUPS:
        if k in ks:
            return g
    return "Other"


def _lines_around(mean: float, width: int = 3) -> list[float]:
    centre = math.floor(mean) + 0.5
    return [centre + d for d in range(-width, width + 1) if centre + d > 0]


def standard_markets(sim: SimResult, exp: dict, squads: dict[str, pd.DataFrame]) -> list[str]:
    specs = ["result:home", "result:draw", "result:away", "dc:1x", "dc:x2", "dc:12",
             "btts:yes", "btts:no"]
    specs += [f"goals:{d}:{l}" for l in (0.5, 1.5, 2.5, 3.5, 4.5) for d in ("over", "under")]
    for s in ("home", "away"):
        specs += [f"team_goals:{s}:over:{l}" for l in (0.5, 1.5, 2.5)]
        specs += [f"team_goals:{s}:under:{l}" for l in (0.5, 1.5)]
        specs += [f"handicap:{s}:-1.5", f"handicap:{s}:+1.5", f"win_to_nil:{s}",
                  f"clean_sheet:{s}", f"score_both_halves:{s}", f"dnb:{s}"]
    specs += [f"ht:{x}" for x in ("home", "draw", "away")]
    specs += [f"htft:{a}/{b}" for a in ("home", "draw", "away") for b in ("home", "draw", "away")]
    specs += ["ht_goals:over:0.5", "ht_goals:over:1.5", "2h_goals:over:0.5", "2h_goals:over:1.5"]
    hg_max = 5
    specs += [f"cs:{h}-{a}" for h in range(hg_max) for a in range(hg_max)]
    tot_c = sum(exp["corners"])
    specs += [f"corners:{d}:{l}" for l in _lines_around(tot_c) for d in ("over", "under")]
    for s, j in (("home", 0), ("away", 1)):
        specs += [f"team_corners:{s}:over:{l}" for l in _lines_around(exp["corners"][j], 2)]
        specs += [f"team_cards:{s}:over:{l}" for l in (0.5, 1.5, 2.5)]
        specs += [f"team_shots:{s}:over:{l}" for l in _lines_around(
            exp["goals"][j] + exp["ngon"][j] + exp["ngoff"][j], 2)]
        specs += [f"team_sot:{s}:over:{l}" for l in _lines_around(exp["goals"][j] + exp["ngon"][j], 2)]
        specs += [f"team_fouls:{s}:over:{l}" for l in _lines_around(exp["fouls"][j], 1)]
    specs += ["corners_1x2:home", "corners_1x2:draw", "corners_1x2:away",
              "corner_handicap:home:-1.5", "corner_handicap:away:-1.5"]
    yw, rw = sim.card_weights
    tot_cards = yw * sum(exp["yellow"]) + rw * sum(exp["red"])
    specs += [f"cards:{d}:{l}" for l in _lines_around(tot_cards, 2) for d in ("over", "under")]
    specs += ["cards_1x2:home", "cards_1x2:draw", "cards_1x2:away", "red_card:yes"]
    tot_sh = sum(exp["goals"]) + sum(exp["ngon"]) + sum(exp["ngoff"])
    specs += [f"shots:{d}:{l}" for l in _lines_around(tot_sh, 2) for d in ("over", "under")]
    tot_sot = sum(exp["goals"]) + sum(exp["ngon"])
    specs += [f"sot:{d}:{l}" for l in _lines_around(tot_sot, 2) for d in ("over", "under")]
    tot_f = sum(exp["fouls"])
    specs += [f"fouls:over:{l}" for l in _lines_around(tot_f, 2)]
    for side, sq in squads.items():
        for _, r in sq[sq["start"]].iterrows():
            nm = r["player"]
            if r["group"] == "GK":
                specs += [f"player:{nm}:saves:{k}+" for k in (1, 2, 3, 4, 5)]
                continue
            specs += [f"player:{nm}:goal", f"player:{nm}:goals:2+", f"player:{nm}:first_goal",
                      f"player:{nm}:assist", f"player:{nm}:goal_or_assist",
                      f"player:{nm}:shots:1+", f"player:{nm}:shots:2+",
                      f"player:{nm}:shots:3+", f"player:{nm}:sot:1+", f"player:{nm}:sot:2+",
                      f"player:{nm}:card"]
    return specs


def build_catalogue(sim: SimResult, specs: list[str], prices: dict[str, float]) -> pd.DataFrame:
    rows = []
    seen = set()
    for spec in specs + [s for s in prices if s not in specs]:
        if spec in seen:
            continue
        seen.add(spec)
        try:
            o = markets.evaluate(spec, sim)
        except markets.MarketError as exc:
            rows.append({"spec": spec, "label": spec, "error": str(exc)})
            continue
        p, push = o.prob(), o.push_prob()
        price = prices.get(spec)
        row = {
            "spec": spec, "label": markets.describe(spec, sim.home, sim.away),
            "group": group_of(spec), "family": market_family(spec),
            "prob": p, "push": push,
            "fair_odds": (1 - push) / p if p > 0 else np.inf,
            "odds": price,
            "builder_ok": not spec.startswith(("cs:", "htft:", "dnb:")) and "first_goal" not in spec,
        }
        if price:
            row["implied"] = 1 / price
            row["ev"] = p * price + push - 1
        rows.append(row)
    cat = pd.DataFrame(rows)
    for c in ("odds", "implied", "ev", "error"):
        if c not in cat:
            cat[c] = np.nan
    return cat


def player_table(sim: SimResult, squads: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for side, sq in sim.players.items():
        team = sim.home if side == "h" else sim.away
        prof = squads[side]
        for j, (pid, nm) in enumerate(zip(sq.ids, sq.names)):
            played = sq.minutes[:, j] > 0
            if played.mean() < 0.05:
                continue
            pr = prof.loc[pid]
            v = played
            rows.append({
                "team": team, "player": nm, "pos": pr.get("group"),
                "starts": bool(sq.started[j]),
                "p_plays": float(played.mean()),
                "exp_min": float(sq.minutes[:, j].mean()),
                "score": float((sq.goals[v, j] >= 1).mean()),
                "score_2+": float((sq.goals[v, j] >= 2).mean()),
                "first_goal": float(sq.first_scorer[v, j].mean()),
                "assist": float((sq.assists[v, j] >= 1).mean()),
                "goal_or_assist": float(((sq.goals[v, j] + sq.assists[v, j]) >= 1).mean()),
                "shots_1+": float((sq.shots[v, j] >= 1).mean()),
                "shots_2+": float((sq.shots[v, j] >= 2).mean()),
                "shots_3+": float((sq.shots[v, j] >= 3).mean()),
                "sot_1+": float((sq.sot[v, j] >= 1).mean()),
                "sot_2+": float((sq.sot[v, j] >= 2).mean()),
                "booked": float(((sq.yellow[v, j] + sq.red[v, j]) >= 1).mean()),
                "exp_shots": float(sq.shots[v, j].mean()),
                "apps_3y": int(pr.get("apps", 0) or 0),
                "minutes_3y": int(pr.get("minutes", 0) or 0),
                "status": pr.get("status"), "news": pr.get("news"),
            })
    return pd.DataFrame(rows)


def _count_scale(base_kwargs: dict, stat: str, line: float, p_target: float,
                 card_weights=(1, 2)) -> float:
    """Multiplier on a stat's means so the simulated P(total > line) hits p_target."""
    from .models.simulate import simulate_counts, simulate_scores
    rng = np.random.default_rng(99)
    n = 30_000
    hg, ag, _, _ = simulate_scores(base_kwargs["lh"], base_kwargs["la"], base_kwargs["rho"],
                                   base_kwargs["ht_frac"], n, rng)
    sm, fx, refk = base_kwargs["stats_model"], base_kwargs["fixture"], base_kwargs["referee_known"]

    def p_over(scale):
        r = np.random.default_rng(5)
        names_ = ("yellow", "red") if stat == "cards" else (stat,)
        ms = {nm: (scale, scale) for nm in names_}
        st = simulate_counts(sm, fx, hg, ag, r, refk, ms)
        if stat == "cards":
            tot = card_weights[0] * (st["h_yellow"] + st["a_yellow"]) \
                + card_weights[1] * (st["h_red"] + st["a_red"])
        else:
            tot = st[f"h_{stat}"] + st[f"a_{stat}"]
        return float((tot > line).mean())

    lo, hi = 0.5, 2.0
    if not (p_over(lo) <= p_target <= p_over(hi)):
        return 1.0
    for _ in range(18):
        mid = math.sqrt(lo * hi)
        if p_over(mid) < p_target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ------------------------------------------------------------------- main


@dataclass
class Bundle:
    """Everything fitted as of one moment, reusable across matches."""
    as_of: pd.Timestamp
    matches: pd.DataFrame
    goals: GoalsModel
    stats: MatchStatsModel
    profiles: pd.DataFrame
    priors: pd.DataFrame
    pm: pd.DataFrame
    shots: pd.DataFrame

    @classmethod
    def fit(cls, as_of: pd.Timestamp) -> "Bundle":
        m = load.matches()
        tr = features.team_table(m)
        pm = load.player_matches()
        profiles, priors = players.build_profiles(as_of, pm)
        return cls(as_of=as_of, matches=m, goals=GoalsModel().fit(tr, as_of),
                   stats=MatchStatsModel().fit(tr, as_of), profiles=profiles, priors=priors,
                   pm=pm, shots=load.shots())


def analyse(spec: MatchSpec, as_of: pd.Timestamp | None = None, seed: int = 7,
            bundle: Bundle | None = None, catalogue: bool = True) -> AnalysisResult:
    notes = list(spec.notes)
    kickoff = spec.kickoff or pd.Timestamp(dt.datetime.now()).floor("h")
    if as_of is None:
        as_of = bundle.as_of if bundle else min(kickoff - pd.Timedelta(hours=1),
                                                pd.Timestamp(dt.datetime.now()))
    bundle = bundle or Bundle.fit(as_of)
    m = bundle.matches
    known = sorted(set(m["home"]) | set(m["away"]))
    home = names.resolve_team(spec.home, known)
    away = names.resolve_team(spec.away, known)
    season = load.season_of(kickoff)

    gm = bundle.goals
    sm = bundle.stats
    prom = features.promoted_sets(m)
    if season not in prom:
        prev = set(m.loc[m["season"] == season - 1, "home"])
        prom[season] = {t for t in (home, away) if t not in prev}
    ref_known = sm.models_["yellow"].refs_
    referee = resolve_referee(spec.referee, ref_known)
    if spec.referee and referee is None:
        notes.append(f"Referee '{spec.referee}' has no Premier League history in the data; "
                     f"treated as an average referee.")
    fixture = features.fixture_rows(home, away, season, prom, referee=referee)
    lh_m, la_m = (float(x) for x in gm.lambdas(fixture))

    # ---- squads
    pm, profiles, priors, shots = bundle.pm, bundle.profiles, bundle.priors, bundle.shots
    fpl = load.fpl_players() if spec.use_fpl else None
    chance_col = load.fpl_chance_column(home, away) if fpl is not None and not fpl.empty \
        else "chance_of_playing_next_round"
    prof_idx = profiles.set_index("player_id")
    squads, pen_info, news, shift = {}, {}, {}, {}
    for side, team in (("h", home), ("a", away)):
        key = "home" if side == "h" else "away"
        sq, sq_notes, _ = players.build_squad(
            team, as_of, profiles, priors, pm, fpl,
            lineup=spec.lineups.get(key), absent=spec.absent.get(key),
            bench=spec.bench.get(key), chance_col=chance_col)
        notes += sq_notes
        squads[side] = sq
        news[side] = players.team_news(team, fpl, pm, as_of, chance_col)
        order = players.pen_takers(team, as_of, sq, shots)
        lam = lh_m if side == "h" else la_m
        share = gm.pen_rate_ * gm.pen_conv_ / max(lam, 0.3)
        pen_info[side] = (float(np.clip(share, 0.02, 0.25)), order)
        # How this XI compares with the side the team's ratings were earned with.
        pres, wsum = players.presence(team, as_of, pm, gm.params.xi)
        shift[side] = players.attack_shift(list(sq.index[sq["start"]]), prof_idx, pres, priors,
                                           sq["group"].to_dict()) \
            if wsum >= MIN_PRESENCE_WEIGHT else 0.0
    lineup_mult = {s: float(np.clip(np.exp(LINEUP_COEF * v), 0.8, 1.2)) for s, v in shift.items()}
    if any(abs(v - 1) >= 0.01 for v in lineup_mult.values()):
        notes.append(
            f"Line-up adjustment to expected goals (before any bookmaker anchor): {home} "
            f"x{lineup_mult['h']:.2f}, {away} x{lineup_mult['a']:.2f}, for missing or returning "
            f"regulars compared with the sides that earned each team's rating.")
    lh_m *= lineup_mult["h"]
    la_m *= lineup_mult["a"]

    # ---- bookmaker anchor for goals
    o = spec.odds
    lam_mkt = None
    if all(k in o for k in ("result:home", "result:draw", "result:away")):
        ph, pdw, pa = odds_mod.demargin([o["result:home"], o["result:draw"], o["result:away"]])
        po = None
        if "goals:over:2.5" in o and "goals:under:2.5" in o:
            po = float(odds_mod.demargin([o["goals:over:2.5"], o["goals:under:2.5"]])[0])
        lam_mkt = implied_lambdas(ph, pdw, pa, po, gm.rho_)
    w = spec.market_weight if spec.market_weight is not None else MARKET_WEIGHT_GOALS
    if lam_mkt:
        lh = math.exp(w * math.log(lam_mkt[0]) + (1 - w) * math.log(lh_m))
        la = math.exp(w * math.log(lam_mkt[1]) + (1 - w) * math.log(la_m))
    else:
        lh, la = lh_m, la_m
        notes.append("No 1X2 odds given, so result and goals probabilities are the model's "
                     "own (no bookmaker anchor).")

    base_kwargs = dict(lh=lh, la=la, rho=gm.rho_, ht_frac=gm.ht_frac_, stats_model=sm,
                       fixture=fixture, referee_known=referee is not None)
    # ---- bookmaker anchor for corners / cards totals
    mean_scale = {}
    for stat in ("corners", "cards"):
        lines = {}
        for k in o:
            p = k.split(":")
            if p[0] == stat and len(p) == 3:
                lines.setdefault(float(p[2]), {})[p[1]] = o[k]
        for line, d in lines.items():
            if "over" in d and "under" in d:
                p_over = float(odds_mod.demargin([d["over"], d["under"]])[0])
                s = _count_scale(base_kwargs, stat, line, p_over, spec.card_weights)
                s = s ** MARKET_WEIGHT_COUNTS
                if stat == "cards":
                    mean_scale["yellow"] = (s, s)
                    mean_scale["red"] = (s, s)
                else:
                    mean_scale[stat] = (s, s)
                notes.append(f"{stat.title()} means scaled x{s:.2f} towards the bookmaker's "
                             f"{line} line.")
                break

    sim = simulate_match(home=home, away=away, squads=squads, pen_info=pen_info,
                         mean_scale=mean_scale or None, n=spec.sims, seed=seed,
                         card_weights=spec.card_weights, **base_kwargs)

    exp = {
        "goals": (float(sim.hg.mean()), float(sim.ag.mean())),
        **{k: (float(sim.stats[f"h_{k}"].mean()), float(sim.stats[f"a_{k}"].mean()))
           for k in ("corners", "yellow", "red", "fouls", "ngon", "ngoff")},
    }
    specs = standard_markets(sim, exp, squads) if catalogue else []
    cat = build_catalogue(sim, specs, o)
    ptab = player_table(sim, squads) if catalogue else pd.DataFrame()

    reports = []
    for b in spec.builders:
        legs = b["legs"]
        leg_odds = [o.get(l) for l in legs] if all(l in o for l in legs) else None
        reports.append(builder.analyse(sim, legs, b.get("odds"), leg_odds))
    legs = builder.strong_legs(cat) if catalogue else pd.DataFrame()
    sugg = builder.suggest(sim, cat) if catalogue else pd.DataFrame()

    return AnalysisResult(
        spec=spec, home=home, away=away, as_of=as_of,
        lambdas_model=(lh_m, la_m), lambdas_market=lam_mkt, lambdas=(lh, la), rho=gm.rho_,
        sim=sim, catalogue=cat, player_table=ptab, squads=squads, news=news,
        lineup_shift=shift, builders=reports,
        strong_legs=legs, suggestions=sugg, expected=exp, referee=referee, notes=notes,
        goals_model=gm, stats_model=sm,
    )
