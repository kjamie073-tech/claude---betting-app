"""Bet builder pricing: joint probabilities, how legs relate, and value checks."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import markets, odds as odds_mod
from .models.simulate import SimResult

# Minimum edge before a builder is worth backing (docs/STAKING.md).
MIN_EDGE_BUILDER = 0.08

# Thresholds for describing how two legs relate. lift = P(A and B) / (P(A) P(B)).
STRONG_POS, POS, NEG, STRONG_NEG = 1.25, 1.07, 0.93, 0.8
NEAR_CERTAIN = 0.92


def relation(lift: float) -> str:
    if not np.isfinite(lift):
        return "n/a"
    if lift == 0:
        return "cannot both win"
    if lift >= STRONG_POS:
        return "strongly linked (help each other)"
    if lift >= POS:
        return "linked (help each other)"
    if lift <= STRONG_NEG:
        return "work strongly against each other"
    if lift <= NEG:
        return "work against each other"
    return "roughly independent"


@dataclass
class BuilderReport:
    legs: list[str]
    labels: list[str]
    leg_probs: list[float]
    joint: float
    joint_se: float
    independent: float
    lift: float
    fair_odds: float
    pairs: pd.DataFrame
    odds: float | None = None
    ev: float | None = None
    leg_odds: list[float | None] = field(default_factory=list)
    bookmaker_lift: float | None = None
    flags: list[str] = field(default_factory=list)
    valid_share: float = 1.0

    def verdict(self, safety: float = MIN_EDGE_BUILDER) -> str:
        if self.odds is None:
            return (f"Fair odds {self.fair_odds:.2f}. Only worth backing at "
                    f"{self.fair_odds * (1 + safety):.2f} or bigger (the staking plan's "
                    f"{safety:.0%} minimum edge for builders).")
        if self.ev is not None and self.ev >= safety:
            return f"Value: model edge {self.ev:+.1%} at {self.odds:.2f} (fair {self.fair_odds:.2f})."
        if self.ev is not None and self.ev > 0:
            return (f"Marginal: edge {self.ev:+.1%} is below the {safety:.0%} the staking plan "
                    f"asks of a builder; skip unless you can get "
                    f"{self.fair_odds * (1 + safety):.2f}+.")
        return (f"No value: {self.odds:.2f} is below the fair price of {self.fair_odds:.2f} "
                f"(edge {self.ev:+.1%}).")


def analyse(sim: SimResult, legs: list[str], builder_odds=None,
            leg_odds: list | None = None) -> BuilderReport:
    outs = [markets.evaluate(l, sim) for l in legs]
    valid = np.ones(sim.n, dtype=bool)
    for o in outs:
        if o.valid is not None:
            valid &= o.valid
    W = np.column_stack([o.win[valid] for o in outs])
    nv = W.shape[0]
    probs = W.mean(axis=0)
    joint_arr = W.all(axis=1)
    joint = float(joint_arr.mean()) if nv else float("nan")
    se = float(np.sqrt(max(joint * (1 - joint), 1e-12) / max(nv, 1)))
    indep = float(np.prod(probs))
    k = len(legs)
    labels = [markets.describe(l, sim.home, sim.away) for l in legs]
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        pij = float((W[:, i] & W[:, j]).mean())
        lift = pij / (probs[i] * probs[j]) if probs[i] * probs[j] > 0 else np.nan
        rows.append({"leg_a": labels[i], "leg_b": labels[j], "p_both": pij,
                     "lift": lift, "relation": relation(lift)})
    pairs = pd.DataFrame(rows)
    rep = BuilderReport(
        legs=legs, labels=labels, leg_probs=[float(p) for p in probs], joint=joint,
        joint_se=se, independent=indep, lift=joint / indep if indep > 0 else np.nan,
        fair_odds=odds_mod.fair_odds(joint), pairs=pairs,
        valid_share=float(valid.mean()),
    )
    if builder_odds is not None:
        rep.odds = odds_mod.to_decimal(builder_odds)
        rep.ev = joint * rep.odds - 1
    if leg_odds:
        rep.leg_odds = [odds_mod.to_decimal(o) if o else None for o in leg_odds]
        if rep.odds and all(rep.leg_odds):
            product = float(np.prod(rep.leg_odds))
            # >1 means the bookmaker cut the price for correlation (or margin).
            rep.bookmaker_lift = product / rep.odds

    # Flags
    for lab, p in zip(labels, probs):
        if p >= NEAR_CERTAIN:
            rep.flags.append(f"'{lab}' is {p:.0%} likely: it barely moves the price but still "
                             f"carries bookmaker margin. Consider dropping it.")
        if p < 0.08:
            rep.flags.append(f"'{lab}' is only {p:.0%} likely on its own.")
    for _, r in pairs.iterrows():
        if r["lift"] <= NEG:
            rep.flags.append(f"'{r['leg_a']}' and '{r['leg_b']}' {r['relation']} "
                             f"(together {r['lift']:.2f}x as likely as if unrelated).")
        elif r["lift"] >= STRONG_POS:
            rep.flags.append(f"'{r['leg_a']}' and '{r['leg_b']}' are {r['relation']}: "
                             f"{r['lift']:.2f}x as likely together as if unrelated, so check "
                             f"the bookmaker has not cut the price by more than that.")
    if rep.bookmaker_lift is not None and np.isfinite(rep.lift):
        if rep.bookmaker_lift > rep.lift * 1.15:
            rep.flags.append(
                f"The bookmaker has cut the price {rep.bookmaker_lift:.2f}x versus multiplying "
                f"the single prices, more than the {rep.lift:.2f}x link the model sees: "
                f"this combination looks overpriced by the bookmaker.")
        elif rep.bookmaker_lift < rep.lift * 0.9:
            rep.flags.append(
                f"The bookmaker cut the price only {rep.bookmaker_lift:.2f}x for the link "
                f"between legs while the model sees {rep.lift:.2f}x: the combination may be "
                f"underpriced (good for you).")
    if rep.valid_share < 0.999:
        rep.flags.append("Player legs assume the player plays; bookmakers usually void a "
                         "player leg if the player does not take part.")
    return rep


# Markets that bookmakers offer as builder legs (spec prefixes).
BUILDER_KINDS = ("result", "dc", "btts", "goals", "team_goals", "corners", "team_corners",
                 "cards", "team_cards", "team_shots", "team_sot", "player")
PLAYER_LEGS = ("goal", "assist", "goal_or_assist", "shots", "sot", "card")


def _builder_legs(catalogue: pd.DataFrame, exclude_players=()) -> pd.DataFrame:
    c = catalogue[catalogue["builder_ok"] & catalogue["prob"].notna()].copy()
    kind = c["spec"].str.split(":").str[0]
    c = c[kind.isin(BUILDER_KINDS)]
    is_pl = c["spec"].str.startswith("player:")
    who = c["spec"].str.split(":").str[1]
    what = c["spec"].str.split(":").str[2]
    c = c[~is_pl | (what.isin(PLAYER_LEGS) & ~who.isin(set(exclude_players)))]
    # Goalkeepers' shot and card lines are not builder material.
    return c[~c["spec"].str.contains(":saves:")]


def _subject(spec: str) -> str:
    """What a leg is about: a player, or a market kind (one leg each per builder)."""
    p = spec.split(":")
    return f"player:{p[1]}" if p[0] == "player" else p[0] + (
        f":{p[1]}" if p[0].startswith("team_") else "")


def strong_legs(catalogue: pd.DataFrame, n: int = 12, min_edge_single: float = 0.03,
                min_edge_player: float = 0.05, exclude_players=()) -> pd.DataFrame:
    """The legs to build around.

    First, legs where your odds beat the model's fair odds by the staking
    plan's minimum edge (``basis`` "edge"), best first. Then, from the
    markets you did not price, the line in each market that is about 70%
    likely (``basis`` "likely"), spread across markets and players: the kind
    of leg builders are made of, worth looking up in the bookmaker's builder.
    Legs you priced without that edge are left out. ``exclude_players``: players with
    too little Premier League data to trust their chances.
    """
    c = _builder_legs(catalogue, exclude_players)
    priced = c[c["odds"].notna()]
    need = np.where(priced["spec"].str.startswith("player:"), min_edge_player, min_edge_single)
    value = priced[priced["ev"] >= need].sort_values("ev", ascending=False).head(n)
    rest = c[c["odds"].isna()]
    rest = rest[(rest["prob"] >= 0.55) & (rest["prob"] <= 0.88)].copy()
    rest["dist"] = (rest["prob"] - 0.70).abs()
    rest = rest.sort_values("dist").drop_duplicates("family")
    rest["subject"] = rest["spec"].map(_subject)
    taken = set(value["spec"].map(_subject))
    rest = rest[~rest["subject"].isin(taken)]
    rest = rest.sort_values("prob", ascending=False).drop_duplicates("subject")
    rest["kind"] = rest["spec"].str.split(":").str[0]
    rest["rank"] = rest.groupby("kind").cumcount()
    cap = np.where(rest["kind"] == "player", n // 3, 2)
    likely = rest[rest["rank"] < cap].sort_values("prob", ascending=False) \
        .drop(columns=["dist", "subject", "kind", "rank"])
    out = pd.concat([value.assign(basis="edge"), likely.assign(basis="likely")])
    return out.head(n).reset_index(drop=True)


def suggest(sim: SimResult, catalogue: pd.DataFrame, n_legs=(2, 3, 4), top: int = 6,
            min_p: float = 0.18, max_p: float = 0.55, n_candidates: int = 12,
            max_implied: float = 0.9, min_pair_lift: float = NEG,
            exclude_players=()) -> pd.DataFrame:
    """Builders worth pricing up.

    Candidates are the strong legs (strong_legs). A combination uses each
    player and each market at most once, leaves out pairs where one leg
    almost guarantees the other (``max_implied``: the second adds bookmaker
    margin but hardly any odds) and pairs that work against each other, and
    lands at a sensible price (model chance ``min_p``..``max_p``, fair odds
    about 1.8 to 5.5). Combinations with more value legs (legs your odds show
    an edge on) come first; then, if every leg is priced, the edge if the
    bookmaker priced the builder as your single prices multiplied together
    (an upper bound: bookmakers trim the price of linked legs), otherwise how
    much the links between legs raise the chance above the legs' chances
    multiplied together.
    """
    cols = ["legs", "labels", "prob", "fair_odds", "lift", "singles_odds", "edge_if_singles",
            "value_legs"]
    cand = strong_legs(catalogue, n=n_candidates, exclude_players=exclude_players)
    if len(cand) < 2:
        return pd.DataFrame(columns=cols)
    priced = (cand["basis"] == "edge").all()
    is_value = dict(zip(cand["spec"], cand["basis"] == "edge"))
    wins = {r.spec: markets.evaluate(r.spec, sim) for r in cand.itertuples()}
    subj = {r.spec: _subject(r.spec) for r in cand.itertuples()}
    rows = []
    for k in n_legs:
        for combo in itertools.combinations(cand.itertuples(), k):
            if len({subj[c.spec] for c in combo}) < k:
                continue
            valid = np.ones(sim.n, dtype=bool)
            for c in combo:
                if wins[c.spec].valid is not None:
                    valid &= wins[c.spec].valid
            W = np.column_stack([wins[c.spec].win[valid] for c in combo])
            pr = W.mean(axis=0)
            ok = True
            for i, j in itertools.combinations(range(k), 2):
                both = (W[:, i] & W[:, j]).mean()
                lift = both / (pr[i] * pr[j]) if pr[i] * pr[j] else 1.0
                implied = max(both / pr[i] if pr[i] else 0, both / pr[j] if pr[j] else 0)
                if lift < min_pair_lift or implied >= max_implied:
                    ok = False
                    break
            if not ok:
                continue
            p = float(W.all(axis=1).mean())
            if not (min_p <= p <= max_p):
                continue
            indep = float(np.prod(pr))
            singles = float(np.prod([c.odds for c in combo])) if priced else np.nan
            rows.append({
                "legs": [c.spec for c in combo],
                "labels": " + ".join(c.label for c in combo),
                "prob": p, "fair_odds": 1 / p, "lift": p / indep if indep else np.nan,
                "singles_odds": singles,
                "edge_if_singles": p * singles - 1 if priced else np.nan,
                "value_legs": int(sum(is_value[c.spec] for c in combo)),
            })
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values(
        ["value_legs", "edge_if_singles" if priced else "lift"], ascending=False)
    # Avoid near-duplicates that share most legs.
    picked, seen = [], []
    for r in df.itertuples():
        legs = set(r.legs)
        if any(len(legs & t) >= max(2, len(legs) - 1) or legs <= t or t <= legs for t in seen):
            continue
        picked.append(r.Index)
        seen.append(legs)
        if len(picked) >= top:
            break
    return df.loc[picked, cols].reset_index(drop=True)
