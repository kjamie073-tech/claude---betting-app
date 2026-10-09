"""Accumulators of bet builders: one builder per match, odds multiplied.

J's usual bet is one builder on each Premier League game of the round, all
stacked into a single accumulator at a fixed stake (Paddy Power style: the
acca price is the builders' prices multiplied together). This module picks a
builder for each match, then prices the whole acca.

Legs from different matches are treated as independent: the acca's chance is
the builders' chances multiplied. Inside each match the builder is priced on
the joint simulation, so the links between its own legs are kept.

Typical use (see CLAUDE.md for the whole workflow):

    python -m plbet acca --gameweek 7 --matches DIR --out DIR
    python -m plbet acca a.yaml b.yaml c.yaml --odds 45.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import builder, odds as odds_mod
from .analysis import LOW_DATA_MINUTES, AnalysisResult

# J's fixed stake per acca (pounds).
DEFAULT_STAKE = 5.0
# Same minimum edge as a single bet builder (docs/STAKING.md).
MIN_EDGE_ACCA = builder.MIN_EDGE_BUILDER
# Suggested builders for an acca: 3 legs (bookmaker promos usually need 3+)
# at a chance that keeps the whole acca from becoming a lottery ticket.
ACCA_LEGS = (3,)
ACCA_MIN_P = 0.30
ACCA_MAX_P = 0.80
# Gap between model-only and bookmaker expected goals that usually means news
# the model lacks (CLAUDE.md, sanity checks).
GOALS_GAP = 0.3


@dataclass
class AccaLeg:
    match: str
    kickoff: pd.Timestamp | None
    legs: list[str]
    labels: list[str]
    prob: float
    se: float
    odds: float | None = None
    source: str = "suggested"
    flags: list[str] = field(default_factory=list)

    @property
    def fair_odds(self) -> float:
        return odds_mod.fair_odds(self.prob)

    @property
    def ev(self) -> float | None:
        return self.prob * self.odds - 1 if self.odds else None


@dataclass
class AccaReport:
    legs: list[AccaLeg]
    prob: float
    se: float
    odds: float | None
    stake: float
    odds_source: str
    notes: list[str] = field(default_factory=list)

    @property
    def fair_odds(self) -> float:
        return odds_mod.fair_odds(self.prob)

    @property
    def min_odds(self) -> float:
        return self.fair_odds * (1 + MIN_EDGE_ACCA)

    @property
    def ev(self) -> float | None:
        return self.prob * self.odds - 1 if self.odds else None

    def verdict(self) -> str:
        if self.odds is None:
            return (f"Fair odds {self.fair_odds:.2f}. Only worth backing at {self.min_odds:.2f} or "
                    f"bigger (the staking plan's {MIN_EDGE_ACCA:.0%} minimum edge for builders).")
        if self.ev >= MIN_EDGE_ACCA:
            return f"Value: model edge {self.ev:+.1%} at {self.odds:.2f} (fair {self.fair_odds:.2f})."
        if self.ev > 0:
            return (f"Marginal: edge {self.ev:+.1%} is below the {MIN_EDGE_ACCA:.0%} minimum; "
                    f"worth it at {self.min_odds:.2f}+.")
        return (f"No value: {self.odds:.2f} is below the fair price of {self.fair_odds:.2f} "
                f"(edge {self.ev:+.1%}).")


def combine(legs: list[AccaLeg], stake: float = DEFAULT_STAKE,
            acca_odds=None) -> AccaReport:
    """Price an acca from one builder per match (matches independent).

    ``acca_odds``: the price the bookmaker shows for the whole acca. Without
    it, the acca price is the builders' prices multiplied together, if every
    builder is priced.
    """
    if not legs:
        raise ValueError("an acca needs at least one match")
    prob = float(np.prod([l.prob for l in legs]))
    # Relative simulation errors add (in quadrature) when chances multiply.
    rel = math.sqrt(sum((l.se / l.prob) ** 2 for l in legs if l.prob > 0))
    if acca_odds is not None:
        o, src = odds_mod.to_decimal(acca_odds), "your acca price"
    elif all(l.odds for l in legs):
        o, src = float(np.prod([l.odds for l in legs])), "builder prices multiplied"
    else:
        o, src = None, "not priced"
    rep = AccaReport(legs=legs, prob=prob, se=prob * rel, odds=o, stake=stake, odds_source=src)
    if acca_odds is not None and all(l.odds for l in legs):
        product = float(np.prod([l.odds for l in legs]))
        if abs(o / product - 1) > 0.02:
            rep.notes.append(f"Your acca price {o:.2f} differs from the builders' prices "
                             f"multiplied ({product:.2f}); the report uses {o:.2f}. Check for a "
                             f"boost, or a builder price that has moved.")
    priced = [l for l in legs if l.odds]
    bad = [l for l in priced if l.ev < 0]
    if bad:
        rep.notes.append(
            "Builders priced below the model's fair odds drag the whole acca down: "
            + "; ".join(f"{l.match} ({l.odds:.2f} v fair {l.fair_odds:.2f}, edge {l.ev:+.0%})"
                        for l in bad)
            + ". Dropping them raises the acca's edge.")
    if priced and len(priced) == len(legs) and bad and len(bad) < len(legs):
        keep = [l for l in legs if l.ev >= 0]
        p_k = float(np.prod([l.prob for l in keep]))
        o_k = float(np.prod([l.odds for l in keep]))
        rep.notes.append(f"Without them: {len(keep)} builders, odds {o_k:.2f}, model chance "
                         f"{p_k:.2%}, edge {p_k * o_k - 1:+.1%}.")
    if prob < 0.01:
        rep.notes.append(f"The acca has about a 1 in {1 / prob:,.0f} chance. Each builder's "
                         f"errors and margin multiply, so the edge on long accas is the least "
                         f"reliable number in the report.")
    if not priced:
        rep.notes.append("No builder prices yet: put each builder in its match file with the "
                         "bookmaker's price (builders: [{legs: [...], odds: X}]) and run again, "
                         "or pass the acca price with --odds.")
    return rep


def _low_data_players(res: AnalysisResult) -> list[str]:
    pt = res.player_table
    if pt.empty:
        return []
    return list(pt.loc[(pt["minutes_3y"] < LOW_DATA_MINUTES) & pt["starts"], "player"])


def pick_builder(res: AnalysisResult, n_legs=ACCA_LEGS, min_p: float = ACCA_MIN_P,
                 max_p: float = ACCA_MAX_P) -> AccaLeg | None:
    """The builder this match puts into the acca.

    A builder from the match file comes first (the best edge among priced
    ones, otherwise the first); without one, the model's top suggestion at an
    acca-friendly chance.
    """
    match = f"{res.home} v {res.away}"
    ko = res.spec.kickoff
    flags = []
    if res.lambdas_market:
        gap = sum(abs(a - b) for a, b in zip(res.lambdas_model, res.lambdas_market))
        if gap > GOALS_GAP:
            flags.append(f"model and bookmaker expected goals differ by {gap:.2f}: check team "
                         f"news before trusting this builder")
    else:
        flags.append("no 1X2 odds in the match file, so goals are the model's alone")
    flags += [n for n in res.notes if "no Premier League history" in n]
    for side in ("h", "a"):
        flags += [n for n in res.workload.get(side, ([], []))[1]
                  if any(k in n for k in ("long-haul", "last played", "days later", "unavailable"))]
    if res.builders:
        priced = [b for b in res.builders if b.odds]
        b = max(priced, key=lambda b: b.ev) if priced else res.builders[0]
        return AccaLeg(match=match, kickoff=ko, legs=list(b.legs), labels=list(b.labels),
                       prob=b.joint, se=b.joint_se, odds=b.odds, source="your builder",
                       flags=flags + b.flags)
    sugg = builder.suggest(res.sim, res.catalogue, n_legs=n_legs, top=1, min_p=min_p,
                           max_p=max_p, exclude_players=_low_data_players(res))
    if sugg.empty:
        return None
    legs = list(sugg["legs"].iat[0])
    rep = builder.analyse(res.sim, legs)
    return AccaLeg(match=match, kickoff=ko, legs=legs, labels=rep.labels, prob=rep.joint,
                   se=rep.joint_se, source="suggested", flags=flags + rep.flags)


def render(rep: AccaReport, skipped: list[str] | None = None,
           out_dir: str | Path | None = None) -> str:
    n = len(rep.legs)
    kos = [l.kickoff for l in rep.legs if l.kickoff is not None]
    first = min(kos) if kos else pd.Timestamp.now()
    lines = [f"# {n}-match bet builder acca — {first:%a %d %b %Y}", ""]
    lines.append(f"_Model run {pd.Timestamp.now():%d %b %Y %H:%M}. One builder per match; "
                 f"matches treated as independent, legs inside a match priced together._\n")
    lines.append("## The acca\n")
    lines.append(f"- **Model chance:** {rep.prob:.3%} (about 1 in {1 / rep.prob:,.0f}; "
                 f"simulation error ±{rep.se:.3%}).")
    lines.append(f"- **Fair odds:** {rep.fair_odds:,.2f}. **Minimum worth taking:** "
                 f"{rep.min_odds:,.2f}.")
    if rep.odds:
        ret = rep.stake * rep.odds
        exp = rep.stake * rep.ev
        lines.append(f"- **Odds:** {rep.odds:,.2f} ({rep.odds_source}); £{rep.stake:.2f} returns "
                     f"£{ret:,.2f} if it wins.")
        sign = "-" if exp < 0 else "+"
        lines.append(f"- **Expected profit at £{rep.stake:.2f}:** {sign}£{abs(exp):.2f} per bet "
                     f"(edge {rep.ev:+.1%}).")
    lines.append(f"- **Verdict:** {rep.verdict()}\n")
    lines.append("## Builders\n")
    rows = []
    for l in rep.legs:
        rows.append({
            "Match": l.match + (f" ({l.kickoff:%a %H:%M})" if l.kickoff is not None else ""),
            "Builder": " + ".join(l.labels),
            "Model chance": f"{l.prob:.1%}",
            "Fair odds": f"{l.fair_odds:.2f}",
            "Min odds": f"{l.fair_odds * (1 + MIN_EDGE_ACCA):.2f}",
            "Your odds": f"{l.odds:.2f}" if l.odds else "–",
            "Edge": f"{l.ev:+.1%}" if l.ev is not None else "–",
            "From": l.source,
        })
    df = pd.DataFrame(rows)
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("|" + "---|" * len(df.columns))
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    lines.append("")
    lines.append("Market specs for each match file (`builders: [{legs: [...], odds: X}]`):\n")
    for l in rep.legs:
        lines.append(f"- {l.match}: `{l.legs}`")
    lines.append("")
    flagged = [l for l in rep.legs if l.flags]
    if flagged:
        lines.append("## Flags\n")
        for l in flagged:
            for f in l.flags:
                lines.append(f"- **{l.match}:** {f}")
        lines.append("")
    lines.append("## Notes\n")
    for nt in rep.notes:
        lines.append(f"- {nt}")
    for s in skipped or []:
        lines.append(f"- Left out: {s}")
    lines.append("- Each builder's chance assumes its player legs' players play; bookmakers "
                 "usually void a player leg (not the acca) if the player does not take part.")
    lines.append("- Main markets are priced efficiently (docs/BACKTEST.md): value, if any, "
                 "comes from props and how the bookmaker prices links between legs.")
    md = "\n".join(lines) + "\n"
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{first:%Y-%m-%d}-acca.md").write_text(md)
    return md


def selection_text(rep: AccaReport) -> str:
    """One line for the tracker's 'Selection / legs' column."""
    return " | ".join(f"{l.match}: " + " + ".join(l.labels) for l in rep.legs)
