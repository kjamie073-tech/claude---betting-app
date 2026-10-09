"""Markdown match report from an AnalysisResult."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import builder, team_stats
from .analysis import AnalysisResult
from .data import load


def _pct(p) -> str:
    return "" if p is None or (isinstance(p, float) and np.isnan(p)) else f"{p:.0%}" if p >= 0.1 \
        else f"{p:.1%}"


def _odds(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:.2f}"


def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    if df is None or df.empty:
        return "_none_\n"
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for v in r.to_numpy():
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def market_table(cat: pd.DataFrame, group: str) -> pd.DataFrame:
    d = cat[cat["group"] == group]
    out = pd.DataFrame({
        "Market": d["label"],
        "Chance": d["prob"].map(_pct),
        "Fair odds": d["fair_odds"].map(_odds),
    })
    if d["odds"].notna().any():
        out["Your odds"] = d["odds"].map(_odds)
        out["Edge"] = d["ev"].map(lambda e: "" if pd.isna(e) else f"{e:+.0%}")
    return out


def render(r: AnalysisResult, out_dir: str | Path | None = None) -> str:
    sim = r.sim
    home, away = r.home, r.away
    ko = r.spec.kickoff
    title = f"# {home} v {away}"
    if ko is not None:
        title += f" — {ko:%a %d %b %Y, %H:%M}"
    st = load.status()
    fresh = st.get("updated_at", "unknown")
    lines = [title, ""]
    lines.append(f"_Model run {pd.Timestamp.now():%d %b %Y %H:%M}, using data downloaded "
                 f"{fresh[:16].replace('T', ' ')} UTC. Probabilities from {sim.n:,} simulated "
                 f"matches._\n")

    # ------------------------------------------------------------- summary
    cat = r.catalogue
    get = lambda s: float(cat.loc[cat["spec"] == s, "prob"].iat[0]) if (cat["spec"] == s).any() else np.nan
    lh, la = r.lambdas
    lines.append("## Summary\n")
    if r.lambdas_market:
        lines.append(f"- **Expected goals:** {home} {lh:.2f}, {away} {la:.2f} (model blended with "
                     f"your bookmaker's prices; model alone {r.lambdas_model[0]:.2f} – "
                     f"{r.lambdas_model[1]:.2f}, bookmaker {r.lambdas_market[0]:.2f} – "
                     f"{r.lambdas_market[1]:.2f}).")
    else:
        lines.append(f"- **Expected goals:** {home} {lh:.2f}, {away} {la:.2f} (model only: no "
                     f"bookmaker 1X2 odds given).")
    lines.append(f"- **Result:** {home} {_pct(get('result:home'))}, draw {_pct(get('result:draw'))}, "
                 f"{away} {_pct(get('result:away'))}.")
    lines.append(f"- **Goals:** over 2.5 {_pct(get('goals:over:2.5'))}, both teams score "
                 f"{_pct(get('btts:yes'))}.")
    e = r.expected
    yw, rw = sim.card_weights
    lines.append(f"- **Corners:** {e['corners'][0]:.1f} – {e['corners'][1]:.1f} "
                 f"(total {sum(e['corners']):.1f}).")
    cards_h = yw * e["yellow"][0] + rw * e["red"][0]
    cards_a = yw * e["yellow"][1] + rw * e["red"][1]
    lines.append(f"- **Cards** (yellow {yw}, red {rw}): {cards_h:.1f} – {cards_a:.1f} "
                 f"(total {cards_h + cards_a:.1f}); referee: {r.referee or 'not known'}.")
    sh = [e["goals"][i] + e["ngon"][i] + e["ngoff"][i] for i in (0, 1)]
    sot = [e["goals"][i] + e["ngon"][i] for i in (0, 1)]
    lines.append(f"- **Shots:** {sh[0]:.1f} – {sh[1]:.1f}; on target {sot[0]:.1f} – {sot[1]:.1f}.")
    cs = cat[cat["spec"].str.startswith("cs:")].sort_values("prob", ascending=False).head(5)
    lines.append("- **Most likely scores:** " + ", ".join(
        f"{s.split(':')[1]} ({p:.0%})" for s, p in zip(cs["spec"], cs["prob"])) + ".\n")

    # ------------------------------------------------------------- team news
    lines.append("## Line-ups and team news\n")
    for side, team in (("h", home), ("a", away)):
        sq = r.squads[side]
        given = bool(r.spec.lineups.get("home" if side == "h" else "away"))
        lines.append(f"**{team}** — {'line-up as given' if given else 'projected XI'}: "
                     + ", ".join(sq.loc[sq["start"], "player"]) + ".\n")
        out, doubt = _news_lines(r.news.get(side))
        if out:
            lines.append(f"- Out: {out}.")
        if doubt:
            lines.append(f"- Doubtful: {doubt}.")
        ab = r.spec.absent.get("home" if side == "h" else "away")
        if ab:
            lines.append(f"- Ruled out (from the match file): {', '.join(ab)}.")
        if not (out or doubt or ab):
            lines.append("- No injury or suspension news in FPL.")
        lines.append("")
    lines.append("_Injury news from the Fantasy Premier League site, with how many of the "
                 "team's last 10 league games each player started._\n")

    # ------------------------------------------------------------- workload
    lines.append("## Games in all competitions\n")
    for side, team in (("h", home), ("a", away)):
        games, wnotes = r.workload.get(side, ([], []))
        lines.append(f"**{team}** — last games: " + ("; ".join(games) if games else
                                                    "none in the last 6 weeks") + ".\n")
        for n in wnotes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("_Cup, European and international games from ESPN. The model's ratings use "
                 "league games only; use this to set `minutes:` or the XI in the match file "
                 "when a player is tired or likely to be rested._\n")

    # ------------------------------------------------------------- form
    as_of = r.as_of
    lines.append("## Form (last 6 league games)\n")
    for team in (home, away):
        lines.append(f"**{team}**\n")
        lines.append(_md_table(team_stats.form(team, as_of)))
    table = team_stats.league_table(as_of)
    if home in table.index or away in table.index:
        lines.append("## League table\n")
        t = table.loc[[x for x in (home, away) if x in table.index],
                      ["Pos", "P", "Pts", "GF", "GA", "GD", "xG", "xGA", "xGD"]]
        lines.append(_md_table(t.assign(xG=t["xG"].round(1), xGA=t["xGA"].round(1)).rename_axis(
            "Team"), index=True))
    lines.append("## Averages per match\n")
    comp = team_stats.comparison(home, away, as_of)
    lines.append(_md_table(comp.map(lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else v)
                           .rename_axis("Per match"), index=True))
    lines.append("## Head-to-head (league, since 2019)\n")
    lines.append(_md_table(team_stats.head_to_head(home, away, as_of)))
    lines.append("_Head-to-head samples are small and old line-ups differ; the model gives them no "
                 "extra weight._\n")
    rp = team_stats.referee_profile(r.referee, as_of)
    lines.append("## Referee\n")
    lg = rp["league"]
    if rp.get("ref"):
        ref = rp["ref"]
        lines.append(f"{r.referee}: {ref['matches']} league games in 3 seasons, "
                     f"{ref['yellows']:.2f} yellows and {ref['reds']:.2f} reds per game "
                     f"(league {lg['yellows']:.2f} / {lg['reds']:.2f}), {ref['fouls']:.1f} fouls "
                     f"(league {lg['fouls']:.1f}); over 3.5 cards in {ref['over_3.5_cards_%']:.0f}% "
                     f"of the games they refereed.\n")
    else:
        lines.append(f"Not known or no recent Premier League games. League average "
                     f"{lg['yellows']:.2f} yellows per game; the model averages over referees, "
                     f"which widens the card ranges.\n")

    # ------------------------------------------------------------- markets
    lines.append("## Market probabilities\n")
    for g in ("Result", "Goals", "Halves", "Corners", "Cards", "Shots", "Fouls"):
        t = market_table(cat, g)
        if g == "Goals":
            t = t[~cat.loc[cat["group"] == g, "spec"].str.startswith("cs:").to_numpy()]
        lines.append(f"### {g}\n")
        lines.append(_md_table(t))

    # ------------------------------------------------------------- players
    lines.append("## Players\n")
    lines.append("_Chances assume the player is in the line-up shown; bench players are "
                 "conditional on coming on. Bookmakers usually void a player bet if the player "
                 "does not play._\n")
    pt = r.player_table.copy()
    if not pt.empty:
        pt = pt.sort_values(["team", "starts", "score"], ascending=[True, False, False])
        show = pd.DataFrame({
            "Team": pt["team"], "Player": pt["player"], "Pos": pt["pos"],
            "Mins": pt["exp_min"].round(0).astype(int),
            "Score": pt["score"].map(_pct), "Assist": pt["assist"].map(_pct),
            "G or A": pt["goal_or_assist"].map(_pct), "1+ shot": pt["shots_1+"].map(_pct),
            "2+ shots": pt["shots_2+"].map(_pct), "1+ SoT": pt["sot_1+"].map(_pct),
            "2+ SoT": pt["sot_2+"].map(_pct), "Booked": pt["booked"].map(_pct),
            "Mins (3y)": pt["minutes_3y"],
        })
        lines.append(_md_table(show))

    # ------------------------------------------------------------- value
    priced = cat[cat["odds"].notna()].sort_values("ev", ascending=False)
    if not priced.empty:
        lines.append("## Your odds against the model\n")
        lines.append(_md_table(pd.DataFrame({
            "Market": priced["label"], "Odds": priced["odds"].map(_odds),
            "Bookmaker %": priced["implied"].map(_pct), "Model %": priced["prob"].map(_pct),
            "Fair odds": priced["fair_odds"].map(_odds),
            "Edge": priced["ev"].map(lambda v: f"{v:+.0%}"),
        })))
    if r.builders:
        lines.append("## Your bet builders\n")
        for b in r.builders:
            lines.append(f"**{' + '.join(b.labels)}**" + (f" @ {b.odds:.2f}" if b.odds else ""))
            lines.append("")
            lines.append(f"- Model chance {b.joint:.1%} (±{1.96 * b.joint_se:.1%} simulation "
                         f"error), fair odds {b.fair_odds:.2f}.")
            lines.append(f"- If the legs were unrelated: {b.independent:.1%}. Links between legs "
                         f"change that by x{b.lift:.2f}.")
            lines.append(f"- Verdict: {b.verdict()}")
            for f in b.flags:
                lines.append(f"- ⚠️ {f}")
            if not b.pairs.empty:
                lines.append("")
                lines.append(_md_table(pd.DataFrame({
                    "Leg A": b.pairs["leg_a"], "Leg B": b.pairs["leg_b"],
                    "Both": b.pairs["p_both"].map(_pct), "Link": b.pairs["lift"].map(lambda v: f"x{v:.2f}"),
                    "Relationship": b.pairs["relation"]})))
            lines.append("")
    sl = r.strong_legs
    if sl is not None and not sl.empty:
        lines.append("## Strongest legs\n")
        n_val = int((sl["basis"] == "edge").sum())
        if n_val:
            lines.append(f"_The first {n_val} are legs where your odds beat the model's fair odds "
                         f"by at least the staking plan's minimum edge (3%, or 5% for player "
                         f"bets)._\n")
        if n_val < len(sl):
            lines.append("_" + ("The rest are" if n_val else "These are") +
                         " the likeliest legs (about 70% lines) in markets you did not price, "
                         "not necessarily value: a leg is only worth having in a builder if the "
                         "bookmaker's own price for it is above the fair odds. Legs you priced "
                         "are left out unless your odds clear the minimum edge._\n")
        tab = pd.DataFrame({"Leg": sl["label"], "Model %": sl["prob"].map(_pct),
                            "Fair odds": sl["fair_odds"].map(_odds)})
        if n_val:
            tab["Your odds"] = sl["odds"].map(_odds)
            tab["Edge"] = sl["ev"].map(lambda v: "" if pd.isna(v) else f"{v:+.0%}")
        lines.append(_md_table(tab))
    if not r.suggestions.empty:
        sg = r.suggestions
        lines.append("## Builders worth pricing up\n")
        lines.append(f"_Built from the legs above, at fair odds of roughly 2 to 5. Back one only "
                     f"at the \"back at\" price or bigger: the fair odds plus the staking plan's "
                     f"{builder.MIN_EDGE_BUILDER:.0%} minimum edge for builders. \"Link\" is how "
                     f"much likelier the legs are together than their chances multiplied: above "
                     f"x1 they help each other, which bookmakers often price in only partly._\n")
        tbl = {"Builder": sg["labels"], "Chance": sg["prob"].map(_pct),
               "Fair odds": sg["fair_odds"].map(_odds),
               "Back at": (sg["fair_odds"] * (1 + builder.MIN_EDGE_BUILDER)).map(_odds),
               "Link": sg["lift"].map(lambda v: f"x{v:.2f}")}
        if "value_legs" in sg and sg["value_legs"].fillna(0).gt(0).any():
            tbl["Legs with an edge"] = sg["value_legs"].fillna(0).astype(int)
        if sg["singles_odds"].notna().all():
            tbl["Your singles multiplied"] = sg["singles_odds"].map(_odds)
        lines.append(_md_table(pd.DataFrame(tbl)))

    lines.append("## Notes and data gaps\n")
    for n in r.notes:
        lines.append(f"- {n}")
    lines.append("- Player fouls, tackles and offsides are not modelled (no reliable free data).")
    lines.append(f"- Cards counted as yellow = {yw}, red = {rw}. Check your bookmaker's rules; "
                 "some count a red as 1, and cards to managers or bench players usually do not count.")
    md = "\n".join(lines) + "\n"
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{(ko or pd.Timestamp.now()):%Y-%m-%d}-{_slug(home)}-v-{_slug(away)}"
        (out / f"{stem}.md").write_text(md)
        cat.to_csv(out / f"{stem}-markets.csv", index=False)
        r.player_table.to_csv(out / f"{stem}-players.csv", index=False)
    return md


def _news_lines(news: pd.DataFrame | None) -> tuple[str, str]:
    """('Out' list, 'Doubtful' list) from players.team_news."""
    if news is None or news.empty:
        return "", ""
    out, doubt = [], []
    for _, n in news.iterrows():
        text = (n["news"] or "").strip().rstrip(".")
        starts = int(n["starts_last10"] or 0)
        bits = [text] if text else []
        if starts:
            bits.append(f"started {starts} of last 10")
        item = n["player"] + (f" ({'; '.join(bits)})" if bits else "")
        ch = n["chance"]
        if n["status"] == "d" or (pd.notna(ch) and 0 < ch < 100):
            doubt.append(item)
        else:
            out.append(item)
    return ", ".join(out), ", ".join(doubt)


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-").replace("--", "-")
