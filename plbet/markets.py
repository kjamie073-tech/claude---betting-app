"""Bet market definitions, evaluated on simulated matches.

A market is written as a short spec string. Team sides can be written as
``home``/``away`` or as the team's name. Lines use .5 so there are no pushes,
except Asian handicaps on whole numbers and draw no bet, which can push.

Match result     result:home | result:draw | result:away   (also 1, X, 2)
Double chance    dc:1x | dc:x2 | dc:12
Draw no bet      dnb:home | dnb:away
Total goals      goals:over:2.5 | goals:under:2.5
Team goals       team_goals:home:over:1.5
Both score       btts:yes | btts:no
Correct score    cs:2-1
Handicap         handicap:home:-1.5   (Asian, whole lines push: handicap:home:-1)
Win to nil       win_to_nil:home
Clean sheet      clean_sheet:away
Half-time        ht:home | ht:draw | ht:away
HT/FT            htft:home/draw
Half goals       ht_goals:over:0.5 | 2h_goals:over:1.5
Score both halves  score_both_halves:home
Corners          corners:over:9.5 | team_corners:home:over:4.5 | corners_1x2:home
                 corner_handicap:home:-1.5
Cards            cards:over:3.5 | team_cards:away:over:1.5 | cards_1x2:home
                 red_card:yes | booking_points:over:45.5
Shots            shots:over:24.5 | team_shots:home:over:12.5
On target        sot:over:8.5 | team_sot:away:over:3.5
Fouls            fouls:over:21.5 | team_fouls:home:over:10.5
Player           player:Bukayo Saka:goal          anytime scorer
                 player:Bukayo Saka:goals:2+      2 or more goals
                 player:Bukayo Saka:first_goal
                 player:Bukayo Saka:assist
                 player:Bukayo Saka:goal_or_assist
                 player:Bukayo Saka:shots:2+      (also shots:over:1.5)
                 player:Bukayo Saka:sot:1+
                 player:Bukayo Saka:card          booked (yellow or red)
                 player:David Raya:saves:3+       goalkeeper saves
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .data import names
from .models.simulate import SimResult


@dataclass
class Outcome:
    win: np.ndarray
    push: np.ndarray | None = None
    valid: np.ndarray | None = None      # sims where the bet stands (player played)

    def prob(self) -> float:
        w = self.win if self.valid is None else self.win[self.valid]
        return float(w.mean()) if len(w) else float("nan")

    def push_prob(self) -> float:
        if self.push is None:
            return 0.0
        p = self.push if self.valid is None else self.push[self.valid]
        return float(p.mean())


class MarketError(ValueError):
    pass


def _side(token: str, sim: SimResult) -> str:
    t = token.strip().lower()
    if t in ("home", "h", "1"):
        return "h"
    if t in ("away", "a", "2"):
        return "a"
    canon = names.team(token)
    if canon == sim.home:
        return "h"
    if canon == sim.away:
        return "a"
    raise MarketError(f"'{token}' is neither {sim.home} nor {sim.away}")


def _cmp(values: np.ndarray, direction: str, line: float) -> np.ndarray:
    d = direction.lower()
    if d in ("over", "o"):
        return values > line
    if d in ("under", "u"):
        return values < line
    raise MarketError(f"expected over/under, got '{direction}'")


def _at_least(token: str) -> int:
    t = token.strip().lower()
    m = re.fullmatch(r"(\d+)\+", t)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"over[: ]?(\d+(?:\.5)?)", t)
    if m:
        return int(np.floor(float(m.group(1)))) + 1
    if t.isdigit():
        return int(t)
    raise MarketError(f"expected a threshold like 2+ or over:1.5, got '{token}'")


def team_stat(sim: SimResult, stat: str, side: str) -> np.ndarray:
    if stat == "goals":
        return sim.hg if side == "h" else sim.ag
    if stat == "shots":
        g = sim.hg if side == "h" else sim.ag
        return g + sim.stats[f"{side}_ngon"] + sim.stats[f"{side}_ngoff"]
    if stat == "sot":
        g = sim.hg if side == "h" else sim.ag
        return g + sim.stats[f"{side}_ngon"]
    if stat == "cards":
        yw, rw = sim.card_weights
        return yw * sim.stats[f"{side}_yellow"] + rw * sim.stats[f"{side}_red"]
    if stat == "booking_points":
        return 10 * sim.stats[f"{side}_yellow"] + 25 * sim.stats[f"{side}_red"]
    return sim.stats[f"{side}_{stat}"]


def evaluate(spec: str, sim: SimResult) -> Outcome:
    parts = [p.strip() for p in spec.split(":")]
    kind = parts[0].lower()
    hg, ag = sim.hg, sim.ag

    if kind in ("1", "x", "2"):
        parts = ["result", {"1": "home", "x": "draw", "2": "away"}[kind]]
        kind = "result"
    if kind == "result":
        t = parts[1].lower()
        if t in ("draw", "x"):
            return Outcome(hg == ag)
        s = _side(parts[1], sim)
        return Outcome(hg > ag if s == "h" else ag > hg)
    if kind == "dc":
        t = parts[1].lower()
        m = {"1x": hg >= ag, "x2": ag >= hg, "12": hg != ag}
        if t not in m:
            raise MarketError(f"double chance must be 1x, x2 or 12, got '{t}'")
        return Outcome(m[t])
    if kind == "dnb":
        s = _side(parts[1], sim)
        return Outcome(hg > ag if s == "h" else ag > hg, push=hg == ag)
    if kind == "goals":
        return Outcome(_cmp(hg + ag, parts[1], float(parts[2])))
    if kind == "team_goals":
        s = _side(parts[1], sim)
        return Outcome(_cmp(team_stat(sim, "goals", s), parts[2], float(parts[3])))
    if kind == "btts":
        both = (hg > 0) & (ag > 0)
        return Outcome(both if parts[1].lower() in ("yes", "y") else ~both)
    if kind == "cs":
        h, a = (int(x) for x in parts[1].split("-"))
        return Outcome((hg == h) & (ag == a))
    if kind == "handicap":
        s = _side(parts[1], sim)
        line = float(parts[2])
        margin = (hg - ag) if s == "h" else (ag - hg)
        adj = margin + line
        if float(line).is_integer():
            return Outcome(adj > 0, push=adj == 0)
        if abs(line * 4 - round(line * 4)) < 1e-9 and not float(line * 2).is_integer():
            raise MarketError("quarter-ball handicaps are not supported; split into two bets")
        return Outcome(adj > 0)
    if kind == "win_to_nil":
        s = _side(parts[1], sim)
        return Outcome((hg > ag) & (ag == 0) if s == "h" else (ag > hg) & (hg == 0))
    if kind == "clean_sheet":
        s = _side(parts[1], sim)
        return Outcome(ag == 0 if s == "h" else hg == 0)
    if kind == "ht":
        t = parts[1].lower()
        if t in ("draw", "x"):
            return Outcome(sim.hht == sim.aht)
        s = _side(parts[1], sim)
        return Outcome(sim.hht > sim.aht if s == "h" else sim.aht > sim.hht)
    if kind == "htft":
        a, b = parts[1].split("/")
        return Outcome(evaluate(f"ht:{a}", sim).win & evaluate(f"result:{b}", sim).win)
    if kind == "ht_goals":
        return Outcome(_cmp(sim.hht + sim.aht, parts[1], float(parts[2])))
    if kind == "2h_goals":
        return Outcome(_cmp((hg - sim.hht) + (ag - sim.aht), parts[1], float(parts[2])))
    if kind == "score_both_halves":
        s = _side(parts[1], sim)
        g, h1 = (hg, sim.hht) if s == "h" else (ag, sim.aht)
        return Outcome((h1 > 0) & (g - h1 > 0))
    if kind in ("corners", "cards", "shots", "sot", "fouls", "booking_points"):
        tot = team_stat(sim, kind, "h") + team_stat(sim, kind, "a")
        return Outcome(_cmp(tot, parts[1], float(parts[2])))
    if kind in ("team_corners", "team_cards", "team_shots", "team_sot", "team_fouls"):
        s = _side(parts[1], sim)
        return Outcome(_cmp(team_stat(sim, kind[5:], s), parts[2], float(parts[3])))
    if kind in ("corners_1x2", "cards_1x2"):
        stat = kind.split("_")[0]
        h, a = team_stat(sim, stat, "h"), team_stat(sim, stat, "a")
        t = parts[1].lower()
        if t in ("draw", "x", "tie"):
            return Outcome(h == a)
        s = _side(parts[1], sim)
        return Outcome(h > a if s == "h" else a > h)
    if kind == "corner_handicap":
        s = _side(parts[1], sim)
        h, a = team_stat(sim, "corners", "h"), team_stat(sim, "corners", "a")
        margin = (h - a) if s == "h" else (a - h)
        adj = margin + float(parts[2])
        return Outcome(adj > 0, push=(adj == 0) if float(parts[2]).is_integer() else None)
    if kind == "red_card":
        any_red = (sim.stats["h_red"] + sim.stats["a_red"]) > 0
        return Outcome(any_red if parts[1].lower() in ("yes", "y") else ~any_red)
    if kind == "player":
        return _player(parts[1], parts[2:], sim)
    raise MarketError(f"unknown market '{spec}'")


def find_player(sim: SimResult, name: str) -> tuple[str, int]:
    sides = [(side, j) for side, sq in sim.players.items() for j in range(len(sq.names))]
    pool = [sim.players[side].names[j] for side, j in sides]
    k, _ = names.best_match(name, pool, threshold=0.75)
    if k is None:
        raise MarketError(f"player '{name}' is not in either simulated squad")
    return sides[k]


def _player(name: str, rest: list[str], sim: SimResult) -> Outcome:
    side, j = find_player(sim, name)
    sq = sim.players[side]
    played = sq.minutes[:, j] > 0
    what = rest[0].lower() if rest else "goal"
    arg = rest[1] if len(rest) > 1 else None
    if what in ("goal", "anytime", "score", "to_score"):
        return Outcome(sq.goals[:, j] >= 1, valid=played)
    if what == "goals":
        return Outcome(sq.goals[:, j] >= _at_least(arg or "1+"), valid=played)
    if what in ("first_goal", "first_scorer", "fgs"):
        return Outcome(sq.first_scorer[:, j], valid=played)
    if what in ("assist", "assists"):
        return Outcome(sq.assists[:, j] >= _at_least(arg or "1+"), valid=played)
    if what in ("goal_or_assist", "g/a", "ga"):
        return Outcome((sq.goals[:, j] + sq.assists[:, j]) >= 1, valid=played)
    if what == "shots":
        return Outcome(sq.shots[:, j] >= _at_least(arg or "1+"), valid=played)
    if what in ("sot", "shots_on_target"):
        return Outcome(sq.sot[:, j] >= _at_least(arg or "1+"), valid=played)
    if what in ("card", "booked", "carded"):
        return Outcome((sq.yellow[:, j] + sq.red[:, j]) >= 1, valid=played)
    if what == "red":
        return Outcome(sq.red[:, j] >= 1, valid=played)
    if what == "saves":
        opp = "a" if side == "h" else "h"
        saves = sim.stats[f"{opp}_ngon"]
        return Outcome(saves >= _at_least(arg or "1+"), valid=played)
    raise MarketError(f"unknown player market '{what}'")


def describe(spec: str, home: str, away: str) -> str:
    """Readable label for a spec."""
    p = [x.strip() for x in spec.split(":")]
    k = p[0].lower()

    def team(tok):
        t = tok.lower()
        return home if t in ("home", "h", "1") else away if t in ("away", "a", "2") else tok

    try:
        if k == "result":
            return "Draw" if p[1].lower() in ("draw", "x") else f"{team(p[1])} to win"
        if k == "dc":
            return {"1x": f"{home} or draw", "x2": f"{away} or draw",
                    "12": f"{home} or {away}"}[p[1].lower()]
        if k == "dnb":
            return f"{team(p[1])} draw no bet"
        if k == "goals":
            return f"{p[1].title()} {p[2]} goals"
        if k == "team_goals":
            return f"{team(p[1])} {p[2]} {p[3]} goals"
        if k == "btts":
            return "Both teams to score" + ("" if p[1].lower() in ("yes", "y") else " - No")
        if k == "cs":
            return f"Correct score {p[1]}"
        if k == "handicap":
            return f"{team(p[1])} {float(p[2]):+g} handicap"
        if k == "win_to_nil":
            return f"{team(p[1])} to win to nil"
        if k == "clean_sheet":
            return f"{team(p[1])} clean sheet"
        if k == "ht":
            return "Draw at half-time" if p[1].lower() in ("draw", "x") \
                else f"{team(p[1])} leading at half-time"
        if k == "htft":
            a, b = p[1].split("/")
            return f"HT/FT {team(a) if a.lower() != 'draw' else 'Draw'}/" \
                   f"{team(b) if b.lower() != 'draw' else 'Draw'}"
        if k == "ht_goals":
            return f"First half {p[1]} {p[2]} goals"
        if k == "2h_goals":
            return f"Second half {p[1]} {p[2]} goals"
        if k == "score_both_halves":
            return f"{team(p[1])} to score in both halves"
        if k in ("corners", "cards", "shots", "sot", "fouls", "booking_points"):
            noun = {"sot": "shots on target", "booking_points": "booking points"}.get(k, k)
            return f"{p[1].title()} {p[2]} {noun}"
        if k.startswith("team_"):
            noun = {"sot": "shots on target"}.get(k[5:], k[5:])
            return f"{team(p[1])} {p[2]} {p[3]} {noun}"
        if k in ("corners_1x2", "cards_1x2"):
            noun = k.split("_")[0]
            return f"Most {noun}: {team(p[1]) if p[1].lower() not in ('draw', 'x', 'tie') else 'tie'}"
        if k == "corner_handicap":
            return f"{team(p[1])} {float(p[2]):+g} corner handicap"
        if k == "red_card":
            return "Red card in match" + ("" if p[1].lower() in ("yes", "y") else " - No")
        if k == "player":
            what = p[2].lower() if len(p) > 2 else "goal"
            arg = p[3] if len(p) > 3 else ""
            labels = {
                "goal": "to score", "goals": f"{arg} goals", "first_goal": "first goalscorer",
                "assist": "to assist", "goal_or_assist": "to score or assist",
                "shots": f"{arg} shots", "sot": f"{arg} shots on target",
                "card": "to be booked", "saves": f"{arg} saves",
            }
            return f"{p[1]} {labels.get(what, what)}"
    except (IndexError, KeyError, ValueError):
        pass
    return spec
