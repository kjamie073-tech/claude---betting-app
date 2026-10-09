"""Command line: python -m plbet <command>.

  sync                         download the latest data branch into ./data
  status                       when each data source was last updated
  fixtures [--days 10]         upcoming Premier League fixtures
  ratings                      current team ratings
  template HOME AWAY           write a match file to fill in
  analyse MATCH.yaml           full report (markdown) for one match
  quick HOME AWAY              report with no odds or line-ups
  builder MATCH.yaml --legs .. price one bet builder
  stake --prob P --odds O      staking-plan stake for one bet
  tracker new|add|settle|summary
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

TEMPLATE = """\
# Match file for python -m plbet analyse. Delete anything you do not know.
home: {home}
away: {away}
kickoff: {kickoff}        # local UK time
referee:                  # e.g. Michael Oliver (announced early in match week)

# Confirmed or predicted starting XIs (11 names each). Leave empty to let the
# model project the XI from recent selections and FPL injury news.
lineups:
  home: []
  away: []
# Optional: named substitutes. Without this, recent bench usage is used.
bench:
  home: []
  away: []
# Players ruled out (injury, suspension, illness) not yet in the FPL news.
absent:
  home: []
  away: []

# Bookmaker odds you can see (decimal 2.5, fractional 6/4, or evs). The 1X2
# and over/under 2.5 prices are used to anchor the model; the rest are
# compared with the model. See plbet/markets.py for every market name.
odds:
  # result:home: 2.10
  # result:draw: 3.60
  # result:away: 3.40
  # goals:over:2.5: 1.80
  # goals:under:2.5: 2.00
  # btts:yes: 1.70
  # corners:over:10.5: 1.90
  # corners:under:10.5: 1.85
  # player:Bukayo Saka:goal: 2.75

# Bet builders to check: legs use the same market names.
builders: []
#  - legs: ["result:home", "player:Bukayo Saka:goal", "team_corners:home:over:5.5"]
#    odds: 6.00

settings:
  sims: 100000
  card_weights: [1, 2]     # how your bookmaker counts a yellow and a red card
"""


def _match_path(home: str, away: str, kickoff: pd.Timestamp | None) -> str:
    from .report import _slug
    d = kickoff.strftime("%Y-%m-%d") if kickoff is not None else dt.date.today().isoformat()
    return f"{d}-{_slug(home)}-v-{_slug(away)}.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plbet", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    sub.add_parser("status")
    f = sub.add_parser("fixtures")
    f.add_argument("--days", type=int, default=10)
    sub.add_parser("ratings")
    t = sub.add_parser("template")
    t.add_argument("home")
    t.add_argument("away")
    t.add_argument("--out", default="matches")
    a = sub.add_parser("analyse")
    a.add_argument("match")
    a.add_argument("--out", default="reports")
    a.add_argument("--sims", type=int)
    q = sub.add_parser("quick")
    q.add_argument("home")
    q.add_argument("away")
    q.add_argument("--kickoff")
    q.add_argument("--referee")
    q.add_argument("--out", default="reports")
    b = sub.add_parser("builder")
    b.add_argument("match")
    b.add_argument("--legs", nargs="+", required=True)
    b.add_argument("--odds")
    s = sub.add_parser("stake")
    s.add_argument("--prob", type=float, required=True)
    s.add_argument("--odds", required=True)
    s.add_argument("--kind", choices=["single", "player", "builder"], default="single")
    s.add_argument("--bank", type=float, default=500)
    tk = sub.add_parser("tracker")
    tk.add_argument("action", choices=["new", "add", "settle", "summary"])
    tk.add_argument("path")
    tk.add_argument("--bank", type=float, default=500)
    tk.add_argument("--date")
    tk.add_argument("--match")
    tk.add_argument("--selection")
    tk.add_argument("--odds", type=float)
    tk.add_argument("--stake", type=float)
    tk.add_argument("--chance", type=float)
    tk.add_argument("--type", default="Single")
    tk.add_argument("--market", default="Other")
    tk.add_argument("--bookmaker", default="")
    tk.add_argument("--row", type=int)
    tk.add_argument("--result")
    tk.add_argument("--closing", type=float)
    args = p.parse_args(argv)

    if args.cmd == "sync":
        from .data import load
        print(load.sync_data())
        print(json.dumps(load.status(), indent=2))
        return 0
    if args.cmd == "status":
        from .data import load
        print(json.dumps(load.status(), indent=2))
        return 0
    if args.cmd == "fixtures":
        from .data import load
        fx = load.fpl_fixtures()
        now = pd.Timestamp.now()
        up = fx[(fx["kickoff"] >= now - pd.Timedelta(hours=3))
                & (fx["kickoff"] <= now + pd.Timedelta(days=args.days))]
        print(up[["event", "kickoff", "home", "away"]].to_string(index=False))
        return 0
    if args.cmd == "ratings":
        from .data import load
        from .models import features
        from .models.goals import GoalsModel
        tr = features.team_table()
        gm = GoalsModel().fit(tr, pd.Timestamp.now())
        cur = set(load.fpl_teams().values()) or None
        rt = gm.ratings()
        if cur:
            rt = rt[rt.index.isin(cur)]
        print(rt.round(3).to_string())
        print(f"home advantage {gm.model_.coef()['is_home']:.3f}, rho {gm.rho_:.3f}")
        return 0
    if args.cmd == "template":
        from .data import load, names
        fx = load.fpl_fixtures()
        home, away = names.team(args.home), names.team(args.away)
        row = fx[(fx["home"] == home) & (fx["away"] == away) & (fx["finished"] == False)]  # noqa: E712
        ko = row["kickoff"].iloc[0] if len(row) else None
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / _match_path(home, away, ko)
        path.write_text(TEMPLATE.format(home=home, away=away,
                                        kickoff=ko.strftime("%Y-%m-%d %H:%M") if ko is not None else ""))
        print(path)
        return 0
    if args.cmd in ("analyse", "quick"):
        from .analysis import MatchSpec, analyse
        from .report import render
        if args.cmd == "analyse":
            spec = MatchSpec.from_yaml(args.match)
            if args.sims:
                spec.sims = args.sims
        else:
            spec = MatchSpec.from_dict({"home": args.home, "away": args.away,
                                        "kickoff": args.kickoff, "referee": args.referee})
            if spec.kickoff is None:
                from .data import load, names
                fx = load.fpl_fixtures()
                h, a = names.team(args.home), names.team(args.away)
                row = fx[(fx["home"] == h) & (fx["away"] == a) & (fx["finished"] == False)]  # noqa: E712
                if len(row):
                    spec.kickoff = row["kickoff"].iloc[0]
        res = analyse(spec)
        md = render(res, args.out)
        print(md)
        return 0
    if args.cmd == "builder":
        from . import builder as bmod
        from .analysis import MatchSpec, analyse
        spec = MatchSpec.from_yaml(args.match)
        spec.builders = [{"legs": args.legs, "odds": args.odds}]
        res = analyse(spec)
        rep = res.builders[0]
        print(" + ".join(rep.labels))
        print(f"model chance {rep.joint:.2%} (independent {rep.independent:.2%}, link x{rep.lift:.2f}); "
              f"fair odds {rep.fair_odds:.2f}")
        print(rep.verdict())
        for fl in rep.flags:
            print("-", fl)
        print(rep.pairs.round(3).to_string(index=False))
        return 0
    if args.cmd == "stake":
        from .staking import StakingPlan
        print(json.dumps(StakingPlan(args.bank).stake(args.prob, args.odds, args.kind), indent=2))
        return 0
    if args.cmd == "tracker":
        from . import tracker
        if args.action == "new":
            print(tracker.create(args.path, args.bank))
        elif args.action == "add":
            row = tracker.add_bet(args.path, date=args.date or dt.date.today(), match=args.match,
                                  selection=args.selection, odds=args.odds, stake=args.stake,
                                  model_chance=args.chance, bet_type=args.type,
                                  market=args.market, bookmaker=args.bookmaker)
            print(f"added row {row}")
        elif args.action == "settle":
            tracker.settle(args.path, args.row, args.result, args.closing)
            print("settled")
        else:
            print(json.dumps(tracker.summary(args.path), indent=2, default=str))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
