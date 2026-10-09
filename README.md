# Premier League bet builder model

A model for analysing Premier League matches and building bet builders. Pick
a match and paste your bookmaker's odds; it produces a full match report:

- **Everything that drives the match:** form, home and away records, goals,
  xG, shots, corners, cards and fouls, injuries and suspensions, projected or
  confirmed line-ups, head-to-head and the referee's record.
- **A chance and fair price for every realistic market:** result, goals, both
  teams to score, halves, corners, cards, shots, fouls, and player markets
  (scorer, 2+ goals, first scorer, assists, shots, shots on target, cards).
- **Your odds checked against the model**, with the edge on each price.
- **Bet builders:** the real chance of every combination, how its legs are
  linked (a team winning and its striker scoring help each other), flags for
  weak legs and combinations the bookmaker has overpriced, and suggestions
  worth pricing up.
- **Stakes** from a bankroll plan, and an **Excel tracker** that shows your
  real ROI, profit against expectation and closing-line value over time.

The day-to-day way to use it is through Claude in this project: say which
match, paste the odds you can see, and Claude does the team-news research,
runs the model and replies with the picks and the full report. CLAUDE.md is
the playbook it follows.

## How good is it?

The short version (details in [docs/BACKTEST.md](docs/BACKTEST.md)):

- On match result and goals the bookmakers are very hard to beat. The model
  on its own is close to, but slightly worse than, Bet365's prices, so when
  you give it the 1X2 odds it leans on them.
- Corners, cards, shots and player markets were checked for calibration
  (when the model says 30%, does it happen about 30% of the time?) on
  seasons it was not tuned on.
- Bet builder chances from the joint simulation were checked against what
  actually happened and against simply multiplying the legs' chances.

No model makes betting a reliable way to make money. Bookmakers' margins on
player markets and builders are large, and most bettors lose. The staking
plan ([docs/STAKING.md](docs/STAKING.md)) is built to keep stakes small and
only bet where the model sees a real edge; "no bet" is often the right
answer.

## Running it yourself

Python 3.11+.

```bash
pip install -r requirements.txt
python -m plbet sync                       # download the data branch into ./data
python -m plbet fixtures                   # upcoming matches
python -m plbet template Liverpool "Man City" --out matches
# edit the match file: referee, line-ups, absentees, your odds, builders
python -m plbet analyse matches/2026-10-11-liverpool-v-man-city.yaml --out reports
```

Other commands:

```bash
python -m plbet quick Arsenal Leeds        # model-only report, no match file
python -m plbet builder MATCH.yaml --legs "result:home" "player:Bukayo Saka:goal" --odds 4.5
python -m plbet stake --prob 0.30 --odds 4.0 --kind builder --bank 500
python -m plbet ratings                    # current team ratings
python -m plbet tracker new bets.xlsx --bank 500
python -m plbet tracker add bets.xlsx --match "Liverpool v Man City" --selection "Over 2.5" --odds 1.9 --stake 5 --chance 0.58 --type Single --market Goals
python -m plbet tracker settle bets.xlsx --row 2 --result Won --closing 1.8
python -m plbet tracker summary bets.xlsx
python -m pytest                           # tests
```

Market names for match files and builders are listed at the top of
[plbet/markets.py](plbet/markets.py).

## Documentation

- [docs/METHOD.md](docs/METHOD.md): how the model works
- [docs/BACKTEST.md](docs/BACKTEST.md): how it performed on past seasons
- [docs/STAKING.md](docs/STAKING.md): bankroll and staking plan
- [docs/DATA.md](docs/DATA.md): data sources and the daily download
- [CLAUDE.md](CLAUDE.md): the playbook Claude follows for each match

## Repository layout

```
plbet/data/        downloading (runs in GitHub Actions), loading, name matching
plbet/models/      goals, corners/cards/fouls/shots, players, simulation
plbet/markets.py   market definitions evaluated on simulated matches
plbet/builder.py   bet builder pricing, leg relationships, suggestions
plbet/analysis.py  one match end to end
plbet/report.py    the markdown report
plbet/staking.py   staking plan
plbet/tracker.py   Excel bet tracker
plbet/backtest.py  walk-forward backtests
tracker/           blank tracker template
```

## Gambling responsibly

Only bet money you can afford to lose, keep a separate bank, and never chase
losses. If betting stops being fun, free and confidential help is available
from [GamCare](https://www.gamcare.org.uk) (0808 8020 133), and
[GAMSTOP](https://www.gamstop.co.uk) lets you block yourself from UK
gambling sites.
