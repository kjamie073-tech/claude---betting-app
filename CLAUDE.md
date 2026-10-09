# Playbook for Claude: Premier League bet builder model

This repo is J's Premier League betting model. J picks a match, pastes
bookmaker odds, and wants a thorough analysis, a probability for every
realistic market, the strongest bet builder picks, a check of how builder legs
relate, staking advice, and a record of bets and ROI. Accuracy matters more
than speed: double-check numbers and say plainly when data is missing or
uncertain rather than guessing.

## Setup (every new session)

```bash
pip install -r requirements.txt        # numpy pandas scipy requests openpyxl pyyaml
python -m plbet sync                   # data branch -> ./data
python -m plbet status                 # when each source was last updated
```

The sandbox cannot reach the data sites directly. The `Update data` GitHub
Action downloads them (daily at 05:20 UTC, plus Thu/Fri evenings) into the
`data` branch. If `status` shows data older than the last round of matches,
trigger the workflow (`actions_run_trigger`, workflow `update-data.yml`, ref
`main`, inputs `{"only": ""}`), wait for the run to finish, then `sync` again.

Private files live in the project's shared folder, never in this public repo:

- match files: `/mnt/project-files/betting/matches/`
- reports: `/mnt/project-files/betting/reports/`
- J's tracker: `/mnt/project-files/betting/bet_tracker.xlsx`

## Analysing a match

1. **Find the fixture.** `python -m plbet fixtures --days 10`.
2. **Research team news with WebSearch** (do not skip; the model cannot know
   about injuries announced today):
   - injuries, suspensions, illness, players returning (club press
     conferences, BBC Sport, Sky Sports, the clubs' sites);
   - predicted line-ups (Sports Mole, Fantasy Football Scout, RotoWire,
     Fantasy Football Hub) — or confirmed line-ups if it is within ~75 minutes
     of kick-off;
   - the referee (the Premier League announces appointments early in the
     week);
   - anything else that changes the game: a manager change, a cup game
     midweek, rotation, weather.
   Note each source; mention conflicts between sources in the report notes.
3. **Write the match file.** `python -m plbet template HOME AWAY --out /mnt/project-files/betting/matches`
   then fill in kick-off, referee, the predicted XIs, anyone ruled out who is
   not already in FPL's injury news, and J's odds. Map each pasted price to a
   market spec (full list at the top of `plbet/markets.py`), for example:
   - "Arsenal 1.80" → `result:home: 1.80`
   - "Over 2.5 goals 1.62 / Under 2.5 2.30" → `goals:over:2.5` and `goals:under:2.5`
   - "Saka anytime 5/2" → `player:Bukayo Saka:goal: 5/2`
   - "Saka 2+ shots on target 3.25" → `player:Bukayo Saka:sot:2+: 3.25`
   - "Over 10.5 corners 1.90" → `corners:over:10.5`
   - "Rice to be carded 4.0" → `player:Declan Rice:card: 4.0`
   - a builder → `builders: [{legs: [...], odds: 6.5}]`
   Always include both sides of a market when J gives them (e.g. over and
   under): the model uses the 1X2 and over/under 2.5 prices, and any total
   corners or cards line, to anchor itself to the market.
   Ask which bookmaker it is if card counting matters (most UK books count a
   yellow as 1 and a red as 2: `card_weights: [1, 2]`).
4. **Run it.** `python -m plbet analyse <file> --out /mnt/project-files/betting/reports`
5. **Sanity-check before replying.**
   - Do the projected/confirmed XIs look right? Players who moved clubs in the
     window or have no Premier League history are listed in the report notes;
     put the real XI in the match file if the projection is off.
   - Compare the model-only goal expectations with the bookmaker's. A gap of
     more than ~0.3 goals usually means news the model lacks (or a data
     problem): find out which before trusting either.
   - Player chances for anyone with under ~600 Premier League minutes in the
     last three seasons lean heavily on position averages; say so.
6. **Reply to J.** Lead with the picks, then the reasoning:
   - the 2–4 strongest picks with model chance, fair odds, J's odds and edge;
   - the best builder(s): legs, model chance, fair odds, the minimum odds worth
     taking (fair × 1.05), and how the legs relate (linked, independent, or
     working against each other);
   - verdicts on any builders J asked about, including flags;
   - stakes from the staking plan (`python -m plbet stake ...`, or the
     tracker's suggested stake);
   - what is uncertain or missing.
   Attach the full report (`attached_outputs`) rather than pasting it.
   If nothing clears the minimum edge, say "no bet" plainly. That is a
   correct and common answer.

## Things to be honest about

- Main markets (result, goals) are priced efficiently: the backtest shows the
  model alone is slightly worse than Bet365's prices, which is why it leans on
  the bookmaker's own odds there. Value is more likely in props and in how a
  bookmaker prices links between builder legs.
- Corners/cards anchors use an even model/market split that has not been
  backtested (there is no free historical corners/cards odds data).
- Player fouls, tackles and offsides are not modelled.
- Results over a few dozen bets are mostly luck (docs/STAKING.md section 5).

## Logging bets and results

J's tracker is an Excel file with live formulas. Create it once:
`python -m plbet tracker new /mnt/project-files/betting/bet_tracker.xlsx --bank <starting bank>`

When J says what they backed, add it (and attach the updated file):
`python -m plbet tracker add <path> --date 2026-10-11 --match "Liverpool v Man City" --selection "..." --odds 6.5 --stake 5 --chance 0.18 --type Builder --market Builder --bookmaker Bet365`

After the match, settle it (`tracker settle <path> --row N --result Won --closing 6.0`)
and report ROI with `tracker summary <path>`.

## Code map

- `plbet/data/` — download (`fetch.py`, runs in Actions), loading (`load.py`), names
- `plbet/models/goals.py` — team ratings from xG and goals, Dixon-Coles scores
- `plbet/models/counts.py` — corners, cards, fouls, shots given the score, plus copula
- `plbet/models/players.py` — player shares, minutes, line-up projection
- `plbet/models/simulate.py` — the joint Monte Carlo simulator
- `plbet/markets.py` — market specs evaluated on simulations
- `plbet/builder.py` — builder pricing, leg relationships, suggestions
- `plbet/analysis.py` — puts it all together for one match
- `plbet/report.py`, `plbet/team_stats.py` — the markdown report
- `plbet/staking.py`, `plbet/tracker.py` — staking plan and Excel tracker
- `plbet/backtest.py` — walk-forward backtests (results in docs/BACKTEST.md)
