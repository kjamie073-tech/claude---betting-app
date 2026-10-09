# Data

Everything the model uses is free and public. A GitHub Action downloads it
and commits it to this repository's `data` branch, so every analysis works
from the same numbers and nothing has to be fetched while you wait.

## Sources

| Source | What it gives | Used for |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (`E0`, Premier League, from 2019-20) | Results, half-time scores, shots, shots on target, corners, fouls, yellow and red cards, referee, and pre-match and closing odds from several bookmakers (Bet365 and the market average; Pinnacle until partway through 2025-26) | Team ratings, corners/cards/fouls/shots models, referee effects, and the backtests against bookmaker prices |
| [Understat](https://understat.com/league/EPL) | Team xG and non-penalty xG per match; for every player in every match: minutes, position, goals, shots, xG, assists, xA, key passes, cards; every shot with its xG and outcome | Team ratings (xG is a better guide to future goals than goals), player shares of team output, penalty takers, line-ups used in the backtests |
| [Fantasy Premier League](https://fantasy.premierleague.com) API | Current squads, injury and suspension news with a chance of playing, set-piece takers, fixtures and kick-off times | Team news, projected line-ups, who has left a club, which gameweek a match is in |
| ESPN's public site API (from 2021-22) | Champions, Europa and Conference League, FA Cup, EFL Cup, Community Shield, Club World Cup and international matches (Nations League, qualifiers, friendlies, tournaments): line-ups, each player's minutes (worked out from substitutions and red cards), goals, assists, shots, shots on target, cards, fouls; team scores and box scores; cup and European fixtures for the next ten days | Rest days, rotation before and after midweek games, international duty and long trips home; the "Games in all competitions" section of each report |

What is **not** available for free, and so is not used:

- confirmed line-ups before kick-off (put them in the match file once they
  are announced, about 75 minutes before kick-off);
- historical odds for corners, cards, shots and player markets, so those
  models are checked for calibration but not against bookmakers;
- player-level fouls, tackles and offsides.

## Refresh schedule

The workflow `.github/workflows/update-data.yml` runs:

- every morning at 05:20 UTC (after the night's results are published);
- Thursday and Friday at 16:20 UTC, after the managers' pre-match press
  conferences, to pick up injury news.

It can also be run by hand from the repository's **Actions** tab ("Update
data" → "Run workflow"). Inputs let you download older seasons
(`first_season`) or re-download everything (`refresh_all`).

Each run only downloads what has changed: finished seasons are kept, the
current season's files are refreshed, and Understat match pages are fetched
only for matches not already stored.

## Layout of the `data` branch

```
status.json                       when each source last updated, and any errors
football-data/E0_2627.csv         one file per season (E1 = Championship, kept for reference)
football-data/fixtures.csv
understat/league_EPL_2026.json    season fixtures, results and team xG
understat/player_matches_2026.csv one row per player per match
understat/shots_2026.csv          one row per shot
fpl/bootstrap-static.json         squads, availability news, gameweeks
fpl/fixtures.json
fpl/player_history_2026.csv       FPL's per-match history for this season
espn/<competition>/2026-09.jsonl.gz one compact match per line, per competition and month
espn/<competition>/days_2026.json  match days already stored
espn/upcoming.json                cup and European fixtures in the next ten days
```

`python -m plbet sync` checks the branch out into `./data` (ignored by git);
`python -m plbet status` shows how fresh it is.

## Name matching

The sources name teams and players differently ("Nott'm Forest",
"Nottingham Forest"; "Mathis Cherki" in Understat, "Rayan Cherki" in FPL).
`plbet/data/names.py` maps team names to one canonical form and matches
player names by tokens, accents removed, falling back to a unique surname
within the team. Players the model cannot match are listed in the report's
notes rather than silently dropped. ESPN players are linked to Understat
players by name within the same club (cup and European games); players seen
only with their country are linked only on an exact, unique name.

## Terms of use

football-data.co.uk asks that its data is used for personal, non-commercial
purposes. Understat, ESPN and the Fantasy Premier League site are public websites
without an official API for third parties: the workflow downloads politely
(one request at a time, with pauses) and only what is new. Keep this project
for personal use.
