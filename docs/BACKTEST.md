# How the model did on past seasons

Every number here comes from **walk-forward testing**: before each test
match the model was refitted using only matches played before it, exactly as
it would have been used at the time. Settings were tuned on seasons up to
2024-25; **2025-26 and the start of 2026-27 were held out** and played no
part in tuning, so they are the fair test.

How to read the numbers:

- **Log loss** measures how much probability the model put on what actually
  happened; lower is better. Small differences matter: bookmakers' prices
  differ from each other by a few thousandths.
- **Skill** compares log loss with a plain base rate learnt from earlier
  seasons (for example "over 2.5 goals happens 55% of the time"). +5% means
  the model's chances carry real information about that match; 0% means it
  does no better than the base rate.
- **Calibration** tables group bets by the chance the model gave them. A
  well-calibrated model's 30% bets win about 30% of the time. Groups with
  fewer than 30 bets are left out; "±" is the luck you would expect in a
  group of that size.

To reproduce: `python -m plbet backtest goals --seasons 2021 2022 2023 2024 2025 2026`
and `python -m plbet backtest full` (about 20 minutes on four cores).

## 1. Summary

- **Result and goals: the bookmakers are better, but only a little.** Over
  2021-22 to 2024-25 the model alone was worse than Bet365's prices in three
  seasons out of four; on the held-out seasons it was roughly level with
  Bet365's early prices and a little worse than the closing prices. Betting
  the model's own result and goals edges against Bet365 lost 5% over four
  seasons. So when you give the 1X2 and over/under prices, the model leans
  75% on them.
- **Corners, cards, shots and fouls are well calibrated**, and the model
  knows something about each match beyond the base rate (1–11% skill),
  most for team shots and fouls and least for total corners.
- **Player markets are well calibrated** after three fixes the backtests
  found (section 4), with 1–7% skill over position averages. The exception:
  on the held-out seasons the leading shooters' shots-on-target chances ran
  high.
- **Bet builder chances from the joint simulation are right**, and much
  better than multiplying the legs: "over 2.5 goals and both teams to score"
  happened 45% of the time; the joint model said 45%, multiplying the legs
  said 32%.
- **Line-ups matter, modestly**: a regular defender or goalkeeper missing
  from the other side adds about 4.5% to a team's expected goals.

None of this shows the model can beat bookmakers' prop and builder prices:
there is no free history of those prices to test against. The model gives
honest chances; whether a price is worth taking depends on the margin the
bookmaker has built in, which is why the staking plan demands a minimum edge.

## 2. Match result and goals against the bookmakers

Model-only forecasts (team ratings, no line-up information, no odds) against
the bookmakers' prices with their margin removed. "Bet365" is the price
football-data.co.uk collects before the weekend; "closing" is the price at
kick-off.

| Season | Matches | 1X2 model | 1X2 Bet365 | 1X2 Bet365 closing | 1X2 average closing | O/U 2.5 model | O/U 2.5 Bet365 | O/U 2.5 Bet365 closing | O/U 2.5 average closing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021-22 | 380 | 0.9567 | 0.9357 | 0.9347 | 0.9358 | 0.6807 | 0.6831 | 0.6908 | 0.6873 |
| 2022-23 | 380 | 0.9791 | 0.9664 | 0.9621 | 0.9620 | 0.6744 | 0.6671 | 0.6667 | 0.6679 |
| 2023-24 | 380 | 0.9317 | 0.9061 | 0.8965 | 0.8984 | 0.6520 | 0.6469 | 0.6442 | 0.6442 |
| 2024-25 | 380 | 0.9646 | 0.9711 | 0.9677 | 0.9671 | 0.6800 | 0.6775 | 0.6780 | 0.6787 |
| 2025-26 | 380 | 1.0248 | 1.0199 | 1.0142 | 1.0128 | 0.6889 | 0.6894 | 0.6844 | 0.6840 |
| 2026-27 | 50 | 1.0417 | 1.0568 | 1.0577 | 1.0596 | 0.6791 | 0.6807 | 0.6843 | 0.6841 |

Pinnacle's closing prices, the sharpest in the market, stop partway through
2025-26 in the data; up to then they scored much like the market average at
kick-off.

**Blending.** With your 1X2 and over/under prices, the model blends its
expected goals with the bookmaker's. Tested on matches with Bet365's early
prices:

| Weight on Bet365 | Result 2021-25 | Over 2.5 2021-25 | BTTS 2021-25 | Result held out | Over 2.5 held out | BTTS held out |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 (model only) | 0.9580 | 0.6718 | 0.6842 | 1.0267 | 0.6878 | 0.6833 |
| 0.50 | 0.9481 | 0.6693 | 0.6828 | 1.0204 | 0.6866 | 0.6830 |
| 0.75 (used) | 0.9459 | 0.6688 | 0.6826 | 1.0205 | 0.6872 | 0.6833 |
| 0.85 | 0.9455 | 0.6687 | 0.6827 | 1.0212 | 0.6877 | 0.6835 |
| 0.90 | 0.9454 | 0.6688 | 0.6827 | 1.0217 | 0.6880 | 0.6836 |
| 1.00 (Bet365 only) | 0.9455 | 0.6688 | 0.6828 | 1.0229 | 0.6887 | 0.6839 |

On 2021-22 to 2024-25 the best weight on the bookmaker was 0.85–0.95 for
the result and 0.7–0.85 for goals and both teams to score; the held-out
seasons preferred less (0.5–0.6), because the model did relatively better
there. 0.75 sits between them and is never more than 0.001 from the best log
loss on any of these markets.

**Betting the model's own edges does not work.** Backing every result or
over/under 2.5 price at Bet365 where the model alone saw an edge of 3% or
more: 

| Minimum edge | Bets 2021-25 | Return 2021-25 | Bets held out | Return held out |
|---|---:|---:|---:|---:|
| 3% | 1944 | -5.3% | 600 | +1.3% |
| 5% | 1653 | -5.1% | 520 | +1.7% |
| 10% | 1134 | -6.7% | 353 | -7.6% |

That is what no edge looks like: a loss close to the bookmaker's margin over
four seasons and a result within luck of zero on the held-out seasons.

## 3. Every team market: corners, cards, shots and fouls

The full simulation (team ratings plus the real starting XIs and the real
referee, no odds) priced every market in 1,570 matches from 2022-23 to
2026-27. The result and goals rows repeat section 2's question without the
bookmaker; the rest have no free price history, so calibration is the test.
Held-out seasons (430 matches):

| Market | Model average | Happened | Skill vs base rate |
|---|---:|---:|---:|
| Home win | 43% | 42% | +7.6% |
| Draw | 25% | 28% | +1.3% |
| Away win | 33% | 30% | +7.4% |
| Over 2.5 goals | 53% | 55% | +0.6% |
| Both teams to score | 54% | 55% | +0.8% |
| Home team ahead at half-time | 34% | 37% | +6.0% |
| Over 0.5 first-half goals | 71% | 70% | +1.6% |
| Over 1.5 first-half goals | 35% | 34% | +0.7% |
| Over 8.5 corners | 66% | 64% | +1.2% |
| Over 9.5 corners | 54% | 55% | +0.4% |
| Over 10.5 corners | 42% | 43% | +1.0% |
| Over 11.5 corners | 31% | 30% | +1.5% |
| Home team over 4.5 corners | 57% | 59% | +4.9% |
| Away team over 3.5 corners | 63% | 61% | +0.6% |
| Home team more corners | 52% | 53% | +2.8% |
| Over 2.5 cards | 73% | 73% | +1.8% |
| Over 3.5 cards | 55% | 56% | +1.7% |
| Over 4.5 cards | 38% | 38% | +1.0% |
| Over 5.5 cards | 24% | 23% | +0.6% |
| Home team over 1.5 cards | 53% | 54% | +2.7% |
| Away team over 1.5 cards | 63% | 67% | +2.3% |
| Over 23.5 shots | 61% | 62% | +1.5% |
| Over 26.5 shots | 39% | 39% | +2.6% |
| Over 7.5 shots on target | 64% | 60% | +2.2% |
| Over 9.5 shots on target | 37% | 34% | +3.0% |
| Home team over 12.5 shots | 56% | 60% | +10.8% |
| Away team over 10.5 shots | 55% | 55% | +6.1% |
| Home team over 4.5 on target | 49% | 46% | +5.6% |
| Away team over 3.5 on target | 55% | 53% | +5.5% |
| Over 20.5 fouls | 57% | 58% | +2.1% |
| Over 23.5 fouls | 35% | 34% | +6.7% |

Calibration, result, goals and half-time (held out, 2,580 bets):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 10–20% | 122 | 16.7% | 12.3% | 3.4% |
| 20–30% | 731 | 25.1% | 26.4% | 1.6% |
| 30–40% | 372 | 35.0% | 34.9% | 2.5% |
| 40–50% | 485 | 45.5% | 44.7% | 2.3% |
| 50–60% | 710 | 54.7% | 58.6% | 1.9% |
| 60–70% | 135 | 62.6% | 57.8% | 4.2% |

Calibration, corners (held out, 2,580 bets):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 20–30% | 199 | 26.6% | 24.1% | 3.1% |
| 30–40% | 438 | 34.9% | 35.8% | 2.3% |
| 40–50% | 472 | 45.0% | 47.2% | 2.3% |
| 50–60% | 549 | 55.0% | 57.0% | 2.1% |
| 60–70% | 558 | 65.1% | 62.7% | 2.0% |
| 70–80% | 317 | 73.8% | 70.3% | 2.5% |
| 80–90% | 36 | 82.0% | 66.7% | 6.4% |

Calibration, cards (held out, 2,580 bets):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 10–20% | 115 | 16.9% | 14.8% | 3.5% |
| 20–30% | 299 | 24.9% | 26.4% | 2.5% |
| 30–40% | 321 | 35.3% | 35.5% | 2.7% |
| 40–50% | 369 | 45.3% | 42.3% | 2.6% |
| 50–60% | 588 | 55.0% | 60.2% | 2.1% |
| 60–70% | 507 | 65.0% | 64.3% | 2.1% |
| 70–80% | 330 | 74.2% | 74.8% | 2.4% |
| 80–90% | 51 | 82.5% | 82.4% | 5.3% |

Calibration, shots and shots on target (held out, 3,440 bets):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 10–20% | 40 | 16.0% | 15.0% | 5.8% |
| 20–30% | 207 | 26.5% | 27.5% | 3.1% |
| 30–40% | 660 | 35.2% | 35.8% | 1.9% |
| 40–50% | 650 | 44.6% | 45.2% | 1.9% |
| 50–60% | 670 | 55.5% | 53.3% | 1.9% |
| 60–70% | 816 | 64.7% | 63.6% | 1.7% |
| 70–80% | 301 | 73.8% | 71.8% | 2.5% |
| 80–90% | 83 | 83.7% | 79.5% | 4.1% |

Calibration, fouls (held out, 860 bets):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 10–20% | 55 | 16.0% | 14.5% | 4.9% |
| 20–30% | 122 | 25.2% | 27.0% | 3.9% |
| 30–40% | 149 | 34.7% | 36.2% | 3.9% |
| 40–50% | 166 | 44.8% | 45.8% | 3.9% |
| 50–60% | 163 | 54.8% | 52.1% | 3.9% |
| 60–70% | 111 | 65.0% | 65.8% | 4.5% |
| 70–80% | 75 | 74.3% | 73.3% | 5.0% |

Average per team per match, model against what happened:

| Per team | 2022-23 | 2023-24 | 2024-25 | 2025-26 | 2026-27 |
|---|---:|---:|---:|---:|---:|
| Goals | 1.42 / 1.43 | 1.53 / 1.64 | 1.58 / 1.47 | 1.40 / 1.38 | 1.42 / 1.41 |
| Shots | 12.59 / 12.63 | 13.16 / 13.79 | 13.35 / 12.96 | 12.59 / 12.50 | 12.86 / 13.92 |
| Shots on target | 4.41 / 4.40 | 4.62 / 4.93 | 4.76 / 4.55 | 4.35 / 4.19 | 4.33 / 4.37 |
| Corners | 5.09 / 5.05 | 5.24 / 5.42 | 5.30 / 5.15 | 5.04 / 5.00 | 5.01 / 4.69 |
| Yellow cards | 1.74 / 1.79 | 1.98 / 2.09 | 2.07 / 2.02 | 1.90 / 1.87 | 1.91 / 1.88 |
| Red cards | 0.06 / 0.04 | 0.06 / 0.07 | 0.06 / 0.07 | 0.06 / 0.05 | 0.06 / 0.06 |
| Fouls | 10.48 / 10.76 | 10.85 / 11.05 | 11.05 / 11.03 | 10.83 / 10.82 | 10.89 / 11.41 |

(model / actual; 2026-27 is the first 50 matches.)

Notes:

- **Corners** were the hardest market. Before the totals calibration
  (docs/METHOD.md, section 3) the model had no skill on total corners at all;
  now it has a little. On the held-out seasons its most confident corner
  calls (70%+) came in a few points less often than it said, so treat very
  confident corner legs with care.
- **Referees new to the Premier League** (26 team-matches here) gave about
  10% more yellow cards and fouls than the model expected for an average
  referee. The sample is small; the model does not adjust for it.

## 4. Player markets

Starters only (the backtest knew the real XIs, as you would after team news),
with outcomes from Understat. "Skill" is against the average for the
player's position.

| Market | Player-matches | Model average | Happened | Skill vs position average |
|---|---:|---:|---:|---:|
| Anytime scorer | 9451 | 9.6% | 9.3% | +3.0% |
| Assist | 9451 | 7.1% | 6.8% | +2.0% |
| Score or assist | 9451 | 15.8% | 15.2% | +2.7% |
| 1+ shots | 9451 | 53.1% | 52.6% | +5.2% |
| 2+ shots | 9451 | 25.7% | 26.2% | +6.5% |
| 3+ shots | 9451 | 11.6% | 11.8% | +7.4% |
| 1+ shots on target | 9451 | 25.6% | 24.7% | +3.5% |
| 2+ shots on target | 9451 | 6.2% | 5.4% | +5.1% |
| Booked | 9451 | 15.1% | 15.0% | +1.4% |

Calibration, anytime scorer (held out):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 0–5% | 3842 | 2.7% | 2.3% | 0.3% |
| 5–10% | 2498 | 7.0% | 7.1% | 0.5% |
| 10–20% | 1765 | 14.3% | 14.3% | 0.8% |
| 20–30% | 941 | 24.3% | 24.5% | 1.4% |
| 30–40% | 314 | 33.8% | 29.3% | 2.7% |
| 40–50% | 82 | 43.6% | 43.9% | 5.5% |

Calibration, 2+ shots (held out):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 0–5% | 1452 | 1.5% | 1.8% | 0.3% |
| 5–10% | 1334 | 7.5% | 8.6% | 0.7% |
| 10–20% | 2128 | 14.4% | 15.2% | 0.8% |
| 20–30% | 1185 | 24.7% | 26.0% | 1.3% |
| 30–40% | 859 | 34.9% | 34.3% | 1.6% |
| 40–50% | 804 | 44.8% | 44.4% | 1.8% |
| 50–60% | 788 | 54.9% | 56.6% | 1.8% |
| 60–70% | 581 | 64.5% | 63.9% | 2.0% |
| 70–80% | 279 | 73.9% | 71.3% | 2.6% |
| 80–90% | 41 | 83.5% | 78.0% | 5.8% |

Calibration, 1+ shots on target (held out):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 0–5% | 925 | 0.4% | 0.5% | 0.2% |
| 5–10% | 929 | 7.9% | 8.1% | 0.9% |
| 10–20% | 2659 | 15.0% | 15.1% | 0.7% |
| 20–30% | 1715 | 24.4% | 22.4% | 1.0% |
| 30–40% | 972 | 34.8% | 34.4% | 1.5% |
| 40–50% | 984 | 45.2% | 44.1% | 1.6% |
| 50–60% | 806 | 54.5% | 51.7% | 1.8% |
| 60–70% | 366 | 64.0% | 59.6% | 2.5% |
| 70–80% | 89 | 73.5% | 73.0% | 4.7% |

Calibration, 2+ shots on target (held out):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 0–5% | 6140 | 1.5% | 1.5% | 0.2% |
| 5–10% | 1113 | 7.2% | 7.0% | 0.8% |
| 10–20% | 1436 | 14.6% | 12.7% | 0.9% |
| 20–30% | 572 | 23.9% | 18.9% | 1.8% |
| 30–40% | 156 | 33.9% | 25.6% | 3.8% |

Calibration, booked (held out):

| Model said | Bets | Average model chance | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| 0–5% | 118 | 4.3% | 4.2% | 1.9% |
| 5–10% | 1279 | 8.0% | 7.7% | 0.8% |
| 10–20% | 6604 | 14.9% | 14.5% | 0.4% |
| 20–30% | 1433 | 22.7% | 24.4% | 1.1% |

Three problems the first backtests found, all fixed:

- **Substitutes' minutes.** A bug in how often bench players come on
  (a pandas quirk that turned the rate into nonsense) gave substitutes about
  four times their real minutes. Starters' chances of scoring, shooting and
  being booked came out 10–17% too low.
- **What substitutes produce.** Substitutes play about 6% of the minutes but
  take about 13% of the goals and shots and 14% of the cards: they come on
  when games are open and play stoppage time that the minute counts leave
  out. Bench players' rates are now scaled up to match (docs/METHOD.md,
  section 4), and a player can only be booked once.
- **Card-prone players.** A player's booking rate is very noisy from one
  season to the next. Given how many cards their team actually got, the most
  card-prone 1% of starters were booked about 25% less often than their
  share said, and the least card-prone about 10% more often. Their rates
  are now pulled harder towards the average for their position (a prior
  worth 3,600 minutes instead of 1,200), which brings every group within
  about 10%. Before the change, the model's booking calls of 30% or more
  (436 player-matches) came in 25% of the time, and 18% on the held-out
  seasons; after it, its 30%+ calls (140) came in 32% of the time against
  32% predicted. The price is a little sharpness: skill on the booked market
  fell from +1.6% to +1.4%.

## 5. Bet builders: joint simulation against multiplying the legs

Each builder below was priced every match: legs multiplied (as if
independent), the joint simulation, and how often it actually won. "Its
likeliest scorer" is the team's starter with the highest scoring chance.

| Builder (every match, 2022-23 to 2026-27) | Legs multiplied | Joint model | Happened | ± (1 s.e.) |
|---|---:|---:|---:|---:|
| Over 2.5 goals + both teams score | 31.7% | 45.0% | 45.4% | 1.3% |
| Home win + its likeliest scorer | 15.9% | 22.1% | 22.2% | 1.0% |
| Away win + its likeliest scorer | 10.4% | 15.9% | 15.4% | 0.9% |
| Home win + that player 1+ on target + over 1.5 goals | 23.3% | 27.5% | 28.5% | 1.1% |
| Away win + that player 1+ on target + over 1.5 goals | 15.7% | 19.0% | 17.8% | 1.0% |
| Home win + home team over 4.5 corners | 28.8% | 27.6% | 27.6% | 1.1% |
| Away win + over 9.5 corners | 18.6% | 18.5% | 18.2% | 1.0% |
| Both teams score + over 3.5 cards | 31.2% | 32.1% | 35.0% | 1.2% |
| Over 2.5 goals + over 9.5 corners + over 3.5 cards | 18.2% | 17.7% | 19.1% | 1.0% |

Linked legs (a team winning and its main striker scoring, goals and both
teams scoring) are priced far better jointly. Where legs are close to
independent (a win and the corner count), the two methods agree. The one
builder the joint model priced clearly short was both teams scoring with over
3.5 cards (32% against 35%): open games bring a few more cards than the model
links to them, so that pairing is slightly better than the model says. The
joint model's log loss over all of these builders was 0.5106 against 0.5193 for multiplying the legs (0.4945 against 0.5025 on the held-out seasons).

## 6. Line-ups and injuries

How much do changes to a team's usual XI move its attack? For 3,828
team-matches from 2021-22 to 2026-27 with the real starting XIs, each team's
non-penalty xG (a steadier measure than goals) was regressed on the model's
expected goals plus:

- the **line-up shift**: the attacking contribution of the XI minus that of
  the players the team usually fields (docs/METHOD.md, section 1);
- the opponent's **missing defenders**: regular goalkeepers and defenders
  not in its XI, weighted by how regularly they start.

Each number is the effect on the log of expected goals (so 0.045 is about
+4.6%), with one standard error. 836 of the team-matches were in the held-out
seasons.

| Effect | 2021-22 to 2024-25 | Held-out seasons | All seasons | Model uses |
|---|---:|---:|---:|---:|
| Line-up shift | 0.25 ± 0.10 | 0.30 ± 0.19 | 0.27 ± 0.09 | 0.27 |
| Each regular defender or goalkeeper the opponent is missing | 0.055 ± 0.014 | 0.031 ± 0.022 | 0.049 ± 0.011 | 0.045 |

- Most line-up changes are small: 90% of team-matches had a shift between
  −0.17 and +0.15, which moves expected goals by about 4% either way. The
  whole adjustment is capped at ±20%.
- Teams are missing about 0.9 regular defenders or goalkeepers on average,
  so the model measures from there: an opponent missing two first-choice
  defenders adds about 5% to a team's expected goals.
- Both effects made the held-out forecasts of xG better, though only a
  little. With real goals as the outcome (far noisier), the line-up shift
  still helped; the defenders effect added nothing measurable on the
  held-out seasons, and was weaker there for xG too, so the model uses a
  value a little below the 2021-25 estimate.
- A missing first-choice goalkeeper looked worth extra on its own (+0.14 ±
  0.05 on goals in 2021-25) but made the held-out forecasts worse, so the
  model treats goalkeepers like other defenders.

## 6b. Fixture congestion and international duty

Tested on 2021-22 to 2026-27 with every Champions, Europa and Conference
League, FA Cup and EFL Cup match and the international qualifiers and
Nations League games from ESPN (6,400 matches), linked to Premier League
line-ups from Understat. "Regulars" started at least 7 of their team's
previous 10 league games.

**Team goals after short rest**, against the goals model and against Bet365's
closing prices (goals scored / expected; ±3% is one standard error):

| Before the league game | Team-matches | Scored v model | Conceded v model | Scored v market | Conceded v market |
|---|---:|---:|---:|---:|---:|
| 5+ days' rest | 2,399 | 1.00 | 0.99 | 1.00 | 0.98 |
| ≤4 days after a league game | 807 | 1.02 | 1.01 | 1.02 | 1.01 |
| ≤4 days after a cup or European game | 692 | 1.00 | 1.03 | 0.98 | 1.04 |

No effect beyond noise, and none on rest advantage either: the market
already prices congestion. **No adjustment is made.**

**Regular starters after midweek club games** (minutes in a cup or
European game in the 4 days before):

| Midweek minutes | Player-matches | Started | Minutes when starting | Played 90 |
|---|---:|---:|---:|---:|
| none | 26,691 | 77% | 86 | 79% |
| 1–45 | 835 | 83% | 86 | 75% |
| 46–75 | 516 | 76% | 83 | 65% |
| 76+ | 2,213 | 82% | 86 | 81% |

Regulars who played the full midweek game start the weekend game *more*
often, not less: managers keep their key players in. Only those taken off
after 46–75 minutes midweek are subbed earlier at the weekend (65% play 90
against 79%), a small group.

**International duty** (qualifiers, Nations League and friendlies in the 12
days before; tournaments excluded, since players at the Africa Cup of
Nations or Asian Cup are simply absent and FPL lists them so):

| Duty | Player-matches | Started | Minutes when starting | Played 90 |
|---|---:|---:|---:|---:|
| none | 28,550 | 77% | 86 | 79% |
| Europe | 1,296 | 84% | 86 | 81% |
| outside Europe (long trip back) | 409 | 77% | 85 | 75% |

Long trips home cost about a minute and a few percent of full 90s, inside
the noise; within 3 days of the trip, 74% started (42 cases). **No automatic
adjustment**: the workload notes in each report remain, and specific news
(a player flagged as tired or carrying a knock) goes into `minutes:` by hand.

## 7. What this means for betting

- On result and goals, trust the bookmaker's price more than the model's;
  the model's job there is to price everything else consistently with it.
- The model's chances for corners, cards, shots, fouls, players and builders
  are honest estimates: on seasons it was not tuned on, its 30% calls
  happened about 30% of the time. That is the foundation for finding value,
  not proof of it.
- Its weakest spots on the held-out seasons were its most confident corner
  calls (80%+) and the leading shooters: players it gave a 20–40% chance of
  2+ shots on target managed it about a fifth less often, and its 30–40%
  scorer calls came in at 29%. Both were fine on 2022-23 to 2024-25, but
  treat an "edge" there with suspicion.
- Bookmakers build much bigger margins into player props and builders than
  into the match result. An edge only exists where a price is wrong by more
  than that margin, which is why the staking plan requires 5% (player
  singles) and 8% (builders) of edge, and why "no bet" is often the answer.
- Judge the model by closing-line value and by actual against expected
  profit over hundreds of bets (docs/STAKING.md), not by a few weeks.
