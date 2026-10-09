# How the model works

The model answers one question for a chosen match: **how likely is every
outcome a bookmaker will let you bet on, including combinations of them?** It
does that by simulating the match 100,000 times, with every simulated match
carrying a full set of numbers (score at half-time and full-time, corners,
cards, fouls, shots for both teams, and goals, assists, shots and cards for
every player), and then counting. Because each leg of a bet builder is read
off the same simulated matches, the links between legs (a team winning and
its striker scoring) come out automatically.

## 1. Goals: team ratings

Each team has an attacking and a defensive rating, fitted with a Poisson
regression to every Premier League match since 2019-20:

    log(expected goals for team i against team j)
        = league average + attack(i) − defence(j) + home advantage
          + promoted-team effect

- **What is being fitted.** Not only actual goals: the target is
  0.3 × goals + 0.7 × (non-penalty xG + the league's average penalty xG).
  xG is a steadier measure of how well a team plays than goals, which are
  noisy. The 30/70 split was chosen by walk-forward testing (section 8).
- **Recent matches count more.** Match weights halve every 173 days
  (decay 0.004 per day).
- **Shrinkage.** Ratings are pulled towards average (ridge penalty), so a few
  results do not swing them wildly. Newly promoted teams get a separate
  effect learnt from previous promoted teams, which matters early in the
  season.
- **Home advantage** is estimated from the data, with a separate term for the
  2020-21 matches played without crowds.

The two expected goals become a full score distribution with the
Dixon-Coles adjustment (it corrects the frequencies of 0-0, 1-0, 0-1 and 1-1,
which plain Poisson gets slightly wrong). Half-time scores split each team's
goals between the halves using the observed share of goals scored before the
break.

### Line-ups and injuries

Team ratings are learnt from the sides that actually played, so they already
include a team's usual players. What matters for a given match is how its XI
differs from usual. For every player the model knows:

- his share of his team's non-penalty xG while he is on the pitch, and his
  usual minutes when he starts (section 3);
- how much of the team's recent history he started (same time weighting as
  the ratings).

The **line-up shift** adds up the attacking contribution of the XI and
subtracts the contribution of the players the team usually fields. A missing
star counts against the team; a returning one counts for it; a like-for-like
swap is close to zero. The team's expected goals are multiplied by
exp(coefficient × shift); the coefficient was estimated from real line-ups
(docs/BACKTEST.md). Injury news comes from the Fantasy Premier League site
and from the match file, which overrides it.

## 2. Anchoring to the bookmaker

Main markets are priced very efficiently (section 8). When you give the 1X2
odds, and ideally both sides of over/under 2.5 goals, the model:

1. removes the bookmaker's margin (the "power" method, which takes more
   margin off long shots than favourites, as bookmakers do);
2. finds the two expected-goals figures that reproduce those probabilities
   under the same Dixon-Coles score model;
3. blends them with its own figures, 75% bookmaker and 25% model (on the log
   scale). The weight was chosen by walk-forward testing (docs/BACKTEST.md).

Everything else (corners, cards, players, builders) is then simulated around
those blended expected goals, so a match the bookmaker expects to be
one-sided produces more corners and shots for the favourite, and so on.
Given a total corners or total cards line with both prices, the model also
moves its corners or cards means halfway towards the bookmaker's line.

## 3. Corners, cards, fouls and shots

Each team's corners, yellow cards, red cards, fouls, non-goal shots on target
and shots off target has its own regression, fitted the same way as goals
(team-for and team-against ratings, home advantage, time decay, shrinkage)
with three additions:

- **Score state.** Each count is modelled given the final score (winning by
  two, winning by one, losing by one, losing by two, total goals). A team
  chasing a game takes more corners and shots; a team defending a lead
  commits more fouls and picks up more cards. In the simulation the score
  comes first, so these effects carry through to every market.
- **Referees.** Cards and fouls include a referee effect, estimated with
  heavy shrinkage (a referee needs many games before the model believes he
  is unusual). If the referee is not known, each simulation draws a random
  referee, which correctly widens the range of card totals.
- **Over-dispersion.** Counts use the negative binomial distribution where
  the data are more variable than Poisson (corners, yellow cards, fouls,
  shots); red cards are Poisson.

After allowing for the score and the two teams, some correlation is left
(an open, end-to-end game raises both teams' shots and corners; a bad-tempered
one raises both teams' cards and fouls). Those leftover correlations between
all twelve counts are measured from the last three seasons and reproduced
with a Gaussian copula.

## 4. Players

From Understat's player match data (last three seasons, recent matches
weighted more), each player gets:

- **shares** of his team's non-penalty xG, xA, non-goal shots on target and
  shots off target while he is on the pitch;
- **card rates** per 90 minutes;
- **minutes**: how long he usually plays when he starts, how often he plays
  the full match, and how long he plays when he comes off the bench;
- **selection**: how often he has started recently and how often he comes
  on when he does not start.

Small samples are shrunk towards the average for the player's position
(600 minutes of play for shares, 1,200 for yellow cards, 6,000 for red
cards), so a centre-back with one lucky goal is not priced like a striker.
Shares rather than per-90 rates are used so that a player's output scales
with how many chances his team is expected to create in this match.

In each simulated match:

1. Every player gets minutes (starters may be substituted; bench players
   may come on).
2. The team's simulated goals are split into own goals (3%), penalties and
   open-play goals. Open-play goals go to players in proportion to
   xG share × minutes; 76% of open-play goals get an assist, given by xA
   share among the other players on the pitch. Penalties go to the
   first-choice taker if he is on the pitch (FPL's penalty order, then recent
   penalties taken).
3. The team's non-goal shots on and off target, yellow cards and red cards
   are shared out the same way.

Player probabilities are therefore **conditional on the player playing**,
which is how bookmakers settle player bets (void if he takes no part).

## 5. Markets

`plbet/markets.py` turns a short text description into a yes/no outcome on
every simulated match: result, double chance, draw no bet, totals, team
totals, both teams to score, correct score, Asian handicaps (whole lines can
push), half-time and HT/FT, half goals, corners and corner handicaps, cards
(with your bookmaker's counting rule, usually yellow 1 and red 2), shots,
shots on target, fouls, and player markets (score, 2+ goals, first scorer,
assist, score or assist, shots, shots on target, booked, goalkeeper saves).
The chance is the share of simulations where it wins, and the fair odds are
one divided by the chance.

## 6. Bet builders

For a builder, the legs are checked together on the same simulated matches:

- **model chance** = share of simulations where every leg wins (player legs
  only count simulations where the player plays);
- **link** = model chance ÷ (the legs' chances multiplied). Above 1 the legs
  help each other; below 1 they work against each other;
- **pairwise links** between every two legs, so you can see which pair is
  doing the work;
- **flags**: legs that are near-certain (they add bookmaker margin but barely
  move the price), very unlikely legs, legs that work against each other,
  and, if you give the single prices too, whether the bookmaker has cut the
  builder price by more than the real link between the legs (overpriced)
  or less (possibly good value).

The report also proposes builders worth pricing up: combinations of the
strongest legs that use different markets and players, leave out legs that
almost guarantee each other, and land at fair odds of about 2 to 5.

## 7. Staking and tracking

See docs/STAKING.md for the bankroll plan (fractional Kelly with minimum
edges and caps) and the Excel tracker, which records every bet with the
model's chance and works out ROI, profit against expectation and closing-line
value.

## 8. How it was tested

Every number in docs/BACKTEST.md comes from walk-forward testing: the model
is refitted using only matches played before each test match, exactly as it
would have been used at the time. Settings were tuned on earlier seasons and
then checked on 2025-26 and 2026-27, which played no part in tuning.

## 9. Known limits

- Player fouls, tackles, offsides and fouls won are not modelled.
- The defensive effect of individual absences (a missing centre-back or
  goalkeeper) is not modelled; only attacking contributions are.
- Red cards do not change the rest of the simulated match (a team down to
  ten men keeps its scoring rate).
- Projected line-ups assume the usual starters; when the real XI is
  announced, put it in the match file and run again.
- Players new to the Premier League start from position averages until they
  have played enough minutes.
- Corners and cards anchoring to bookmaker lines uses an even split that
  could not be tested, because historical odds for those markets are not
  freely available.
