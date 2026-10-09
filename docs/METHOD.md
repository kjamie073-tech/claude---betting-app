# How the model works

The model answers one question for a chosen match: **how likely is every
outcome a bookmaker will let you bet on, including combinations of them?** It
does that by simulating the match 100,000 times, with every simulated match
carrying a full set of numbers (score at half-time and full-time, corners,
cards, fouls, shots for both teams, and goals, assists, shots and cards for
every player), and then counting. Because each leg of a bet builder is read
off the same simulated matches, the links between legs (a team winning and
its striker scoring) come out automatically.

Every setting below was chosen by walk-forward testing on seasons up to
2024-25 and then checked on 2025-26 and 2026-27, which played no part in
tuning (section 8 and [BACKTEST.md](BACKTEST.md)).

## 1. Goals: team ratings

Each team has an attacking and a defensive rating, fitted with a Poisson
regression to every Premier League match since 2019-20:

    log(expected goals for team i against team j)
        = league average + attack(i) − defence(j) + home advantage
          + promoted-team effect

- **What is being fitted.** Not only actual goals: the target is
  0.3 × goals + 0.7 × (non-penalty xG + the league's average penalty xG).
  xG is a steadier measure of how well a team plays than goals, which are
  noisy.
- **Recent matches count more.** Match weights halve every 173 days
  (decay 0.004 per day).
- **Shrinkage.** Ratings are pulled towards average (ridge penalty), so a few
  results do not swing them wildly. Newly promoted teams get a separate
  effect learnt from previous promoted teams, which matters early in the
  season.
- **Home advantage** is estimated from the data, with a separate term for the
  2020-21 matches played without crowds.

### Calibrating the totals

Two things the regression gets wrong on its own, both found in the
backtest:

- **Level.** Understat's xG has run 5–13% above the goals actually scored in
  recent seasons, so a model fitted mostly to xG expects too many goals.
- **Spread.** Ratings differences that are partly noise make the predicted
  match totals too far apart: matches predicted to be high-scoring are less
  high-scoring than predicted, and low ones less low.

So each match's predicted total is moved towards the league's real goals per
game over the last 365 days:

    calibrated total = level × (predicted total ÷ typical predicted total)^0.73

where *level* is the real average and *typical predicted total* is the
model's average over the same matches. The split between the two teams is
kept. The 0.73 was fitted on 2021-22 to 2024-25; on 2025-26 and 2026-27 it
brought the average from 2.98 predicted goals per game (against 2.76
scored) to 2.79.

### Score and half-time

The two expected goals become a full score distribution with the
Dixon-Coles adjustment (it corrects the frequencies of 0-0, 1-0, 0-1 and 1-1,
which plain Poisson gets slightly wrong). Half-time scores split each team's
goals between the halves using the observed share of goals scored before the
break.

### Line-ups and injuries

Team ratings are learnt from the sides that actually played, so they already
include a team's usual players. What matters for a given match is how its XI
differs from usual. For every player the model knows:

- their share of the team's non-penalty xG while on the pitch, and their
  usual minutes when starting (section 4);
- how much of the team's recent history they started (same time weighting as
  the ratings).

The **line-up shift** adds up the attacking contribution of the XI and
subtracts the contribution of the players the team usually fields. A missing
star counts against the team; a returning one counts for it; a like-for-like
swap is close to zero.

The other side's defence matters too: the model counts the opponent's
regular goalkeepers and defenders (those who started at least half of its
recent games) who are not in its XI, each weighted by how regularly they
start. About 0.9 are missing in a typical match, and the team ratings
already include that.

A team's expected goals are multiplied by

    exp(0.27 × line-up shift + 0.045 × (opponent's missing defenders − 0.94))

kept between ×0.8 and ×1.2. Both numbers were estimated from the real XIs
of 3,800 team-matches ([BACKTEST.md](BACKTEST.md), section 6). The effects
are modest: in 90% of matches the line-up shift moves a team's expected
goals by less than 5% (a missing striker's chances partly go to whoever
replaces them), and each extra regular defender missing from the other side
adds about 4.5%.
Injury news comes from the Fantasy Premier League site and from the match
file, which overrides it.

## 2. Anchoring to the bookmaker

Main markets are priced very efficiently ([BACKTEST.md](BACKTEST.md),
section 2). When you give the 1X2 odds, and ideally both sides of over/under
2.5 goals, the model:

1. removes the bookmaker's margin (the "power" method, which takes more
   margin off long shots than favourites, as bookmakers do);
2. finds the two expected-goals figures that reproduce those probabilities
   under the same Dixon-Coles score model;
3. blends them with its own figures on the log scale, with weight
   MARKET_WEIGHT_GOALS on the bookmaker. The weight was chosen by
   walk-forward testing.

Everything else (corners, cards, players, builders) is then simulated around
those blended expected goals, so a match the bookmaker expects to be
one-sided produces more corners and shots for the favourite, and so on.
Given a total corners or total cards line with both prices, the model also
moves its corners or cards means halfway towards the bookmaker's line (an
even split: there are no free historical corners or cards odds to tune it
on).

## 3. Corners, cards, fouls and shots

Each team's corners, yellow cards, red cards, fouls, non-goal shots on target
and shots off target has its own regression, fitted the same way as goals
(team-for and team-against ratings, home advantage, time decay, shrinkage)
with these additions:

- **Score state.** Each count is modelled given the final score (winning by
  two, winning by one, losing by one, losing by two, total goals). A team
  chasing a game takes more corners and shots; a team defending a lead
  commits more fouls and picks up more cards. In the simulation the score
  comes first, so these effects carry through to every market.
- **Referees.** Cards and fouls include a referee effect, estimated with
  heavy shrinkage (a referee needs many games before the model treats them
  as unusual). If the referee is not known, each simulation draws a random
  referee, which correctly widens the range of card totals.
- **Over-dispersion.** Counts use the negative binomial distribution where
  the data are more variable than Poisson (corners, yellow cards, fouls,
  shots); red cards are Poisson.
- **Calibrated totals.** As with goals, predicted match totals were too far
  apart, so each is moved towards the typical total with the same formula:
  exponent 0.59 for corners, 0.58 for non-goal shots on target, 0.65 for
  shots off target. Cards and fouls needed no correction (exponent 1). Red
  cards are too rare to tell matches apart, so every match gets the typical
  rate.

After allowing for the score and the two teams, some correlation is left
(an open, end-to-end game raises both teams' shots and corners; a bad-tempered
one raises both teams' cards and fouls). Those leftover correlations between
all twelve counts are measured from the last three seasons and reproduced
with a Gaussian copula.

## 4. Players

From Understat's player match data (last three seasons, recent matches
weighted more), each player gets:

- **shares** of the team's non-penalty xG, xA, non-goal shots on target and
  shots off target while on the pitch;
- **card rates** per 90 minutes;
- **minutes**: how long they usually play when starting, how often they play
  the full match, and how long they play when coming off the bench;
- **selection**: how often they have started recently and how often they
  come on when not starting.

Small samples are shrunk towards the average for the player's position
(600 minutes of play for shares, 3,600 for yellow cards, 6,000 for red
cards), so a centre-back with one lucky goal is not priced like a striker.
Card rates get the strongest pull because they are the noisiest: with a
weaker one the backtest found the most card-prone players booked about 25%
less often than predicted.
Shares rather than per-90 rates are used so that a player's output scales
with how many chances the team is expected to create in this match.

In each simulated match:

1. Every player gets minutes. Starters may be substituted; bench players
   may come on. Eleven players are on the pitch throughout, so the bench's
   chances of coming on are scaled until their expected minutes fill the
   minutes the starters are expected to leave (substitutes play about 6% of
   a team's minutes).
2. The team's simulated goals are split into own goals (3.8%), penalties
   and open-play goals. Open-play goals go to players in proportion to
   xG share × minutes; 76% of open-play goals get an assist, given by xA
   share among the other players on the pitch. Penalties go to the
   first-choice taker if on the pitch (FPL's penalty order, then recent
   penalties taken).
3. The team's non-goal shots on and off target, yellow cards and red cards
   are shared out the same way. Each yellow card goes to a different player
   (a second yellow is a red card).

Substitutes produce far more per minute than their usual rates suggest:
they play about 6% of the minutes but take about 13% of the goals and shots
and 14% of the yellow cards, because they come on when games are open and
play stoppage time that the minute counts leave out. So in step 2 and 3 a
substitute's rates are multiplied by about 1.7 to 1.9 for goals, assists and
shots, 2.45 for yellow cards and 1.6 for red cards, the factors that make
the simulated substitutes' share match 2022-23 to 2024-25. Without this the
starters' scorer and shots chances came out 5–10% too high.

Player probabilities are therefore **conditional on the player playing**,
which is how bookmakers settle player bets (void if the player takes no
part). Players with under 600 Premier League minutes in the last three
seasons are priced mostly from position averages; the report says so and
leaves them out of suggested builder legs.

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

The report also lists the strongest legs (legs your odds show an edge on,
then the likeliest line in each market you did not price) and proposes
builders worth pricing up: combinations that use different markets and
players, leave out legs that almost guarantee each other, and land at fair
odds of about 2 to 5.

## 7. Staking and tracking

See [STAKING.md](STAKING.md) for the bankroll plan (fractional Kelly with
minimum edges and caps) and the Excel tracker, which records every bet with
the model's chance and works out ROI, profit against expectation and
closing-line value.

## 8. How it was tested

Every number in [BACKTEST.md](BACKTEST.md) comes from walk-forward testing:
the model is refitted using only matches played before each test match,
exactly as it would have been used at the time, with the real starting XIs
and referee. Settings were tuned on earlier seasons and then checked on
2025-26 and 2026-27, which played no part in tuning.

## 9. Known limits

- Player fouls, tackles and offsides are not modelled (no reliable free
  data). Goalkeeper saves come from the opponent's shots on target and have
  not been backtested.
- Red cards do not change the rest of the simulated match (a team down to
  ten men keeps its scoring rate).
- Projected line-ups assume the usual starters; when the real XI is
  announced, put it in the match file and run again.
- Players new to the Premier League start from position averages until they
  have played enough minutes.
- Corners and cards anchoring to bookmaker lines uses an even split that
  could not be tested, because historical odds for those markets are not
  freely available.
- Referees new to the Premier League are treated as average. In the
  backtest they gave about 10% more yellow cards and fouls than that, but
  the sample is small (26 team-matches).
- On the held-out seasons the model's most confident corner calls (80%+)
  came in less often than it said (docs/BACKTEST.md, section 3).
