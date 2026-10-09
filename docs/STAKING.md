# Bankroll and staking plan

The model tells you what it thinks the chance of a bet is. The staking plan
decides how much (if anything) to put on it. Most betting losses come from
staking, not from picks: betting too much, chasing losses, or piling onto
long-odds builders. This plan is built to keep you in the game long enough
for any real edge to show.

## 1. The bank

- Set aside a **betting bank**: money used only for betting that you could
  lose in full without it affecting anything else. Never top it up to chase
  losses.
- **1 unit = 1% of the bank.** On the 1st of each month, re-set the unit to
  1% of the bank at that moment. Stakes shrink after a bad month and grow
  after a good one, automatically and without emotion.

## 2. Only bet when there is an edge

For every bet the match report gives a **model chance** and **fair odds**
(1 ÷ chance). The **edge** is

    edge = model chance × your odds − 1

| Bet type | Minimum edge | Why |
|---|---|---|
| Result, goals, corners, cards singles | 3% | Main markets are priced sharply; the model's anchor to the bookmaker keeps these honest |
| Player singles (scorer, shots, cards) | 5% | More uncertainty: line-ups, minutes, role |
| Bet builders | 8% | Errors in each leg multiply, and bookmakers build big margins into builders |

If the edge is below the minimum, **don't bet**, however good the pick looks.
"Likely to win" and "worth betting" are different things: a 75% leg at odds
of 1.25 is a losing bet.

## 3. How much: fractional Kelly with caps

The Kelly stake is the share of the bank that grows it fastest *if the
probabilities are exactly right*:

    Kelly % = edge ÷ (odds − 1)

The model's probabilities are estimates, not facts, and full Kelly is far too
aggressive when they are off. So:

| Bet type | Stake | Cap |
|---|---|---|
| Singles on main markets | ¼ Kelly | 2 units |
| Player singles | ⅛ Kelly | 2 units |
| Bet builders | ⅛ Kelly | 1 unit |
| Everything on one match day | | 5 units in total |

Example: bank £500 (unit £5). The model gives a builder a 30% chance and the
bookmaker offers 4.00. Edge = 0.30 × 4 − 1 = 20%. Kelly = 0.20 ÷ 3 = 6.7% of
the bank. Eighth Kelly = 0.83% = **£4.17** (under the 1-unit cap).

`python -m plbet stake --prob 0.30 --odds 4.0 --kind builder --bank 500` does
this sum for you, and the tracker's *Suggested stake* column does it
automatically.

## 4. Rules that protect the bank

- **No chasing.** Never raise stakes to win back losses. The unit only changes
  on the 1st of the month.
- **One builder per match** unless the report shows two separate value
  builders that do not share legs. Two builders on the same game are mostly the
  same bet twice.
- **Stop and review at −40%.** If the bank falls 40% below where it started,
  stop betting and review the record (see section 6) before going on.
- **Long shots are entertainment.** Builders with under an 8% chance (odds
  above about 12) are outside the plan. If you want one for fun, use a fixed
  £1–2 that is not counted as part of the bank.

## 5. What to expect

These numbers come from simulating 300 bets at the stated odds and edge with
this plan's stake sizes (20,000 runs each):

| Typical odds | True edge | Chance of being in profit after 300 bets | Typical longest losing run | Chance of a 20%+ drawdown on the way |
|---|---|---|---|---|
| 2.0 (singles) | +5% | 79% | 7 (often 9–10) | 39% |
| 2.0 | 0% | 48% | 7 | 68% |
| 5.0 (builders) | +5% | 63% | 19 (often 26+) | 54% |
| 5.0 | 0% | 46% | 20 | 68% |
| 10.0 (big builders) | +5% | 57% | 34 (often 50) | 58% |

The takeaways:

- **Even with a real edge, results are noisy.** At builder odds, losing runs
  of 20 or more are normal and say very little about whether the model works.
- **Judge the model on hundreds of bets, not dozens.** Below about 200 bets
  luck dominates ROI.
- **Watch closing-line value (CLV) instead.** If the odds you take are
  usually better than the odds just before kick-off, the market is moving
  towards your view. That shows up far sooner than profit does. The tracker
  works it out when you enter closing odds.

## 6. Reviewing the record

Every month (or every 50 settled bets), look at the tracker's Summary sheet:

1. **ROI** by bet type and market. Stop betting any market that is clearly
   negative after 100+ bets in it.
2. **Actual minus expected profit.** Persistently negative over 200+ bets means
   the model is overrating its picks: raise the minimum edges.
3. **CLV.** Average CLV below 0% over 100+ bets means the edge is probably not
   real, whatever the short-term profit says.

## 7. Accumulators of bet builders

J's usual bet is one builder per game stacked into one acca at £5. The maths
of that is harsh, and worth knowing exactly:

- An acca's chance is the builders' chances multiplied, and so is the
  bookmaker's margin. If each builder is priced 20% below fair (a typical
  builder margin), a 10-game acca returns on average 0.8^10 ≈ 11% of the
  stake: an expected loss of about £4.45 per £5 acca. A 4-game acca at the
  same margin returns about 41%.
- So a long acca only makes sense when most of its builders are priced
  *above* the model's fair odds. `plbet acca` names the builders priced
  below fair and shows the acca without them. Dropping them is the single
  biggest thing that improves the expected return.
- **Offers are the most dependable edge a recreational bettor has.** A free
  £2 builder is worth roughly £1–£2 of real money; acca insurance (stake
  back as a free bet if one leg loses) and winnings boosts can turn a slightly
  negative acca positive. `plbet acca --boost 0.25 --insurance` and
  `--free-bet` price them in. Use every free bet on the builder with the
  highest *chance × (odds − 1)*, not the safest one.
- Fewer, better legs beat more legs: the same £5 on a 3–4 game acca of
  value builders has a far better expected return than a 10-game acca, and
  wins often enough to tell within a season whether the edge is real.

## 8. A word on the house edge

Bookmakers build a margin into every price: around 5% on match results,
10–15% on player markets, and often 20–30% or more on bet builders. Most
bettors lose over time because of it. This model's job is to find the minority
of bets where the price is wrong by more than the margin, and the plan's job
is to make sure you only bet those and survive the swings. If it is not fun,
or you are betting money you need, stop. Free, confidential support is
available from GamCare (0808 8020 133) and GAMSTOP lets you self-exclude from
UK bookmakers.
