"""Tests that need no downloaded data (synthetic simulations and tables)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plbet import builder, markets, odds
from plbet.data import names
from plbet.models import players
from plbet.models.goals import implied_lambdas, outcome_probs, score_matrix, total_over
from plbet.models.simulate import SimResult, SquadSim
from plbet.staking import StakingPlan


# ---------------------------------------------------------------- odds

def test_to_decimal_formats():
    assert odds.to_decimal("6/4") == pytest.approx(2.5)
    assert odds.to_decimal("evs") == pytest.approx(2.0)
    assert odds.to_decimal(2.1) == pytest.approx(2.1)
    assert odds.to_decimal("+150") == pytest.approx(2.5)
    assert odds.to_decimal("-200") == pytest.approx(1.5)


@pytest.mark.parametrize("method", ["power", "shin", "multiplicative"])
def test_demargin_sums_to_one_and_keeps_order(method):
    p = odds.demargin([2.0, 3.6, 4.2], method)
    assert sum(p) == pytest.approx(1.0, abs=1e-9)
    assert p[0] > p[1] > p[2]
    # removing the margin lowers every implied probability
    assert all(pi < 1 / o for pi, o in zip(p, [2.0, 3.6, 4.2]))


# ---------------------------------------------------------------- goals

def test_score_matrix_and_implied_lambdas_round_trip():
    lh, la, rho = 1.7, 1.1, -0.05
    mat = score_matrix(lh, la, rho)
    assert mat.sum() == pytest.approx(1.0, abs=1e-6)
    ph, pd_, pa = outcome_probs(mat)
    assert ph + pd_ + pa == pytest.approx(1.0, abs=1e-9)
    po = total_over(mat, 2.5)
    lh2, la2 = implied_lambdas(ph, pd_, pa, po, rho)
    assert lh2 == pytest.approx(lh, abs=0.02)
    assert la2 == pytest.approx(la, abs=0.02)


# ---------------------------------------------------------------- markets

def _sim(n=20000, seed=1) -> SimResult:
    rng = np.random.default_rng(seed)
    hg = rng.poisson(1.6, n).astype(np.int16)
    ag = rng.poisson(1.1, n).astype(np.int16)
    hht = rng.binomial(hg, 0.45).astype(np.int16)
    aht = rng.binomial(ag, 0.45).astype(np.int16)
    stats = {f"{s}_{k}": rng.poisson(m, n).astype(np.int16)
             for s in ("h", "a")
             for k, m in (("corners", 5.0), ("yellow", 1.8), ("red", 0.08), ("fouls", 10.5),
                          ("ngon", 3.0), ("ngoff", 7.0))}
    # A striker who plays 90% of simulations and scores half the home goals.
    played = rng.random(n) < 0.9
    goals = np.where(played, rng.binomial(hg, 0.5), 0)
    sq = SquadSim(
        ids=[1, 2], names=["Test Striker", "Test Keeper"], started=np.array([True, True]),
        minutes=np.column_stack([np.where(played, 90, 0), np.full(n, 90)]),
        goals=np.column_stack([goals, np.zeros(n, int)]),
        assists=np.zeros((n, 2), int), shots=np.column_stack([goals + 1, np.zeros(n, int)]),
        sot=np.column_stack([goals, np.zeros(n, int)]),
        yellow=np.zeros((n, 2), int), red=np.zeros((n, 2), int),
        first_scorer=np.zeros((n, 2), bool))
    return SimResult(n=n, home="Arsenal", away="Chelsea", hg=hg, ag=ag, hht=hht, aht=aht,
                     stats=stats, players={"h": sq})


def test_result_and_goal_markets_are_consistent():
    sim = _sim()
    p = {s: markets.evaluate(s, sim).prob() for s in ("result:home", "result:draw", "result:away")}
    assert sum(p.values()) == pytest.approx(1.0)
    over = markets.evaluate("goals:over:2.5", sim).prob()
    under = markets.evaluate("goals:under:2.5", sim).prob()
    assert over + under == pytest.approx(1.0)
    # team names work as sides
    assert markets.evaluate("result:Arsenal", sim).prob() == pytest.approx(p["result:home"])
    assert markets.evaluate("dc:1x", sim).prob() == pytest.approx(p["result:home"] + p["result:draw"])
    # draw no bet pushes on a draw
    o = markets.evaluate("dnb:home", sim)
    assert o.push_prob() == pytest.approx(p["result:draw"])


def test_cards_use_bookmaker_weights():
    sim = _sim()
    h, a = sim.stats["h_yellow"], sim.stats["a_yellow"]
    r = sim.stats["h_red"] + sim.stats["a_red"]
    expected = ((h + a + 2 * r) > 3.5).mean()
    assert markets.evaluate("cards:over:3.5", sim).prob() == pytest.approx(expected)
    sim.card_weights = (1, 1)
    expected = ((h + a + r) > 3.5).mean()
    assert markets.evaluate("cards:over:3.5", sim).prob() == pytest.approx(expected)


def test_player_markets_are_conditional_on_playing():
    sim = _sim()
    o = markets.evaluate("player:Test Striker:goal", sim)
    played = sim.players["h"].minutes[:, 0] > 0
    assert o.valid is not None and o.valid.mean() == pytest.approx(played.mean())
    assert o.prob() == pytest.approx((sim.players["h"].goals[played, 0] >= 1).mean())
    # name matching tolerates a missing first name
    assert markets.evaluate("player:Striker:goal", sim).prob() == pytest.approx(o.prob())
    with pytest.raises(markets.MarketError):
        markets.evaluate("player:Nobody Atall:goal", sim)


# ---------------------------------------------------------------- builders

def test_builder_link_between_nested_legs():
    sim = _sim()
    # Winning implies at least one goal: the legs are linked.
    rep = builder.analyse(sim, ["result:home", "team_goals:home:over:0.5"], builder_odds=2.5)
    p_win = markets.evaluate("result:home", sim).prob()
    assert rep.joint == pytest.approx(p_win)
    assert rep.lift > 1.1
    assert rep.ev == pytest.approx(rep.joint * 2.5 - 1)


def test_builder_independent_legs_have_lift_near_one():
    sim = _sim()
    rep = builder.analyse(sim, ["corners:over:9.5", "fouls:over:20.5"])
    assert rep.lift == pytest.approx(1.0, abs=0.05)
    assert "roughly independent" in set(rep.pairs["relation"])


def test_builder_flags_bookmaker_cutting_more_than_the_link():
    sim = _sim()
    legs = ["result:home", "goals:over:2.5"]
    rep = builder.analyse(sim, legs, builder_odds=2.0, leg_odds=[1.9, 1.8])
    assert rep.bookmaker_lift == pytest.approx(1.9 * 1.8 / 2.0)
    assert any("overpriced" in f for f in rep.flags)


# ---------------------------------------------------------------- staking

def test_staking_example_from_docs():
    s = StakingPlan(500).stake(0.30, 4.0, "builder")
    assert s["stake"] == pytest.approx(4.17, abs=0.01)
    assert StakingPlan(500).stake(0.30, 3.4, "builder")["stake"] == 0.0   # edge 2% < 8%
    # singles are capped at 2 units
    assert StakingPlan(500).stake(0.6, 2.0, "single")["stake"] == pytest.approx(10.0)


# ---------------------------------------------------------------- names

def test_team_aliases():
    assert names.team("Spurs") == "Tottenham"
    assert names.team("Man Utd") == "Man United"
    assert names.team("Nottingham Forest") == "Nott'm Forest"
    assert names.team("Wolverhampton Wanderers") == "Wolves"


def test_player_name_scores():
    assert names.name_score("Bukayo Saka", "Bukayo Saka") == 1.0
    assert names.name_score("Gabriel", "Gabriel Magalhaes") >= 0.9
    assert names.name_score("Joško Gvardiol", "Josko Gvardiol") == 1.0
    assert names.name_score("Erling Haaland", "Bukayo Saka") < 0.6


# ---------------------------------------------------------------- players

def test_availability_from_fpl_status():
    assert players.availability("a", np.nan) == 1.0
    assert players.availability("d", 75.0) == 0.75
    assert players.availability("i", np.nan) == 0.0
    assert players.availability("s", np.nan) == 0.0


def test_attack_shift_is_zero_for_the_usual_xi():
    prof = pd.DataFrame({"share_npxg": [0.3, 0.2, 0.1], "min_st": [90.0, 90.0, 90.0]},
                        index=["a", "b", "c"])
    pres = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})
    assert players.attack_shift(["a", "b", "c"], prof, pres) == pytest.approx(0.0)
    # without the main striker the XI is 0.3 short
    assert players.attack_shift(["b", "c"], prof, pres) == pytest.approx(-0.3)
    # a player who usually starts half the games counts half
    pres2 = pd.Series({"a": 0.5, "b": 1.0, "c": 1.0})
    assert players.attack_shift(["b", "c"], prof, pres2) == pytest.approx(-0.15)


def _player_matches(rows):
    """Minimal Understat-style player match table: (match, day, player, started, minutes)."""
    d = pd.DataFrame(rows, columns=["match_id", "day", "player_id", "started", "minutes"])
    d["date"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(d["day"], unit="D")
    d["team"] = "Team"
    return d


def test_selection_probs_are_probabilities():
    # p1 starts every game; p2 starts two, comes on once, sits out once;
    # p3 comes off the bench in two of four games
    rows = []
    for k in range(4):
        rows.append((k, 7 * k, "p1", True, 90))
    rows += [(0, 0, "p2", True, 90), (1, 7, "p2", True, 80), (2, 14, "p2", False, 10)]
    rows += [(1, 7, "p3", False, 10), (3, 21, "p3", False, 20)]
    sel = players.selection_probs("Team", pd.Timestamp("2026-02-01"), _player_matches(rows))
    sel = sel.set_index("player_id")
    assert sel["p_start_hist"].between(0, 1).all()
    assert sel["p_sub_hist"].between(0, 1).all()
    assert sel.at["p1", "p_start_hist"] == pytest.approx(1.0)
    assert sel.at["p1", "p_sub_hist"] == 0.0
    assert 0 < sel.at["p2", "p_sub_hist"] < 1
    assert 0 < sel.at["p3", "p_sub_hist"] < 1


def test_bench_minutes_fill_what_the_starters_leave():
    n_bench = 12
    sq = pd.DataFrame({
        "start": [True] * 11 + [False] * n_bench,
        "full": [0.5] * 11 + [0.0] * n_bench,
        "min_st": [80.0] * 11 + [0.0] * n_bench,
        "min_sb": [0.0] * 11 + [20.0] * n_bench,
        "p_sub": [0.0] * 11 + [0.9] * n_bench,
    })
    p = players.balance_bench(sq)
    full, off = players.start_minutes(np.full(11, 0.5), np.full(11, 80.0))
    left = float(np.sum(90 - (full * 90 + (1 - full) * off)))
    assert float((p * sq["min_sb"]).sum()) == pytest.approx(left)
    assert (p[:11] == 0).all() and (p <= 0.95).all()
    # a short bench is scaled up, but no one beyond the cap
    short = sq.iloc[:13].assign(p_sub=[0.0] * 11 + [0.1, 0.1])
    p2 = players.balance_bench(short)
    assert (p2 <= 0.95 + 1e-12).all() and p2.iloc[11] > 0.1


# ---------------------------------------------------------------- tracker

def test_tracker_round_trip(tmp_path):
    from plbet import tracker
    path = tracker.create(tmp_path / "t.xlsx", starting_bank=200)
    r1 = tracker.add_bet(path, date="2026-10-11", match="Liverpool v Man City",
                         selection="Over 2.5 goals", odds=1.9, stake=4, model_chance=0.58,
                         bet_type="Single", market="Goals", bookmaker="Bet365")
    r2 = tracker.add_bet(path, date="2026-10-11", match="Liverpool v Man City",
                         selection="Haaland to score + City win", odds=4.0, stake=2,
                         model_chance=0.28, bet_type="Builder", market="Builder")
    tracker.settle(path, r1, "Won", closing_odds=1.8)
    tracker.settle(path, r2, "Lost")
    s = tracker.summary(path)
    assert s["settled"] == 2
    assert s["staked"] == pytest.approx(6.0)
    assert s["profit"] == pytest.approx(4 * 0.9 - 2)
    assert s["bank"] == pytest.approx(200 + 1.6)
    assert s["avg_clv"] == pytest.approx(1.9 / 1.8 - 1)
    with pytest.raises(ValueError):
        tracker.settle(path, r1, "Maybe")


def test_yellow_cards_go_to_different_players():
    from plbet.models.simulate import _distinct
    rng = np.random.default_rng(3)
    counts = rng.poisson(3.0, 5000)
    w = np.tile(np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05]), (5000, 1))
    out = _distinct(counts, w, rng)
    assert out.max() <= 1
    assert (out.sum(axis=1) == np.minimum(counts, 6)).all()
    # the most card-prone player is booked most often
    assert out[:, 0].mean() > out[:, 1].mean() > out[:, 5].mean()


def test_acca_multiplies_builders_and_flags_bad_legs():
    from plbet.acca import AccaLeg, combine
    legs = [AccaLeg("A v B", None, ["x"], ["x"], 0.5, 0.001, odds=2.2),
            AccaLeg("C v D", None, ["y"], ["y"], 0.4, 0.001, odds=2.0)]
    rep = combine(legs, stake=5)
    assert rep.prob == pytest.approx(0.2)
    assert rep.odds == pytest.approx(4.4)
    assert rep.ev == pytest.approx(0.2 * 4.4 - 1)
    assert rep.min_odds == pytest.approx(5 * 1.08)
    assert any("C v D" in n for n in rep.notes)  # priced below fair: flagged
    assert combine(legs, acca_odds="9/2").odds == pytest.approx(5.5)
    unpriced = combine([AccaLeg("A v B", None, ["x"], ["x"], 0.5, 0.001)])
    assert unpriced.odds is None and unpriced.ev is None
