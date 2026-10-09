"""Stake sizing: fractional Kelly with hard caps.

The plan (docs/STAKING.md has the reasoning):

* Keep a separate betting bank you can afford to lose. One unit = 1% of it,
  re-set on the 1st of each month to 1% of the bank at that point.
* Stake = Kelly fraction x bank x edge / (odds - 1), where
  edge = model chance x odds - 1.
  Kelly fraction: 1/4 for singles on main markets, 1/8 for bet builders and
  player bets (their probabilities are less certain).
* Only bet when the edge clears the minimum: 3% for main-market singles, 5%
  for player singles, 8% for bet builders.
* Caps: 2 units per single, 1 unit per builder, 5 units per match day.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import odds as odds_mod


@dataclass
class StakingPlan:
    bank: float
    unit_pct: float = 0.01
    kelly_single: float = 0.25
    kelly_builder: float = 0.125
    min_edge_single: float = 0.03
    min_edge_player: float = 0.05
    min_edge_builder: float = 0.08
    cap_single_units: float = 2.0
    cap_builder_units: float = 1.0
    cap_day_units: float = 5.0
    min_prob_builder: float = 0.08

    @property
    def unit(self) -> float:
        return self.bank * self.unit_pct

    def stake(self, prob: float, odds, kind: str = "single") -> dict:
        """kind: 'single' (result/goals/corners/cards), 'player' or 'builder'."""
        o = odds_mod.to_decimal(odds)
        edge = prob * o - 1
        if kind == "builder":
            frac, min_edge, cap = self.kelly_builder, self.min_edge_builder, self.cap_builder_units
        elif kind == "player":
            frac, min_edge, cap = self.kelly_builder, self.min_edge_player, self.cap_single_units
        else:
            frac, min_edge, cap = self.kelly_single, self.min_edge_single, self.cap_single_units
        reason = ""
        if edge < min_edge:
            reason = f"edge {edge:+.1%} is below the {min_edge:.0%} minimum"
            amount = 0.0
        elif kind == "builder" and prob < self.min_prob_builder:
            reason = f"only {prob:.0%} likely; below the {self.min_prob_builder:.0%} floor"
            amount = 0.0
        else:
            kelly = edge / (o - 1)
            amount = min(frac * kelly * self.bank, cap * self.unit)
            reason = "capped" if amount == cap * self.unit else "fractional Kelly"
        return {
            "odds": o, "prob": prob, "edge": edge, "fair_odds": 1 / prob if prob else float("inf"),
            "stake": round(amount, 2), "units": round(amount / self.unit, 2) if self.unit else 0,
            "reason": reason,
        }
