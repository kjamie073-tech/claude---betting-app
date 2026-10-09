"""Odds conversion and bookmaker-margin removal."""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq


def to_decimal(odds: str | float | int) -> float:
    """Parse decimal (2.5), fractional (6/4, 'evs') or American (+150, -200) odds."""
    if isinstance(odds, (int, float)):
        if odds <= 1:
            raise ValueError(f"decimal odds must be above 1, got {odds}")
        return float(odds)
    s = str(odds).strip().lower()
    if s in ("evs", "evens", "even", "ev"):
        return 2.0
    if re.fullmatch(r"\d+\s*/\s*\d+", s):
        return float(Fraction(s.replace(" ", ""))) + 1.0
    if re.fullmatch(r"[+-]\d+(\.\d+)?", s):
        v = float(s)
        return 1 + v / 100 if v > 0 else 1 + 100 / abs(v)
    v = float(s)
    if v <= 1:
        raise ValueError(f"cannot read odds '{odds}'")
    return v


def implied(odds) -> np.ndarray:
    return 1.0 / np.asarray([to_decimal(o) for o in np.atleast_1d(odds)], float)


def overround(odds) -> float:
    return float(implied(odds).sum() - 1.0)


def demargin(odds, method: str = "power") -> np.ndarray:
    """Fair probabilities from a complete set of odds for one market.

    power (default) and shin both take more margin off longshots than
    favourites, which matches how bookmakers shade prices; multiplicative
    scales every price equally.
    """
    pi = implied(odds)
    if len(pi) < 2:
        raise ValueError("need every outcome of the market to remove the margin")
    if method == "multiplicative":
        return pi / pi.sum()
    if method == "power":
        if abs(pi.sum() - 1) < 1e-12:
            return pi
        k = brentq(lambda k: np.sum(pi ** k) - 1.0, 0.2, 5.0)
        return pi ** k
    if method == "shin":
        s = pi.sum()

        def probs(z):
            return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / s) - z) / (2 * (1 - z))

        if s <= 1:
            return pi / s
        z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.4)
        return probs(z)
    raise ValueError(method)


def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p


def expected_value(p: float, odds: float) -> float:
    """Expected profit per unit staked."""
    return p * to_decimal(odds) - 1.0
