"""
Fractional Kelly bankroll calculator — Daniel / Modelo de Predicciones
========================================================================
kelly.py — standalone, no model dependency.

    f* = (b*p - q) / b        full Kelly, b = decimal_odds - 1,
                              p = model probability, q = 1 - p
    stake = fraction * f* * dispersion_penalty, capped at `cap`.

CORRECTION (expert): fractional Kelly does NOT mathematically
guarantee the user never goes bust. It sharply reduces ruin risk
vs full Kelly (quarter Kelly cuts variance ~4x and drawdowns a lot),
but with a finite bankroll, miscalibrated probabilities, or a long
enough losing run, ruin is still possible — the probability just
becomes small. Any claim of a guarantee is wrong; this module
documents the risk honestly and caps stakes for that reason.

Dispersion penalty: the three base models (Poisson / LightGBM /
XGBoost) disagreeing means the edge is uncertain, so the stake is
shrunk. dispersion = max-min of the three models' probability for
the chosen side. penalty = clip(1 - dispersion / 0.30, 0, 1):
a 0.30 spread (e.g. Bayesian 0.75 vs trees 0.55, as in Daniel's
example) shrinks the stake to zero; a 0.10 spread keeps 2/3.
"""
import numpy as np


def dispersion_penalty(dispersion, scale=0.30):
    """1.0 when models agree, 0.0 when spread >= scale."""
    return float(np.clip(1.0 - float(dispersion) / scale, 0.0, 1.0))


def kelly_stake(p_model, odds_decimal, fraction=0.25, dispersion=0.0,
                cap=0.02, dispersion_scale=0.30):
    """Return a dict with the stake as a fraction of bankroll.
    fraction=0.25 is Quarter Kelly (default). cap=0.02 limits any
    single bet to 2% of bankroll even if Kelly says more."""
    p = float(np.clip(p_model, 0.001, 0.999))
    odds = float(odds_decimal)
    if odds <= 1.0:
        return {"stake_frac": 0.0, "edge": 0.0, "full_kelly": 0.0,
                "penalty": 0.0, "reason": "odds <= 1"}
    b = odds - 1.0
    edge = p * odds - 1.0  # expected profit per 1 unit staked
    full = (b * p - (1.0 - p)) / b
    if full <= 0:
        return {"stake_frac": 0.0, "edge": float(edge),
                "full_kelly": float(full), "penalty": 1.0,
                "reason": "no edge (full Kelly <= 0)"}
    pen = dispersion_penalty(dispersion, dispersion_scale)
    stake = float(np.clip(fraction * full * pen, 0.0, cap))
    return {"stake_frac": stake, "edge": float(edge),
            "full_kelly": float(full),
            "fractional_kelly_uncapped": float(fraction * full),
            "penalty": pen, "capped": bool(stake >= cap),
            "reason": "ok" if stake > 0 else "dispersion penalty = 0"}


def example(p_model=0.60, odds_decimal=2.0, dispersion=0.05):
    """Worked example: 60% model vs decimal odds 2.00 (fair 50%)."""
    return kelly_stake(p_model, odds_decimal, fraction=0.25,
                       dispersion=dispersion)
