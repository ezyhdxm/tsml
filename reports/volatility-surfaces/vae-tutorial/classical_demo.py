#!/usr/bin/env python3
"""Deterministic teaching examples; no market data, model training or VAE reruns.

1. One non-lognormal terminal distribution yields a non-flat implied-vol smile.
2. Bid/ask-aware convex projection repairs inconsistent call-price midpoints.
3. Two discrete martingales share European prices but differ on a path payoff.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, brentq, minimize
from scipy.special import ndtr


def black_call(strike: np.ndarray | float, vol: float, tau: float = 1.0,
               forward: float = 100.0) -> np.ndarray:
    """Undiscounted European call; r=q=0 in the demonstrations."""
    k = np.asarray(strike, dtype=float)
    if np.any(k <= 0) or np.any(np.asarray(vol) <= 0) or tau <= 0 or forward <= 0:
        raise ValueError('Strike, vol, maturity and forward must be positive.')
    root = vol * np.sqrt(tau)
    d1 = np.log(forward / k) / root + 0.5 * root
    return forward * ndtr(d1) - k * ndtr(d1 - root)


def mixture_call(k: np.ndarray | float) -> np.ndarray:
    return 0.5 * black_call(k, 0.1) + 0.5 * black_call(k, 0.4)


def implied_vol(k: float, price: float) -> float:
    intrinsic = max(100.0 - k, 0.0)
    if not intrinsic < price < 100.0:
        raise ValueError('IV inversion requires an interior Black price.')
    return brentq(lambda vol: float(black_call(k, vol)) - price,
                  1e-6, 5.0, xtol=1e-13)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    k = np.linspace(60.0, 160.0, 201)
    price = mixture_call(k)
    iv = np.array([implied_vol(float(ki), float(ci)) for ki, ci in zip(k, price)])
    assert np.max(np.abs(black_call(k, iv) - price)) < 1e-9
    picked = np.array([70., 80., 90., 100., 110., 120., 130., 140.])
    picked_price = mixture_call(picked)
    picked_iv = np.array([implied_vol(float(ki), float(ci)) for ki, ci in zip(picked, picked_price)])
    pd.DataFrame({'strike': picked, 'call_price': picked_price,
                  'implied_vol': picked_iv}).to_csv(output / 'mixture_example.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(k, 100 * iv, label='One mixture law: Black-equivalent IV')
    ax.axhline(25, linestyle='--', label='Arithmetic average of 10% and 40%')
    ax.set(xlabel='Strike K; forward = 100', ylabel='Implied volatility (%)',
           title='Different IVs do not require different underlyings')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output / 'mixture_smile.png', dpi=180); plt.close(fig)

    strikes = np.arange(70.0, 131.0, 10.0)
    truth = mixture_call(strikes)
    # Deliberately inconsistent illustrative midpoints, not empirical market data.
    mid = truth + np.array([.01, -.02, .015, 2., 0., -.015, .01])
    half = np.array([.08, .08, .08, 2.10, .08, .08, .08])
    n = len(strikes)
    slopes = np.zeros((n-1, n))
    for j, width in enumerate(np.diff(strikes)):
        slopes[j, j] = -1.0 / width
        slopes[j, j+1] = 1.0 / width
    convex = np.diff(slopes, axis=0)
    matrix = np.vstack([slopes, convex])
    lower = np.r_[-np.ones(n-1), np.zeros(n-2)]
    upper = np.r_[np.zeros(n-1), np.full(n-2, np.inf)]
    lo = np.maximum(mid-half, np.maximum(100-strikes, 0))
    hi = np.minimum(mid+half, 100.)
    if np.any(lo > hi):
        raise ValueError('Bid/ask box does not intersect individual price bounds.')
    def objective(c):
        error = (c-mid)/half
        return .5 * error @ error
    def grad(c):
        return (c-mid)/half**2
    # A generic feasible initialization: constant vol Black calls near ATM IV.
    start = np.clip(black_call(strikes, .25), lo, hi)
    opt = minimize(objective, start, jac=grad, method='SLSQP',
                   bounds=Bounds(lo, hi),
                   constraints=[LinearConstraint(matrix, lower, upper)],
                   options={'maxiter': 1000, 'ftol': 1e-12})
    if not opt.success:
        raise RuntimeError(f'Projection failed: {opt.message}')
    c = opt.x
    tol = 1e-8
    assert np.all(c >= lo-tol) and np.all(c <= hi+tol)
    assert np.all(slopes@c >= -1-tol) and np.all(slopes@c <= tol)
    assert np.all(convex@c >= -tol)
    table = pd.DataFrame({'strike': strikes, 'truth': truth, 'mid': mid,
                          'bid': mid-half, 'ask': mid+half, 'projected': c})
    table.to_csv(output / 'price_projection.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(strikes, mid, yerr=half, fmt='o', capsize=4, label='Illustrative bid/ask and midpoint')
    ax.plot(strikes, truth, '--', label='Synthetic truth (not used in objective)')
    ax.plot(strikes, c, 's-', label='Constrained price projection')
    ax.set(xlabel='Strike K', ylabel='Undiscounted call price',
           title='Classical construction in price space: no VAE required')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output / 'price_projection.png', dpi=180); plt.close(fig)

    s1 = np.array([2.5, 3.5]); s2 = np.array([1., 3., 5.])
    transitions = {'A': np.array([[.25, .75, 0.], [.25, .25, .5]]),
                   'B': np.array([[.5, .25, .25], [0., .75, .25]])}
    path = ((s1[:, None] < 3) & (s2[None, :] > 4)).astype(float)
    models = {}
    for name, p in transitions.items():
        assert np.allclose(p.sum(1), 1) and np.all(p >= 0)
        assert np.allclose(p @ s2, s1)
        terminal = .5 * p.sum(0)
        assert np.allclose(terminal, [.25, .5, .25])
        models[name] = {'transition': p.tolist(), 'conditional_mean': (p@s2).tolist(),
                        'terminal_probabilities': terminal.tolist(),
                        'path_digital_price': float((.5*p*path).sum())}
    result = {'scope': 'Deterministic illustrative calculations, not market benchmarks.',
              'mixture': {'forward': 100, 'tau': 1, 'component_vols': [.1, .4],
                          'weights': [.5, .5],
                          'atm_iv': float(picked_iv[3]),
                          'root_mean_square_vol': float(np.sqrt((.1**2+.4**2)/2))},
              'price_projection': {'solver': 'SLSQP', 'solver_success': True,
                                   'mid_min_convexity_margin': float((convex@mid).min()),
                                   'projected_min_convexity_margin': float((convex@c).min()),
                                   'all_inside_bid_ask': True,
                                   'continuous_global_arbitrage_certificate': False},
              'same_marginals_different_paths': models,
              'models_retrained': False}
    (output/'classical_examples.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent/'classical_examples')
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
