"""Synthetic cash-settled call hedging; no market data or Heston simulation.
Run: python hedging_experiments.py --paths 32768 --seed 20260905
Requires NumPy and SciPy. All P&L/costs are discounted to t=0.
The simulation measure is a specified Q (mu=r), not a fitted P.
Nested grids share exact GBM paths. Stock sampling has no Euler error.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ndtr, roots_hermitenorm

ROOT = Path(__file__).resolve().parent
S0, K, T, RATE, VOL = 100.0, 100.0, 1.0, 0.03, 0.25
HALF_SPREAD = 0.0005  # Stock half-spread 5 bp; full spread 10 bp.
GRIDS = (16, 64, 256, 1024)
REVEAL, LOW, HIGH, PROB_HIGH = 0.5, 0.10, 0.40, 0.5


def call_variance(spot, tau: float, variance):
    """Call price and delta with known integrated variance; q=0."""
    s, w = np.asarray(spot, dtype=float), np.asarray(variance, dtype=float)
    if np.any(s <= 0) or tau < 0 or np.any(w < 0):
        raise ValueError('Positive spot, nonnegative time/variance required')
    if tau == 0:
        return np.maximum(s-K, 0), (s > K).astype(float)
    if np.any(w <= 0):
        raise ValueError('Positive integrated variance required before expiry')
    root = np.sqrt(w)
    d1 = (np.log(s/K) + RATE*tau + w/2)/root
    d2 = d1-root
    return s*ndtr(d1)-K*np.exp(-RATE*tau)*ndtr(d2), ndtr(d1)


def price_delta(spot, t: float, mode: str, high):
    tau = T-t
    if mode == 'constant':
        return call_variance(spot, tau, VOL**2*tau)
    if mode != 'revelation':
        raise ValueError('Unknown model')
    if t < REVEAL:
        # Future regime is not observed by the hedger before announcement.
        common = VOL**2*(REVEAL-t)
        lo, dlo = call_variance(spot, tau, common+LOW**2*(T-REVEAL))
        hi, dhi = call_variance(spot, tau, common+HIGH**2*(T-REVEAL))
        return (1-PROB_HIGH)*lo+PROB_HIGH*hi, (1-PROB_HIGH)*dlo+PROB_HIGH*dhi
    return call_variance(spot, tau, np.where(high, HIGH, LOW)**2*tau)


def summary(x):
    x = np.asarray(x)
    loss = -x
    cutoff = np.quantile(loss, 0.95)
    return {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)),
            'mean_mcse': float(x.std(ddof=1)/np.sqrt(x.size)),
            'loss_var95': float(cutoff),
            'loss_es95': float(loss[loss >= cutoff].mean())}


def simulate(mode: str, paths: int, seed: int):
    rng = np.random.default_rng(seed)
    high = np.random.default_rng(seed+1).random(paths) < PROB_HIGH
    spot = np.full(paths, S0)
    premium, initial_delta = price_delta(S0, 0.0, mode, False)
    premium, initial_delta = float(premium), float(initial_delta)
    states = {}
    for n in GRIDS:
        theta = np.full(paths, initial_delta)
        opening = HALF_SPREAD*S0*abs(theta)
        cash = np.full(paths, premium)-theta*S0
        states[n] = dict(theta=theta, cash=cash, net_cash=cash-opening,
                         cost_pv=opening.copy(), opening=opening.copy())
    fine = max(GRIDS)
    dt = T/fine
    error_limit = np.zeros(paths)
    for i in range(1, fine+1):
        # The interval ending at REVEAL still uses pre-announcement volatility.
        sig = VOL if mode == 'constant' or i <= fine*REVEAL else np.where(high, HIGH, LOW)
        spot *= np.exp((RATE-0.5*sig**2)*dt+sig*np.sqrt(dt)*rng.standard_normal(paths))
        t, discount = i*dt, np.exp(-RATE*i*dt)
        if mode == 'revelation' and i == fine*REVEAL:
            lo, _ = call_variance(spot, T-REVEAL, LOW**2*(T-REVEAL))
            hi, _ = call_variance(spot, T-REVEAL, HIGH**2*(T-REVEAL))
            before = (1-PROB_HIGH)*lo+PROB_HIGH*hi
            after = np.where(high, hi, lo)
            error_limit = discount*(before-after)
        if i < fine:
            _, target = price_delta(spot, t, mode, high)
        for n, a in states.items():
            if i % (fine//n):
                continue
            growth = np.exp(RATE*T/n)
            a['cash'] *= growth
            a['net_cash'] *= growth
            if i < fine:
                trade = target-a['theta']
                cost = HALF_SPREAD*spot*np.abs(trade)
                a['cash'] -= trade*spot
                a['net_cash'] -= trade*spot+cost
                a['cost_pv'] += discount*cost
                a['theta'] = target.copy()
    payoff = np.maximum(spot-K, 0)
    rows = []
    max_ledger_error = 0.0
    for n, a in states.items():
        closing = np.exp(-RATE*T)*HALF_SPREAD*spot*np.abs(a['theta'])
        cost_pv = a['cost_pv']+closing
        gross = np.exp(-RATE*T)*(a['cash']+a['theta']*spot-payoff)
        net = gross-cost_pv
        net_direct = np.exp(-RATE*T)*(a['net_cash']+a['theta']*spot-payoff)-closing
        max_ledger_error = max(max_ledger_error, float(np.max(np.abs(net-net_direct))))
        row = {'steps': n, 'gross_pv': summary(gross), 'net_pv': summary(net),
               'mean_cost_pv': float(cost_pv.mean()),
               'mean_rebalance_cost_pv': float((cost_pv-a['opening']-closing).mean()),
               'cost_mean_mcse': float(cost_pv.std(ddof=1)/np.sqrt(paths)),
               'mean_cost_plus_half_gross_sd': float(cost_pv.mean()+0.5*gross.std(ddof=1))}
        if mode == 'revelation':
            row['rmse_to_continuous_limit'] = float(np.sqrt(np.mean((gross-error_limit)**2)))
            row['correlation_with_limit'] = float(np.corrcoef(gross, error_limit)[0, 1])
        rows.append(row)
    return {'initial_premium': premium, 'initial_delta': initial_delta,
            'rows': rows, 'max_cash_ledger_identity_error': max_ledger_error,
            'continuous_limit_sample': summary(error_limit) if mode == 'revelation' else None}


def main(paths: int, seed: int):
    if paths < 2000 or paths % 2:
        raise ValueError('Use an even number of paths >= 2000')
    const = simulate('constant', paths, seed)
    reveal = simulate('revelation', paths, seed)
    nodes, weights = roots_hermitenorm(160)
    shalf = S0*np.exp((RATE-0.5*VOL**2)*REVEAL+VOL*np.sqrt(REVEAL)*nodes)
    lo, _ = call_variance(shalf, T-REVEAL, LOW**2*(T-REVEAL))
    hi, _ = call_variance(shalf, T-REVEAL, HIGH**2*(T-REVEAL))
    limit_var = (np.exp(-2*RATE*REVEAL)*PROB_HIGH*(1-PROB_HIGH)
                 *np.sum(weights*(hi-lo)**2)/np.sqrt(2*np.pi))
    reveal['continuous_limit_sd_quadrature'] = float(np.sqrt(limit_var))
    term_vol = np.sqrt(0.5*0.15**2+0.5*0.40**2)
    time_example = {'sigma_first_half': 0.15, 'sigma_second_half': 0.40,
        'effective_sigma': float(term_vol),
        'initial_price_either_order': float(call_variance(S0, 1, term_vol**2)[0]),
        'initial_price_using_arithmetic_vol': float(call_variance(S0, 1, 0.275**2)[0]),
        'half_time_price_remaining_vol_15': float(call_variance(S0, 0.5, 0.15**2*0.5)[0]),
        'half_time_price_remaining_vol_40': float(call_variance(S0, 0.5, 0.40**2*0.5)[0])}
    n = np.array(GRIDS, dtype=float)
    risk_slope = float(np.polyfit(np.log(n), np.log([x['gross_pv']['sd'] for x in const['rows']]), 1)[0])
    cost_slope = float(np.polyfit(np.log(n), np.log([x['mean_rebalance_cost_pv'] for x in const['rows']]), 1)[0])
    checks = {
        'stock_and_cash_ledgers_agree': max(const['max_cash_ledger_identity_error'], reveal['max_cash_ledger_identity_error']) < 1e-9,
        'constant_vol_gross_means_within_5_mcse': all(abs(x['gross_pv']['mean']) < 5*x['gross_pv']['mean_mcse'] for x in const['rows']),
        'revelation_gross_means_within_5_mcse': all(abs(x['gross_pv']['mean']) < 5*x['gross_pv']['mean_mcse'] for x in reveal['rows']),
        'constant_vol_sd_scales_near_inverse_sqrt_N': -0.65 < risk_slope < -0.35,
        'rebalancing_cost_scales_near_sqrt_N': 0.35 < cost_slope < 0.65,
        'revelation_limit_remains_positive': limit_var > 1.0,
        'revelation_limit_quadrature_matches_mc_sd': abs(reveal['continuous_limit_sample']['sd']-np.sqrt(limit_var)) < 0.08,
        'finest_revelation_error_tracks_unspanned_jump': reveal['rows'][-1]['correlation_with_limit'] > 0.98,
        'prices_and_costs_finite': all(np.isfinite(x['mean_cost_pv']) for x in const['rows']+reveal['rows'])}
    result = {'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'parameters': {'S0': S0, 'K': K, 'T': T, 'r': RATE, 'q': 0, 'sigma': VOL,
            'half_spread': HALF_SPREAD, 'paths': paths, 'seed': seed, 'grids': GRIDS,
            'simulation_measure': 'specified Q; mu=r', 'reveal_time': REVEAL,
            'low_vol': LOW, 'high_vol': HIGH, 'Q_probability_high': PROB_HIGH},
        'constant': const, 'revelation': reveal, 'time_dependent': time_example,
        'log_log_slopes': {'hedging_sd_vs_N': risk_slope, 'rebalance_cost_vs_N': cost_slope},
        'checks': {k: bool(v) for k, v in checks.items()}}
    (ROOT/'hedging_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    assert all(checks.values()), checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=int, default=32768)
    parser.add_argument('--seed', type=int, default=20260905)
    args = parser.parse_args()
    main(args.paths, args.seed)
