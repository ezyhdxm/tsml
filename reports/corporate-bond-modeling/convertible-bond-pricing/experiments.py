"""Synthetic convertible-bond experiments used in SOURCE.md.

Jump-to-zero stock; independent constant default intensity; immediate recovery
of face; constant rates/volatility; optional grid-date exercise. No market data.
Requires Python >= 3.10, NumPy, SciPy. Run directly to regenerate results.json.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr

BASE = dict(S=50., F=100., m=2., T=3., r=.04, q=.01, sigma=.30, h=.025, R=.40)


def validate(S, F, m, T, r, q, sigma, h, R):
    if not all(math.isfinite(x) for x in (S, F, m, T, r, q, sigma, h, R)):
        raise ValueError("All scalar inputs must be finite")
    if min(S, F, T, sigma) <= 0 or m < 0 or h < 0 or not 0 <= R <= 1:
        raise ValueError("Invalid model inputs")


def recovery(F: float, R: float, h: float, r: float, t: float) -> float:
    a = r + h
    integral = -math.expm1(-a * t) / a if abs(a) > 1e-12 else t
    return R * F * h * integral


def analytic(S=50., F=100., m=2., T=3., r=.04, q=.01,
             sigma=.30, h=.025, R=.40) -> float:
    """European, zero-coupon convertible; m=0 returns the straight bond."""
    validate(S, F, m, T, r, q, sigma, h, R)
    a = r + h
    bond = F * math.exp(-a * T) + recovery(F, R, h, r, T)
    if m == 0:
        return bond
    d1 = (math.log(S / (F / m)) + (a - q + .5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(bond + m * (S * math.exp(-q*T) * ndtr(d1)
                            - (F/m) * math.exp(-a*T) * ndtr(d2)))


def price(S=50., F=100., m=2., T=3., r=.04, q=.01, sigma=.30,
          h=.025, R=.40, N=600, american=False, coupon_rate=0.,
          call_price=None, call_from=1.5, put_price=None, put_time=1.5) -> float:
    """Survival-conditioned CRR tree, per F face amount.

    Synthetic convention: semiannual coupon paid to survivors before each
    same-day decision; call has no notice period and allows immediate conversion.
    Holder put must not exceed issuer call price where both can apply.
    Call is available from call_from to (but not including) terminal maturity.
    Put/coupon event dates must be on the tree grid; no silent date snapping.
    """
    validate(S, F, m, T, r, q, sigma, h, R)
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer")
    if not math.isfinite(coupon_rate) or coupon_rate < 0:
        raise ValueError("Coupon rate must be finite and nonnegative")
    for value in (call_price, put_price):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Exercise prices must be finite and nonnegative")
    if call_price is not None and not 0 <= call_from < T:
        raise ValueError("Call start must precede maturity")
    if put_price is not None and not 0 <= put_time < T:
        raise ValueError("Put date must precede maturity")
    if call_price is not None and put_price is not None and put_price > call_price:
        raise ValueError("Overlapping put > call requires a separate priority rule")
    dt = T/N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1/u
    p = (math.exp((r + h - q) * dt) - d) / (u - d)
    if not 0 <= p <= 1:
        raise ValueError("Probability outside [0,1]; refine grid")

    def date_index(t):
        k = round(t / dt)
        if abs(k * dt - t) > 1e-9:
            raise ValueError("Contract event is off grid")
        return k

    coupons = {}
    if coupon_rate:
        if abs(T/.5 - round(T/.5)) > 1e-9:
            raise ValueError("Synthetic coupon schedule requires whole half-years")
        coupons = {date_index(t): F * coupon_rate * .5
                   for t in np.arange(.5, T+1e-9, .5)}
    put_index = date_index(put_time) if put_price is not None else None
    call_index = date_index(call_from) if call_price is not None else None
    stocks = S * np.exp((2 * np.arange(N+1) - N) * math.log(u))
    values = np.maximum(F, m * stocks) + coupons.get(N, 0.)
    discount = math.exp(-(r+h) * dt)
    recovery_step = recovery(F, R, h, r, dt)
    for i in range(N-1, -1, -1):
        stocks = S * np.exp((2 * np.arange(i+1) - i) * math.log(u))
        conversion = m * stocks
        values = discount * (p * values[1:] + (1-p) * values[:-1]) + recovery_step
        if call_index is not None and i >= call_index:
            values = np.minimum(values, np.maximum(call_price, conversion))
        if i == put_index:
            values = np.maximum(values, put_price)
        if american:
            values = np.maximum(values, conversion)
        values = values + coupons.get(i, 0.)
    return float(values[0])


def run_experiments() -> dict:
    out = {"parameters": BASE, "unit": "per 100 face, synthetic only"}
    out['analytic'] = {
        "risk_free_european": analytic(**(BASE | {'h': 0})),
        "defaultable_european": analytic(**BASE),
        "defaultable_straight": analytic(**(BASE | {'m': 0})),
        "conversion_parity": BASE['m'] * BASE['S'],
    }
    out['convergence'] = [dict(N=n, risk_free=price(**(BASE | {'h': 0}), N=n),
                              defaultable=price(**BASE, N=n))
                          for n in [60, 120, 300, 600, 1200, 2400]]
    scenarios = [
        ('European, zero coupon', {}),
        ('American, zero coupon', {'american': True}),
        ('American, 2% coupon', {'american': True, 'coupon_rate': .02}),
        ('American, 2% coupon, call 105 from 1.5y',
         {'american': True, 'coupon_rate': .02, 'call_price': 105}),
        ('American, 2% coupon, put 100 at 1.5y',
         {'american': True, 'coupon_rate': .02, 'put_price': 100}),
        ('American, coupon, call and put',
         {'american': True, 'coupon_rate': .02, 'call_price': 105, 'put_price': 100}),
    ]
    out['scenarios'] = [dict(name=name, price_600=price(**BASE, N=600, **kw),
                             price_1200=price(**BASE, N=1200, **kw))
                        for name, kw in scenarios]
    out['greeks'] = {}
    v = analytic(**BASE)
    for name, key, bump in [('delta', 'S', .05), ('vega_1volpt', 'sigma', .01),
                            ('rate_1bp', 'r', .0001), ('hazard_1bp', 'h', .0001)]:
        up = analytic(**(BASE | {key: BASE[key] + bump}))
        dn = analytic(**(BASE | {key: BASE[key] - bump}))
        out['greeks'][name] = (up-dn)/(2*bump) if name == 'delta' else (up-dn)/2
    out['greeks']['gamma'] = (analytic(**(BASE | {'S': 50.05}))
                               + analytic(**(BASE | {'S': 49.95})) - 2*v)/.05**2
    out['greeks']['jump_hedged_pnl'] = BASE['R']*BASE['F'] - v + out['greeks']['delta']*BASE['S']
    s = out['scenarios']
    checks = {
        'analytic_match_at_2400_abs_error_lt_0_003':
            abs(out['convergence'][-1]['defaultable'] - v) < .003,
        'no_conversion_matches_bond_1e_10':
            abs(price(**(BASE | {'m': 0})) - analytic(**(BASE | {'m': 0}))) < 1e-10,
        'american_ge_european': s[1]['price_600'] >= s[0]['price_600'],
        'call_le_noncall': s[3]['price_600'] <= s[2]['price_600'],
        'put_ge_no_put': s[4]['price_600'] >= s[2]['price_600'],
        'call_plus_put_between': s[3]['price_600'] <= s[5]['price_600'] <= s[4]['price_600'],
        'american_ge_immediate_conversion':
            all(price(**(BASE | {'S': x}), american=True, N=120) >= 2*x-1e-10
                for x in [10, 25, 50, 100, 150]),
        'coupon_bond_matches_discounted_coupon_cashflows':
            abs(price(**(BASE | {'m': 0}), coupon_rate=.02)
                - analytic(**(BASE | {'m': 0}))
                - sum(math.exp(-(.04+.025)*i/2) for i in range(1, 7))) < 1e-10,
        'default_free_call_conv_no_dividend_no_early_premium':
            abs(price(**(BASE | {'h': 0, 'q': 0}), american=True)
                - price(**(BASE | {'h': 0, 'q': 0}))) < 1e-9,
    }
    try:
        price(**(BASE | {'h': 5}), N=1)
        checks['invalid_probability_rejected'] = False
    except ValueError:
        checks['invalid_probability_rejected'] = True
    out['checks'] = {k: bool(v) for k, v in checks.items()}
    assert all(checks.values()), checks
    return out


if __name__ == '__main__':
    result = run_experiments()
    (Path(__file__).resolve().parent / 'results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
