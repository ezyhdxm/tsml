"""Numerical consistency checks for the rates term-structure report.

The script uses only synthetic parameters and public mathematical formulas.
It is not a pricing library; it is a compact regression test for the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Check:
    name: str
    error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return bool(self.error <= self.tolerance)


def vasicek_exact_step(
    r_t: float,
    dt: float,
    kappa: float,
    theta_q: float,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    if dt < 0 or kappa <= 0 or sigma < 0:
        raise ValueError("Require dt >= 0, kappa > 0, sigma >= 0.")
    decay = math.exp(-kappa * dt)
    mean = theta_q + (r_t - theta_q) * decay
    variance = sigma**2 * (1.0 - decay**2) / (2.0 * kappa)
    return float(mean + math.sqrt(max(variance, 0.0)) * rng.standard_normal())


def cir_exact_step(
    r_t: float,
    dt: float,
    kappa: float,
    theta_q: float,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    if r_t < 0 or dt < 0 or kappa <= 0 or theta_q < 0 or sigma <= 0:
        raise ValueError("Invalid CIR inputs.")
    if dt == 0:
        return float(r_t)

    decay = math.exp(-kappa * dt)
    one_minus_decay = 1.0 - decay
    scale = sigma**2 * one_minus_decay / (4.0 * kappa)
    degrees_of_freedom = 4.0 * kappa * theta_q / sigma**2
    noncentrality = 4.0 * kappa * decay * r_t / (
        sigma**2 * one_minus_decay
    )
    draw = rng.noncentral_chisquare(degrees_of_freedom, noncentrality)
    return float(scale * draw)


def g2pp_exact_step(
    x_t: float,
    y_t: float,
    dt: float,
    a: float,
    b: float,
    sigma: float,
    eta: float,
    rho: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if dt < 0 or a <= 0 or b <= 0 or sigma < 0 or eta < 0:
        raise ValueError("Invalid G2++ inputs.")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1].")

    var_x = sigma**2 * (1.0 - math.exp(-2.0 * a * dt)) / (2.0 * a)
    var_y = eta**2 * (1.0 - math.exp(-2.0 * b * dt)) / (2.0 * b)
    cov_xy = rho * sigma * eta * (
        1.0 - math.exp(-(a + b) * dt)
    ) / (a + b)
    covariance = np.array([[var_x, cov_xy], [cov_xy, var_y]])
    shock_x, shock_y = rng.multivariate_normal(np.zeros(2), covariance)

    x_next = math.exp(-a * dt) * x_t + shock_x
    y_next = math.exp(-b * dt) * y_t + shock_y
    return float(x_next), float(y_next)


def _relative_error(actual: float, expected: float, floor: float = 1e-12) -> float:
    return abs(actual - expected) / max(abs(expected), floor)


def _vasicek_bond(t: float, maturity: float, r: float, kappa: float, theta: float, sigma: float) -> float:
    tau = maturity - t
    b = (1.0 - math.exp(-kappa * tau)) / kappa
    log_a = (
        (theta - sigma**2 / (2.0 * kappa**2)) * (b - tau)
        - sigma**2 * b**2 / (4.0 * kappa)
    )
    return math.exp(log_a - b * r)


def _cir_bond(t: float, maturity: float, r: float, kappa: float, theta: float, sigma: float) -> float:
    tau = maturity - t
    gamma = math.sqrt(kappa**2 + 2.0 * sigma**2)
    eg = math.exp(gamma * tau)
    denominator = (gamma + kappa) * (eg - 1.0) + 2.0 * gamma
    b = 2.0 * (eg - 1.0) / denominator
    a = (
        2.0 * gamma * math.exp((kappa + gamma) * tau / 2.0)
        / denominator
    ) ** (2.0 * kappa * theta / sigma**2)
    return a * math.exp(-b * r)


def _finite_difference_pde_residual(
    price,
    t: float,
    maturity: float,
    state: float,
    drift: float,
    variance_rate: float,
    short_rate: float,
    h_t: float = 1e-5,
    h_x: float = 1e-5,
) -> float:
    v = price(t, maturity, state)
    v_t = (price(t + h_t, maturity, state) - price(t - h_t, maturity, state)) / (2.0 * h_t)
    v_x = (price(t, maturity, state + h_x) - price(t, maturity, state - h_x)) / (2.0 * h_x)
    v_xx = (
        price(t, maturity, state + h_x)
        - 2.0 * v
        + price(t, maturity, state - h_x)
    ) / h_x**2
    residual = v_t + drift * v_x + 0.5 * variance_rate * v_xx - short_rate * v
    return abs(residual) / max(abs(v), 1e-12)


def run_checks(seed: int = 20260827, n_paths: int = 300_000) -> list[Check]:
    rng = np.random.default_rng(seed)
    checks: list[Check] = []

    # Vasicek exact transition moments.
    r0, dt, kappa, theta, sigma = 0.037, 0.75, 0.42, 0.031, 0.014
    decay = math.exp(-kappa * dt)
    mean = theta + (r0 - theta) * decay
    variance = sigma**2 * (1.0 - decay**2) / (2.0 * kappa)
    draws = mean + math.sqrt(variance) * rng.standard_normal(n_paths)
    checks.append(Check("Vasicek exact mean", _relative_error(float(draws.mean()), mean), 2.5e-3))
    checks.append(Check("Vasicek exact variance", _relative_error(float(draws.var()), variance), 8.0e-3))

    # CIR exact transition moments.
    r0, dt, kappa, theta, sigma = 0.026, 0.6, 0.55, 0.038, 0.16
    decay = math.exp(-kappa * dt)
    scale = sigma**2 * (1.0 - decay) / (4.0 * kappa)
    df = 4.0 * kappa * theta / sigma**2
    nc = 4.0 * kappa * decay * r0 / (sigma**2 * (1.0 - decay))
    draws = scale * rng.noncentral_chisquare(df, nc, size=n_paths)
    mean = theta + (r0 - theta) * decay
    variance = (
        sigma**2 * r0 * decay * (1.0 - decay) / kappa
        + theta * sigma**2 * (1.0 - decay) ** 2 / (2.0 * kappa)
    )
    checks.append(Check("CIR exact mean", _relative_error(float(draws.mean()), mean), 5.0e-3))
    checks.append(Check("CIR exact variance", _relative_error(float(draws.var()), variance), 1.5e-2))
    checks.append(Check("CIR nonnegativity", float(max(0.0, -draws.min())), 0.0))

    # G2++ endpoint innovation covariance.
    dt, a, b, sig, eta, rho = 0.8, 0.17, 0.61, 0.012, 0.018, -0.42
    var_x = sig**2 * (1.0 - math.exp(-2.0 * a * dt)) / (2.0 * a)
    var_y = eta**2 * (1.0 - math.exp(-2.0 * b * dt)) / (2.0 * b)
    cov_xy = rho * sig * eta * (1.0 - math.exp(-(a + b) * dt)) / (a + b)
    cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])
    shocks = rng.multivariate_normal(np.zeros(2), cov, size=n_paths)
    empirical = np.cov(shocks, rowvar=False, ddof=0)
    checks.append(Check("G2++ Var[x]", _relative_error(float(empirical[0, 0]), var_x), 1.2e-2))
    checks.append(Check("G2++ Var[y]", _relative_error(float(empirical[1, 1]), var_y), 1.2e-2))
    checks.append(Check("G2++ Cov[x,y]", _relative_error(float(empirical[0, 1]), cov_xy), 2.0e-2))

    # Vasicek and CIR bond formulas satisfy their pricing PDEs.
    vp = lambda t, T, r: _vasicek_bond(t, T, r, 0.31, 0.042, 0.017)
    residual = _finite_difference_pde_residual(
        vp, 1.2, 7.0, 0.035, 0.31 * (0.042 - 0.035), 0.017**2, 0.035
    )
    checks.append(Check("Vasicek bond PDE", residual, 2.0e-6))

    cp = lambda t, T, r: _cir_bond(t, T, r, 0.63, 0.041, 0.12)
    residual = _finite_difference_pde_residual(
        cp, 0.9, 6.5, 0.032, 0.63 * (0.041 - 0.032), 0.12**2 * 0.032, 0.032
    )
    checks.append(Check("CIR bond PDE", residual, 2.0e-6))

    # HJM drift is the maturity derivative of one half the squared bond loading.
    t = 0.4
    def sigma_hjm(T: float) -> np.ndarray:
        tau = T - t
        return np.array([0.011 * math.exp(-0.22 * tau), 0.007 * math.exp(-0.73 * tau)])

    def integral_sigma(T: float) -> np.ndarray:
        tau = T - t
        return np.array([
            0.011 * (1.0 - math.exp(-0.22 * tau)) / 0.22,
            0.007 * (1.0 - math.exp(-0.73 * tau)) / 0.73,
        ])

    T = 8.0
    analytical = float(sigma_hjm(T) @ integral_sigma(T))
    h = 1e-5
    primitive = lambda u: 0.5 * float(integral_sigma(u) @ integral_sigma(u))
    numerical = (primitive(T + h) - primitive(T - h)) / (2.0 * h)
    checks.append(Check("HJM drift restriction", _relative_error(numerical, analytical), 2.0e-8))

    # Deterministic shifts exactly recover an arbitrary synthetic initial curve.
    grid = np.linspace(0.0, 20.0, 200_001)
    f0 = 0.022 + 0.006 * np.exp(-grid / 4.0) + 0.0004 * grid
    market_log_discount = -float(np.trapezoid(f0, grid))

    a, sig = 0.19, 0.013
    phi_hw = f0 + sig**2 * (1.0 - np.exp(-a * grid)) ** 2 / (2.0 * a**2)
    variance_hw = sig**2 / a**2 * (
        grid[-1]
        - 2.0 * (1.0 - math.exp(-a * grid[-1])) / a
        + (1.0 - math.exp(-2.0 * a * grid[-1])) / (2.0 * a)
    )
    hw_log_discount = -float(np.trapezoid(phi_hw, grid)) + 0.5 * variance_hw
    checks.append(Check("Hull-White initial-curve fit", abs(hw_log_discount - market_log_discount), 2.0e-10))

    a, b, sig, eta, rho = 0.17, 0.68, 0.012, 0.019, -0.35
    q = (
        sig**2 / a**2 * (1.0 - np.exp(-a * grid)) ** 2
        + eta**2 / b**2 * (1.0 - np.exp(-b * grid)) ** 2
        + 2.0 * rho * sig * eta / (a * b)
        * (1.0 - np.exp(-a * grid)) * (1.0 - np.exp(-b * grid))
    )
    phi_g2 = f0 + 0.5 * q
    tau = grid[-1]
    variance_g2 = (
        sig**2 / a**2 * (tau - 2.0 * (1.0 - math.exp(-a * tau)) / a + (1.0 - math.exp(-2.0 * a * tau)) / (2.0 * a))
        + eta**2 / b**2 * (tau - 2.0 * (1.0 - math.exp(-b * tau)) / b + (1.0 - math.exp(-2.0 * b * tau)) / (2.0 * b))
        + 2.0 * rho * sig * eta / (a * b)
        * (tau - (1.0 - math.exp(-a * tau)) / a - (1.0 - math.exp(-b * tau)) / b + (1.0 - math.exp(-(a + b) * tau)) / (a + b))
    )
    g2_log_discount = -float(np.trapezoid(phi_g2, grid)) + 0.5 * variance_g2
    checks.append(Check("G2++ initial-curve fit", abs(g2_log_discount - market_log_discount), 2.0e-10))

    return checks


def main() -> None:
    checks = run_checks()
    width = max(len(check.name) for check in checks)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{status}  {check.name:<{width}}  error={check.error:.3e}  tol={check.tolerance:.3e}")
    failed = [check for check in checks if not check.passed]
    if failed:
        raise SystemExit(f"{len(failed)} validation check(s) failed")
    print(f"All {len(checks)} numerical checks passed.")


if __name__ == "__main__":
    main()
