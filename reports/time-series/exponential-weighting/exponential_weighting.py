"""Exponentially weighted estimators used by the accompanying TSML report.

The module deliberately separates three objects that are often conflated:

1. a normalized exponentially weighted average of observed samples;
2. a recursive exponential smoother interpreted as a dynamic state;
3. a RiskMetrics-style conditional covariance recursion.

Only NumPy and pandas are required.  The functions are written for clarity and
point-in-time use rather than for maximum throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, log, sqrt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ArrayLike1D = Sequence[float] | np.ndarray | pd.Series


@dataclass(frozen=True)
class WeightGeometry:
    """Interpretable summaries of an infinite geometric weighting kernel."""

    beta: float
    alpha: float
    half_life: float
    mean_age: float
    effective_sample_size: float
    horizon_95pct: int
    horizon_99pct: int


def _validate_beta(beta: float) -> float:
    beta = float(beta)
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly between 0 and 1")
    return beta


def beta_from_alpha(alpha: float) -> float:
    """Convert smoothing gain alpha to retention/forgetting factor beta."""
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    if alpha == 1.0:
        return 0.0
    return 1.0 - alpha


def alpha_from_beta(beta: float) -> float:
    """Convert retention factor beta to smoothing gain alpha."""
    return 1.0 - _validate_beta(beta)


def beta_from_half_life(half_life: float) -> float:
    """Return beta such that an observation's weight halves after half_life periods."""
    half_life = float(half_life)
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    return exp(-log(2.0) / half_life)


def alpha_from_half_life(half_life: float) -> float:
    """Return the regular-grid smoothing gain associated with a half-life."""
    return 1.0 - beta_from_half_life(half_life)


def half_life_from_beta(beta: float) -> float:
    """Return the geometric-weight half-life in observation periods."""
    beta = _validate_beta(beta)
    return log(0.5) / log(beta)


def mean_age(beta: float) -> float:
    """Mean lag under infinite normalized weights (1-beta) beta**j."""
    beta = _validate_beta(beta)
    return beta / (1.0 - beta)


def effective_sample_size(beta: float) -> float:
    """Kish effective sample size of infinite geometric weights."""
    beta = _validate_beta(beta)
    return (1.0 + beta) / (1.0 - beta)


def tail_mass_horizon(beta: float, remaining_mass: float = 0.01) -> int:
    """Smallest N for which the total weight at lags N, N+1, ... is <= epsilon."""
    beta = _validate_beta(beta)
    remaining_mass = float(remaining_mass)
    if not 0.0 < remaining_mass < 1.0:
        raise ValueError("remaining_mass must lie strictly between 0 and 1")
    return int(ceil(log(remaining_mass) / log(beta)))


def weight_geometry(*, beta: float | None = None, half_life: float | None = None) -> WeightGeometry:
    """Return a compact set of memory diagnostics for one decay parameter."""
    if (beta is None) == (half_life is None):
        raise ValueError("provide exactly one of beta or half_life")
    beta_value = beta_from_half_life(half_life) if half_life is not None else _validate_beta(beta)  # type: ignore[arg-type]
    return WeightGeometry(
        beta=beta_value,
        alpha=1.0 - beta_value,
        half_life=half_life_from_beta(beta_value),
        mean_age=mean_age(beta_value),
        effective_sample_size=effective_sample_size(beta_value),
        horizon_95pct=tail_mass_horizon(beta_value, 0.05),
        horizon_99pct=tail_mass_horizon(beta_value, 0.01),
    )


def _as_finite_vector(values: ArrayLike1D) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if x.size == 0:
        return x.copy()
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite; handle missing data explicitly before calling")
    return x


def ewm_mean_regular(
    values: ArrayLike1D,
    *,
    beta: float | None = None,
    alpha: float | None = None,
    half_life: float | None = None,
    adjust: bool = True,
    initial: float | None = None,
) -> np.ndarray:
    """Compute a regular-grid exponentially weighted mean.

    Parameters
    ----------
    adjust=True
        Returns the normalized finite-history weighted average.  This is the
        literal weighted-average estimand and removes start-up mass imbalance.
    adjust=False
        Returns the state recursion m_t = beta*m_{t-1} + (1-beta)*x_t.
        With initial=None, the first observation initializes the state.
    """
    supplied = sum(v is not None for v in (beta, alpha, half_life))
    if supplied != 1:
        raise ValueError("provide exactly one of beta, alpha, or half_life")
    if alpha is not None:
        beta_value = beta_from_alpha(alpha)
        if beta_value == 0.0:
            return _as_finite_vector(values)
    elif half_life is not None:
        beta_value = beta_from_half_life(half_life)
    else:
        beta_value = _validate_beta(beta)  # type: ignore[arg-type]

    x = _as_finite_vector(values)
    if x.size == 0:
        return x.copy()

    out = np.empty_like(x)
    if adjust:
        numerator = 0.0
        denominator = 0.0
        for i, value in enumerate(x):
            numerator = beta_value * numerator + value
            denominator = beta_value * denominator + 1.0
            out[i] = numerator / denominator
        return out

    if initial is None:
        state = float(x[0])
        out[0] = state
        start = 1
    else:
        state = float(initial)
        start = 0

    gain = 1.0 - beta_value
    for i in range(start, x.size):
        state = beta_value * state + gain * float(x[i])
        out[i] = state
    return out


def _time_coordinates(times: Iterable[object], half_life: object) -> tuple[np.ndarray, float]:
    """Convert numeric or datetime times and half-life to common floating units."""
    t_values = list(times)
    if len(t_values) == 0:
        return np.empty(0, dtype=float), 1.0

    t_array = np.asarray(t_values)
    if np.issubdtype(t_array.dtype, np.datetime64) or isinstance(t_values[0], (pd.Timestamp, np.datetime64)):
        t_ns = pd.to_datetime(t_values).view("int64").astype(float)
        h_ns = float(pd.Timedelta(half_life).value)
        if h_ns <= 0.0:
            raise ValueError("half_life must be positive")
        return t_ns, h_ns

    t = np.asarray(t_values, dtype=float)
    h = float(half_life)
    if h <= 0.0:
        raise ValueError("half_life must be positive")
    return t, h


def ewm_mean_irregular(
    times: Iterable[object],
    values: ArrayLike1D,
    *,
    half_life: object,
    adjusted_observation_weights: bool = True,
    initial: float | None = None,
) -> np.ndarray:
    """Compute one of two clock-time exponential smoothers on irregular observations.

    adjusted_observation_weights=True
        At time t_i, estimate the normalized weighted average with weights
        exp(-log(2)*(t_i-t_j)/half_life) on observed samples j <= i.

    adjusted_observation_weights=False
        Use the dynamic-state recursion
        m_i = d_i*m_{i-1} + (1-d_i)*x_i,
        d_i = exp(-log(2)*(t_i-t_{i-1})/half_life).

    These coincide asymptotically on a regular grid but are different finite-sample
    objects on an irregular grid.  The report explains which interpretation fits
    which question.
    """
    x = _as_finite_vector(values)
    t, h = _time_coordinates(times, half_life)
    if t.size != x.size:
        raise ValueError("times and values must have the same length")
    if x.size == 0:
        return x.copy()
    gaps = np.diff(t)
    if np.any(gaps < 0.0):
        raise ValueError("times must be monotonically nondecreasing")

    out = np.empty_like(x)
    if adjusted_observation_weights:
        numerator = float(x[0])
        denominator = 1.0
        out[0] = numerator
        for i in range(1, x.size):
            decay = exp(-log(2.0) * float(gaps[i - 1]) / h)
            numerator = decay * numerator + float(x[i])
            denominator = decay * denominator + 1.0
            out[i] = numerator / denominator
        return out

    state = float(x[0]) if initial is None else float(initial)
    if initial is None:
        out[0] = state
        start = 1
    else:
        decay0 = 0.0
        state = decay0 * state + (1.0 - decay0) * float(x[0])
        out[0] = state
        start = 1

    for i in range(start, x.size):
        decay = exp(-log(2.0) * float(gaps[i - 1]) / h)
        state = decay * state + (1.0 - decay) * float(x[i])
        out[i] = state
    return out


def ewm_mean_covariance(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    beta: float,
    unbiased: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-history normalized EW mean and covariance at every observation.

    The covariance is the weighted central second moment.  If unbiased=True,
    the fixed-weight iid correction 1/(1-sum(w_i**2)) is applied whenever it
    is defined.  This is a descriptive sample covariance, not a conditional
    volatility model.
    """
    beta = _validate_beta(beta)
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("values must be a finite 1D or 2D array")

    n, p = x.shape
    means = np.empty((n, p), dtype=float)
    covariances = np.empty((n, p, p), dtype=float)
    first_moment = np.zeros(p, dtype=float)
    second_moment = np.zeros((p, p), dtype=float)
    weight_sum = 0.0
    squared_weight_sum = 0.0

    for i, row in enumerate(x):
        first_moment = beta * first_moment + row
        second_moment = beta * second_moment + np.outer(row, row)
        weight_sum = beta * weight_sum + 1.0
        squared_weight_sum = beta * beta * squared_weight_sum + 1.0

        mean = first_moment / weight_sum
        covariance = second_moment / weight_sum - np.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)

        if unbiased:
            sum_normalized_squared = squared_weight_sum / (weight_sum * weight_sum)
            correction_denominator = 1.0 - sum_normalized_squared
            if correction_denominator > np.finfo(float).eps:
                covariance = covariance / correction_denominator
            else:
                covariance[:] = np.nan

        means[i] = mean
        covariances[i] = covariance

    return means, covariances


def riskmetrics_covariance(
    returns: Sequence[Sequence[float]] | np.ndarray,
    *,
    beta: float = 0.94,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """RiskMetrics-style conditional covariance recursion.

    Sigma_t = beta*Sigma_{t-1} + (1-beta)*r_t r_t'.
    The model treats returns as conditionally mean zero.  It is distinct from
    an unbiased sample covariance estimator.
    """
    beta = _validate_beta(beta)
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if r.ndim != 2 or not np.all(np.isfinite(r)):
        raise ValueError("returns must be a finite 1D or 2D array")
    n, p = r.shape
    if initial is None:
        sigma = np.zeros((p, p), dtype=float)
    else:
        sigma = np.asarray(initial, dtype=float).copy()
        if sigma.shape != (p, p):
            raise ValueError("initial covariance has the wrong shape")
        sigma = 0.5 * (sigma + sigma.T)

    out = np.empty((n, p, p), dtype=float)
    gain = 1.0 - beta
    for i, row in enumerate(r):
        sigma = beta * sigma + gain * np.outer(row, row)
        sigma = 0.5 * (sigma + sigma.T)
        out[i] = sigma
    return out


def local_level_optimal_alpha(signal_to_noise: float) -> float:
    """Steady-state Kalman gain for q = process_variance / measurement_variance."""
    q = float(signal_to_noise)
    if q < 0.0:
        raise ValueError("signal_to_noise must be nonnegative")
    if q == 0.0:
        return 0.0
    return 0.5 * (sqrt(q * q + 4.0 * q) - q)


def local_level_state_mse_ratio(alpha: float, signal_to_noise: float) -> float:
    """Post-update state MSE divided by measurement-noise variance.

    This is the stationary MSE of a constant-gain EMA used to track a random
    walk local level with q = Q/R.
    """
    alpha = float(alpha)
    q = float(signal_to_noise)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if q < 0.0:
        raise ValueError("signal_to_noise must be nonnegative")
    return (((1.0 - alpha) ** 2) * q + alpha * alpha) / (alpha * (2.0 - alpha))


def one_step_sse_by_half_life(values: ArrayLike1D, half_lives: Sequence[float]) -> pd.DataFrame:
    """Tune regular-grid SES half-life using point-in-time one-step squared errors."""
    x = _as_finite_vector(values)
    if x.size < 2:
        raise ValueError("at least two observations are required")
    records: list[dict[str, float]] = []
    for half_life in half_lives:
        smoothed = ewm_mean_regular(x, half_life=float(half_life), adjust=False)
        errors = x[1:] - smoothed[:-1]
        records.append(
            {
                "half_life": float(half_life),
                "alpha": alpha_from_half_life(float(half_life)),
                "sse": float(np.dot(errors, errors)),
                "mse": float(np.mean(errors * errors)),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("mse", ignore_index=True)


def exponential_expert_weights(
    losses: Sequence[Sequence[float]] | np.ndarray,
    *,
    eta: float,
    prior: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Hedge/exponential-weights probabilities before each period's loss is seen.

    This weights experts by cumulative loss.  It is not an EMA over observations,
    even though both constructions use exponentials.
    """
    loss = np.asarray(losses, dtype=float)
    if loss.ndim != 2 or not np.all(np.isfinite(loss)):
        raise ValueError("losses must be a finite T-by-K array")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    _, k = loss.shape
    if prior is None:
        log_weight = np.full(k, -log(k), dtype=float)
    else:
        p = np.asarray(prior, dtype=float)
        if p.shape != (k,) or np.any(p <= 0.0):
            raise ValueError("prior must contain K strictly positive entries")
        p = p / p.sum()
        log_weight = np.log(p)

    probabilities = np.empty_like(loss)
    for t in range(loss.shape[0]):
        shifted = log_weight - np.max(log_weight)
        p = np.exp(shifted)
        p /= p.sum()
        probabilities[t] = p
        log_weight -= eta * loss[t]
    return probabilities
