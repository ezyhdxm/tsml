import numpy as np
import pandas as pd

from exponential_weighting import (
    alpha_from_half_life,
    effective_sample_size,
    ewm_mean_covariance,
    ewm_mean_irregular,
    ewm_mean_regular,
    exponential_expert_weights,
    local_level_optimal_alpha,
    local_level_state_mse_ratio,
    riskmetrics_covariance,
    tail_mass_horizon,
)


def test_adjusted_regular_matches_direct_weights() -> None:
    x = np.array([1.0, 4.0, 2.0, 8.0])
    beta = 0.8
    got = ewm_mean_regular(x, beta=beta, adjust=True)
    expected = []
    for t in range(len(x)):
        w = beta ** np.arange(t, -1, -1)
        expected.append(np.dot(w, x[: t + 1]) / w.sum())
    np.testing.assert_allclose(got, expected)


def test_unadjusted_regular_matches_pandas() -> None:
    x = pd.Series([1.0, 4.0, 2.0, 8.0])
    alpha = 0.2
    got = ewm_mean_regular(x.to_numpy(), alpha=alpha, adjust=False)
    expected = x.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(got, expected)


def test_irregular_adjusted_matches_direct_clock_weights() -> None:
    t = np.array([0.0, 1.0, 4.0, 9.0])
    x = np.array([2.0, 1.0, 5.0, 4.0])
    h = 3.0
    got = ewm_mean_irregular(t, x, half_life=h, adjusted_observation_weights=True)
    expected = []
    for i in range(len(x)):
        w = 2.0 ** (-(t[i] - t[: i + 1]) / h)
        expected.append(np.dot(w, x[: i + 1]) / w.sum())
    np.testing.assert_allclose(got, expected)


def test_unbiased_covariance_matches_direct_formula() -> None:
    x = np.array([[1.0, 2.0], [3.0, 0.0], [2.0, 5.0]])
    beta = 0.7
    means, covs = ewm_mean_covariance(x, beta=beta, unbiased=True)
    w = beta ** np.arange(2, -1, -1)
    w /= w.sum()
    mu = np.sum(w[:, None] * x, axis=0)
    centered = x - mu
    direct = (centered.T * w) @ centered / (1.0 - np.dot(w, w))
    np.testing.assert_allclose(means[-1], mu)
    np.testing.assert_allclose(covs[-1], direct)


def test_riskmetrics_covariance_stays_psd() -> None:
    rng = np.random.default_rng(12)
    r = rng.normal(size=(100, 4))
    covs = riskmetrics_covariance(r, beta=0.94)
    assert np.min(np.linalg.eigvalsh(covs[-1])) > -1e-12


def test_local_level_gain_minimizes_mse() -> None:
    q = 0.08
    optimum = local_level_optimal_alpha(q)
    grid = np.linspace(0.001, 0.999, 5000)
    mse = np.array([local_level_state_mse_ratio(a, q) for a in grid])
    assert abs(grid[np.argmin(mse)] - optimum) < 5e-4


def test_geometry_identities() -> None:
    h = 10.0
    alpha = alpha_from_half_life(h)
    beta = 1.0 - alpha
    assert abs(beta**h - 0.5) < 1e-12
    assert effective_sample_size(beta) > h
    assert tail_mass_horizon(beta, 0.01) > h


def test_expert_weights_are_probabilities_and_react_to_loss() -> None:
    losses = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    p = exponential_expert_weights(losses, eta=1.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[-1, 0] > p[0, 0]
