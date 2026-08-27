# 13. Numerical SDE Implementation Details

## 13.1 Brownian Correlation

If

$$
dX_t=\mu dt+L_tdZ_t,
\qquad d\langle Z\rangle_t=I dt,
$$

then covariance is $L_tL_t^\top dt$. If instead $W$ is correlated with

$$
d\langle W\rangle_t=Rdt,
$$

then covariance is $\Sigma R\Sigma^\top dt$. Correlation must not be encoded both in the loading matrix and again in the Brownian correlation matrix.

A numerical implementation should verify that $R$ is symmetric, has unit diagonal, and is positive semidefinite. A Cholesky failure should not be hidden by an arbitrarily large diagonal jitter.

## 13.2 Strong and Weak Error

- **Strong convergence** controls pathwise error and matters for pathwise Greeks, barriers, hitting times, and exposure trajectories.
- **Weak convergence** controls expectation error and is often the relevant criterion for European prices.

Euler--Maruyama has strong order $1/2$ and, under suitable smoothness, weak order 1. Exact endpoint transitions remove endpoint discretization bias but not path-integral, early-exercise, or interpolation error.

## 13.3 Exact Gaussian Step

```python
from __future__ import annotations

import math
import numpy as np


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
```

## 13.4 Exact CIR Step

```python
from __future__ import annotations

import math
import numpy as np


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
    noncentrality = (
        4.0 * kappa * decay * r_t
        / (sigma**2 * one_minus_decay)
    )
    draw = rng.noncentral_chisquare(degrees_of_freedom, noncentrality)
    return float(scale * draw)
```

## 13.5 Exact G2++ Endpoint Step

```python
from __future__ import annotations

import math
import numpy as np


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
    cov_xy = (
        rho * sigma * eta
        * (1.0 - math.exp(-(a + b) * dt))
        / (a + b)
    )
    covariance = np.array([[var_x, cov_xy], [cov_xy, var_y]])
    shock_x, shock_y = rng.multivariate_normal(np.zeros(2), covariance)

    x_next = math.exp(-a * dt) * x_t + shock_x
    y_next = math.exp(-b * dt) * y_t + shock_y
    return float(x_next), float(y_next)
```

## 13.6 HJM and LMM Validation Tests

For HJM:

- numerically integrated $f$, $P$, and $r$ must remain mutually consistent;
- under $\mathbb Q$, the sample mean of $P(t,T)/B_t$ should have no systematic drift;
- prices should converge under maturity-grid refinement;
- factor shocks must be shared across maturities.

For LMM:

- under each $T_{i+1}$-forward measure, the sample mean of $L_i$ should remain constant;
- terminal-measure prices should agree with natural-measure prices when the full drift is used;
- the recursion

  $$
  P(t,T_i)=P(t,T_{i+1})[1+\delta_iL_i(t)]
  $$

  should remain numerically consistent;
- the correlation matrix and factor loadings must remain PSD at every time point.

# 14. Model-Selection Map

| Task | Good starting model | Why | Main risk |
|---|---|---|---|
| Teaching, analytical benchmarks | Vasicek | Fully Gaussian, exact transition, closed-form bonds | One factor, negative rates, constant volatility |
| Nonnegative short rates | CIR / CIR++ | Square-root diffusion, noncentral-$\chi^2$ transition | Boundary handling, curve fit, limited smile |
| Callable/Bermudan low-dimensional pricing | Hull--White 1F | Exact initial curve, convenient trees/PDE/MC | Overly restrictive one-factor correlation |
| Richer slope and curvature risk | G2++ | Two Gaussian factors, analytical bond price | No smile, parameter identification |
| Curve modeling from volatility structure | HJM | No-arbitrage drift fixed by volatility | Infinite dimension, discretization and calibration |
| Caps, swaptions, and rates exotics | LMM/BGM and extensions | Direct market-forward dynamics | Multiple-measure drift, negative rates, high dimension |
| Yield forecasting | DNS / VAR / ML factor model | Parsimonious and forecast oriented | Not automatically arbitrage free |
| Term premia and joint historical/pricing analysis | Gaussian ATSM, AFNS, ACM/JSZ | Explicit $\mathbb P$ versus $\mathbb Q$ distinction | Persistent-state and price-of-risk identification |

There is no universally best term-structure model. The right question is which observables must be matched, under which measure the model will be used, whether early exercise and smiles matter, and how many state variables the computational budget permits.

# 15. Recommended Learning Sequence

1. Derive $P(t,T)=\mathbb E^Q[e^{-\int r}]$ and the pricing PDE independently.
2. Derive the exact Vasicek transition, integrated-rate variance, and $A/B$ bond formula.
3. Derive the CIR Riccati equation and understand the Feller boundary and noncentral-$\chi^2$ transition.
4. Derive the Hull--White shift $\phi(t)$ and drift function $\theta(t)$ rather than memorizing them.
5. Reproduce the HJM drift restriction in full; it is the central no-arbitrage result for rates models.
6. Derive the sign and summation range in the LMM terminal-measure drift from numeraire change.
7. Only then move to calibration, state-space estimation, AFNS/ACM, and smile extensions.

# 16. Conclusion

The unified logic of term-structure modeling is

$$
\text{state or curve SDE under }\mathbb P
\quad\xrightarrow{\text{price of risk}}\quad
\text{SDE under }\mathbb Q,
$$

$$
\text{numeraire plus no arbitrage}
\quad\Longrightarrow\quad
\text{pricing drift restriction},
$$

$$
P(t,T)
=
\mathbb E_t^{\mathbb Q}\left[e^{-\int_t^T r_sds}\right],
$$

and

$$
\text{initial-curve fit}
\neq
\text{historical-dynamics fit}
\neq
\text{option-volatility fit}.
$$

Vasicek and CIR show how a short-rate SDE generates a full term structure. Hull--White and G2++ show how deterministic shifts reproduce today's curve. HJM shows how forward volatility determines risk-neutral drift. LMM shows how to model market forwards under natural forward measures. A reliable implementation must preserve the measure, numeraire, correlation, boundary, integration, and discretization details—not just the headline SDE.
