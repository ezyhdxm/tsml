---
title: "Interest-Rate Term-Structure Models: From Short-Rate SDEs to Affine Pricing, HJM, and Market Models"
subtitle: "A rigorous guide to measures, drift restrictions, analytical pricing, calibration, and simulation"
author: "TSML Research Notes"
date: "2026-08-27"
lang: en-US
bibliography: references.bib
link-citations: true
reference-section-title: "References"
toc-title: "Contents"
---

<nav class="language-switch"><strong>Language:</strong> <a href="index.html">中文</a> · <span aria-current="page">English</span></nav>

# Abstract

A “term-structure model” is not a single problem. At least four tasks must be separated:

1. **Representing and bootstrapping today's curve:** constructing discount factors $P(0,T)$, zero rates, and forward rates from market quotes.
2. **Modeling future curve dynamics under the physical measure:** forecasting, risk management, scenario generation, and term-premium analysis.
3. **Pricing derivatives under a risk-neutral measure:** ensuring that tradable assets divided by the chosen numeraire are martingales.
4. **Matching option-market volatility:** deciding how many short-rate factors, forward-rate factors, correlations, and volatility factors are required.

This report starts from a common no-arbitrage framework and gives the SDEs, measure changes, pricing equations, analytical solutions, boundary conditions, exact transition laws, and discretization details for Vasicek, CIR, Hull--White, G2++, HJM, and LMM/BGM. It then connects these pricing models to Dynamic Nelson--Siegel, Gaussian affine term-structure models, AFNS, and term-premium estimation.

The central messages are:

- **Short-rate models** compress the curve into a small Markov state, enabling PDE, tree, and low-dimensional Monte Carlo methods.
- **HJM** models the full instantaneous-forward curve; once volatility is specified, the risk-neutral drift is fixed by no arbitrage.
- **LMM/BGM** models market forward rates on a discrete tenor and makes each forward a martingale under its own forward measure.
- **Physical-measure and risk-neutral drifts must not be mixed.** The former governs forecasting; the latter governs pricing. Their difference is the price of risk.
- **Exact fit to today's curve, accurate historical dynamics, and fit to the option smile are distinct identification problems.**

Notation: $t$ is current time, $T$ is maturity, $\tau=T-t$ is time to maturity, $P(t,T)$ is the price of a default-free zero-coupon bond paying one at $T$, and $W$ is Brownian motion. Unless stated otherwise, pricing SDEs are written under the money-market risk-neutral measure $\mathbb Q$.

# 1. What a Complete Interest-Rate SDE Specification Must Contain

Writing only

$$
dr_t=\kappa(\theta-r_t)dt+\sigma dW_t
$$

is incomplete. A production-quality model definition should state at least:

1. **Probability space and filtration:** $(\Omega,\mathcal F,(\mathcal F_t)_{t\ge0},\mathbb M)$.
2. **Measure:** physical $\mathbb P$, money-market risk-neutral $\mathbb Q$, a $T$-forward measure $\mathbb Q^T$, terminal measure, or another equivalent martingale measure.
3. **Numeraire:** money-market account, zero-coupon bond, or rolling spot numeraire.
4. **State variables and initial condition:** $r_0$, $X_0$, or the initial forward curve $f(0,T)$.
5. **Brownian dimension:** one factor or multiple factors.
6. **Correlation structure:** $d\langle W^i,W^j\rangle_t=\rho_{ij}dt$, or an equivalent loading-matrix representation.
7. **Drift and diffusion:** dependence on time, state, and maturity.
8. **Parameter restrictions and boundary behavior:** mean reversion, positivity, the Feller condition, nonexplosion, and existence of a strong solution.
9. **Pricing restriction:** discounted tradables must be martingales; HJM drift restrictions must hold where applicable.
10. **Numerical scheme:** exact transition, Euler, log-Euler, full truncation, path integration, and generation of correlated shocks.

The remainder of the report follows this checklist.

# 2. Term-Structure Objects and No-Arbitrage Foundations

## 2.1 Discount Factors, Zero Rates, and Instantaneous Forwards

A zero-coupon bond satisfies

$$
P(t,T)>0,\qquad P(T,T)=1.
$$

The continuously compounded zero rate is

$$
y(t,T)=-\frac{\log P(t,T)}{T-t}.
$$

The instantaneous forward rate is

$$
f(t,T)=-\partial_T\log P(t,T).
$$

Hence

$$
P(t,T)=\exp\left(-\int_t^T f(t,u)du\right),
$$

and the short rate is the diagonal of the forward surface:

$$
r_t=f(t,t).
$$

These objects must be mutually consistent. Common implementation errors include treating a simple-compounded market quote as a continuously compounded zero rate, mixing day-count conventions, or comparing forwards built with inconsistent payment lags and business-day adjustments.

## 2.2 Money-Market Numeraire and Risk-Neutral Pricing

Define the money-market account

$$
B_t=\exp\left(\int_0^t r_sds\right),
\qquad dB_t=r_tB_tdt.
$$

Under the measure $\mathbb Q$ associated with $B_t$, a tradable asset divided by $B_t$ is a local martingale. Under standard integrability conditions,

$$
P(t,T)
=
\mathbb E_t^{\mathbb Q}\left[
\exp\left(-\int_t^T r_sds\right)
\right].
$$

This identity is the starting point of every short-rate model.

## 2.3 Physical and Risk-Neutral Measures

Suppose a $d$-dimensional state follows, under $\mathbb P$,

$$
dX_t
=
\mu^{\mathbb P}(t,X_t)dt
+
\Sigma(t,X_t)dW_t^{\mathbb P},
$$

where $W^{\mathbb P}$ is an $m$-dimensional standard Brownian motion. Let $\lambda_t\in\mathbb R^m$ denote the market price of risk. Under conditions such as Novikov's condition,

$$
\left.\frac{d\mathbb Q}{d\mathbb P}\right|_{\mathcal F_t}
=
\exp\left(
-\int_0^t\lambda_s^\top dW_s^{\mathbb P}
-\frac12\int_0^t\|\lambda_s\|^2ds
\right),
$$

and

$$
dW_t^{\mathbb Q}=dW_t^{\mathbb P}+\lambda_tdt.
$$

Therefore

$$
dX_t
=
\underbrace{\left(\mu^{\mathbb P}-\Sigma\lambda\right)}_{\mu^{\mathbb Q}}dt
+
\Sigma dW_t^{\mathbb Q},
$$

so that

$$
\boxed{
\mu^{\mathbb Q}=\mu^{\mathbb P}-\Sigma\lambda
}.
$$

Equivalent measure changes normally preserve diffusion and alter drift. Historical forecasting requires $\mathbb P$-dynamics; derivative valuation requires $\mathbb Q$-dynamics. A risk-neutral long-run mean calibrated from options is generally not a historical forecast.

## 2.4 Pricing PDE

If $r_t=r(t,X_t)$ and the terminal payoff is $g(X_T)$, then

$$
V(t,x)
=
\mathbb E_{t,x}^{\mathbb Q}\left[
\exp\left(-\int_t^T r(s,X_s)ds\right)g(X_T)
\right].
$$

Let $a=\Sigma\Sigma^\top$. Feynman--Kac gives

$$
\partial_tV
+(\mu^{\mathbb Q})^\top\nabla_xV
+\frac12\operatorname{tr}\left(a\nabla_x^2V\right)
-rV=0,
$$

with $V(T,x)=g(x)$. A zero-coupon bond has $g\equiv1$.

# 3. The Unified Affine Term-Structure Form

## 3.1 Affine Diffusion and Exponential-Affine Bond Prices

Let the risk-neutral state satisfy

$$
dX_t=(b_0+BX_t)dt+\Sigma(X_t)dW_t^{\mathbb Q},
$$

with affine covariance

$$
a(x)=\Sigma(x)\Sigma(x)^\top
=a_0+\sum_{i=1}^d x_i a_i,
$$

and affine short rate

$$
r_t=\delta_0+\delta_1^\top X_t.
$$

Under admissibility conditions, bond prices take the exponential-affine form [@duffiekan1996]

$$
P(t,T)=\exp\left(\phi(\tau)+\psi(\tau)^\top X_t\right),
\qquad \tau=T-t.
$$

Substitution into the pricing PDE yields

$$
\phi'(\tau)
=b_0^\top\psi(\tau)
+\frac12\psi(\tau)^\top a_0\psi(\tau)
-\delta_0,
$$

and, for $i=1,\ldots,d$,

$$
\psi_i'(\tau)
=\left(B^\top\psi(\tau)\right)_i
+\frac12\psi(\tau)^\top a_i\psi(\tau)
-\delta_{1,i},
$$

with

$$
\phi(0)=0,\qquad \psi(0)=0.
$$

Vasicek and CIR are one-dimensional special cases. In Gaussian ATSMs, $a_i=0$ and the Riccati system becomes linear. In CIR-type models, state-dependent variance produces the quadratic term.

## 3.2 Affine Does Not Mean Every Parameter Is Admissible

If a state must remain nonnegative, drift at the boundary must point inward and the diffusion cannot instantly push the process outside its state space. Dai and Singleton classify the restrictions that affine families impose on conditional variances, correlations, and prices of risk [@daisingleton2000]. It is useful to separate:

- mathematical admissibility;
- cross-sectional yield fit;
- time-series identification;
- option-volatility fit;
- economic interpretation.
