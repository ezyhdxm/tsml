# 10. SOFR, OIS Discounting, and Multiple Curves

SOFR is a broad measure of overnight Treasury-repo financing, and USD rates markets have transitioned from LIBOR conventions toward SOFR and OIS discounting [@nyfedsofr; @arrcsofr].

## 10.1 Idealized Single-Curve Representation

If one OIS curve is used for both discounting and forecasting, an idealized forward compounded overnight rate is

$$
F_i^{OIS}(t)
=
\frac1{\delta_i}
\left[
\frac{P^{OIS}(t,T_i)}{P^{OIS}(t,T_{i+1})}-1
\right].
$$

Actual compounded-in-arrears SOFR is fixed daily through the accrual period and depends on day count, holidays, lookback or lockout, and payment delay. A simple forward ratio does not replace those contractual details.

## 10.2 Multi-Curve Modeling

Collateralized pricing generally distinguishes:

- a discount curve $P^d(t,T)$;
- a forecast pseudo-curve $P^x(t,T)$ for an index or tenor;
- stochastic basis spreads.

One may define

$$
F_i^x(t)
=
\frac1{\delta_i}
\left[
\frac{P^x(t,T_i)}{P^x(t,T_{i+1})}-1
\right],
$$

but the pricing measure is determined by the discount numeraire $P^d(t,T)$. A pseudo-discount curve is generally not a tradable bond curve, so its ratio cannot automatically be treated as a martingale under a “natural” measure. Modern multi-curve HJM/LMM models must explicitly model basis or multiplicative spreads and preserve no arbitrage under the discounting numeraire.

# 11. Forecasting Versus Pricing: DNS, AFNS, and Gaussian ATSMs

## 11.1 Dynamic Nelson--Siegel

The Dynamic Nelson--Siegel representation is

$$
\begin{aligned}
y_t(\tau)
={}&\beta_{1t}
+\beta_{2t}\frac{1-e^{-\lambda\tau}}{\lambda\tau}\\
&+\beta_{3t}
\left[
\frac{1-e^{-\lambda\tau}}{\lambda\tau}
-e^{-\lambda\tau}
\right]
+\varepsilon_t(\tau).
\end{aligned}
$$

The three factors are commonly interpreted as level, slope, and curvature. Diebold and Li evolve the factors with a VAR or related time-series model for forecasting [@dieboldli2006]. A continuous-time version is

$$
d\beta_t
=K^{\mathbb P}(\mu^{\mathbb P}-\beta_t)dt
+\Sigma_\beta dW_t^{\mathbb P}.
$$

This observation equation plus a physical-measure SDE does not automatically eliminate dynamic arbitrage. DNS is useful for forecasting and compression, but should not be used for complex derivative pricing without an arbitrage-free extension.

## 11.2 Gaussian Affine Term-Structure Models

Let

$$
dX_t
=K^{\mathbb P}(\theta^{\mathbb P}-X_t)dt
+\Sigma dW_t^{\mathbb P},
$$

and under $\mathbb Q$,

$$
dX_t
=K^{\mathbb Q}(\theta^{\mathbb Q}-X_t)dt
+\Sigma dW_t^{\mathbb Q}.
$$

With

$$
r_t=\delta_0+\delta_1^\top X_t,
$$

zero yields are affine:

$$
y_t(\tau)=A_y(\tau)+B_y(\tau)^\top X_t.
$$

Differences between $\mathbb P$ and $\mathbb Q$ parameters determine term premia and expected excess bond returns. The main difficulty is identification rather than solving Riccati equations: state rotations, risk-price parameters, and measurement error can generate similar yield fits. JSZ and ACM use observable yield portfolios or regression structure to improve estimation [@joslinsingletonzhu2011; @adriancrumpmoench2013].

## 11.3 AFNS

Arbitrage-Free Nelson--Siegel imposes special risk-neutral dynamics inside a Gaussian ATSM so that factor loadings retain an approximately Nelson--Siegel shape, while an arbitrage-free convexity adjustment is added [@christensendieboldrudebusch2011]. It combines interpretable level/slope/curvature factors with cross-maturity pricing restrictions.

# 12. Calibration, Estimation, and Identification

## 12.1 Recommended Layered Workflow

**Layer 1: Market curves.** Bootstrap $P^M(0,T)$ from OIS, futures, swaps, and other relevant instruments. Record collateral, day count, business-day convention, payment lag, and interpolation variable.

**Layer 2: Today's cross section.** Hull--White and G2++ fit the initial curve through $\phi(t)$. Constant-parameter Vasicek and CIR generally require a deterministic shift to fit an arbitrary curve exactly.

**Layer 3: Risk-neutral volatility.** Calibrate $a,b,\sigma,\eta,\rho$, or HJM/LMM loadings, to liquid cap/floor and swaption prices. Weight residuals by bid--ask, vega, or economically meaningful price scales.

**Layer 4: Physical dynamics.** Estimate $K^{\mathbb P}$, physical long-run means, and innovation covariance from historical yields or factors.

**Layer 5: Prices of risk.** Identify term premia through differences between physical and risk-neutral dynamics. Avoid placing every parameter into one unconstrained nonlinear optimization.

## 12.2 Price, Yield, and Implied-Volatility Errors

A one-basis-point yield error has different price impact at 2Y and 30Y. The same option-price error has different meaning at low and high vega. Possible objectives include

$$
\sum_kw_k(P_k^{model}-P_k^{mkt})^2,
$$

$$
\sum_kw_k(y_k^{model}-y_k^{mkt})^2,
$$

and

$$
\sum_kw_k(\sigma_{imp,k}^{model}-\sigma_{imp,k}^{mkt})^2.
$$

The objective should match the use case: price and Greeks for hedging, implied vol for quoting, or yield error for macro forecasting.

## 12.3 State-Space Observation Error

Observed yields need not equal latent model yields exactly. Coupon-bond bootstrapping, liquidity, bid--ask, taxes, and interpolation create measurement error:

$$
y_t^{obs}(\tau_j)
=A_y(\tau_j)+B_y(\tau_j)^\top X_t+\varepsilon_{t,j}.
$$

Gaussian models permit Kalman filtering. Nonlinear or non-Gaussian models may require extended or unscented filters, particle filters, or simulation-based methods. Measurement errors should not automatically be assumed independent across maturities; curve-construction noise is often correlated.

## 12.4 Identification Risks

- A small mean-reversion speed can be difficult to distinguish from a highly persistent long-run factor.
- Factor scale, volatility, and loading normalization may not be unique.
- One swaption diagonal cannot identify an unrestricted correlation surface.
- An exact deterministic shift can hide cross-sectional misfit without improving stochastic dynamics.
- Highly persistent physical-measure drift parameters have substantial finite-sample bias.
