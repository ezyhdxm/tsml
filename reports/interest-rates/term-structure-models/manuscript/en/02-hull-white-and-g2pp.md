# 6. Hull--White One-Factor: Exact Fit to Today's Curve

Hull and White extend Gaussian short-rate models with time-inhomogeneous drift so the model can fit an arbitrary initial curve [@hullwhite1990].

## 6.1 Two Equivalent Parameterizations

One form is

$$
\boxed{
dr_t=[\theta(t)-ar_t]dt+\sigma dW_t^{\mathbb Q}}
$$

with $a>0$ and $\sigma\ge0$.

A more transparent shifted form is

$$
r_t=x_t+\phi(t),
$$

$$
\boxed{dx_t=-ax_tdt+\sigma dW_t^{\mathbb Q}},
\qquad x_0=0.
$$

The two are related by

$$
\theta(t)=\phi'(t)+a\phi(t).
$$

## 6.2 Choosing the Deterministic Shift

Let

$$
f^M(0,t)=-\partial_t\log P^M(0,t)
$$

be the market instantaneous-forward curve. Since $x$ is a zero-mean Gaussian OU process,

$$
\boxed{
\phi(t)
=f^M(0,t)
+
\frac{\sigma^2}{2a^2}(1-e^{-at})^2
}.
$$

For the parameterization $dr=[\theta(t)-ar]dt+\sigma dW$,

$$
\boxed{
\theta(t)
=
\partial_t f^M(0,t)
+a f^M(0,t)
+
\frac{\sigma^2}{2a}(1-e^{-2at})
}.
$$

Published formulas can differ by a factor of $a$ because some authors write $dr=a[\bar\theta(t)-r]dt+\sigma dW$. The SDE convention must be checked before using a memorized formula.

## 6.3 Bond Price

Define

$$
B_a(t,T)=\frac{1-e^{-a(T-t)}}a.
$$

Then

$$
P(t,T)=A_{HW}(t,T)e^{-B_a(t,T)r_t},
$$

where

$$
A_{HW}(t,T)
=
\frac{P^M(0,T)}{P^M(0,t)}
\exp\left[
B_a(t,T)f^M(0,t)
-
\frac{\sigma^2}{4a}(1-e^{-2at})B_a(t,T)^2
\right].
$$

At $t=0$, the model exactly reproduces $P^M(0,T)$.

## 6.4 Exact State and Discount-Integral Simulation

For the shifted state,

$$
x_{t+\Delta}
=e^{-a\Delta}x_t
+
\sigma\sqrt{\frac{1-e^{-2a\Delta}}{2a}}Z_1.
$$

Let

$$
J_{t,\Delta}=\int_t^{t+\Delta}x_sds.
$$

Then

$$
\mathbb E_t[J_{t,\Delta}]=B_a(\Delta)x_t,
$$

$$
\operatorname{Var}_t(J_{t,\Delta})
=
\frac{\sigma^2}{a^2}
\left[
\Delta
-\frac{2(1-e^{-a\Delta})}{a}
+\frac{1-e^{-2a\Delta}}{2a}
\right],
$$

and

$$
\operatorname{Cov}_t(x_{t+\Delta},J_{t,\Delta})
=
\frac{\sigma^2}{2a^2}(1-e^{-a\Delta})^2.
$$

The pair can therefore be sampled exactly as a bivariate Gaussian, after which

$$
\int_t^{t+\Delta}r_sds
=J_{t,\Delta}+\int_t^{t+\Delta}\phi(s)ds.
$$

This is more internally consistent than simulating only the endpoint exactly and approximating discounting with a left-endpoint rule.

## 6.5 Interpretation

The deterministic shift fits today's curve but does not add a stochastic factor. Hull--White 1F remains a one-shock model. It is often a good starting point for callable and Bermudan products, tree/PDE methods, and low-dimensional Monte Carlo. It is less suitable when independent level, slope, and curvature movements or a volatility smile are essential.

# 7. G2++: A Two-Factor Gaussian Exact-Fit Model

G2++ adds a second correlated OU factor:

$$
r_t=x_t+y_t+\phi(t),
$$

$$
dx_t=-ax_tdt+\sigma dW_t^{1,\mathbb Q},
$$

$$
dy_t=-by_tdt+\eta dW_t^{2,\mathbb Q},
$$

$$
d\langle W^1,W^2\rangle_t=\rho dt,
\qquad -1\le\rho\le1.
$$

Require $a,b>0$ and $\sigma,\eta\ge0$.

## 7.1 Shift for the Initial Curve

Define

$$
q(t)
=
\frac{\sigma^2}{a^2}(1-e^{-at})^2
+
\frac{\eta^2}{b^2}(1-e^{-bt})^2
+
\frac{2\rho\sigma\eta}{ab}(1-e^{-at})(1-e^{-bt}).
$$

With $x_0=y_0=0$,

$$
\boxed{
\phi(t)=f^M(0,t)+\frac12q(t)
}
$$

fits the initial discount curve exactly.

## 7.2 Bond Price

Let

$$
B_a(\tau)=\frac{1-e^{-a\tau}}a,
\qquad
B_b(\tau)=\frac{1-e^{-b\tau}}b.
$$

The conditional variance of the integrated stochastic short rate is

$$
\begin{aligned}
\mathcal V(\tau)
={}&
\frac{\sigma^2}{a^2}
\left[
\tau-\frac{2(1-e^{-a\tau})}{a}
+\frac{1-e^{-2a\tau}}{2a}
\right]\\
&+
\frac{\eta^2}{b^2}
\left[
\tau-\frac{2(1-e^{-b\tau})}{b}
+\frac{1-e^{-2b\tau}}{2b}
\right]\\
&+
\frac{2\rho\sigma\eta}{ab}
\left[
\tau-\frac{1-e^{-a\tau}}a
-\frac{1-e^{-b\tau}}b
+\frac{1-e^{-(a+b)\tau}}{a+b}
\right].
\end{aligned}
$$

Therefore

$$
\boxed{
P(t,T)
=
\exp\left[
-\int_t^T\phi(s)ds
-B_a(\tau)x_t
-B_b(\tau)y_t
+\frac12\mathcal V(\tau)
\right]
}.
$$

## 7.3 Exact Endpoint Correlation

One exact step is

$$
x_{t+\Delta}=e^{-a\Delta}x_t+\varepsilon_x,
$$

$$
y_{t+\Delta}=e^{-b\Delta}y_t+\varepsilon_y,
$$

where $(\varepsilon_x,\varepsilon_y)$ is zero-mean Gaussian with

$$
\operatorname{Var}(\varepsilon_x)
=
\frac{\sigma^2(1-e^{-2a\Delta})}{2a},
$$

$$
\operatorname{Var}(\varepsilon_y)
=
\frac{\eta^2(1-e^{-2b\Delta})}{2b},
$$

and

$$
\operatorname{Cov}(\varepsilon_x,\varepsilon_y)
=
\frac{\rho\sigma\eta(1-e^{-(a+b)\Delta})}{a+b}.
$$

The endpoint innovation correlation is generally **not equal** to the instantaneous Brownian correlation $\rho$. The correct implementation samples from this covariance matrix, rather than imposing correlation $\rho$ directly on endpoint shocks.

## 7.4 What G2++ Adds

Two mean-reversion speeds allow short- and long-horizon components to move with different persistence. The correlation parameter enriches slope and curvature dynamics. Analytical bond pricing and Gaussian exact simulation remain available. Negative rates, weak smile dynamics, and parameter identification remain limitations.
