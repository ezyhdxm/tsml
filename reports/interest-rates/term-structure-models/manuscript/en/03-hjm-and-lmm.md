# 8. HJM: Modeling the Entire Forward Curve

The Heath--Jarrow--Morton framework specifies dynamics for every instantaneous forward rate [@hjm1992].

## 8.1 HJM SDE

Let $W^{\mathbb Q}$ be $m$-dimensional. For each fixed maturity $T$ and $t\le T$,

$$
\boxed{
df(t,T)
=\alpha(t,T)dt
+\sigma(t,T)^\top dW_t^{\mathbb Q}
}.
$$

Here $T$ indexes a family of coupled SDEs, so the state is the full forward curve unless the volatility structure admits a finite-dimensional realization.

## 8.2 Full Derivation of the HJM Drift Restriction

Define the bond-volatility integral

$$
\Sigma_P(t,T)=\int_t^T\sigma(t,u)du.
$$

Since

$$
\log P(t,T)=-\int_t^T f(t,u)du,
$$

Leibniz's rule gives

$$
\begin{aligned}
d\left(\int_t^T f(t,u)du\right)
={}&-f(t,t)dt+\int_t^Tdf(t,u)du\\
={}&
\left[-r_t+\int_t^T\alpha(t,u)du\right]dt
+\Sigma_P(t,T)^\top dW_t^{\mathbb Q}.
\end{aligned}
$$

Hence

$$
d\log P(t,T)
=
\left[r_t-\int_t^T\alpha(t,u)du\right]dt
-
\Sigma_P(t,T)^\top dW_t^{\mathbb Q}.
$$

Applying Itô to the exponential,

$$
\frac{dP(t,T)}{P(t,T)}
=
\left[
 r_t
-\int_t^T\alpha(t,u)du
+\frac12\|\Sigma_P(t,T)\|^2
\right]dt
-
\Sigma_P(t,T)^\top dW_t^{\mathbb Q}.
$$

Under the money-market risk-neutral measure, the drift of a tradable bond must be $r_t$. Therefore

$$
\int_t^T\alpha(t,u)du
=
\frac12\left\|\int_t^T\sigma(t,u)du\right\|^2.
$$

Differentiating with respect to $T$ yields

$$
\boxed{
\alpha(t,T)
=
\sigma(t,T)^\top\int_t^T\sigma(t,u)du
}.
$$

Thus, under $\mathbb Q$, the volatility specification fixes the drift. An independently selected “mean-reverting” forward drift generally creates arbitrage.

The resulting bond SDE is

$$
\boxed{
\frac{dP(t,T)}{P(t,T)}
=r_tdt-\Sigma_P(t,T)^\top dW_t^{\mathbb Q}
}.
$$

## 8.3 HJM Under the Physical Measure

If

$$
dW_t^{\mathbb Q}=dW_t^{\mathbb P}+\lambda_tdt,
$$

then

$$
df(t,T)
=
\left[
\sigma(t,T)^\top\int_t^T\sigma(t,u)du
+
\sigma(t,T)^\top\lambda_t
\right]dt
+
\sigma(t,T)^\top dW_t^{\mathbb P}.
$$

The physical forward drift equals the no-arbitrage drift plus a maturity-dependent risk premium.

## 8.4 Forward Measures

Using $P(t,U)$ as numeraire defines $\mathbb Q^U$. Because the bond numeraire has $\mathbb Q$-volatility $-\Sigma_P(t,U)$,

$$
dW_t^{\mathbb Q^U}
=dW_t^{\mathbb Q}+\Sigma_P(t,U)dt.
$$

Therefore

$$
df(t,T)
=
\left[
\alpha^{\mathbb Q}(t,T)
-
\sigma(t,T)^\top\Sigma_P(t,U)
\right]dt
+
\sigma(t,T)^\top dW_t^{\mathbb Q^U}.
$$

Measure changes alter drift, not instantaneous covariance.

## 8.5 Musiela Parameterization

Define time-to-maturity coordinates

$$
g_t(x)=f(t,t+x),\qquad x\ge0.
$$

Then

$$
\boxed{
dg_t(x)
=
\left[
\partial_xg_t(x)
+
\sigma_t(x)^\top\int_0^x\sigma_t(u)du
\right]dt
+
\sigma_t(x)^\top dW_t^{\mathbb Q}
}.
$$

The $\partial_xg$ term is a coordinate-shift effect: as calendar time passes, a fixed maturity moves left along the time-to-maturity axis. A Musiela grid is often easier to maintain numerically than a fixed calendar-maturity grid.

HJM is infinite-dimensional in general. Only special volatility families can be represented by finitely many recursively updated Markov states [@filipovicteichmann2001].

## 8.6 Exponentially Decaying Volatility and Hull--White

Choose one-factor deterministic volatility

$$
\sigma(t,T)=\sigma e^{-a(T-t)}.
$$

Then

$$
\int_t^T\sigma(t,u)du
=
\frac{\sigma}{a}(1-e^{-a(T-t)}),
$$

and

$$
\alpha(t,T)
=
\frac{\sigma^2}{a}
 e^{-a(T-t)}(1-e^{-a(T-t)}).
$$

This HJM has a finite-dimensional Gaussian Markov representation and is equivalent to one-factor Hull--White. Short-rate models are therefore often special finite-dimensional HJM realizations rather than unrelated alternatives.

## 8.7 Maturity-Grid Discretization

On time grid $t_n$ and maturity grid $T_j$, for $T_j\ge t_n$:

1. Compute the loading $\sigma_{n,j}$.
2. Approximate the maturity integral

   $$
   \Sigma_{n,j}
   \approx
   \sum_{k:\,T_k\in[t_n,T_j]}
   \sigma_{n,k}\Delta T_k.
   $$

3. Set

   $$
   \alpha_{n,j}=\sigma_{n,j}^\top\Sigma_{n,j}.
   $$

4. Use the same $m$-dimensional Brownian increment for all maturities:

   $$
   f_{n+1,j}
   =f_{n,j}
   +\alpha_{n,j}\Delta t
   +\sigma_{n,j}^\top\Delta W_n.
   $$

5. Reintegrate the forward curve to obtain $P(t_{n+1},T_j)$.

Independent noise by maturity is incorrect: the entire curve must share the same factor shocks. A basic validation is that the Monte Carlo mean of $P(t,T)/B_t$ should have no systematic drift.

# 9. LMM/BGM: Market Forwards on a Discrete Tenor

The Brace--Gatarek--Musiela market model specifies dynamics directly for observable discrete forward rates [@bgm1997].

## 9.1 Tenor and Simple Forward Rates

Let

$$
0=T_0<T_1<\cdots<T_N,
$$

with accrual fractions

$$
\delta_i=T_{i+1}-T_i
$$

in an idealized year-fraction notation. The simple forward rate for $[T_i,T_{i+1}]$ is

$$
\boxed{
L_i(t)
=
\frac1{\delta_i}
\left[
\frac{P(t,T_i)}{P(t,T_{i+1})}-1
\right]
}.
$$

## 9.2 SDE Under the Natural Forward Measure

With $P(t,T_{i+1})$ as numeraire,

$$
1+\delta_iL_i(t)
=
\frac{P(t,T_i)}{P(t,T_{i+1})}
$$

is a martingale. The classical lognormal LMM sets

$$
\boxed{
dL_i(t)
=L_i(t)\lambda_i(t)^\top dW_t^{T_{i+1}}
}.
$$

For deterministic $\lambda_i$,

$$
L_i(T_i)
=
L_i(t)
\exp\left[
-\frac12\int_t^{T_i}\|\lambda_i(s)\|^2ds
+
\int_t^{T_i}\lambda_i(s)^\top dW_s^{T_{i+1}}
\right].
$$

The natural forward is lognormal under its own measure, which leads directly to Black-style caplet pricing.

## 9.3 Joint Dynamics Under the Terminal Measure

The natural measures differ across forwards, so joint simulation needs a common numeraire. Under the terminal measure $\mathbb Q^{T_N}$,

$$
\boxed{
\frac{dL_i(t)}{L_i(t)}
=
-\sum_{j=i+1}^{N-1}
\frac{\delta_jL_j(t)}{1+\delta_jL_j(t)}
\lambda_i(t)^\top\lambda_j(t)dt
+
\lambda_i(t)^\top dW_t^{T_N}
}.
$$

The negative sign is a numeraire-change effect. The drift is state dependent and coupled across maturities. Treating every forward as driftless lognormal in a joint simulation is inconsistent.

## 9.4 Correlation Through Factor Loadings

With $m$ factors,

$$
\lambda_i(t)=\nu_i(t)b_i(t),
\qquad b_i(t)\in\mathbb R^m,
$$

and

$$
\frac{d\langle L_i,L_j\rangle_t}
{L_i(t)L_j(t)}
=
\lambda_i(t)^\top\lambda_j(t)dt.
$$

The implied correlation matrix must be positive semidefinite. Fitting pairwise correlations independently can produce an invalid matrix. Direct low-rank loading parameterization or PSD projection is safer.

## 9.5 Terminal-Measure Log-Euler Scheme

Freeze drift and loadings over one step:

$$
\mu_i^n
=
-\sum_{j=i+1}^{N-1}
\frac{\delta_jL_j^n}{1+\delta_jL_j^n}
(\lambda_i^n)^\top\lambda_j^n.
$$

Then

$$
L_i^{n+1}
=
L_i^n
\exp\left[
\left(\mu_i^n-\frac12\|\lambda_i^n\|^2\right)\Delta t
+(\lambda_i^n)^\top\sqrt{\Delta t}Z_n
\right],
$$

using the same $m$-dimensional $Z_n$ for every $i$. A predictor--corrector drift can reduce freezing bias.

## 9.6 Negative Rates and Extensions

A pure lognormal LMM requires $L_i>0$. A shifted model defines

$$
F_i=L_i+s_i>0,
$$

and under the natural measure uses

$$
dF_i=F_i\lambda_i^\top dW^{T_{i+1}}.
$$

The terminal-measure drift must be rederived; mechanically replacing every $L_i$ in the classical drift by $L_i+s_i$ is not generally valid.

Local volatility, stochastic volatility, or SABR-like extensions can fit smiles, but the martingale condition, measure changes, and moment behavior must be rechecked.
