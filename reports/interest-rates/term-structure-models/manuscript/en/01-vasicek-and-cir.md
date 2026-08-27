# 4. Vasicek: A Gaussian One-Factor Short-Rate Model

The Vasicek model is the canonical mean-reverting Gaussian interest-rate model [@vasicek1977].

## 4.1 Risk-Neutral SDE

Under $\mathbb Q$,

$$
\boxed{
dr_t=\kappa(\theta-r_t)dt+\sigma dW_t^{\mathbb Q}
}
$$

with

$$
\kappa>0,\qquad \sigma\ge0,\qquad \theta\in\mathbb R.
$$

Here $\kappa$ is the mean-reversion speed, $\log2/\kappa$ is the half-life, $\theta$ is the risk-neutral long-run mean, and $\sigma$ is instantaneous volatility. The state space is $\mathbb R$, so negative rates occur with positive probability.

## 4.2 Exact Solution and Transition Law

The integrating-factor solution is

$$
r_{t+\Delta}
=
\theta+(r_t-\theta)e^{-\kappa\Delta}
+
\sigma\int_t^{t+\Delta}
 e^{-\kappa(t+\Delta-s)}dW_s^{\mathbb Q}.
$$

Thus

$$
r_{t+\Delta}\mid\mathcal F_t
\sim
\mathcal N\left(
\theta+(r_t-\theta)e^{-\kappa\Delta},
\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa\Delta})
\right),
$$

and an exact step is

$$
r_{t+\Delta}
=
\theta+(r_t-\theta)e^{-\kappa\Delta}
+
\sigma\sqrt{\frac{1-e^{-2\kappa\Delta}}{2\kappa}}Z,
\qquad Z\sim\mathcal N(0,1).
$$

Euler--Maruyama is unnecessary for this transition.

## 4.3 Joint Law of the Endpoint and Short-Rate Integral

Let

$$
I_{t,\Delta}=\int_t^{t+\Delta}r_sds,
\qquad
B_\kappa(\Delta)=\frac{1-e^{-\kappa\Delta}}{\kappa}.
$$

Then

$$
\mathbb E_t[I_{t,\Delta}]
=
\theta\Delta+(r_t-\theta)B_\kappa(\Delta),
$$

$$
\operatorname{Var}_t(I_{t,\Delta})
=
\frac{\sigma^2}{\kappa^2}
\left[
\Delta
-\frac{2(1-e^{-\kappa\Delta})}{\kappa}
+\frac{1-e^{-2\kappa\Delta}}{2\kappa}
\right],
$$

and

$$
\operatorname{Cov}_t\left(r_{t+\Delta},I_{t,\Delta}\right)
=
\frac{\sigma^2}{2\kappa^2}(1-e^{-\kappa\Delta})^2.
$$

The pair is jointly Gaussian, allowing exact simulation of both the endpoint and the discount integral. This matters for path-dependent payoffs and exposure simulation.

## 4.4 Analytical Zero-Coupon Bond Price

With $\tau=T-t$,

$$
P(t,T)=A_V(\tau)e^{-B_V(\tau)r_t},
$$

where

$$
B_V(\tau)=\frac{1-e^{-\kappa\tau}}{\kappa},
$$

and

$$
\boxed{
A_V(\tau)
=
\exp\left[
\left(\theta-\frac{\sigma^2}{2\kappa^2}\right)
\left(B_V(\tau)-\tau\right)
-
\frac{\sigma^2}{4\kappa}B_V(\tau)^2
\right]
}.
$$

The long-maturity zero rate is

$$
\lim_{\tau\to\infty}y(t,t+\tau)
=
\theta-\frac{\sigma^2}{2\kappa^2},
$$

not simply $\theta$; Gaussian convexity lowers the asymptotic yield.

## 4.5 Mapping Physical to Risk-Neutral Parameters

Suppose under $\mathbb P$,

$$
dr_t=\kappa_P(\theta_P-r_t)dt+\sigma dW_t^{\mathbb P},
$$

and let

$$
\lambda_t=\lambda_0+\lambda_1r_t.
$$

Then

$$
\kappa_Q=\kappa_P+\sigma\lambda_1,
$$

and

$$
\theta_Q
=
\frac{\kappa_P\theta_P-\sigma\lambda_0}{\kappa_Q}.
$$

This makes explicit why historical and option-implied mean reversion can differ.

## 4.6 Strengths and Limitations

Vasicek is analytically transparent, exactly simulable, and convenient for Gaussian state-space estimation. Its limitations are one-factor curve dynamics, constant conditional volatility, possible negative rates, and the inability of the constant-parameter model to fit an arbitrary initial curve exactly.

# 5. CIR: Square-Root Diffusion and Nonnegative Short Rates

The Cox--Ingersoll--Ross model uses state-dependent volatility to preserve nonnegativity [@cir1985].

## 5.1 Risk-Neutral SDE

$$
\boxed{
dr_t=\kappa(\theta-r_t)dt+\sigma\sqrt{r_t}\,dW_t^{\mathbb Q}
}
$$

with

$$
\kappa>0,\qquad \theta\ge0,\qquad \sigma>0,\qquad r_0\ge0.
$$

Volatility shrinks near zero and rises with the rate level.

## 5.2 Feller Condition and the Zero Boundary

Define

$$
\nu=\frac{4\kappa\theta}{\sigma^2}.
$$

- If $2\kappa\theta\ge\sigma^2$, equivalently $\nu\ge2$, zero is unattainable from a positive initial value.
- If $0<2\kappa\theta<\sigma^2$, zero is attainable, but the standard CIR process remains nonnegative and reflects instantaneously rather than crossing below zero.
- If $\theta=0$, zero may be absorbing.

The Feller condition is not required for existence of the model; it determines boundary accessibility. Enforcing it during calibration is an economic or numerical choice, not a universal mathematical necessity.

## 5.3 Exact Transition Distribution

Let

$$
c_\Delta
=
\frac{\sigma^2(1-e^{-\kappa\Delta})}{4\kappa},
$$

and

$$
\lambda_\Delta
=
\frac{4\kappa e^{-\kappa\Delta}r_t}
{\sigma^2(1-e^{-\kappa\Delta})}.
$$

Then

$$
\boxed{
r_{t+\Delta}=c_\Delta Y,
\qquad Y\sim\chi'^2_\nu(\lambda_\Delta)
}.
$$

The conditional moments are

$$
\mathbb E_t[r_{t+\Delta}]
=
\theta+(r_t-\theta)e^{-\kappa\Delta},
$$

and

$$
\operatorname{Var}_t(r_{t+\Delta})
=
\frac{\sigma^2r_te^{-\kappa\Delta}(1-e^{-\kappa\Delta})}{\kappa}
+
\frac{\theta\sigma^2(1-e^{-\kappa\Delta})^2}{2\kappa}.
$$

## 5.4 Analytical Bond Price

Let

$$
\gamma=\sqrt{\kappa^2+2\sigma^2}.
$$

Then

$$
P(t,T)=A_C(\tau)e^{-B_C(\tau)r_t},
$$

with

$$
B_C(\tau)
=
\frac{2(e^{\gamma\tau}-1)}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma},
$$

and

$$
A_C(\tau)
=
\left[
\frac{2\gamma\exp((\kappa+\gamma)\tau/2)}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}
\right]^{2\kappa\theta/\sigma^2}.
$$

The Riccati equations are

$$
B_C'=1-\kappa B_C-\frac12\sigma^2B_C^2,
\qquad B_C(0)=0,
$$

and

$$
\frac{A_C'}{A_C}=-\kappa\theta B_C,
\qquad A_C(0)=1.
$$

## 5.5 Price of Risk

If the physical dynamics are also CIR, a common affine-preserving specification is

$$
\lambda_t=\lambda_r\sqrt{r_t}.
$$

Then

$$
\kappa_Q=\kappa_P+\sigma\lambda_r,
\qquad
\theta_Q=\frac{\kappa_P\theta_P}{\kappa_Q}.
$$

More general prices of risk can change the constant drift term as well, but may involve $1/\sqrt{r_t}$ and require care near zero.

## 5.6 Numerical Discretization

Naive Euler,

$$
r_{n+1}=r_n+\kappa(\theta-r_n)\Delta t
+\sigma\sqrt{r_n}\sqrt{\Delta t}Z_n,
$$

can produce a negative value at finite step size. Alternatives include:

- **exact noncentral-$\chi^2$ transition**, preferred for endpoints;
- **full-truncation Euler**, using $r_n^+=\max(r_n,0)$ in drift and diffusion;
- **quadratic-exponential methods**, designed to balance speed and moment matching in large simulations [@andersen2008].

Even an exact endpoint draw does not exactly sample $\int r_sds$. For path-dependent discounting, use a finer grid, a conditional transform/inversion method, or the analytical bond formula when only a zero-coupon price is needed.
