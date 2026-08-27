# 4. Vasicek：Gaussian 一因子短利率模型

Vasicek 模型是最基础的连续时间均值回复利率模型 [@vasicek1977]。

## 4.1 风险中性 SDE

在 $\mathbb Q$ 下，

$$
\boxed{
dr_t=\kappa(\theta-r_t)dt+\sigma dW_t^{\mathbb Q}
}
$$

其中

$$
\kappa>0,\qquad \sigma\ge0,\qquad \theta\in\mathbb R.
$$

- $\kappa$：均值回复速度；half-life 为 $\log2/\kappa$。
- $\theta$：风险中性长期均值。
- $\sigma$：瞬时波动率。
- 状态空间是整个 $\mathbb R$，所以负利率有正概率。

## 4.2 精确解与精确转移

积分因子法给出

$$
r_{t+\Delta}
=
\theta+(r_t-\theta)e^{-\kappa\Delta}
+
\sigma\int_t^{t+\Delta}
 e^{-\kappa(t+\Delta-s)}dW_s^{\mathbb Q}.
$$

因此条件分布为

$$
r_{t+\Delta}\mid\mathcal F_t
\sim
\mathcal N\left(
\theta+(r_t-\theta)e^{-\kappa\Delta},
\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa\Delta})
\right).
$$

精确模拟一步为

$$
r_{t+\Delta}
=
\theta+(r_t-\theta)e^{-\kappa\Delta}
+
\sigma\sqrt{\frac{1-e^{-2\kappa\Delta}}{2\kappa}}Z,
\qquad Z\sim\mathcal N(0,1).
$$

没有必要对 Vasicek 使用 Euler--Maruyama；精确转移既更快又无离散偏差。

## 4.3 短利率积分的联合正态分布

定价需要

$$
I_{t,\Delta}=\int_t^{t+\Delta}r_sds.
$$

定义

$$
B_\kappa(\Delta)=\frac{1-e^{-\kappa\Delta}}{\kappa}.
$$

则

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

并且

$$
\operatorname{Cov}_t
\left(r_{t+\Delta},I_{t,\Delta}\right)
=
\frac{\sigma^2}{2\kappa^2}
(1-e^{-\kappa\Delta})^2.
$$

所以可以一次生成二维相关正态数，同时精确模拟 endpoint 和区间积分。这对 path-dependent payoff、discount factor 和 exposure simulation 很重要。

## 4.4 零息债券解析解

令 $\tau=T-t$。债券价格具有

$$
P(t,T)=A_V(\tau)e^{-B_V(\tau)r_t},
$$

其中

$$
B_V(\tau)=\frac{1-e^{-\kappa\tau}}{\kappa},
$$

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

对应的长期零息收益率为

$$
\lim_{\tau\to\infty}y(t,t+\tau)
=
\theta-\frac{\sigma^2}{2\kappa^2}.
$$

注意长期收益率不是简单的 $\theta$；Gaussian convexity adjustment 会把它向下移动。

## 4.5 $\mathbb P$ 与 $\mathbb Q$ 参数的关系

假设真实测度下

$$
dr_t=\kappa_P(\theta_P-r_t)dt+\sigma dW_t^{\mathbb P},
$$

风险价格取仿射形式

$$
\lambda_t=\lambda_0+\lambda_1r_t.
$$

因为 $dW^{\mathbb Q}=dW^{\mathbb P}+\lambda_tdt$，有

$$
\kappa_Q=\kappa_P+\sigma\lambda_1,
$$

$$
\theta_Q
=
\frac{\kappa_P\theta_P-\sigma\lambda_0}{\kappa_Q}.
$$

这清楚展示了为什么历史均值回复与期权隐含均值回复可以不同。

## 4.6 优点与局限

优点：解析性强、精确模拟、Gaussian state-space estimation 简单。局限：一因子导致所有 maturity 主要由同一个冲击驱动；条件波动率常数；收益率可能为负；任意初始曲线不能由常参数 Vasicek 精确拟合。

# 5. CIR：平方根扩散与非负短利率

Cox--Ingersoll--Ross 模型用状态依赖波动率保持短利率非负 [@cir1985]。

## 5.1 风险中性 SDE

$$
\boxed{
dr_t=\kappa(\theta-r_t)dt+\sigma\sqrt{r_t}\,dW_t^{\mathbb Q}
}
$$

参数限制为

$$
\kappa>0,\qquad \theta\ge0,\qquad \sigma>0,\qquad r_0\ge0.
$$

当 $r_t$ 很低时，扩散项自动减小；当 $r_t$ 较高时，波动率增大。

## 5.2 Feller condition 与零边界

令

$$
\nu=\frac{4\kappa\theta}{\sigma^2}.
$$

- 若 $2\kappa\theta\ge\sigma^2$，即 $\nu\ge2$，则从正初值出发，零边界不可达。
- 若 $0<2\kappa\theta<\sigma^2$，零边界可达，但过程仍保持非负；标准 CIR 边界是瞬时反射型，而不是穿越到负数。
- 若 $\theta=0$，零可以成为吸收边界。

Feller condition 不是“模型存在”的必要条件；它决定的是零边界是否不可达。校准时强行要求它成立可能显著降低对市场的拟合能力，因此需要明确这是经济约束还是数值偏好。

## 5.3 精确转移分布

定义

$$
c_\Delta
=
\frac{\sigma^2(1-e^{-\kappa\Delta})}{4\kappa},
$$

$$
\lambda_\Delta
=
\frac{4\kappa e^{-\kappa\Delta}r_t}
{\sigma^2(1-e^{-\kappa\Delta})}.
$$

则

$$
\boxed{
r_{t+\Delta}
=c_\Delta Y,
\qquad
Y\sim\chi'^2_{\nu}(\lambda_\Delta)
}
$$

其中 $\chi'^2$ 是非中心卡方分布。条件均值和方差为

$$
\mathbb E_t[r_{t+\Delta}]
=
\theta+(r_t-\theta)e^{-\kappa\Delta},
$$

$$
\operatorname{Var}_t(r_{t+\Delta})
=
\frac{\sigma^2r_te^{-\kappa\Delta}(1-e^{-\kappa\Delta})}{\kappa}
+
\frac{\theta\sigma^2(1-e^{-\kappa\Delta})^2}{2\kappa}.
$$

## 5.4 零息债券解析解

令

$$
\gamma=\sqrt{\kappa^2+2\sigma^2}.
$$

则

$$
P(t,T)=A_C(\tau)e^{-B_C(\tau)r_t},
$$

$$
B_C(\tau)
=
\frac{2(e^{\gamma\tau}-1)}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma},
$$

$$
A_C(\tau)
=
\left[
\frac{2\gamma\exp((\kappa+\gamma)\tau/2)}
{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}
\right]^{2\kappa\theta/\sigma^2}.
$$

Riccati 方程为

$$
B_C'=1-\kappa B_C-\frac12\sigma^2B_C^2,
\qquad B_C(0)=0,
$$

$$
\frac{A_C'}{A_C}=-\kappa\theta B_C,
\qquad A_C(0)=1.
$$

## 5.5 风险价格

若 $\mathbb P$ 下仍是 CIR，常见的 affine-preserving 选择是

$$
\lambda_t=\lambda_r\sqrt{r_t}.
$$

于是

$$
\kappa_Q=\kappa_P+\sigma\lambda_r,
\qquad
\theta_Q=\frac{\kappa_P\theta_P}{\kappa_Q}.
$$

更一般的风险价格可以同时改变常数漂移项，但可能包含 $1/\sqrt{r_t}$，在零附近需要额外处理。

## 5.6 数值离散

朴素 Euler

$$
r_{n+1}=r_n+\kappa(\theta-r_n)\Delta t
+\sigma\sqrt{r_n}\sqrt{\Delta t}Z_n
$$

会在有限步长下生成负数，使下一步的平方根失效。可选方案：

- **精确非中心卡方转移**：模拟 endpoint 时首选。
- **full-truncation Euler**：在漂移和扩散中使用 $r_n^+=\max(r_n,0)$，并明确最终状态是否截断。
- **quadratic-exponential 类方法**：在极端参数或大规模 Monte Carlo 中兼顾速度与矩匹配 [@andersen2008]。

但即使 endpoint 精确，$\int r_sds$ 也没有像 Gaussian 模型那样简单的联合正态抽样。若 payoff 依赖路径积分，可加密网格、使用 conditional transform/inversion，或在只需要零息债券时直接使用解析公式，而不是 Monte Carlo。
