# 10. SOFR、OIS discounting 与多曲线框架

SOFR 是以美国国债回购交易为基础的隔夜担保融资利率；美元利率市场已从 LIBOR 框架转向 SOFR/OIS discounting [@nyfedsofr; @arrcsofr]。

## 10.1 单曲线理想化

若同一 OIS curve 同时用于 discount 和 forecast，则可以用 OIS discount factors 定义 forward compounded overnight rate 的理想化版本：

$$
F_i^{OIS}(t)
=
\frac1{\delta_i}
\left[
\frac{P^{OIS}(t,T_i)}{P^{OIS}(t,T_{i+1})}-1
\right].
$$

实际 compounded-in-arrears SOFR 在 accrual period 内逐日 fixing，包含 day-count、holiday、lookback/lockout、payment delay 等 contract details；这些细节不应被一个简单 forward ratio 掩盖。

## 10.2 多曲线

一般 collateralized pricing 中，至少区分：

- discount curve $P^d(t,T)$；
- 某个 index/tenor 的 forecast pseudo-curve $P^x(t,T)$；
- basis spread dynamics。

定义

$$
F_i^x(t)
=
\frac1{\delta_i}
\left[
\frac{P^x(t,T_i)}{P^x(t,T_{i+1})}-1
\right].
$$

但定价测度由 discount numeraire $P^d(t,T)$ 决定。$P^x$ 通常不是可交易零息债券，所以不能未经证明就把 $P^x(t,T_i)/P^x(t,T_{i+1})$ 当作某个自然 measure 下的鞅。现代 multi-curve LMM/HJM 需要显式建模 basis 或 multiplicative spread，并保证 discounting numeraire 下的无套利关系。

# 11. 预测模型与定价模型：DNS、AFNS 和 Gaussian ATSM

## 11.1 Dynamic Nelson--Siegel

Dynamic Nelson--Siegel 写成

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

三个因子通常解释为 level、slope、curvature。Diebold--Li 令这些因子服从 VAR 或其他时间序列动态，用于收益率预测 [@dieboldli2006]。

连续时间版本可写为

$$
d\beta_t
=K^{\mathbb P}(\mu^{\mathbb P}-\beta_t)dt
+\Sigma_\beta dW_t^{\mathbb P}.
$$

但仅有这个 observation equation 与 $\mathbb P$-SDE，并不能保证不存在动态套利。DNS 适合 forecasting/factor compression，不应未经修正直接用于复杂衍生品定价。

## 11.2 Gaussian affine term-structure model

设

$$
dX_t
=K^{\mathbb P}(\theta^{\mathbb P}-X_t)dt
+\Sigma dW_t^{\mathbb P},
$$

在风险中性测度下

$$
dX_t
=K^{\mathbb Q}(\theta^{\mathbb Q}-X_t)dt
+\Sigma dW_t^{\mathbb Q},
$$

短利率

$$
r_t=\delta_0+\delta_1^\top X_t.
$$

债券收益率为

$$
y_t(\tau)=A_y(\tau)+B_y(\tau)^\top X_t.
$$

$\mathbb P$ 与 $\mathbb Q$ 参数差异决定风险溢价和 expected excess bond return。Gaussian ATSM 的困难主要不是解 Riccati，而是 identification：状态旋转、风险价格参数和 measurement error 可以产生相似的收益率拟合。JSZ、ACM 等方法通过选择 observable yield portfolios 或回归结构改善估计 [@joslinsingletonzhu2011; @adriancrumpmoench2013]。

## 11.3 AFNS

Arbitrage-Free Nelson--Siegel 在 Gaussian ATSM 中施加特殊的 $\mathbb Q$-dynamics，使 factor loading 近似保留 Nelson--Siegel 形状，同时加入无套利 convexity adjustment [@christensendieboldrudebusch2011]。它连接了两种需求：

- DNS 的可解释 factor shape 与预测便利；
- affine model 的跨 maturity pricing restrictions。

# 12. 校准、估计与识别

## 12.1 推荐的分层流程

**第一层：市场曲线。** 从 OIS、futures、swaps 等工具 bootstrap $P^M(0,T)$，明确 collateral、day count、business-day convention、payment lag 和 interpolation variable。

**第二层：今天的横截面。** 对 Hull--White/G2++ 通过 $\phi(t)$ 精确拟合初始 curve；对 Vasicek/CIR 若不加 deterministic shift，通常只能近似拟合。

**第三层：风险中性波动率。** 使用 cap/floor、swaption 等 liquid option prices 校准 $a,b,\sigma,\eta,\rho$ 或 HJM/LMM loading。损失函数应按 bid--ask、vega 或价格尺度加权。

**第四层：真实测度动态。** 从历史 yield/factor time series 估计 $K^{\mathbb P}$、长期均值和 innovation covariance。

**第五层：风险价格。** 通过 $\mathbb P$ 与 $\mathbb Q$ 参数差异识别 term premia。不要把所有参数一次性塞进一个未加约束的 nonlinear optimizer。

## 12.2 Price error、yield error 与 implied-vol error

同一个 1bp yield error 对 2Y 和 30Y 债券的价格影响不同。同一个期权 price error 在低 vega 和高 vega 区域意义不同。常见目标包括

$$
\sum_k w_k(P_k^{model}-P_k^{mkt})^2,
$$

$$
\sum_k w_k(y_k^{model}-y_k^{mkt})^2,
$$

$$
\sum_k w_k(\sigma_{imp,k}^{model}-\sigma_{imp,k}^{mkt})^2.
$$

权重应与用途一致：hedging 更关心 price/Greek，surface quoting 更关心 implied vol，宏观预测更关心 yield。

## 12.3 State-space observation error

市场收益率不是精确的 latent-model yield：coupon-bond bootstrap、liquidity、bid--ask、税收和 interpolation 都会产生误差。可写

$$
y_t^{obs}(\tau_j)
=A_y(\tau_j)+B_y(\tau_j)^\top X_t+\varepsilon_{t,j}.
$$

Gaussian model 可用 Kalman filter；非线性/非 Gaussian 模型可用 extended/unscented filter、particle filter 或 simulation-based estimation。measurement error 不应自动假设独立同分布；不同 maturity 的 bootstrap noise 往往相关。

## 12.4 识别风险

- $a$ 很小与长期 factor 高 persistence 可互相替代。
- $\sigma$、factor scale 和 loading normalization 可能不唯一。
- 一条 swaption diagonal 难以识别完整 correlation surface。
- 精确初始 curve fit 会把某些 misfit 隐藏进 deterministic shift，而不是改善 stochastic dynamics。
- $\mathbb P$ 中高度 persistent 的 drift 参数有严重 finite-sample bias。
