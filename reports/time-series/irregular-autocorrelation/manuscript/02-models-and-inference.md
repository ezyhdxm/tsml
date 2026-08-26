# 6. 第三类方法：连续时间参数模型

## 6.1 OU / continuous-time AR(1)

最简单模型是

$$
dX_t=-\lambda X_t\,dt+\sigma\,dW_t,
\qquad \lambda>0.
$$

它的 stationary ACF 为

$$
\rho_c(\tau)=e^{-\lambda|\tau|}.
$$

对任意 irregular interval $\Delta_j=t_j-t_{j-1}$，exact transition 是

$$
X_{t_j}
=e^{-\lambda\Delta_j}X_{t_{j-1}}+\eta_j,
$$

其中

$$
\operatorname{Var}(\eta_j)
=
\frac{\sigma^2}{2\lambda}
\left(1-e^{-2\lambda\Delta_j}\right).
$$

因此完全不需要 resample。常用摘要是 half-life：

$$
\tau_{1/2}=\frac{\log 2}{\lambda}.
$$

## 6.2 IAR formulation

Eyheramendy, Elorrieta, and Palma 写成 irregular autoregressive（IAR）形式 [@eyheramendy2018]：

$$
y_{t_j}
=
\phi^{\Delta_j}y_{t_{j-1}}
+
\sigma\sqrt{1-\phi^{2\Delta_j}}\,\varepsilon_{t_j},
\qquad 0<\phi<1,
$$

从而

$$
\rho_c(\tau)=\phi^\tau.
$$

令 $\phi=e^{-\lambda}$ 就与 OU decay 对应。它的价值不是发明了新的 ACF 定义，而是给出可直接处理 irregular gaps 的 likelihood/model-checking framework。

## 6.3 什么时候 OU 不够？

若经验 ACF：

- 有明显负相关；
- 有振荡；
- 先快降、再有长尾；
- 不同 market regime 有不同 decay；

单指数

$$
e^{-\lambda\tau}
$$

就过于僵硬。可以考虑：

$$
\rho(\tau)
=a_1e^{-\lambda_1\tau}+a_2e^{-\lambda_2\tau},
$$

或者 continuous-time ARMA（CARMA）/state-space model。CARMA 是离散 ARMA 在 continuous-time stationary process 上的对应物，可表达更丰富的 spectral 与 autocovariance 结构 [@brockwell2014]。

参数模型的核心交换是：

> 用结构假设换取 gap 内插值、unsupported lag 外推、以及更低维的 half-life/oscillation summary。

因此应先画 nonparametric curve 和 support，再决定参数模型是否可信，而不是先假设 OU 后只报告一个 half-life。

# 7. 为什么“先 resample 再 ACF”经常误导？

## 7.1 Forward fill 会制造 piecewise-constant persistence

若 10:00 有 observation，下一笔在 13:00，1-minute forward fill 会生成 179 个完全相同的伪值。普通 ACF 看到的是长平台，自然得到很高的短 lag correlation。

这不代表 resampling 永远不能用。如果 $X(t)$ 本来就是“最新可见状态”——例如一个真实会持有至下次更新的 displayed quote state——last observation carried forward 可能就是你定义的过程。但对 latent price/residual，它通常不是中性的缺失值处理。

## 7.2 Linear interpolation 会加入 smooth-path assumption

Linear interpolation 假定 gap 内沿直线变化；cubic spline 加入更强 smoothness。若 gap 相对 process time scale 很短，bias 可能可接受；若 gap 与 decay scale 同量级或更长，插值实际上决定了 ACF。Edelson--Krolik 明确提出只在有真实 measured pairs 的 lag 上估计，以避免“inventing data” [@edelson1988]；Rehfeld et al. 的 benchmark 则展示了高度 irregular 时 interpolation 对 persistence 的上偏 [@rehfeld2011]。

## 7.3 一个合成实验

下面模拟 stationary OU，真实 half-life 为 30 分钟；5,000 个 observation 的 gap median 为 2.42 分钟、mean 为 9.44 分钟、95% quantile 为 37.54 分钟，并加入 session-like long gaps。

![直接 pair estimators 能追踪真实 OU ACF；1-minute forward fill 明显抬高短 lag persistence。](figures/ou_simulation_acf.svg)

在这一次固定-seed simulation 中：

| quantity | value |
|---|---:|
| true $\rho_c(5\text{ min})$ | 0.891 |
| Gaussian-kernel estimate | 0.880 |
| 1-minute forward-fill ACF | 0.934 |
| true half-life | 30.0 min |
| kernel curve 的 OU descriptive fit | 33.2 min |
| empirical event-lag-1 ACF | 0.873 |
| $E[e^{-\lambda\Delta}]$ | 0.876 |
| 错误的 mean-gap plug-in $e^{-\lambda E\Delta}$ | 0.804 |

这个实验不是证明 Gaussian kernel 在所有数据上最优；它只清楚展示了三个概念：

1. event lag 1 不是 mean-gap clock lag；
2. forward fill 可以机械抬高短 lag correlation；
3. 每个 lag 的有效 empirical support 不同。

![同一条 ACF 曲线上，每个 lag 由不同数量和不同权重集中的 pairs 支撑。](figures/pair_support.svg)

# 8. 统计推断：最容易出错的部分

## 8.1 不能 bootstrap individual pairs

Pair table 中同一个 $z_i$ 会出现很多次。若独立重采样 pairs，就破坏了真实 dependence graph，通常把 uncertainty 压得过小。

更合理的 resampling unit 取决于数据结构：

- 多条近似独立短 series：按整条 series bootstrap；
- 同一 series 分成 sessions：按 session 或 clock-time blocks bootstrap；
- bond panel 同时有 CUSIP persistence 与 date common shock：考虑 two-way cluster/multiplier bootstrap，或按日期 block 重采样并保留日内全部 bonds；
- 单条长 series：moving-block 或 stationary bootstrap，且在每次 bootstrap 内重新构造 pairs。

本文代码提供了一个按 cluster 汇总 pair contributions 的快速 bootstrap。它比 pair bootstrap 正确得多，但仍把 centering/scaling 视作固定；严谨 inference 应在 original-observation block bootstrap 内重新估计均值、scale 和 pairs。

## 8.2 Pointwise bands 与 simultaneous band 不同

对每个 lag 单独取 2.5%--97.5% bootstrap quantiles 得到 pointwise 95% interval。若你同时扫描 40 个 lags 并问“有没有任何一个显著”，需要 max-$t$ 或 sup-norm bootstrap 构造 simultaneous band，否则会有 multiple-comparison inflation。

## 8.3 Kernel ACF 未必是合法 covariance function

一个合法 stationary covariance 必须对任意时间点集合产生 positive-semidefinite covariance matrix。逐 lag smoothing 出来的曲线即使每一点都在 $[-1,1]$，也不保证整体 PSD。Bjørnstad and Falck 在 nonparametric covariance function 与 confidence envelope 中明确讨论了这一点 [@bjorstad2001]。

因此：

- 用于 residual diagnostic：pointwise kernel curve 通常足够；
- 用于 GLS、Gaussian process 或 likelihood covariance matrix：应拟合/投影到有效 covariance family，例如 OU、Matérn、CARMA 或非负 spectrum。

## 8.4 Nonstationarity 会让“一个 ACF”本身失去意义

若 mean、variance 或 dependence 随时间变化，则真正对象是

$$
\rho(t,\tau)
=
\operatorname{Corr}\{X(t),X(t+\tau)\},
$$

不再只依赖 $\tau$。交易 residual 常见的 nonstationarity 包括：

- time-of-day volatility；
- rating/sector/tenor heteroskedasticity；
- crisis 与 calm regime；
- model version 或 data pipeline change；
- buy/sell side 或 quantity regime。

至少应先检查 conditional mean 与 scale，并用 standardized residual

$$
z_i=\frac{e_i-\widehat\mu_i}{\widehat\sigma_i}
$$

做 ACF。若 dependence 本身也随 regime 变化，则分别估计，而不是强行 pool 成一条曲线。
