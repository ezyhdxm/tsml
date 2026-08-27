# 13. 数值 SDE 实现细节

## 13.1 Brownian correlation

若模型写为

$$
dX_t=\mu dt+L_tdZ_t,
\qquad d\langle Z\rangle_t=I dt,
$$

则 covariance 是 $L_tL_t^\top dt$。若直接使用 correlated Brownian $W$，

$$
d\langle W\rangle_t=Rdt,
$$

则 covariance 是 $\Sigma R\Sigma^\top dt$。不要同时在 loading 中和 Brownian correlation matrix 中重复加入相关性。

数值生成时应检查 $R$：

- 对称；
- 对角线为 1；
- 最小特征值非负；
- Cholesky 失败时，不要简单增加很大的 diagonal jitter 掩盖模型问题。

## 13.2 Strong 与 weak error

- **strong convergence** 关心同一路径上的误差，重要于 pathwise Greeks、barrier/hitting-time 和 exposure trajectory。
- **weak convergence** 关心期望误差，重要于普通 European payoff price。

Euler--Maruyama 通常 strong order $1/2$、在足够光滑条件下 weak order 1。精确转移消除 endpoint discretization bias，但不自动消除 path integral、early exercise 和 interpolation error。

## 13.3 Gaussian 模型的精确一步代码

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

## 13.4 CIR 精确一步代码

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

## 13.5 G2++ 精确 endpoint 代码

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

## 13.6 HJM 与 LMM 的实现测试

HJM：

- 数值积分后的 $f$、$P$、$r$ 三者一致；
- 在 $\mathbb Q$ 下，$P(t,T)/B_t$ 的 sample mean 无显著 drift；
- maturity quadrature 加密后 price 收敛；
- factor shocks 在 maturity 间共享。

LMM：

- 在各自 $T_{i+1}$-forward measure 下，$L_i$ sample mean 保持不变；
- terminal measure 下使用完整 drift 后，各 measure 价格一致；
- forward-to-discount recursion

  $$
  P(t,T_i)=P(t,T_{i+1})[1+\delta_iL_i(t)]
  $$

  保持数值一致；
- correlation matrix 和 loading factorization 在所有时间点 PSD。

# 14. 如何选择模型

| 任务 | 首选起点 | 原因 | 主要风险 |
|---|---|---|---|
| 教学、解析推导、基准测试 | Vasicek | 完全 Gaussian、精确转移、闭式债券价格 | 一因子、负利率、常波动率 |
| 非负短利率与状态依赖波动 | CIR / CIR++ | 平方根扩散、非中心卡方转移 | 边界与离散、曲线拟合、smile 有限 |
| Callable/Bermudan 的低维定价 | Hull--White 1F | 精确初始曲线、树/PDE/MC 方便 | 一因子相关结构太强 |
| 更丰富的 slope/curvature 风险 | G2++ | 两个 Gaussian 因子、解析债券价格 | 仍无 smile、参数识别 |
| 从 volatility structure 出发建模整条曲线 | HJM | 无套利漂移由波动率唯一决定 | 无限维、离散与校准复杂 |
| Cap/floor、swaption 与 rates exotic | LMM/BGM 及扩展 | 直接建模市场 forward、自然接 Black quote | 多测度 drift、负利率、维度高 |
| 收益率预测 | DNS / VAR / ML factor model | 简洁、易估计、预测导向 | 不自动无套利 |
| 期限溢价与历史/定价联合分析 | Gaussian ATSM、AFNS、ACM/JSZ | 明确区分 $\mathbb P$ 与 $\mathbb Q$ | 高 persistence、risk-price 识别 |

没有“最好的 term-structure model”。正确问题是：模型需要同时匹配哪些 observables，在哪个 measure 下使用，是否需要 early exercise，是否要拟合 smile，以及计算预算允许多少状态维度。

# 15. 建议的学习顺序

1. 先独立推导 $P(t,T)=\mathbb E^Q[e^{-\int r}]$ 与定价 PDE。
2. 手算 Vasicek 的 exact transition、积分方差和 $A/B$ 债券公式。
3. 推导 CIR Riccati ODE，理解 Feller condition 与 exact noncentral-$\chi^2$ transition。
4. 从 shifted OU 推导 Hull--White 的 $\phi(t)$ 和 $\theta(t)$，不要背公式。
5. 完整复核 HJM drift restriction；这是理解所有无套利 rates model 的核心。
6. 从 numeraire change 推导 LMM terminal-measure drift 的负号和求和范围。
7. 最后再学习 calibration、state-space estimation、AFNS/ACM 和 smile extensions。

# 16. 总结

期限结构建模的统一逻辑可以压缩成四行：

$$
\text{state/curve SDE under }\mathbb P
\quad\xrightarrow{\text{risk price}}\quad
\text{SDE under }\mathbb Q,
$$

$$
\text{numeraire + no arbitrage}
\quad\Longrightarrow\quad
\text{pricing drift restriction},
$$

$$
P(t,T)
=
\mathbb E_t^{\mathbb Q}
\left[e^{-\int_t^T r_sds}\right],
$$

$$
\text{initial curve fit}
\neq
\text{historical forecast fit}
\neq
\text{option-volatility fit}.
$$

Vasicek/CIR 展示了 short-rate SDE 如何产生整个 term structure；Hull--White/G2++ 展示了 deterministic shift 如何精确匹配今天的曲线；HJM 展示了 forward volatility 如何通过无套利决定 drift；LMM 展示了如何在自然 forward measure 下直接建模市场利率。真正可靠的实现必须同时保留测度、numeraire、相关性、边界、积分和离散化这些细节。
