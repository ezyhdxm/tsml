# 10. 一套不跳步的实务流程

::: {.decision}
### Step 1：写一句 estimand {.unnumbered .unlisted}

例如：

> 在同一 CUSIP、同一交易日内，对 variance-standardized out-of-sample residual，估计随机 observed trade pair 在 clock-time lag $\tau$ 附近的 pair-weighted correlation。

这句话确定了 series、session、conditioning、weighting 和 time clock。
:::

### Step 2：构造可审计 residual

- 只用真正 out-of-sample / point-in-time predictions；
- 明确 sign：$e=y-\hat y$；
- 去除或标记 duplicates/as-of；
- 用预测 uncertainty 或稳定 group scale 得到 $z=e/\hat\sigma$；
- 保留 side、quantity、liquidity、target interval 等 audit columns。

### Step 3：先看 sampling，不先看 ACF

- 相邻 gap distribution；
- 每 CUSIP/session observation count；
- 各目标 lag 的 pair support；
- active bursts 与 overnight gaps。

### Step 4：算 event-time ACF

这是最简单 sanity check，也能与普通 `.shift(k)` 对齐。不要把它改名为 “5-minute ACF”。

### Step 5：算 clock-time kernel ACF

- pairs 限制在同一 CUSIP/session；
- 选择业务可解释的 lag grid；
- Gaussian bandwidth 先取相邻 lag grid spacing 的 0.5--1 倍，再做 sensitivity；
- 同时输出 product-mean 与 local-normalized ACF。

### Step 6：画 support 和 uncertainty

任何没有 support curve 的 irregular ACF 都是不完整的。置信区间按 session/date blocks，而不是 pairs。

### Step 7：做 confound audits

- target overlap；
- duplicate/clustered reports；
- common market factor；
- informative sampling；
- regime/heteroskedasticity；
- calendar time vs business time。

### Step 8：最后才拟合 half-life

只有经验 curve 与

$$
\rho(\tau)=e^{-\lambda\tau}
$$

大致一致时才报告 OU half-life。若有 fast + slow 两段 decay，使用 two-scale model；若有负值/振荡，转向 CARMA/state-space，而不是对负 correlation 取 log。

# 11. Python 实现

同目录的 [`irregular_acf.py`](irregular_acf.py) 提供：

- `standardize_within`：series/group 内标准化；
- `build_within_series_pairs`：用 sliding cutoff 只构造 `max_lag` 内 pairs；
- `kernel_acf_from_pairs`：Gaussian/rectangular/Epanechnikov ACF；
- `event_time_acf`：event-index ACF；
- `cluster_bootstrap_acf_from_pairs`：cluster-level 快速 bootstrap；
- `ou_half_life_from_acf`：对正且单调的 curve 做 descriptive OU fit。

最小用法如下：

```python
import pandas as pd

from irregular_acf import (
    build_within_series_pairs,
    cluster_bootstrap_acf_from_pairs,
    event_time_acf,
    kernel_acf_from_pairs,
    standardize_within,
)

# 1. Residual and session identifiers.
df = predictions.copy()
df["resid"] = df["y_true"] - df["y_pred"]
df["trade_date"] = pd.to_datetime(df["EFFECTIVE_DATETIME_TS"]).dt.date
df["series_session"] = (
    df["CUSIP"].astype(str) + "|" + df["trade_date"].astype(str)
)
df["bootstrap_cluster"] = df["trade_date"].astype(str)

# 2. Illustration only: group standardization.  In a production residual
# diagnostic, predicted conditional scale is often preferable.
df = standardize_within(
    df,
    value_col="resid",
    group_cols=["CUSIP"],
    output_col="z",
)

# 3. Event-time ACF: order rows first.
event_acf = event_time_acf(
    df.sort_values(["series_session", "EFFECTIVE_DATETIME_TS"]),
    value_col="z",
    series_cols=["series_session"],
    max_event_lag=10,
)

# 4. Clock-time pairs only within a CUSIP-day session.
pairs, pair_summary = build_within_series_pairs(
    df,
    time_col="EFFECTIVE_DATETIME_TS",
    value_col="z",
    series_cols=["series_session"],
    cluster_cols=["bootstrap_cluster"],
    max_lag=260,          # enough for 4h plus kernel truncation
    time_unit="minutes",
)

lags = [0, 1, 2, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240]
clock_acf = kernel_acf_from_pairs(
    pairs,
    lags=lags,
    bandwidth=5,
    kernel="gaussian",
    truncate=4,
)

# 5. Date-cluster bootstrap as an example.  Reconsider the cluster unit
# when common shocks persist across dates or when a single bond dominates.
clock_acf_ci = cluster_bootstrap_acf_from_pairs(
    pairs,
    cluster_col="bootstrap_cluster",
    lags=lags,
    bandwidth=5,
    n_boot=500,
    random_state=7,
)
```

## 11.1 复杂度

Naive all-pairs 是 $O(n^2)$。代码使用 sorted timestamps 和 `searchsorted`，只枚举

$$
d_{ij}\le L_{\max}
$$

的 pairs，所以复杂度更接近

$$
O(n+P_L),
$$

其中 $P_L$ 是 max-lag window 内的真实 pair 数量。对极高频 burst，$P_L$ 仍可能很大；生产实现应使用 Numba/C++、按 fine lag bins 预聚合，或在业务上限制同 session/max lag。

## 11.2 代码没有替你决定的事情

代码无法自动决定：

- 何为一条 series；
- 是否允许 overnight pairs；
- 用哪个 conditional scale；
- bootstrap unit；
- pair-weighted 还是 bond-equal；
- observation-time endogeneity 是否可忽略；
- target overlap 是否机械制造 ACF。

这些正是 estimand 的一部分，不能藏进函数默认值。
