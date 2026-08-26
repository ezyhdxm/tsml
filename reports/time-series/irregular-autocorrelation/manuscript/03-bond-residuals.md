# 9. Corporate-bond residual：如何把问题定义对

设每一笔 out-of-sample prediction residual 为

$$
e_{b,i}=y_{b,i}-\widehat y_{b,i},
$$

其中 $b$ 表示 CUSIP，timestamp 为 $t_{b,i}$。

## 9.1 Pair 必须在同一条 series 内

最基本约束是

$$
b_i=b_j.
$$

把 CUSIP A 在 10:00 的 residual 与 CUSIP B 在 10:30 的 residual 组成 pair，估计的是 cross-sectional/common-factor dependence，不是同 bond autocorrelation。

还要决定是否允许跨 session：

- 若研究 wall-clock persistence，overnight pair 可以保留，但要解释 closed hours；
- 若研究 active-market correction，通常在同一 trading session 内构造 pairs；
- 也可以同时报告 calendar-clock 与 business-clock：

$$
s(t)=\int_0^t\mathbf 1\{u\text{ is active market time}\}\,du.
$$

## 9.2 同时报告 event-time 与 clock-time

这两条曲线回答不同业务问题：

| curve | lag | operational interpretation |
|---|---|---|
| event-time ACF | 1, 2, 5 trades | 需要几笔新交易，模型误差才被吸收？ |
| clock-time ACF | 5m, 30m, 1h | mispricing 在真实时间中持续多久？ |

如果 event-time decay 很快但 clock-time decay 很慢，可能只是该 bond 很少交易；如果 clock-time decay 快但 event-time decay 慢，可能在活跃时期连续 trades 带有相似 microstructure effect。

## 9.3 Pooled ACF 有至少两种不同 estimand

### Pair-weighted pooling

$$
\widehat\rho_{\text{pair}}(\tau)
\propto
\sum_b\sum_{i<j\in b}w_{bij}(\tau)z_{bi}z_{bj}.
$$

活跃 bonds 产生更多 pairs，因此主导结果。它回答：“随机抽取一个 eligible observed pair 时的平均 dependence 是什么？”

### Bond-equal pooling

先估计每只 bond 的 $\widehat\rho_b(\tau)$，再平均：

$$
\widehat\rho_{\text{bond}}(\tau)
=
\frac1B\sum_{b=1}^B\widehat\rho_b(\tau).
$$

它回答：“随机抽取一只 bond 时的平均 dependence 是什么？”但 sparse bonds 的估计非常噪声，通常需要 minimum support 或 hierarchical shrinkage。

这两条都可以报告；不能只写“pooled ACF”而不说明 weighting。

## 9.4 Observation process 是数据的一部分

Trade intensity 在市场变化时上升，意味着 pair-weighted ACF 还会给 high-activity states 更大权重。可以做三层敏感性分析：

1. **raw pair-weighted；**
2. **anchor-balanced：** 每个 anchor 对每个 lag 最多选一个 nearest match，接近 S-ACF 思路；
3. **inverse-intensity weighted：** 在有可信 intensity model 时降低高交易强度时段的权重。

第三种依赖 observation-process model，本身也可能错；因此不应隐藏 raw observed-pair result。

## 9.5 Overlapping target 会机械制造 autocorrelation

如果 target 是固定 horizon return/spread change

$$
Y_t^{(H)}=P_{t+H}-P_t,
$$

相邻 samples 的 target windows 可能高度重叠。即使 $P_t$ 是 independent-increment process，若 $0<\tau<H$，也有

$$
\operatorname{Corr}
\left(Y_t^{(H)},Y_{t+\tau}^{(H)}\right)
=
1-\frac{\tau}{H}
$$

（Brownian/increment-homoskedastic idealization）。这不是模型遗漏了 persistent signal，而是两个 labels 共享同一段未来 price movement。

因此 residual ACF 报告必须附带：

- pair 的 target-window overlap rate；
- non-overlapping subset 的 ACF；
- 在一个 “independent increments + same sampling/target construction” null simulation 下的 mechanical baseline。

## 9.6 Near-zero-lag correlation 先排查数据问题

特别高的短 lag residual correlation 可能来自：

- dealer-to-dealer double reporting 或 duplicate trade；
- 同一 client trade 被拆成多个中间 reports；
- as-of/corrected trades；
- stale benchmark 或 quote snapshot；
- 相同 prediction 被复用于多笔近邻 trades；
- bid/ask direction 未处理；
- prediction target 或 features 在多个 rows 上重叠。

所以 ACF 是 diagnostic 的入口，不是自动的 economic conclusion。

## 9.7 推荐的主要输出

对每个 model/version/test window，建议生成：

1. sampling-gap distribution：median、quantiles、burst/gap 图；
2. event-time ACF：$k=1,2,3,5,10$；
3. clock-time Gaussian-kernel ACF：例如 1m--4h；
4. pair count 与 Kish support curve；
5. cluster/block bootstrap pointwise band；
6. pair-weighted 与 bond-equal curve；
7. same-session、calendar-clock、business-clock sensitivity；
8. target-overlap audit 与 non-overlap ACF；
9. OU half-life，只在 curve positive、monotone 且单指数近似合理时；
10. residual ACF 按 side、liquidity、tenor、rating、time-of-day 分层。
