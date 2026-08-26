# 4. 第一类方法：直接对 observation pairs 平滑

## 4.1 统一 kernel estimator

给定目标 lag $\tau$ 和 bandwidth $h$，定义权重

$$
w_{ij}(\tau)
=
K\left(\frac{d_{ij}-\tau}{h}\right).
$$

最直接的 covariance estimator 是

$$
\widehat\gamma_h(\tau)
=
\frac{
\sum_{i<j}w_{ij}(\tau)(X_i-\widehat\mu)(X_j-\widehat\mu)
}{
\sum_{i<j}w_{ij}(\tau)
}.
$$

再用 variance 归一化：

$$
\widehat\rho_h(\tau)
=
\frac{\widehat\gamma_h(\tau)}{\widehat\gamma(0)}.
$$

这个 estimator 的含义非常明确：目标 lag 附近的 pairs 越近，权重越大。

## 4.2 Rectangular kernel：slotting / DCF

若

$$
K(u)=\mathbf 1\{|u|\le1\},
$$

则

$$
\widehat\gamma_h(\tau)
=
\operatorname{average}
\left
\{
(X_i-\widehat\mu)(X_j-\widehat\mu):
|d_{ij}-\tau|\le h
\right\}.
$$

这就是 correlation slotting 的核心。Edelson--Krolik DCF 先为每个 pair 构造 unbinned discrete correlation，再按 pairwise lag 分箱，因此不要求时间插值 [@edelson1988]。

优点是可解释、容易审计；缺点是 bin 边界不连续，而且 bandwidth 太小时方差大、太大时会把不同 decay scale 混在一起。

## 4.3 Gaussian kernel：fuzzy slotting

若

$$
K(u)=e^{-u^2/2},
$$

每个 pair 对邻近 lag 都有平滑贡献。Rehfeld et al. 系统比较了 interpolation、Lomb--Scargle、slotting 与 kernel methods；在他们高度 irregular 的模拟中，Gaussian kernel 对 lag-1 ACF 的 RMSE 比 linear interpolation 低约 40%，而 interpolation 会显著高估 persistence time [@rehfeld2011]。

这里最重要的不是把“40%”当作普遍常数，而是理解其机制：当 gap distribution 很偏斜时，插值产生的大量伪观测会主导普通 ACF；kernel estimator 只使用真实 observation pairs。

## 4.4 Bandwidth 是 estimand-resolution 的一部分

小 $h$：

- temporal resolution 高；
- eligible pairs 少；
- 曲线噪声大，可能出现断点。

大 $h$：

- 方差较小；
- 但短时 microstructure decay 与长时 stale-model decay 会被混合；
- 相邻 lag estimates 高度平滑且相关。

因此 bandwidth 不应只靠“图看起来顺”。至少要报告：

1. ACF curve；
2. 每个 lag 的 raw pair count；
3. 权重集中度；
4. 对多个 bandwidth 的 sensitivity。

一个常用的权重集中度诊断是

$$
n_{\text{Kish}}(\tau)
=
\frac{\left(\sum w_{ij}(\tau)\right)^2}
{\sum w_{ij}(\tau)^2}.
$$

但它只描述 weights 有多集中，**不是**考虑 pair dependence 后的 inferential effective sample size。

## 4.5 Global normalization 还是 local correlation？

如果 residual 已经在 series 内标准化，可以直接估计 weighted product mean：

$$
\widehat\rho_{\text{prod}}(\tau)
=
\frac{\sum w_{ij}z_i z_j}{\sum w_{ij}}.
$$

另一个做法是对每个 lag 使用 weighted Pearson normalization：

$$
\widehat\rho_{\text{local}}(\tau)
=
\frac{\sum w_{ij}z_i z_j}
{\sqrt{\sum w_{ij}z_i^2}\sqrt{\sum w_{ij}z_j^2}}.
$$

后者逐点落在 $[-1,1]$，数值更稳定；但因为不同 lag 的 left/right endpoint composition 不同，它也在局部改变 scale。本文附带代码同时输出两者，让 normalization choice 可见。对近似平稳、已经 variance-standardized 的 residual，两者通常应接近；若差异很大，本身就是 composition/heteroskedasticity 的警报。

## 4.6 Equal-width bins 与 equal-population bins

固定时间宽度保留统一 temporal resolution，但远端 lag 可能 pair 很少。ZDCF 路线使用 equal-population binning 和 Fisher $z$ transform，主要为 sparse uneven light curves 改进 DCF 的有限样本表现 [@alexander2013]。

它适合“每个点都希望有接近的 pair support”的场景，但代价是不同位置的 lag resolution 不同。对交易 residual，我更建议把 equal-width kernel 作为主图，再把 equal-population bin 作为 robustness check，而不要把两种 x-axis resolution 混在同一条曲线里。

## 4.7 S-ACF 不是“另一个 kernel 名字”

Kreutzer et al. 提出的 S-ACF 对每个 anchor $t_i$ 和目标 $t_i+\tau$ 使用 selection function 选择最接近的 observed time，再用 mismatch distance 进行加权 [@kreutzer2023]。它在 regular sampling 上退化为标准 ACF，并在作者研究的 signal-shape/periodicity 问题上表现很好。

它与 all-pairs kernel estimator 的关键差别是 pair weighting：

- **all-pairs kernel：** 目标 lag 附近的所有 pairs 都贡献；交易密集区会产生很多 pairs；
- **S-ACF：** 每个 anchor 通常只选择一个接近目标的 observation；anchor 的影响更均衡。

因此两者不只是 computational variants；它们对应不同的 finite-sample averaging scheme。对“周期与形状恢复”，S-ACF 很自然；对 bond residual 的 conditional pair persistence，我会先用 all-pairs kernel，并额外做 anchor-balanced/S-ACF sensitivity。

# 5. 第二类方法：先估计 spectrum，再反变换

对连续时间弱平稳过程，autocovariance 与 spectral density 由 Fourier transform 联系。Scargle 发展了针对 unevenly spaced data 的 Fourier、autocorrelation 与 cross-correlation 工具 [@scargle1989]；之后也有专门的 irregular spectral estimators [@stoica2006; @geoga2025]。

这条路线的优点是：

- 周期性和多尺度频率结构更自然；
- 若 spectral estimate 被约束为非负，反变换得到的 covariance 更容易满足 positive-semidefinite 要求。

缺点是：

- window、aliasing 与 spectral smoothing 的选择不比 time-domain bandwidth 简单；
- 对“30 分钟后 residual 还剩多少”这种诊断，time-domain pair estimator 更直接；
- 交易时点内生和日内非平稳不会因为进入 frequency domain 自动消失。

所以 spectral method 应被视为补充路线，而不是 irregular ACF 的默认捷径。
