# 16. Binary logloss：第一棵 stump 就是 maximally selected two-proportion score test

考虑无权重 binary response

$$
y_i\in\{0,1\},
$$

以及 canonical logistic loss

$$
\ell(y,f)=\log(1+e^f)-yf.
$$

令

$$
p(f)=\frac{e^f}{1+e^f}.
$$

intercept-only initialization 为

$$
\widehat p_0=\bar y,
\qquad
\widehat F_0=\log\frac{\bar y}{1-\bar y}.
$$

第一轮 gradient 和 Hessian 是

$$
g_i=\bar y-y_i,
\qquad
h_i=\bar y(1-\bar y)=h_0.
$$

Hessian 对所有 observation 相同。对 root split，$G_R=-G_L$，代数化简得到

$$
\boxed{
\Gamma_s
=
\frac{n_Ln_R}{n\bar y(1-\bar y)}
(\bar y_L-\bar y_R)^2.
}
$$

右侧正是 pooled two-proportion score/Wald statistic 的平方形式。对预先固定 split，在 regular asymptotic regime 下

$$
\Gamma_s\Rightarrow\chi_1^2.
$$

但是 LightGBM 选择

$$
\max_{s\in\mathcal S}\Gamma_s,
$$

因此它是一个 **maximally selected two-proportion score statistic**。

## 16.1 条件于总正例数的精确 null

在 global null 下，给定

$$
N_+=\sum_i y_i,
$$

一个固定左子节点中的正例数

$$
N_{+,L}=\sum_{i\in L}y_i
$$

服从 hypergeometric distribution。因而 fixed split 可以做 exact conditional test。对整个 candidate scan，只需在固定 $N_+$ 下随机 permutation binary labels，并重跑同一 root scanner，就可精确纳入：

- thresholds 之间的相关性；
- features 之间的相关性；
- candidate count 差异；
- missing directions；
- 最大值选择。

如果使用 high-cardinality categorical ordering，而 ordering 本身由 gradients 决定，也必须在每次 permutation 中重新排序 categories，不能固定真实数据得到的 category order。

## 16.2 类别权重与 `scale_pos_weight`

一旦引入 observation weights、class weights 或 `scale_pos_weight`，$g_i,h_i$ 不再只由一个公共 $p_0$ 决定。weighted contrast 表示仍成立，但普通 hypergeometric permutation 不再自动精确；需要保持权重结构的 conditional randomization、parametric bootstrap 或直接模拟 fitted null model。

# 17. Poisson loss：homogeneous-rate score scan

对 count response $y_i\in\{0,1,2,\ldots\}$，使用 log link 的 Poisson negative log-likelihood：

$$
\ell(y,f)=e^f-yf+\text{constant}.
$$

intercept-only initialization 为

$$
\widehat F_0=\log\bar y.
$$

因此

$$
g_i=\bar y-y_i,
\qquad
h_i=\bar y.
$$

root split raw gain 为

$$
\boxed{
\Gamma_s
=
\frac{n_Ln_R}{n\bar y}
(\bar y_L-\bar y_R)^2.
}
$$

它是检验左右 Poisson rates 是否相同的 score statistic。给定总 count，在 homogeneous Poisson null 下，左侧 count 有 binomial/multinomial conditional law，因此也可通过 conditional simulation 校准整个 threshold maximum。

若 observation 有不同 exposure $e_i$，应把 offset $\log e_i$ 纳入初始模型；否则 raw split gain 会把 exposure 差异误当成 rate signal。

# 18. 相关、cluster 与时间序列数据：raw gain 不再自动可比

对你的 corporate-bond 场景，iid row 假设往往是最不可信的一层。考虑

$$
y=\mu\mathbf 1+\varepsilon,
\qquad
\varepsilon\mid X\sim N(0,\Sigma).
$$

对任意两个 split contrasts，

$$
\operatorname{Cov}(a_s^\top\varepsilon,a_t^\top\varepsilon\mid X)
=
\boxed{a_s^\top\Sigma a_t}.
$$

单个 raw gain 的期望是

$$
\mathbb E\Gamma_s
=a_s^\top\Sigma a_s,
$$

不再统一等于 $\sigma^2$。因此标准 LightGBM 会偏爱那些在真实 dependence structure 下 noise variance 更大的 contrasts。

## 18.1 Row count 与 effective sample size

一个 leaf 可能有 2,000 rows，但如果它们主要来自：

- 3 个 CUSIP；
- 2 个 issuers；
- 4 个 trading dates；
- 高度重叠的 rolling windows；

那么 $n_A=2000$ 并不意味着有 2,000 个独立信息单位。`min_data_in_leaf` 只约束 row count，不能阻止“cluster 很少但 rows 很多”的 split。

更合理的 leaf support diagnostics 应同时报告：

$$
(n_A,
\#\text{CUSIP},
\#\text{issuer},
\#\text{date},
\text{time span},
\widehat n_{\mathrm{eff}}).
$$

## 18.2 Covariance-aware fixed-split statistic

若 $\Sigma$ 已知，fixed split 可 studentize 为

$$
T_s
=
\frac{(a_s^\top r)^2}{a_s^\top\Sigma a_s}.
$$

在 Gaussian null 下

$$
T_s\sim\chi_1^2.
$$

全部 candidates 的 standardized scores 具有 correlation

$$
\widetilde K_{st}
=
\frac{a_s^\top\Sigma a_t}
{\sqrt{(a_s^\top\Sigma a_s)(a_t^\top\Sigma a_t)}}.
$$

于是最大统计量仍可通过相关 Gaussian simulation 精确校准。

Rabinowicz 与 Rosset 从 splitting criterion、stopping rule 和 leaf fit 三个层面研究了 correlated-data trees，说明忽略 correlation 不只是 standard error 问题，也会改变 tree structure 本身 [@rabinowicz2022correlated]。

## 18.3 实际可用的 resampling

$\Sigma$ 通常未知。可根据数据生成机制选择：

- **Cluster wild bootstrap**：以 issuer、CUSIP 或 date 为 cluster，对 cluster residual 乘 Rademacher/Mammen weights；
- **Moving-block / stationary bootstrap**：保留局部时间依赖；
- **Date-level permutation**：交换整日 residual blocks，而不是逐行打乱；
- **Parametric bootstrap**：先拟合一个 null covariance model，再模拟完整 error process；
- **Two-way clustered bootstrap**：当 issuer 与 date 两个方向都重要时使用多维 cluster scheme。

关键原则是：每个 bootstrap replicate 都要重新执行完整 candidate search。只对已经选中的 split bootstrap leaf means，无法校准 split-selection bias。

## 18.4 Point-in-time validation 仍然不可替代

即使 bootstrap null 校准正确，它也不替代 chronological out-of-sample validation。真实目标通常不是在 stationary null 下控制 Type I error，而是预测未来 RFQ/trade。因而至少需要：

1. training period 内用 resampling 诊断搜索偏差；
2. validation period 做 walk-forward tuning；
3. 最终 locked test period 只使用一次；
4. feature engineering、dedup、target definition 与 hyperparameter research 都视为对 validation 的 adaptive queries。

# 19. 真实 LightGBM 实现会在哪些地方偏离理想理论

本文的基准理论并不是伪装成“标准 LightGBM 全部细节的闭式定理”。下面逐项说明实现差异。

## 19.1 Histogram binning

数值特征先被离散成 bins，threshold scan 实际发生在 bin boundaries 上。若 binning 只依赖 $X$，则条件于 binned $X$ 后，候选集合仍是固定的；第 6 节有限样本理论直接适用于这个有限 dictionary。

影响是：

- candidate 数从所有 unique values 降为至多约 `max_bin`；
- 相邻 bins 仍高度相关；
- 粗 binning 会降低 signal resolution；
- 稀疏边界 bins 仍可能产生高方差 split。

## 19.2 Missing-value direction

对每个 threshold，算法可能比较 missing values 走左或走右。若 missing pattern 视为 $X$ 的一部分，那么这只是额外 candidates，可以放入 $\mathcal S$ 和 Gram matrix $K$。若 missingness 本身由 outcome 或未来信息产生，则问题变成 data leakage，而不是普通 multiplicity。

## 19.3 Leaf-wise / best-first growth

LightGBM source 在每一步先寻找各 leaf 的最佳 split，然后从 `best_split_per_leaf_` 中选 gain 最大的 leaf；若最大 gain 非正则停止 [@lightgbm_source_2026]。这正对应第 10.5 节的全局 leaf–candidate maximization。

## 19.4 Learning rate 在结构之后应用

ordinary GBDT source 先训练 tree、必要时 renew leaf outputs，然后调用 tree shrinkage，再更新 score [@lightgbm_source_2026]。因此第 13 节“$\eta$ 不改变第一棵树结构、只缩放 increment”的结论适用于普通 GBDT 路径。DART、特殊 refit 或自定义流程需要单独分析。

## 19.5 Native categorical features

低基数 category 可做 one-vs-rest scan。高基数 category 时，LightGBM 会使用 gradient/Hessian 构造 category score，例如当前 source 中的排序量近似

$$
\frac{G_c}{H_c+\texttt{cat\_smooth}},
$$

再对排序后的 categories 扫描前缀/后缀，并加入 `cat_l2` 等规则 [@lightgbm_source_2026]。

这里 category order 本身依赖 $Y$，所以候选 dictionary 不再只由 $X$ 决定。若先固定真实数据 category order，再套第 6 节 Gaussian max law，会漏掉一层选择。正确校准方法是：在每个 null simulation/permutation 中重新计算 gradients、重新排序 categories、重新扫描。

## 19.6 GOSS

Gradient-based One-Side Sampling 保留大 gradient observations，并抽样小 gradient observations [@ke2017lightgbm]。即使在第一棵树，大 gradient selection 也依赖 $Y$。因此：

- sample membership 不是 fixed design；
- 简单 fixed-dictionary Gaussian law 不再直接适用；
- 完整 null bootstrap 必须重跑 GOSS sampling 与 reweighting。

GOSS 原论文主要分析的是 gain approximation efficiency，而不是在 exhaustive adaptive search 后的显著性或 post-selection coverage。

## 19.7 Feature subsampling、bagging 与 `extra_trees`

若随机数独立于 $Y$，可以：

- 条件于实际 RNG realization，分析条件分布；或
- 同时对 data noise 与 algorithmic randomness 取平均，分析 unconditional risk。

随机化通常降低候选间和树间相关性，也减少每一步搜索范围；但它并不自动产生 p-values。

## 19.8 Monotone constraints、path smoothing 与 output clipping

- monotone constraints 会排除或修改不满足方向要求的 child outputs；
- `path_smooth` 把 child output 向 parent output 平滑；
- `max_delta_step` 截断 leaf output；
- L1/L2 改变 gain 为 piecewise quadratic/quadratic forms。

这些操作破坏简单的 orthogonal projection identity。它们仍可看成确定性算法的一部分，在 null simulation 中完整复现，但需要新的解析几何才能得到闭式选择分布。

## 19.9 Quantized gradients 与 floating-point ties

gradient quantization、histogram subtraction、distributed reduction 和 floating-point tie-breaking 会造成细微实现差异。若目标是验证理论而非复刻 production bit-level result，应先用 deterministic CPU、关闭量化与复杂 sampling；若目标是给 production model 做 null audit，则应尽可能重跑同一 executable 与同一 seed。
