# 20. 三套可执行的第一棵树审计

## 20.1 Audit A：root exact/null scan

目标：回答“仅仅因为扫描很多 feature–threshold，root 在纯噪声下会产生多大 gain？”

**固定项：**

- binned feature matrix；
- missing values；
- sample weights；
- candidate admissibility；
- feature subsampling realization；
- LightGBM parameters。

**每个 replicate：**

1. 从 intercept-only null 模型生成 $y^{(b)}$；
2. 用与真实数据完全相同的 objective 计算 $g^{(b)},h^{(b)}$；
3. 训练 `num_boost_round=1, num_leaves=2`；
4. 记录 root raw gain、selected feature、threshold、child sizes；
5. 比较 observed gain 与 null distribution。

输出至少包括：

$$
\widehat p_{\max},
\quad
q_{0.95},q_{0.99},
\quad
\Pr_0(\widehat j=j),
\quad
\text{null child-size distribution}.
$$

后两项可直接揭示 high-cardinality feature bias 与 endpoint preference。

## 20.2 Audit B：完整第一棵 tree null distribution

目标：量化 leaf-wise repeated search 的总复杂度。

对每个 null replicate，按 production 参数训练完整一棵 tree，记录：

- leaf 数、最大深度、tree shape；
- 每一步 split gain；
- total gain $\Gamma_{\mathrm{tree}}$；
- leaf row count 与 cluster support；
- feature usage；
- 预测范数 $\|\widehat u\|^2$；
- empirical effective df
  $$
  \widehat{\operatorname{df}}
  =B^{-1}\sum_b\Gamma_{\mathrm{tree}}^{(b)}/\widehat\sigma_b^2.
  $$

如果 observed training gain 只处在 null 分布中部，它不是“没有预测价值”的充分证明，但说明训练 gain 本身几乎没有证据价值。

## 20.3 Audit C：honest split-gain audit

把可用 training history 按时间分成 structure sample $A$ 与 honest sample $B$：

1. 只在 $A$ 上选择完整 tree structure；
2. 固定每一个 selected split；
3. 在 $B$ 上重新计算同一 split 的 gain；
4. 可选：只用 $B$ 重估 leaf outputs；
5. 报告
   $$
   (\Gamma_{m,A},\Gamma_{m,B})
   $$
   与 sign stability。

典型 winner's curse 表现是

$$
\Gamma_{m,A}>0,
\qquad
\Gamma_{m,B}\approx0
\ \text{或}<0
$$

（“负 gain”指在 honest sample 上固定 split 后，相对于 honest parent mean 的 SSE improvement 经噪声校正或 out-of-sample loss difference 为负；普通 in-sample SSE split gain本身不会为负）。

honesty 把 structure selection 与 leaf estimation 分开，是降低 adaptive bias 的经典思路 [@athey2016recursive]。对时间序列必须 chronological split，不能随机把未来 observation 放进 structure sample。

# 21. 对 corporate-bond LightGBM 的直接含义

结合高维 features、重复 CUSIP、issuer dependence 和 irregular time，第一棵树理论给出以下优先级。

## 21.1 不要把 root gain 当作“最强 feature 的显著性”

root gain 是

$$
\max_{j,k}\text{score}_{j,k}^2,
$$

不是预先固定 feature/threshold 的 score。它可以用于 prediction，但不能被解释成 ordinary one-df test。

## 21.2 强正则首先应限制 search，而不只是 shrink leaf values

更直接的组合是：

$$
\texttt{max\_bin}\downarrow,
\quad
\texttt{feature\_fraction\_bynode}\downarrow,
\quad
\texttt{num\_leaves}\downarrow,
\quad
\texttt{min\_data\_in\_leaf}\uparrow,
$$

再配合 `extra_trees` robustness benchmark 与 walk-forward early stopping。单独把 `lambda_l2` 从 1 提到 10，若 root 有几十万 rows，理论影响可能很小。

## 21.3 `min_data_in_leaf` 应补充 cluster support rule

production LightGBM 未必能原生约束 unique CUSIP/issuer/date。至少在训练后做 leaf audit；若某 leaf 的 prediction 主要由极少 cluster 支撑，应将其视为高风险结构。也可以预先降低重复 observation 权重，使每个 cluster 的总权重更接近可控。

## 21.4 用 null distribution 校准“正常的训练 gain”

对 667 个高度相关 features，简单用 $M=667\times254$ 的独立近似会太粗。最可靠的是固定真实 feature matrix 与 pipeline，模拟/重采样 null target，重跑第一棵树。这样 category cardinality、missingness、binning 和 feature correlation 都被自动纳入。

## 21.5 SHAP 解释的是 selected predictive function，不是发现证据

即使第一棵树的 SHAP decomposition algebraically correct，它也没有修正该树是从大量 candidates 中筛选出来的事实。稳定性应通过：

- 时间 folds 的 feature/split recurrence；
- block bootstrap selection probability；
- honest gain；
- locked future test contribution；

来衡量，而不是仅看一次训练的 gain importance 或 mean absolute SHAP。

# 22. 哪些结果是“完整”的，哪些仍然开放

| 设定 | 本文可得到的结论 | 精确性 |
|---|---|---|
| L2、fixed split、iid Gaussian、fixed $X$ | $\Gamma/\sigma^2\sim\chi_1^2$ | 有限样本精确 |
| L2、root exhaustive scan | $\Gamma_{\max}/\sigma^2=\max Z_s^2$, $Z\sim N(0,K)$ | 有限样本精确 |
| 单连续 feature、interior thresholds | Brownian-bridge supremum | 渐近 |
| 多 features | Gaussian maximum；union bound；独立 extreme-value benchmark | 精确表示 / 上界 / benchmark |
| L2、完整 first tree、无 penalty | orthogonal tree-Haar basis；total gain identity | 有限样本精确 |
| 同上、fixed structure | $\chi^2_{L-1}$ | 有限样本精确 |
| 同上、adaptive path + signs | polyhedral selection event | 有限样本精确 |
| Gaussian selective target | truncated-normal law | 有限样本精确（已知 $\Sigma$） |
| 同上、effective df | $\mathrm{df}=E\Gamma/\sigma^2$ | 有限样本精确 |
| learning rate 下 train/test risk | 闭式 incremental formulas | null + fixed-design 条件下精确 |
| Ridge/L1 | exact gain formula；selection region quadratic/piecewise quadratic | gain 精确，推断通常需模拟 |
| Logistic/Poisson fixed split | score statistic | 渐近；部分条件 null 可精确 |
| Correlated Gaussian | covariance-aware contrast law | 已知 $\Sigma$ 时精确 |
| Native categorical / GOSS | 完整算法 null simulation | Monte Carlo / bootstrap |
| 第 2 棵及以后 | gradient 已依赖此前 adaptive path | 本文不提供完整闭式理论 |

这里“完整”必须理解为**在明确假设下完整**。实际 LightGBM 的每一个工程开关都可以被放入一个可重复的 null simulator，但不一定都能压缩成一条漂亮闭式公式。

# 23. 总结

第一棵 boosting tree 的统计结构可以被完整地分解为：

$$
\boxed{
\text{initial score}
\longrightarrow
\text{fixed gradients/Hessians}
\longrightarrow
\text{weighted pseudo-response}
\longrightarrow
\text{greedy max over split contrasts}.
}
$$

平方损失下，它进一步变成

$$
\boxed{
\Gamma_s=(a_s^\top r)^2.
}
$$

由此几乎所有关键现象都能统一解释：

- fixed split 的 $\chi_1^2$；
- threshold scan 的 Brownian bridge；
- 跨 feature 搜索的 $O(\log M)$ 最大噪声 gain；
- complete first tree 的正交 tree-Haar expansion；
- path selection 的 polyhedral geometry；
- search degrees of freedom；
- learning rate 下的 train/test risk gap；
- ridge 对大 root 相对较弱、对小 leaves 较强；
- correlated rows 使 raw gains 的 noise scale 不再统一。

最重要的结论不是“LightGBM 的统计性质很差，所以不能用”，而是：

> **第一棵树是一台可精确刻画的自适应最大化机器。它在 prediction 上可以非常有效，但训练 gain、selected threshold 和 feature importance 的解释必须支付 search cost。**

对 prediction，合理目标是让真实 signal gain $O(n)$ 稳定压过 search noise $O(\log M)$，并用严格的未来样本检验。对 inference，则必须使用 multiplicity adjustment、selective inference、honesty 或完整算法 bootstrap。对你的 bond model，第一步最有价值的不是再猜一个更大的 `lambda_l2`，而是实际跑出：

$$
\boxed{
\text{root null law}
+
\text{full-first-tree null law}
+
\text{honest gain audit}
+
\text{leaf cluster support report}.
}
$$

这四项会把“强正则是否必要”从直觉争论变成可以测量的问题。

# 附录 A. Split gain identity 的逐步证明

对 parent $A=L\cup R$，无 ridge 时

$$
\Gamma_s
=
\frac{G_L^2}{H_L}
+
\frac{G_R^2}{H_R}
-
\frac{(G_L+G_R)^2}{H_L+H_R}.
$$

通分：

$$
\begin{aligned}
\Gamma_s
&=
\frac{
G_L^2H_R(H_L+H_R)
+G_R^2H_L(H_L+H_R)
-(G_L+G_R)^2H_LH_R
}
{H_LH_R(H_L+H_R)}\\
&=
\frac{
G_L^2H_R^2
+G_R^2H_L^2
-2G_LG_RH_LH_R
}
{H_LH_R(H_L+H_R)}\\
&=
\frac{H_LH_R}{H_L+H_R}
\left(
\frac{G_L}{H_L}-\frac{G_R}{H_R}
\right)^2.
\end{aligned}
$$

又因为 weighted leaf pseudo-response mean 为

$$
\bar z_{L,H}=-G_L/H_L,
\qquad
\bar z_{R,H}=-G_R/H_R,
$$

所以

$$
\Gamma_s
=
\frac{H_LH_R}{H_A}
(\bar z_{L,H}-\bar z_{R,H})^2
=
\langle z,b_s\rangle_H^2.
$$

# 附录 B. Tree-Haar orthogonality 的形式化证明

设 $a_m$ 对应把 node $A_m$ 分成 $L_m,R_m$：

$$
a_m=c_m
\left(
\frac{\mathbf1_{L_m}}{|L_m|}
-
\frac{\mathbf1_{R_m}}{|R_m|}
\right),
\qquad
c_m=\sqrt{\frac{|L_m||R_m|}{|A_m|}}.
$$

对 $m<k$，recursive partitioning 保证 $A_k$ 要么与 $A_m$ 不交，要么完全包含在 $L_m$ 或 $R_m$ 中。

若不交，支持不交，故内积为零。

若 $A_k\subseteq L_m$，则 $a_m$ 在 $A_k$ 上为常数 $c_m/|L_m|$。而

$$
\sum_{i\in A_k}a_k(i)
=
\sqrt{\frac{|L_k||R_k|}{|A_k|}}
(1-1)=0.
$$

所以

$$
a_m^\top a_k
=
\frac{c_m}{|L_m|}
\sum_{i\in A_k}a_k(i)
=0.
$$

$A_k\subseteq R_m$ 同理。故所有 selected contrasts 两两正交。

# 附录 C. Gaussian max simulation 的伪代码

```text
Input:
    fixed feature matrix X
    exact LightGBM one-tree parameters theta
    null model P0(y | X)
    number of replicates B

Observed:
    fit one tree to y_obs
    T_obs <- selected root gain or total first-tree gain

For b = 1,...,B:
    y_b <- simulate from P0(y | X)
    rebuild every outcome-dependent object:
        gradients / Hessians
        categorical ordering
        GOSS sample
        split path
    fit exactly one tree with theta
    T_b <- same statistic

p_hat <- (1 + count(T_b >= T_obs)) / (B + 1)
quantiles <- empirical quantiles of {T_b}
```

若只想检验 root scan，设置 `num_leaves=2`；若想估计完整第一棵树的 search df，使用 production leaf constraints 并保存 total raw gain。

# 附录 D. 未知 $\sigma$ 的处理

LightGBM raw gain 没有除以 $\sigma^2$。若目标只是 prediction，不需要估计 $\sigma$；若要把 gain 解释成 test statistic，则 scale 必须处理。

可选方案：

1. 在一个独立 calibration sample 估计 $\sigma$；
2. 在 Gaussian parametric bootstrap 中，对 observed 和每个 replicate 使用相同的 $\widehat\sigma$ procedure；
3. 在 exchangeable null 下 permutation labels/residuals；
4. 用 studentized cluster/wild bootstrap；
5. 避免把 raw-gain threshold 跨不同 target scaling 直接比较。

对于 squared-error target，若将 $y$ 乘常数 $c$，raw gains 会乘 $c^2$。因此固定 `min_gain_to_split` 对 target units 极其敏感。

# 附录 E. 复现实验的最小参数建议

为了让理论与实现尽量对齐，第一轮实验建议：

```python
params = {
    "objective": "regression_l2",
    "boosting": "gbdt",
    "num_leaves": 2,          # root-only theory first
    "learning_rate": 1.0,     # inspect unshrunk gain; structure is unchanged
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
    "path_smooth": 0.0,
    "max_delta_step": 0.0,
    "feature_fraction": 1.0,
    "feature_fraction_bynode": 1.0,
    "bagging_fraction": 1.0,
    "extra_trees": False,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
```

先验证：

$$
\text{model-reported root gain}
\approx
\max_s\frac{n_Ln_R}{n}(\bar y_L-\bar y_R)^2.
$$

再按顺序加入：

1. `num_leaves > 2`；
2. `lambda_l2`；
3. `min_gain_to_split`；
4. feature/node subsampling；
5. categorical features；
6. GOSS；
7. time/cluster bootstrap。

这种 staged validation 可以精确定位每个工程选项破坏了哪一条理论 identity。
