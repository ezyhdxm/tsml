---
title: "第一棵 LightGBM 树的统计理论"
subtitle: "从二阶增益、最大统计量到选择后推断与有效自由度"
author: "TSML Research Notes"
date: "2026-08-23"
lang: zh-CN
bibliography: references.bib
link-citations: true
reference-section-title: "参考文献"
toc-title: "目录"
abstract-title: "摘要"
abstract: |
  本文对 gradient boosting 的第一棵树进行一套尽可能完整、并且与 LightGBM 实现口径一致的统计分析。第一棵树的特殊性在于：初始化常数模型确定后，整棵树生长期间使用的 gradient 与 Hessian 固定不变，因此它可以被还原为在一组固定 pseudo-response 上进行的 greedy weighted regression-tree search。对平方损失、Gaussian noise、固定设计以及不使用 leaf penalty 的基准模型，本文给出 root split 的有限样本精确分布、连续特征 threshold scan 的 Brownian-bridge 极限、跨特征最大统计量、signal 下的非中心分布与选对 split 的概率界；进一步证明完整第一棵 leaf-wise tree 对应一组自适应选择的正交 tree-Haar contrasts，其累计训练 gain、投影范数和有效自由度之间存在精确恒等式。本文还刻画了整条 tree path 的 polyhedral selection event，并据此说明选择后检验与置信区间如何构造；然后推导 learning rate 下的训练误差改善、纯噪声样本外风险和 optimism。最后，我们扩展到 ridge/L1、一般 smooth loss、binary logloss、Poisson loss、相关与时间序列数据，并逐项解释 histogram、categorical ordering、GOSS、missing-value direction、path smoothing 等 LightGBM 实现细节会破坏哪一层理想化理论。本文的目标不是宣称标准 LightGBM 自带统计推断，而是精确回答：第一棵树中哪些结论是有限样本精确的，哪些只是渐近的，哪些必须通过条件模拟或 block bootstrap 才能校准。
---

> **Source of truth.** 请编辑本文件 `report.md`。渲染结果是同目录下的 `index.html`。本文使用普通 Markdown 与 LaTeX delimiters；HTML 由 Pandoc 渲染为内嵌 CSS、浏览器原生 MathML 的单文件版本，不依赖外部 MathJax CDN。

# 1. 问题、结论与分析边界

Gradient boosting 通常被描述为“每轮拟合 residual”，但这个说法掩盖了第一棵树内部真正发生的统计搜索：对每个可分 leaf、每个 feature、每个 admissible threshold，算法计算一个经验 gain，然后选取最大的候选。LightGBM 又采用 leaf-wise / best-first growth，因此在 root 之后，它继续从整棵当前树的所有 leaf–feature–threshold 候选中选最大 gain。这个过程天然包含 max-statistic、winner's curse、selection bias 与 search degrees of freedom。

第一棵树却比后续树容易得多。设初始常数预测为 $F_0$。在一个 boosting iteration 内，LightGBM 先计算一次

$$
(g_i,h_i)
=
\left(
\left.\frac{\partial \ell(y_i,f)}{\partial f}\right|_{f=F_0},
\left.\frac{\partial^2 \ell(y_i,f)}{\partial f^2}\right|_{f=F_0}
\right),
$$

然后整棵树都用这组固定 gradient 与 Hessian 生长。只有树结构与 leaf output 确定后，learning rate 才缩放整棵树。这个“一个 iteration 内 gradient/Hessian 固定”的事实，是本文能够做完整分析的根本原因。它也与 gradient boosting 的函数梯度观点、二阶 tree boosting 的标准目标和 LightGBM 的实现一致 [@friedman2001; @chen2016xgboost; @ke2017lightgbm; @lightgbm_source_2026]。

本文最核心的结论可以先压缩为五行。对平方损失、无 penalty 的第一棵树，令

$$
r=y-\bar y\mathbf 1.
$$

任意候选 split $s$ 对应一个单位长度、与当前 parent leaf 常数方向正交的 contrast $a_s$。则 LightGBM 口径的 raw split gain 为

$$
\boxed{\Gamma_s=(a_s^\top r)^2.}
$$

在 Gaussian global null 下，固定 split 满足

$$
\boxed{\Gamma_s/\sigma^2\sim\chi_1^2,}
$$

而算法实际使用的是

$$
\boxed{\Gamma_{\max}/\sigma^2=\max_{s\in\mathcal S}Z_s^2,\qquad Z\mid X\sim N(0,K).}
$$

若完整第一棵树选出 $L$ 个 leaves，则所选 $L-1$ 个 contrasts 两两正交，并且

$$
\boxed{
\Gamma_{\mathrm{tree}}
=
\sum_{m=1}^{L-1}(a_{\widehat s_m}^\top r)^2
=
\|\widehat u\|_2^2,
}
$$

其中 $\widehat u$ 是第一棵树对 centered response 的未缩放拟合。纯噪声下的有效自由度满足精确恒等式

$$
\boxed{
\operatorname{df}_{\mathrm{tree}}
=
\frac{1}{\sigma^2}\mathbb E\Gamma_{\mathrm{tree}}.
}
$$

因此，一棵只有两个 leaves 的 stump 虽然只估计一个 contrast coefficient，但如果它从很多候选中选最大值，其有效自由度可以接近 $2\log M$，远大于名义参数个数 $1$。

## 1.1 本文覆盖什么

本文分四层分析：

1. **有限样本精确层。** 固定 $X$、Gaussian noise、平方损失、数值型候选 split、无 L1/L2/path smoothing 时，root 的联合 null law、完整 tree path 的选择事件、累计 gain 与自由度恒等式均是有限样本精确的。
2. **渐近层。** 连续 feature 的 threshold process 收敛到 Brownian bridge；一般正确设定 likelihood 的 fixed-split gain 渐近为 score statistic。
3. **条件模拟层。** Ridge、categorical ordering、GOSS、复杂 missing handling 或完整 LightGBM 实现可以通过“固定 $X$，重新模拟 null $Y$，重跑完全相同的一棵树”获得有限样本近似校准。
4. **依赖数据层。** CUSIP、issuer、date 或相邻时间 observation 相关时，必须显式建模 $\Sigma$，或者使用 cluster/wild/block bootstrap；逐行 iid permutation 通常不再有效。

## 1.2 本文不声称什么

本文不声称：

- 第一棵 greedy tree 是给定 leaf 数下的全局最优 tree；它只是 best-first greedy path。
- 标准 LightGBM 的 feature importance、SHAP 或 split count 自动具有显著性解释。
- 第一棵树的漂亮理论会自动延伸到第 $t\ge2$ 棵树。后续 gradient 已经依赖前面所有自适应选择，理论复杂度显著上升。
- 固定的 `min_gain_to_split` 等同于统一的 p-value cutoff。若没有按候选集合、节点大小和依赖结构校准，它只是一个经验 complexity penalty。

# 2. 记号与 LightGBM 的精确 gain 口径

观测为

$$
\mathcal D_n=\{(x_i,y_i):i=1,\ldots,n\},
$$

经验风险写为

$$
\mathcal L_n(F)=\sum_{i=1}^n\ell(y_i,F(x_i)).
$$

第一轮之前的常数模型为

$$
\widehat F_0
=
\arg\min_{c\in\mathbb R}
\sum_{i=1}^n\ell(y_i,c).
$$

第一棵 tree 定义一个 partition

$$
\Pi=\{A_1,\ldots,A_L\},
$$

并在 leaf $A_\ell$ 上输出常数 $w_\ell$：

$$
f_1(x)=\sum_{\ell=1}^L w_\ell\mathbf 1\{x\in A_\ell\}.
$$

记初始点的 gradient 与 Hessian 为

$$
g_i
=
\left.\partial_f\ell(y_i,f)\right|_{f=\widehat F_0},
\qquad
h_i
=
\left.\partial_f^2\ell(y_i,f)\right|_{f=\widehat F_0}.
$$

对任意 observation set $A$，定义

$$
G_A=\sum_{i:x_i\in A}g_i,
\qquad
H_A=\sum_{i:x_i\in A}h_i.
$$

由于 $\widehat F_0$ 是 intercept-only empirical minimizer，通常有一阶条件

$$
\sum_{i=1}^n g_i=0.
$$

这意味着 root node 的 gradient sum 为零；后续 node 的 $G_A$ 一般不为零。

## 2.1 二阶 surrogate 与 leaf output

在 $F_0$ 处做二阶 Taylor expansion：

$$
\mathcal L_n(F_0+f)
\approx
\mathcal L_n(F_0)
+
\sum_{i=1}^n
\left[g_if_i+\frac12h_if_i^2\right].
$$

加入 LightGBM/XGBoost 风格的 leaf penalties，固定 partition 后的 surrogate 是

$$
Q_\Pi(w)
=
\sum_{\ell=1}^L
\left[
G_\ell w_\ell
+
\frac12(H_\ell+\lambda_2)w_\ell^2
+
\lambda_1|w_\ell|
\right].
$$

定义 soft-threshold operator

$$
S_{\lambda_1}(u)
=
\operatorname{sign}(u)(|u|-\lambda_1)_+.
$$

逐 leaf 最小化得到

$$
\boxed{
\widehat w_A
=-\frac{S_{\lambda_1}(G_A)}{H_A+\lambda_2}.
}
$$

当 $\lambda_1=0$ 时，profiled surrogate contribution 为

$$
\min_{w_A}
\left[
G_Aw_A+\frac12(H_A+\lambda_2)w_A^2
\right]
=
-\frac12\frac{G_A^2}{H_A+\lambda_2}.
$$

## 2.2 必须澄清的 $1/2$ 因子

许多论文把一个 split 带来的二阶目标下降写为

$$
\Delta_Q(s)
=
\frac12
\left[
\frac{G_L^2}{H_L+\lambda_2}
+
\frac{G_R^2}{H_R+\lambda_2}
-
\frac{G_A^2}{H_A+\lambda_2}
\right].
$$

LightGBM 内部用于比较 threshold 的 **raw gain** 则是上式的两倍：

$$
\boxed{
\Gamma_s
=
\frac{S_{\lambda_1}(G_L)^2}{H_L+\lambda_2}
+
\frac{S_{\lambda_1}(G_R)^2}{H_R+\lambda_2}
-
\frac{S_{\lambda_1}(G_A)^2}{H_A+\lambda_2}.
}
$$

在没有 path smoothing、monotone constraint 或 max output clipping 的基本情形中，LightGBM 要求

$$
\Gamma_s>\tau,
\qquad
\tau=\texttt{min\_gain\_to\_split},
$$

并报告约为 $\Gamma_s-\tau$ 的有效 split gain。本文所有与 LightGBM 参数对接的阈值都使用 $\Gamma$ 口径；讨论真实二阶 objective reduction 时才使用 $\Delta_Q=\Gamma/2$。这个区分会消除实践中常见的两倍误差。

# 3. 第一棵树等价于 weighted pseudo-response tree

假设 $h_i>0$，定义 Newton pseudo-response

$$
z_i=-\frac{g_i}{h_i},
$$

以及 Hessian-weighted inner product

$$
\langle u,v\rangle_H
=
\sum_{i=1}^nh_i u_iv_i,
\qquad
\|u\|_H^2=\langle u,u\rangle_H.
$$

逐 observation 完成平方：

$$
g_if_i+\frac12h_if_i^2
=
\frac12h_i(f_i-z_i)^2
-
\frac12\frac{g_i^2}{h_i}.
$$

因此，在不考虑与 $f$ 无关的常数时，第一棵树的二阶问题就是

$$
\min_{f\in\mathcal T}
\frac12\sum_{i=1}^nh_i(f_i-z_i)^2
+
\frac{\lambda_2}{2}\sum_{\ell=1}^Lw_\ell^2
+
\lambda_1\sum_{\ell=1}^L|w_\ell|,
$$

其中 $\mathcal T$ 是允许的 piecewise-constant tree class。

> **命题 1（pseudo-response 等价）.** 在第一 boosting round 内，只要 gradient 与 Hessian 固定，二阶 tree learner 等价于在 pseudo-response $z_i=-g_i/h_i$ 上拟合一棵 Hessian-weighted、带 leaf-wise L1/L2 penalty 的 regression tree。

无 penalty 时，leaf $A$ 的最优值就是 weighted mean：

$$
\widehat w_A
=
\bar z_{A,H}
=
\frac{\sum_{i\in A}h_iz_i}{H_A}
=-\frac{G_A}{H_A}.
$$

有 ridge 时，weighted mean 被缩小：

$$
\widehat w_A
=
\frac{H_A}{H_A+\lambda_2}\bar z_{A,H}.
$$

这个表示非常重要。它说明第一棵树并不是一个神秘的 nonlinear boosting object；它首先是一棵普通 weighted least-squares tree，只是 response 和 weights 由 loss 在 intercept-only model 处诱导出来。

# 4. 任意 fixed split 的标准化 contrast 表示

设当前 parent leaf 为 $A$，候选 split $s$ 把它分成 $L$ 与 $R$。令

$$
H_A=H_L+H_R.
$$

定义 weighted split contrast $b_s\in\mathbb R^n$：

$$
\boxed{
b_s(i)
=
\sqrt{\frac{H_LH_R}{H_A}}
\left[
\frac{\mathbf 1\{i\in L\}}{H_L}
-
\frac{\mathbf 1\{i\in R\}}{H_R}
\right].
}
$$

它满足

$$
\|b_s\|_H=1,
\qquad
\langle b_s,\mathbf 1_A\rangle_H=0.
$$

并且无 penalty 时，weighted ANOVA identity 给出

$$
\begin{aligned}
\Gamma_s
&=
\frac{G_L^2}{H_L}
+
\frac{G_R^2}{H_R}
-
\frac{G_A^2}{H_A}\\
&=
\frac{H_LH_R}{H_A}
\left(
-\frac{G_L}{H_L}+\frac{G_R}{H_R}
\right)^2\\
&=
\boxed{\langle z,b_s\rangle_H^2}.
\end{aligned}
$$

> **命题 2（fixed split = squared score contrast）.** 第一棵树中任意一个 fixed candidate split 的 raw gain，都是 pseudo-response 在一个单位 weighted contrast 方向上的投影平方。

这使 multiple split search 的本质完全显露出来：算法在一个由 feature–threshold 生成的巨大 contrast dictionary 中寻找

$$
\max_{s\in\mathcal S_A}|\langle z,b_s\rangle_H|.
$$

# 5. 平方损失：从 LightGBM gain 到普通 SSE reduction

取

$$
\ell(y,f)=\frac12(y-f)^2.
$$

则

$$
\widehat F_0=\bar y,
\qquad
r_i=y_i-\bar y,
\qquad
g_i=-r_i,
\qquad
h_i=1,
\qquad
z_i=r_i.
$$

对 parent leaf $A$，令 $n_A=|A|$，split 后样本量为 $n_L,n_R$。定义 Euclidean unit contrast

$$
\boxed{
a_s
=
\sqrt{\frac{n_Ln_R}{n_A}}
\left[
\frac{\mathbf 1_L}{n_L}
-
\frac{\mathbf 1_R}{n_R}
\right].
}
$$

则

$$
\|a_s\|_2=1,
\qquad
a_s^\top\mathbf 1_A=0,
$$

并且

$$
a_s^\top r
=
\sqrt{\frac{n_Ln_R}{n_A}}
(\bar r_L-\bar r_R).
$$

无 penalty 时，raw gain 精确为

$$
\boxed{
\Gamma_s
=(a_s^\top r)^2
=
\frac{n_Ln_R}{n_A}
(\bar y_L-\bar y_R)^2.
}
$$

另一方面，parent-only fit 的 SSE 与 split fit 的 SSE 之差是

$$
\operatorname{SSE}(A)-\operatorname{SSE}(L)-\operatorname{SSE}(R)
=
\frac{n_Ln_R}{n_A}
(\bar y_L-\bar y_R)^2.
$$

因此平方损失下存在一个精确而非近似的等式：

$$
\boxed{
\Gamma_s=\text{training SSE reduction}.
}
$$

二阶 surrogate objective reduction 是 $\Gamma_s/2$，只是因为我们的 loss 定义里有 $1/2$。
