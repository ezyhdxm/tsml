---
title: "Corporate Bond Quote Quantity 的可分离性与交互结构"
subtitle: "经济机制、识别、统计检验与生产化路线"
author: "TSML Research Note"
date: "2026-08-26"
lang: zh-CN
abstract-title: "摘要"
abstract: |
  本报告研究一个先于模型选择的结构性问题：corporate-bond dealer quote 的 quantity effect 能否从 bond、dealer 与 market-state features 中分离，还是必须被建模为条件交互。我们首先在 quantity-neutral spread concession 上定义可分离性，然后说明常见的“两阶段 residual correction”在 quantity 与其他 covariates 相关时一般不识别 additive quantity curve。结合 corporate-bond transaction cost、dealer inventory、block trading、dealer runs、electronic RFQ 与 semiparametric regression 文献，报告提出一个分层结构：quantity-independent baseline、由 liquidity capacity 决定的横向尺度、由 risk/capital state 决定的纵向幅度，以及少量真正改变曲线形状的 interaction。最后给出 point-in-time target、quantity 语义、endogeneity、overlap、cross-fitting、purified functional ANOVA、out-of-time model comparison 与生产化验证方案。核心建议不是预先假定 quantity 可分离或不可分离，而是把“additive → scale → amplitude → sparse shape interaction → unrestricted challenger”设计为可证伪的模型阶梯。
---

# 执行摘要

## 核心结论

本报告的出发点是正确的：在为 $100\text{k},250\text{k},500\text{k},1\text{mm},2\text{mm},5\text{mm}$ 分别产生 fair value 之前，应先问 quantity 是否携带一个可独立校正的低维效应。若成立，最简结构为

$$
E[Y\mid X,Q=q]=m(X)+h(q),
$$

其中 $Y$ 是 quote 相对于 quantity-neutral mid 的 signed execution concession，$X$ 包含除 quantity 以外的 point-in-time bond、dealer 与 market-state information。此时从观测 quantity $q$ normalize 到 reference quantity $q_0$ 只需

$$
Y^{\mathrm{norm}}(q_0)
=
Y(q)-\{h(q)-h(q_0)\}.
$$

但公开文献与市场结构都提示，**raw dollar quantity 的全局可分离性不应被当作默认事实**：

1. pooled 数据中的 larger-trade discount 可能由 client composition 与 bargaining power 驱动；控制 client identity 后，size slope 可以反转为正，即同一客户的大单更贵 [@pinter2024size]；
2. dealer inventory capacity、information risk、market stress、competition 与 hedgeability 对大 size 的边际影响通常不同 [@bessembinder2018capital; @goldberg2021liquidity; @jacobsen2025receiving]；
3. dealer-run 的 displayed quantity 可能是 nonbinding “up-to size”，并非某个精确 $q$ 的成交价格。BondCliQ 研究中仅一部分 quote 带 size，而且带 size 的 quote 与真实 client-dealer trade-size distribution 差异很大 [@hendershott2026quote]。

因此，更可信的工作假设是

$$
\boxed{
E[Y\mid X,Q]
=
m(X)
+A(X)\,
h\!\left(\log Q-\log Q^\star(X)\right)
+r(X,Q),
}
$$

其中：

- $m(X)$：quantity-independent baseline concession；
- $Q^\star(X)$：当前 bond/state 的 liquidity capacity，控制 quantity curve 的**水平位置**；
- $A(X)$：risk、inventory-capital 与 stress state，控制 curve 的**纵向幅度**；
- $r(X,Q)$：在 scale 与 amplitude 归一化后仍存在的**真正 shape interaction**。

报告的主要实证任务是依次检验：

$$
\begin{aligned}
\mathcal M_0 &: m(X)+h(Q),\\
\mathcal M_1 &: m(X)+h(Q/Q^\star(X)),\\
\mathcal M_2 &: m(X)+A(X)h(Q/Q^\star(X)),\\
\mathcal M_3 &: \mathcal M_2+\text{sparse varying-curve interactions},\\
\mathcal M_4 &: g(X,Q)\quad\text{unrestricted challenger}.
\end{aligned}
$$

只有当更复杂模型在严格 out-of-time、matched-pair 和 conditional-calibration tests 中产生稳定增益，才保留 interaction。

## 一个必须先纠正的方法论问题

“先用其他 features 拟合，再对 residual 做 quantity correction”只有在 $Q$ 与 $X$ 近似独立，或第一阶段被特别设计为不吸收 quantity signal 时才可靠。即使真实模型完全 additive，

$$
Y=m(X)+h(Q)+\varepsilon,
$$

普通第一阶段得到

$$
E[Y\mid X]
=m(X)+E[h(Q)\mid X],
$$

因此 residual 是

$$
R
=Y-E[Y\mid X]
=h(Q)-E[h(Q)\mid X]+\varepsilon,
$$

而不是 $h(Q)+\varepsilon$。当 large quantity 更常出现在 liquid bonds、特定 clients、特定 dealers 或平稳市场中时，强大的 tree model 会通过这些 proxy 吸收 quantity effect，随后对 $R$ 拟合 $Q$ 会产生 attenuation，甚至错误曲线。

可行替代是：

- joint additive backfitting；
- 对 quantity spline basis 做 Robinson-style double residualization [@robinson1988root];
- 使用 cross-fitting 和 Neyman-orthogonal score 降低 nuisance-model overfit bias [@chernozhukov2018double];
- 对 interaction 做可识别的 functional-ANOVA purification，而不是直接解释 tree SHAP interactions [@hooker2007generalized; @lengerich2020purifying]。

## 最值得优先检验的 hypotheses

按预计强度排序：

1. **relative-liquidity interaction**：$Q/Q^\star$ 比 raw $Q$ 更稳定；
2. **risk-amplitude interaction**：quantity 应与 spread risk、CS01、unwind horizon 和 hedgeability 共同决定 dollar risk；
3. **side × inventory/capacity**：正常区间可能主要改变 intercept，接近 limit 时才改变 slope 与 convexity；
4. **market stress**：同时使 $Q^\star$ 下降、$A$ 上升，大 size 端更陡；
5. **information intensity**：最可能产生不可由简单 scale/amplitude 吸收的 genuine shape interaction；
6. **competition/protocol/relationship**：大 quantity 下参与 dealer 数量可能下降，导致竞争效应随 quantity 改变；
7. **client/dealer composition**：可能制造 pooled size discount，却不代表同一 state 下的 counterfactual quantity effect。

# 研究问题与统计对象

## 我们究竟要 normalize 什么

bond 的 quantity-neutral latent mid spread 记为 $M_{i,t}$。设：

- $i$：bond；
- $t$：point-in-time timestamp；
- $d$：dealer；
- $s\in\{\mathrm{bid},\mathrm{offer}\}$：side；
- $Q$：quote quantity；
- $S^{\mathrm{quote}}_{i,t,d,s}(Q)$：quote spread；
- $a_s\in\{+1,-1\}$：把 bid/offer 转为“越大越差”的统一方向。

定义 signed concession：

$$
Y_{i,t,d,s}(Q)
=
a_s\left(
S^{\mathrm{quote}}_{i,t,d,s}(Q)-M_{i,t}
\right),
$$

并选择 $a_s$ 使得 $Y>0$ 始终表示客户面对更差的 execution term。实现时必须明确 spread 字段的单位、side convention 与 benchmark timestamp。

<div class="warning">
<strong>目标不是 raw quote spread。</strong> raw spread 同时承载 issuer credit level、benchmark movement、bond-specific fair value 与 execution concession。若直接拟合 raw spread 对 quantity，quantity curve 会被 fundamental cross-section 污染。$M_{i,t}$ 必须是 point-in-time、尽可能 quantity-neutral，并避免包含当前 dealer 当前 quote。
</div>

一个可操作的 $M_{i,t}$ 可以是：

- leave-one-dealer-out contemporaneous consensus；
- cross-fitted fair-mid model；
- pre-event composite/reference mid；
- 在研究 quote ladder 时，由同 snapshot 其他 levels 提供的 internal anchor。

CPP 或其他 vendor mark 可以作为 feature 或 benchmark，但若它本身对应未知 size，不应无条件当作 quantity-neutral truth。

## “没有 interaction”的精确定义

令 $X$ 包含所有在 quote 时点已知且不含 $Q$ 的 features。quantity effect 可分离的 population null 是

$$
H_0:
\quad
\mu(X,Q)
:=E[Y\mid X,Q]
=m(X)+h(Q).
$$

等价地，对任意 reference $q_0$，条件 quantity contrast

$$
\Delta(q,q_0\mid X)
:=
\mu(X,q)-\mu(X,q_0)
$$

不依赖 $X$：

$$
H_0:
\quad
\Delta(q,q_0\mid X)=\Delta(q,q_0).
$$

这一定义比“quantity 的 SHAP interaction 很小”更直接，因为它正好对应业务上需要的 normalization adjustment。

## 三种不同层次的 interaction

即使 $H_0$ 被拒绝，也不代表需要完全自由的 $g(X,Q)$。应先区分：

### 横向尺度 interaction

不同 bond/state 的曲线形状相同，但变陡的位置不同：

$$
h(Q)
\longrightarrow
h\!\left(\frac{Q}{Q^\star(X)}\right).
$$

### 纵向幅度 interaction

不同 state 只改变 quantity penalty 的强度：

$$
h(Q)
\longrightarrow
A(X)h(Q).
$$

### 真正的 shape interaction

在 quantity 轴和 amplitude 都归一化后，曲率、turning point 或 tail behavior 仍不同：

$$
r(X,Q)\neq 0.
$$

前两类 interaction 仍然允许一个低维、解释性强、可稳定 extrapolate 的 quantity engine；第三类才真正需要 varying curve 或 tree interactions。

## 预测问题与因果问题必须分开

本报告主要服务于**predictive normalization**：给定实际可见 state 和 quote quantity，预测 equivalent quote at $q_0$。这只要求在目标 deployment distribution 上校准。

更强的因果问题是：

> 固定同一 bond、时刻、dealer、side、client 与 information set，只改变 requested quantity，dealer 会如何改变 quote？

由于 clients 选择 trade size，dealers 选择 displayed size，而且大 size 常在更有利状态下出现，普通 observational regression 不自动识别这一 counterfactual。报告会分别标注哪些结论是 predictive、哪些需要额外 identification assumptions。

# 报告边界与术语

1. 所有 rolling statistics 必须以 quote timestamp 为截止点，严格 point-in-time。
2. `quantity=0`、quantity missing、minimum size、maximum/up-to size、requested size 与 incremental ladder size 是不同数据语义，不得合并。
3. 本报告关注 spread concession；若使用 price concession，应显式建模 duration/CS01 的机械转换。
4. 固定 grid $100\text{k}$ 至 $5\text{mm}$ 是输出坐标，不意味着每个节点都有 observational support。
5. interpolation 优先在 $\log Q$ 上进行；超出 conditional support 的节点必须标为 extrapolation，并输出 uncertainty/availability。
