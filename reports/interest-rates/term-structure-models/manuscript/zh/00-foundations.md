---
title: "利率期限结构模型：从短利率 SDE、仿射定价到 HJM 与市场模型"
subtitle: "一份强调测度、漂移限制、解析定价、校准与数值仿真的严谨指南"
author: "TSML Research Notes"
date: "2026-08-27"
lang: zh-CN
bibliography: references.bib
link-citations: true
reference-section-title: "参考文献"
toc-title: "目录"
---

<nav class="language-switch"><strong>语言：</strong><span aria-current="page">中文</span> · <a href="index.en.html">English</a></nav>

# 摘要

“期限结构模型”并不是一个单一问题。至少要区分四件事：

1. **今天的曲线如何表示与 bootstrap**：给定市场报价，构造折现因子 $P(0,T)$、零息收益率和远期利率。
2. **未来曲线如何在真实测度下变化**：用于预测、风险管理、情景生成和期限溢价分析。
3. **衍生品如何在风险中性测度下定价**：要求贴现后的可交易资产价格为鞅。
4. **模型如何与期权市场波动率相匹配**：决定短利率、远期利率、相关性和随机波动率需要多少因子。

本报告从统一的无套利框架出发，完整给出 Vasicek、CIR、Hull--White、G2++、HJM 和 LMM/BGM 的 SDE、测度变化、债券定价方程、解析解、边界条件、精确转移分布和离散化方法。最后把这些定价模型与 Dynamic Nelson--Siegel、Gaussian affine term-structure model、AFNS 和期限溢价估计联系起来。

核心结论是：

- **短利率模型**把整条曲线压缩成少量 Markov 状态，便于 PDE、树和低维 Monte Carlo。
- **HJM**直接对整条瞬时远期曲线建模；一旦给定波动率，无套利漂移被唯一确定。
- **LMM/BGM**对离散 tenor 上的市场远期利率建模，并在各自 forward measure 下令其成为鞅。
- **真实测度 $\mathbb P$ 与风险中性测度 $\mathbb Q$ 的漂移不能混用**。前者决定预测，后者决定定价；两者之差就是风险价格。
- **精确拟合今天的曲线不等于拟合未来动态，也不等于拟合期权 smile**。这是三组不同的识别问题。

本报告沿用以下记号：$t$ 是当前时间，$T$ 是到期日，$\tau=T-t$ 是剩余期限；$P(t,T)$ 是到期支付 1 的无违约零息债券价格；$W$ 是 Brownian motion；除非特别说明，所有定价 SDE 都写在风险中性测度 $\mathbb Q$ 下。

# 1. 每次写利率 SDE 时必须交代什么

一个完整的模型定义至少应包含以下十项。只写

$$
dr_t=\kappa(\theta-r_t)dt+\sigma dW_t
$$

是不够的。

1. **概率空间与 filtration**：$(\Omega,\mathcal F,(\mathcal F_t)_{t\ge 0},\mathbb M)$。
2. **测度**：是历史/真实测度 $\mathbb P$，还是某个定价测度 $\mathbb Q$、$T$-forward measure $\mathbb Q^T$、terminal measure 等。
3. **numeraire**：money-market account、零息债券还是 rolling spot numeraire。
4. **状态变量与初值**：例如 $r_0$、$X_0$ 或整条初始 forward curve $f(0,T)$。
5. **Brownian motion 的维度**：一因子还是多因子。
6. **相关结构**：$d\langle W^i,W^j\rangle_t=\rho_{ij}dt$，或者把相关性吸收到 loading matrix 中。
7. **漂移与扩散系数**：它们是否依赖时间、状态和 maturity。
8. **参数限制与边界行为**：均值回复、非负性、Feller condition、爆炸条件、强解是否存在。
9. **可交易资产定价规则**：贴现后是否为鞅，HJM drift restriction 是否满足。
10. **数值实现**：Euler、exact transition、log-Euler、full truncation、时间积分与相关正态数如何生成。

后文对每个模型都按这个清单展开。

# 2. 期限结构对象与无套利基础

## 2.1 折现因子、零息收益率与瞬时远期利率

零息债券价格为

$$
P(t,T)>0,\qquad P(T,T)=1.
$$

连续复利零息收益率定义为

$$
y(t,T)=-\frac{\log P(t,T)}{T-t}.
$$

瞬时远期利率为

$$
f(t,T)=-\partial_T\log P(t,T).
$$

于是

$$
P(t,T)=\exp\left(-\int_t^T f(t,u)\,du\right),
$$

短利率是 forward curve 在对角线上的值：

$$
r_t=f(t,t).
$$

这三个量必须保持一致。实践中常见的错误是用 simple-compounded market quote 构造曲线，却在模型中把它当作 continuously compounded zero rate；或者在 year fraction、business-day adjustment、payment lag 不一致时直接比较两个 forward rate。

## 2.2 Money-market account 与风险中性定价

定义 money-market account

$$
B_t=\exp\left(\int_0^t r_s\,ds\right),
\qquad dB_t=r_tB_tdt.
$$

如果 $\mathbb Q$ 是以 $B_t$ 为 numeraire 的风险中性测度，则任意可交易资产 $S_t$ 的贴现价格 $S_t/B_t$ 是局部鞅。在通常的可积性条件下，零息债券满足

$$
P(t,T)
=
\mathbb E_t^{\mathbb Q}\left[
\exp\left(-\int_t^T r_s\,ds\right)
\right].
$$

这条公式是所有短利率模型的出发点。

## 2.3 真实测度与风险中性测度

设 $X_t\in\mathbb R^d$ 在真实测度下满足

$$
dX_t
=
\mu^{\mathbb P}(t,X_t)dt
+
\Sigma(t,X_t)dW_t^{\mathbb P},
$$

其中 $W^{\mathbb P}$ 是 $m$ 维标准 Brownian motion。给定市场风险价格 $\lambda_t\in\mathbb R^m$，若 Novikov 条件等保证指数鞅成立，则

$$
\left.\frac{d\mathbb Q}{d\mathbb P}\right|_{\mathcal F_t}
=
\exp\left(
-\int_0^t \lambda_s^\top dW_s^{\mathbb P}
-\frac12\int_0^t\|\lambda_s\|^2ds
\right),
$$

并且

$$
dW_t^{\mathbb Q}=dW_t^{\mathbb P}+\lambda_tdt.
$$

因此

$$
dX_t
=
\underbrace{\left(\mu^{\mathbb P}-\Sigma\lambda\right)}_{\mu^{\mathbb Q}}dt
+
\Sigma dW_t^{\mathbb Q}.
$$

所以：

$$
\boxed{
\mu^{\mathbb Q}=\mu^{\mathbb P}-\Sigma\lambda
}
$$

扩散通常在两个等价测度下相同，漂移通过风险价格改变。历史预测应估计 $\mathbb P$-dynamics；衍生品定价使用 $\mathbb Q$-dynamics。直接拿期权校准得到的 $\mathbb Q$ 均值回复水平预测未来利率，通常没有经济含义。

## 2.4 定价 PDE

若 $r_t=r(t,X_t)$，支付函数为 $g(X_T)$，则

$$
V(t,x)
=
\mathbb E_{t,x}^{\mathbb Q}
\left[
\exp\left(-\int_t^T r(s,X_s)ds\right)g(X_T)
\right].
$$

令 $a=\Sigma\Sigma^\top$，Feynman--Kac 给出

$$
\partial_tV
+
(\mu^{\mathbb Q})^\top\nabla_xV
+
\frac12\operatorname{tr}\left(a\nabla_x^2V\right)
-rV
=0,
$$

终值为 $V(T,x)=g(x)$。零息债券对应 $g\equiv1$。

# 3. 仿射期限结构模型的统一形式

## 3.1 仿射扩散与指数仿射债券价格

设风险中性状态满足

$$
dX_t=(b_0+BX_t)dt+\Sigma(X_t)dW_t^{\mathbb Q},
$$

协方差矩阵为

$$
a(x)=\Sigma(x)\Sigma(x)^\top
=a_0+\sum_{i=1}^d x_i a_i,
$$

短利率为

$$
r_t=\delta_0+\delta_1^\top X_t.
$$

在 admissibility 条件成立时，零息债券通常具有指数仿射形式 [@duffiekan1996]：

$$
P(t,T)=\exp\left(\phi(\tau)+\psi(\tau)^\top X_t\right),
\qquad \tau=T-t.
$$

代入 PDE，得到 Riccati 方程

$$
\phi'(\tau)
=b_0^\top\psi(\tau)
+\frac12\psi(\tau)^\top a_0\psi(\tau)
-\delta_0,
$$

以及对 $i=1,\ldots,d$，

$$
\psi_i'(\tau)
=
\left(B^\top\psi(\tau)\right)_i
+
\frac12\psi(\tau)^\top a_i\psi(\tau)
-\delta_{1,i},
$$

初值是

$$
\phi(0)=0,\qquad \psi(0)=0.
$$

Vasicek 和 CIR 都是这个系统的一维特例。Gaussian ATSM 中 $a_i=0$，Riccati 方程退化为线性 ODE；CIR 中扩散方差与状态成正比，出现二次项。

## 3.2 仿射不代表“任意参数都合法”

若某个状态被要求非负，则漂移必须在边界处指向状态空间内部，扩散矩阵也不能把过程瞬间推到非法区域。Dai--Singleton 的 canonical families 正是在研究不同 affine diffusion 对条件方差、相关性和风险价格施加了什么限制 [@daisingleton2000]。实践中应把以下问题分开：

- 数学上的 admissibility；
- 横截面收益率拟合；
- 时间序列识别；
- 期权波动率拟合；
- 参数的经济解释。
