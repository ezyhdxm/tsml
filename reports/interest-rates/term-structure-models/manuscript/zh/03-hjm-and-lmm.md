# 8. HJM：直接对整条 forward curve 建模

Heath--Jarrow--Morton 框架不先假设短利率只有一两个状态，而是直接规定每个 maturity 的瞬时远期利率动态 [@hjm1992]。

## 8.1 HJM SDE

设 $W^{\mathbb Q}$ 是 $m$ 维 Brownian motion。对每个固定 $T$，在 $t\le T$ 时，

$$
\boxed{
df(t,T)
=\alpha(t,T)dt
+\sigma(t,T)^\top dW_t^{\mathbb Q}
}
$$

其中 $\sigma(t,T)\in\mathbb R^m$。这里 maturity $T$ 是一个索引，所以 HJM 实际上是一族耦合 SDE。

## 8.2 HJM 漂移限制的完整推导

定义

$$
\Sigma_P(t,T)=\int_t^T\sigma(t,u)du.
$$

由

$$
\log P(t,T)=-\int_t^T f(t,u)du,
$$

Leibniz rule 给出

$$
\begin{aligned}
d\left(\int_t^T f(t,u)du\right)
={}&-f(t,t)dt
+\int_t^Tdf(t,u)du\\
={}&
\left[-r_t+\int_t^T\alpha(t,u)du\right]dt
+\Sigma_P(t,T)^\top dW_t^{\mathbb Q}.
\end{aligned}
$$

因此

$$
d\log P(t,T)
=
\left[r_t-\int_t^T\alpha(t,u)du\right]dt
-
\Sigma_P(t,T)^\top dW_t^{\mathbb Q}.
$$

应用 Itô，

$$
\frac{dP(t,T)}{P(t,T)}
=
\left[
 r_t
-\int_t^T\alpha(t,u)du
+\frac12\|\Sigma_P(t,T)\|^2
\right]dt
-
\Sigma_P(t,T)^\top dW_t^{\mathbb Q}.
$$

风险中性测度下，债券 drift 必须是 $r_t$。所以

$$
\int_t^T\alpha(t,u)du
=
\frac12\left\|\int_t^T\sigma(t,u)du\right\|^2.
$$

对 $T$ 求导得到 HJM drift restriction：

$$
\boxed{
\alpha(t,T)
=
\sigma(t,T)^\top
\int_t^T\sigma(t,u)du
}.
$$

这意味着在 $\mathbb Q$ 下，**波动率结构决定漂移**。不能独立选择一个“看起来均值回复”的 forward drift；否则一般会产生套利。

对应的债券 SDE 是

$$
\boxed{
\frac{dP(t,T)}{P(t,T)}
=r_tdt-\Sigma_P(t,T)^\top dW_t^{\mathbb Q}
}.
$$

## 8.3 真实测度下的 HJM

若

$$
dW_t^{\mathbb Q}=dW_t^{\mathbb P}+\lambda_tdt,
$$

则

$$
df(t,T)
=
\left[
\sigma(t,T)^\top\int_t^T\sigma(t,u)du
+
\sigma(t,T)^\top\lambda_t
\right]dt
+
\sigma(t,T)^\top dW_t^{\mathbb P}.
$$

所以真实测度下 forward drift 等于无套利 drift 加上 maturity-dependent risk premium。

## 8.4 Forward measure

以 $P(t,U)$ 为 numeraire 定义 $U$-forward measure $\mathbb Q^U$。因为债券 numeraire 在 $\mathbb Q$ 下的波动率是 $-\Sigma_P(t,U)$，

$$
dW_t^{\mathbb Q^U}
=
dW_t^{\mathbb Q}+\Sigma_P(t,U)dt.
$$

于是

$$
df(t,T)
=
\left[
\alpha^{\mathbb Q}(t,T)
-
\sigma(t,T)^\top\Sigma_P(t,U)
\right]dt
+
\sigma(t,T)^\top dW_t^{\mathbb Q^U}.
$$

测度变化只改漂移，不改 instantaneous covariance。

## 8.5 Musiela 参数化

定义 time-to-maturity coordinate

$$
g_t(x)=f(t,t+x),\qquad x\ge0.
$$

则在风险中性测度下

$$
\boxed{
dg_t(x)
=
\left[
\partial_x g_t(x)
+
\sigma_t(x)^\top\int_0^x\sigma_t(u)du
\right]dt
+
\sigma_t(x)^\top dW_t^{\mathbb Q}
}.
$$

$\partial_xg$ 项不是经济 drift，而是“时间过去后，固定到期日沿 maturity axis 向左移动”的坐标效应。数值上 Musiela grid 比固定 calendar-maturity grid 更容易维护。

HJM 在数学上是无限维的；只有特定 volatility structure 才能压缩为有限维 Markov state [@filipovicteichmann2001]。本报告不使用 semigroup 语言：直观上，若未来整条曲线能够由有限个递推状态完全重构，就存在 finite-dimensional realization。

## 8.6 指数衰减波动率与 Hull--White 的等价

取一因子 deterministic volatility

$$
\sigma(t,T)=\sigma e^{-a(T-t)}.
$$

则

$$
\int_t^T\sigma(t,u)du
=
\frac{\sigma}{a}(1-e^{-a(T-t)}),
$$

因此

$$
\alpha(t,T)
=
\frac{\sigma^2}{a}
 e^{-a(T-t)}(1-e^{-a(T-t)}).
$$

这个 HJM 可以写成有限维 Gaussian Markov system，本质上就是 Hull--White 一因子模型。这个例子说明：short-rate model 不是 HJM 的竞争者，而常常是 HJM 在特殊 volatility family 下的有限维实现。

## 8.7 HJM maturity-grid 离散

给定 time grid $t_n$ 和 maturity grid $T_j$，对 $T_j\ge t_n$：

1. 计算 volatility loading $\sigma_{n,j}$。
2. 通过 maturity quadrature 计算

   $$
   \Sigma_{n,j}
   \approx
   \sum_{k:\,T_k\in[t_n,T_j]}
   \sigma_{n,k}\Delta T_k.
   $$

3. 令

   $$
   \alpha_{n,j}=\sigma_{n,j}^\top\Sigma_{n,j}.
   $$

4. 用同一个 $m$ 维 Brownian increment $\Delta W_n$ 更新所有 maturity：

   $$
   f_{n+1,j}
   =
   f_{n,j}
   +\alpha_{n,j}\Delta t
   +\sigma_{n,j}^\top\Delta W_n.
   $$

5. 重新积分 forward curve 得到 $P(t_{n+1},T_j)$。

必须使用同一组 factor shocks 驱动整条曲线，不能为每个 maturity 独立抽噪声。实现后应检查 $P(t,T)/B_t$ 的 Monte Carlo 均值是否近似保持不变。

# 9. LMM/BGM：在离散 tenor 上直接建模市场远期利率

Brace--Gatarek--Musiela 市场模型直接建模可观察的离散远期利率 [@bgm1997]。

## 9.1 Tenor 与简单远期利率

给定

$$
0=T_0<T_1<\cdots<T_N,
$$

令 accrual fraction 为

$$
\delta_i=T_{i+1}-T_i.
$$

单曲线理想化下，区间 $[T_i,T_{i+1}]$ 的简单远期利率是

$$
\boxed{
L_i(t)
=
\frac1{\delta_i}
\left[
\frac{P(t,T_i)}{P(t,T_{i+1})}-1
\right]
}.
$$

## 9.2 自然 forward measure 下的 SDE

以 $P(t,T_{i+1})$ 为 numeraire。在 $\mathbb Q^{T_{i+1}}$ 下，

$$
\frac{P(t,T_i)}{P(t,T_{i+1})}
=1+\delta_iL_i(t)
$$

是鞅。经典 lognormal LMM 规定

$$
\boxed{
dL_i(t)
=L_i(t)\lambda_i(t)^\top dW_t^{T_{i+1}}
}.
$$

若 $\lambda_i$ deterministic，则

$$
L_i(T_i)
=
L_i(t)
\exp\left[
-\frac12\int_t^{T_i}\|\lambda_i(s)\|^2ds
+
\int_t^{T_i}\lambda_i(s)^\top dW_s^{T_{i+1}}
\right].
$$

因此 caplet 可用 Black-style 公式定价。这是 LMM 受欢迎的主要原因：直接把市场报价对象设为 lognormal martingale。

## 9.3 Terminal measure 下的联合动态

不同 $L_i$ 的自然 measure 不同，联合模拟时需要统一测度。选取 terminal numeraire $P(t,T_N)$。在 $\mathbb Q^{T_N}$ 下，

$$
\boxed{
\frac{dL_i(t)}{L_i(t)}
=
-\sum_{j=i+1}^{N-1}
\frac{\delta_jL_j(t)}{1+\delta_jL_j(t)}
\lambda_i(t)^\top\lambda_j(t)dt
+
\lambda_i(t)^\top dW_t^{T_N}
}.
$$

负号来自从较早的 forward measure 变换到 terminal measure。这个 drift 是状态依赖且跨 maturity 耦合的；不能在联合模拟时把所有 forward 都当成 driftless lognormal。

## 9.4 相关性与 factor loading

如果 $m$ 远小于 forward 数量，可以写

$$
\lambda_i(t)
=
\nu_i(t)b_i(t),
\qquad b_i(t)\in\mathbb R^m.
$$

则 instantaneous covariance 为

$$
\frac{d\langle L_i,L_j\rangle_t}
{L_i(t)L_j(t)}
=
\lambda_i(t)^\top\lambda_j(t)dt.
$$

相关矩阵必须 positive semidefinite。对逐 maturity 单独拟合 pairwise correlation 再拼成矩阵，很容易得到不可 Cholesky 的结果。更稳健的做法是直接参数化低维 loading 或对经验相关矩阵做 PSD projection。

## 9.5 Terminal-measure log-Euler

在一步 $[t_n,t_{n+1}]$ 内冻结 drift 和 loading：

$$
\mu_i^n
=
-\sum_{j=i+1}^{N-1}
\frac{\delta_jL_j^n}{1+\delta_jL_j^n}
(\lambda_i^n)^\top\lambda_j^n.
$$

然后

$$
L_i^{n+1}
=
L_i^n
\exp\left[
\left(\mu_i^n-\frac12\|\lambda_i^n\|^2\right)\Delta t
+(\lambda_i^n)^\top\sqrt{\Delta t}Z_n
\right],
$$

其中同一个 $m$ 维 $Z_n$ 用于所有 $i$。可以采用 predictor--corrector 重新计算 drift，减少冻结偏差。

## 9.6 负利率、shifted LMM 与 stochastic volatility

纯 lognormal LMM 要求 $L_i>0$。允许负利率时常用

$$
F_i=L_i+s_i>0,
$$

并在自然 measure 下建模

$$
dF_i=F_i\lambda_i^\top dW^{T_{i+1}}.
$$

注意 terminal-measure drift 必须从 numeraire change 重新推导，不能机械地把经典公式中的 $L_i$ 全部替换为 $L_i+s_i$。

为了拟合 volatility smile，可加入 local volatility、stochastic volatility 或 SABR-like dynamics。此时应重新检查 martingale condition、measure change 和 moment explosion，而不是只修改边际分布。
