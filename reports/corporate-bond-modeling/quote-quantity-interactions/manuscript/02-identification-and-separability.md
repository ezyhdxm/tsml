# 识别问题：如何判断 quantity 是否真的可分离

# 为什么 naive residual correction 一般不成立

考虑 population additive model

$$
Y=m(X)+h(Q)+\varepsilon,
\qquad
E[\varepsilon\mid X,Q]=0.
$$

若第一步忽略 $Q$、用任意一致 learner 估计 $E[Y\mid X]$，则

$$
E[Y\mid X]
=m(X)+E[h(Q)\mid X].
$$

第一阶段 residual 为

$$
R_Y
=Y-E[Y\mid X]
=h(Q)-E[h(Q)\mid X]+\varepsilon.
$$

如果再直接拟合 $R_Y\sim Q$，得到的是经过 selection/composition 投影后的曲线，不是 $h(Q)$。

## 线性特例中的 attenuation

若

$$
Y=m(X)+\beta Q+\varepsilon,
$$

则

$$
R_Y
=\beta\{Q-E[Q\mid X]\}+\varepsilon.
$$

记

$$
R_Q=Q-E[Q\mid X].
$$

若错误地把 $R_Y$ 回归到 raw $Q$，population slope 为

$$
\frac{\operatorname{Cov}(Q,R_Y)}{\operatorname{Var}(Q)}
=
\beta\frac{\operatorname{Var}(R_Q)}{\operatorname{Var}(Q)}.
$$

只要 $Q$ 可由 $X$ 部分预测，比例就小于 1。极端情况下，liquidity、dealer、client 与 market-state features 几乎完全预测 displayed size，第一阶段 tree model 会把 quantity effect 间接吸收，residual correction 接近零。

<div class="decision">
<strong>结论：</strong>“先拟合 $X$，再对 residual 拟合 raw $Q$”不能作为 separability test。可以保留两阶段思想，但必须同时 residualize outcome 与 quantity basis，并用 out-of-fold nuisance predictions。
</div>

# Additive quantity curve 的三种估计方式

## Joint additive backfitting

直接最小化

$$
\min_{m,h}
\sum_{n=1}^N
\left\{
y_n-m(x_n)-h(q_n)
\right\}^2
+\lambda_h J(h),
$$

其中 $J(h)$ 是 spline roughness 或 second-difference penalty。可交替更新：

1. 固定 $h$，拟合 $m(X)$ 到 $Y-h(Q)$；
2. 固定 $m$，对 $Y-m(X)$ 拟合 smooth $h(Q)$；
3. 对 $h$ 加识别约束，例如 $E[h(Q)]=0$；
4. 迭代至 out-of-fold objective 收敛。

这种方法直观，但如果每次在同一数据上用高容量 learner 更新 $m$，仍可能发生 overfit leakage。生产研究应把 nuisance update 做成 cross-fitted 或 honest folds。

## Orthogonalized quantity basis

令

$$
B(Q)=\left(B_1(\log Q),\ldots,B_K(\log Q)\right)^\top
$$

为 spline / hinge basis，模型为

$$
Y=m(X)+\theta^\top B(Q)+\varepsilon.
$$

分别估计

$$
\mu_Y(X)=E[Y\mid X],
\qquad
\mu_B(X)=E[B(Q)\mid X].
$$

构造 out-of-fold residuals：

$$
\widetilde Y
=Y-\widehat\mu_Y(X),
\qquad
\widetilde B
=B(Q)-\widehat\mu_B(X).
$$

然后估计

$$
\widehat\theta
=
\arg\min_\theta
\sum_n
(\widetilde Y_n-\theta^\top\widetilde B_n)^2
+\lambda\theta^\top\Omega\theta.
$$

这推广了 Robinson partially linear estimator [@robinson1988root]。用 cross-fitting 和 orthogonal score 可以降低 nuisance ML 的 regularization bias 与 own-observation overfit bias [@chernozhukov2018double]。

### 为什么 residualize 整个 basis

若 $h$ 非线性，只 residualize $\log Q$ 不够。需要对每个 basis function $B_k(Q)$ 估计其条件期望。否则 $X$ 仍可能通过 nonlinear quantity distribution 吸收 curve 的局部形状。

## Profiled spline + black-box nuisance

另一种实现是对给定 $\theta$，令

$$
Y^{(\theta)}=Y-\theta^\top B(Q),
$$

用 black-box learner 拟合 $m_\theta(X)$，再最小化 out-of-fold profile loss：

$$
\widehat\theta
=
\arg\min_\theta
\sum_n
\left\{
y_n-\theta^\top B(Q_n)-
\widehat m_{\theta,-k(n)}(X_n)
\right\}^2.
$$

它更计算密集，但可以直接对业务 loss 优化。

# Interaction 本身需要可识别定义

## 任意 main/interaction decomposition 不唯一

若写成

$$
\mu(X,Q)=m(X)+h(Q)+r(X,Q),
$$

没有约束时，可以把任意只依赖 $Q$ 的函数从 $r$ 移到 $h$，或把只依赖 $X$ 的函数移到 $m$，prediction 不变。因此“interaction magnitude”本身没有定义。

一种 population identification 是施加

$$
E[r(X,Q)\mid X]=0,
\qquad
E[r(X,Q)\mid Q]=0,
$$

并令

$$
E[m(X)]=E[h(Q)]=0.
$$

在 independent features 下，这对应标准 functional ANOVA；在 $X$ 与 $Q$ 依赖时，应使用 dependence-aware generalized functional ANOVA [@hooker2007generalized]。对 tree-based piecewise-constant model，可以用 purification 把可由低阶 main effects 表示的部分从 interaction 中移除 [@lengerich2020purifying]。

## 一个业务导向的 interaction 定义

对于 normalization，最直接的对象不是 $r$ 的全局 variance，而是 conditional contrast heterogeneity：

$$
H_Z(q,q_0)
=
E\left[
\Delta(q,q_0\mid X)
\mid Z
\right]
-
\Delta(q,q_0),
$$

其中 $Z$ 是预先指定的 modifier，例如 liquidity state、stress state 或 inventory pressure。若

$$
H_Z(q,q_0)=0
$$

在全 support 上近似成立，则该 modifier 不需要进入 quantity curve，即使 black-box model 的某些 local interaction attribution 非零。

# Scale 与 amplitude 的识别

模型

$$
\mu(X,Q)
=m(X)+A(X)h(\log Q-\log Q^\star(X))
$$

存在 scale normalization ambiguity。例如把 $A$ 乘常数、把 $h$ 除以同一常数不改变 prediction。需要约束：

$$
E[A(X)]=1,
\qquad
h(0)=0,
$$

或

$$
\int h(u)^2w(u)du=1
$$

并把 sign 固定为某个 tail contrast 为正。$Q^\star$ 也需要 anchor，例如

$$
E[\log Q^\star(X)]=\log(1\text{mm})
$$

或把一个 liquidity reference bucket 的 $Q^\star$ 固定为 $1\text{mm}$。

更稳妥的第一版不是同时自由估计 $A,Q^\star,h$，而是：

1. 预定义几种经济含义清晰的 $Q^\star$ proxy；
2. 比较哪种 proxy 最能压缩 conditional-curve heterogeneity；
3. 再在低维参数化 family 中估计 scale coefficients；
4. 最后才允许 flexible $A(X)$。

# quantity endogeneity：predictive 与 causal estimand

## Predictive estimand

部署目标是

$$
\mu_{\mathrm{obs}}(X,q)
=E[Y\mid X,Q=q],
$$

它描述 observational distribution 下的 conditional quote。若 deployment 中同样的 selection mechanism 持续存在，预测可以校准，即使不是 causal。

## Counterfactual estimand

更强的 quantity-normalization 解释需要

$$
\mu_{\mathrm{cf}}(X,q)
=E[Y(q)\mid X],
$$

其中 $Y(q)$ 是把 requested size 设置为 $q$ 的潜在 quote。要从 observational data 识别，至少需要：

1. consistency：观测 $Q=q$ 时 $Y=Y(q)$；
2. conditional exchangeability：$Y(q)\perp Q\mid X$；
3. positivity：对相关 $X$，每个目标 $q$ 都有非零概率；
4. 正确处理 nonresponse / no-quote outcome。

在 OTC bond data 中，第二、第三条件往往很强。client urgency、information 与 dealer inventory 可能未观测；$5\text{mm}$ 在 illiquid bonds 中几乎没有 support。因此报告推荐把 causal language 限制在有 matched variation 或准实验设计的区域。

# identification designs 的证据等级

## Level 1：同一真实 quantity ladder

同一 dealer、bond、side、timestamp 下多个 executable levels：

$$
Y(q_2)-Y(q_1)
$$

会消除同 snapshot 的 $m(X)$。这是最干净的 relative-curve evidence，但必须按 ladder 语义计算 cumulative VWAP，并确认 levels firm/executable。

## Level 2：同 dealer–bond–side 的近同时不同 requested quantity

在很短时间内市场状态近似不变：

$$
\Delta Y
=
h(q_2)-h(q_1)
+\Delta r
+\Delta\varepsilon.
$$

需要控制 quote update sequence，避免 quantity change 其实响应了 market move。

## Level 3：重复 RFQ / sequential inquiries

相同或高度相似 security/client state 下 quantity 变化，可利用 fixed effects 和 timing。必须同时处理 failed RFQ、search stopping 与 dealer participation selection。sequential-search evidence 表明未成交 inquiries 与后续 offers 本身含有状态信息 [@kargar2026sequential]。

## Level 4：high-dimensional matched pairs

在 $X$、dealer、side、time bucket、bond liquidity 上匹配不同 $Q$。应报告 balance、effective sample size 与 common support，而不是只给 regression coefficient。

## Level 5：普通 cross-sectional regression

可用于 predictive challenger，但最容易把 client/dealer composition 当作 quantity effect。Pooled discount 与 within-client penalty 的反转正说明这一风险 [@pinter2024size]。

# support、overlap 与 extrapolation

## 不能只看全样本 quantity histogram

需要估计 conditional support：

$$
\mathcal S_Q(X)
=\{q: p(Q=q\mid X)>0\}
$$

或连续情形的 conditional density。至少按以下维度检查：

- IG/HY、liquidity bucket；
- side；
- dealer class；
- protocol；
- stress regime；
- quote semantic type；
- time-of-day；
- issue age/size。

若某 state 下 $5\text{mm}$ 几乎不出现，则该节点不是 interpolation，而是 model-based extrapolation。

## 输出 support score

对每个 $(X,q)$ 输出：

$$
\mathrm{support}(X,q)
=\widehat p(Q\in\mathcal N(q)\mid X),
$$

或 propensity/density score、nearest-neighbor distance、effective sample count。最终 quantity curve 应同时显示：

- fair bid/offer；
- uncertainty；
- availability probability；
- interpolation/extrapolation flag；
- support score。

## overlap trimming 的作用

在比较 additive 与 interaction models 时，应同时报告：

1. full deployment distribution；
2. common-support subset；
3. matched-pair subset。

若 interaction 只在 low-support tails 提升 in-sample fit，却在 common support 中消失，不应据此建立复杂 production curve。

# reference mid 与 point-in-time identification

## leave-one-dealer-out

若 $M_{i,t}$ 包含当前 dealer quote，则 target 与 feature 机械相关，quantity curve 会被 shrink。对 dealer $d$ 应使用

$$
M_{i,t}^{(-d)}
=
\operatorname{Consensus}
\left\{
S_{i,t,d',s}:d'\neq d
\right\}.
$$

## cross-fitting

若 reference mid 来自 ML model，应对当前 observation 使用不含该 observation、最好不含同 event cluster 的 fitted model。对于 closely related quotes，同一 RFQ/snapshot 的 rows 必须在同一 fold，否则 sibling quote 泄漏。

## label availability

若 $M_{i,t}$ 使用 future trade 或 post-event quote，研究对象就不再是 arrival-time normalization。所有 feature 和 target anchor 都应标注：

- event time；
- observation time；
- label availability time。

## benchmark movement

优先在 spread space 估计。若 source 的 price/yield timestamp 不同步，应先构造同 timestamp benchmark-adjusted spread；否则 quantity 与 time-of-day/latency interaction 会伪装为 execution effect。

# dependent observations 与 uncertainty

quotes 在 bond、dealer、issuer、day 与 event cluster 上强相关。IID standard errors 会过窄。推荐：

- out-of-time folds 作为首要 generalization evidence；
- event/snapshot group 不跨 folds；
- cluster bootstrap 至少以 day 为外层，并保留 bond/dealer dependence；
- matched-pair analysis 按 pair/event cluster resample；
- simultaneous curve bands 而非每个 quantity node 单独置信区间。

若要测试多个 modifier、多个 nodes 与多个 regimes，应采用 hierarchical testing：先检验整个 modifier family 是否改变 curve，再检验具体 nodes，避免大量局部显著性。

# missingness 与 no-quote outcome

可执行 quantity curve 实际上由两个过程组成：

$$
P(\text{quote available at }q\mid X)
$$

和

$$
E[Y(q)\mid X,\text{quote available at }q].
$$

只在 observed quotes 上拟合第二项会产生 selection-on-availability。尤其 large size 下，“没有报价”本身比一个很差的 price 更重要。生产 engine 应分别建模：

1. depth/availability model；
2. conditional price/concession model；
3. missing-size semantic model。

这也解释了为什么 $5\text{mm}$ fair value 不能只输出一个数：必须同时说明在当前 state 下真正获得该 size quote 的概率。
