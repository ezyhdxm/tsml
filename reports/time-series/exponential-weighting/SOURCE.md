---
title: "指数加权方法：从 EMA 到状态空间、EWMA 协方差与不规则采样"
subtitle: "一条连续的理论主线：定义、记忆长度、滤波、最优预测、风险模型与 point-in-time 实现"
author: "TSML Research Notes"
date: "2026-08-25"
lang: zh-CN
abstract: |
  Exponential moving average（EMA）、exponentially weighted average（EWA）、simple exponential smoothing（SES）和 RiskMetrics EWMA 常被混称为同一个公式。它们的代数形式相近，但 estimand、初始化、概率模型和不规则时间处理并不相同。本报告先固定统一记号，再从有限历史带权平均推到递归滤波，解释 half-life、mean age 和 effective sample size，随后给出频域性质、local-level/Kalman 最优性、SES–ARIMA 等价、EWMA 方差与协方差、irregular sampling，以及 corporate-bond trade/quote/residual 数据上的 point-in-time 用法。
---

<div class="takeaway">

### 核心结论

指数加权并不是一个没有理论的 heuristic。它至少有四种严格解释：

1. **加权最小二乘：** normalized EWA 是指数折扣平方损失的精确最优解；
2. **线性滤波：** recursive EMA 是一阶单极点 low-pass filter；
3. **状态空间预测：** SES 是 local-level model 的稳态 Kalman filter，并与 ARIMA\((0,1,1)\) 相连；
4. **条件风险更新：** RiskMetrics EWMA 是零截距、单位持久性的 GARCH/IGARCH 边界递推。

真正的问题不是“要不要 EMA”，而是：**你要估计什么状态、用哪一种时间、何时允许更新，以及怎样选择 decay。**

</div>

# 1. 先把几个同名对象分开

Brown、Holt、Winters 和 Muth 的传统主要研究需求预测与动态 level [@brown1959; @holt2004; @winters1960; @muth1960]；Roberts 把几何加权用于过程监控 [@roberts1959]；RiskMetrics 把相同衰减结构用于收益的条件方差与协方差 [@riskmetrics1996]；后来的 state-space 文献给 exponential smoothing 建立了 likelihood、模型选择和 prediction interval [@ord1997; @hyndman2002; @hyndman2008]。

| 名称 | 被加权的对象 | 典型 estimand |
|---|---|---|
| EWA | 历史 observations | 局部平均水平 |
| EMA | 在线递归状态 | 当前平滑状态 |
| SES | level state 与 forecast errors | 下一期 level forecast |
| EWMA volatility/covariance | 平方收益或 outer products | 下一期条件风险 |
| Exponential weights | experts/models | 接近 hindsight 最优 expert |

本报告统一使用

- \(\beta\in(0,1)\)：retention / forgetting factor；
- \(\alpha=1-\beta\)：新 observation 的 gain。

递推写为

$$
m_t=\beta m_{t-1}+\alpha x_t.
$$

RiskMetrics 常把 \(\beta\) 写成 \(\lambda\)，forecasting 文献通常把新信息权重写成 \(\alpha\)。因此不要只看参数名称，要看它在公式里乘的是旧状态还是新 observation。

# 2. Normalized EWA 与 recursive EMA 不是完全同一个对象

给定 \(x_1,\ldots,x_t\)，有限历史的 normalized EWA 是

$$
\bar x_t
=
\frac{\sum_{i=1}^t \beta^{t-i}x_i}
     {\sum_{i=1}^t \beta^{t-i}}
=
\frac{1-\beta}{1-\beta^t}
\sum_{i=1}^t\beta^{t-i}x_i.
$$

它是严格的 convex combination。定义

$$
A_t=\beta A_{t-1}+x_t,
\qquad
W_t=\beta W_{t-1}+1,
$$

即可用 \(O(1)\) 内存计算 \(\bar x_t=A_t/W_t\)。所以“使用全部历史”和“保存全部历史”是两回事。

Normalized EWA 也可以写成变动 gain 的递推：

$$
\bar x_t=(1-\kappa_t)\bar x_{t-1}+\kappa_t x_t,
\qquad
\kappa_t=\frac{1-\beta}{1-\beta^t}.
$$

而常数 gain EMA 为

$$
m_t=\beta m_{t-1}+(1-\beta)x_t
=(1-\beta)\sum_{j=0}^{t-1}\beta^j x_{t-j}+\beta^t m_0.
$$

差异全部集中在初始化与有限样本归一化：recursive EMA 保留一项 \(\beta^t m_0\)。pandas 的 `adjust=True` 对应 normalized finite-history weights；`adjust=False` 对应常数 gain recursion [@pandas2026]。在短历史、较长 half-life、频繁更换 universe 或 group 时，这个差异不能忽略。

## 2.1 加权最小二乘解释

对当前常数 level \(m\) 最小化指数折扣平方损失

$$
Q_t(m)=\sum_{i=1}^t\beta^{t-i}(x_i-m)^2,
$$

一阶条件立即给出

$$
\arg\min_m Q_t(m)=\bar x_t.
$$

因此 EWA 是一个精确的 weighted least-squares estimator。注意：这只说明它对**指定的折扣 objective**最优，并未说明该 objective 对真实预测任务最优；后者需要概率模型或 out-of-sample validation。

# 3. Decay 参数究竟代表多长的记忆？

无限几何权重为

$$
w_j=(1-\beta)\beta^j,
\qquad j=0,1,2,\ldots
$$

单报 \(\alpha\) 很难解释。以下四个量更直观。

## 3.1 Half-life

令 \(\beta^H=1/2\)，则

$$
H=\frac{\log(1/2)}{\log\beta},
\qquad
\beta=2^{-1/H}.
$$

Half-life 回答：“相隔多少个时间单位后，单个 observation 的相对权重减半？”

## 3.2 Mean age

把 \(w_j\) 看成 observation age 的概率分布，则

$$
E[J]=\sum_{j=0}^{\infty}j(1-\beta)\beta^j
=\frac{\beta}{1-\beta}.
$$

这正是 pandas 参数 `com` 的几何意义。

## 3.3 Kish effective sample size

归一化权重的 Kish ESS 为

$$
N_{\mathrm{eff}}
=
\frac{1}{\sum_jw_j^2}
=
\frac{1+\beta}{1-\beta}.
$$

这恰好对应 pandas 的 `span` 映射。它不是窗口长度，而是：对于 iid、同方差 observations，这组不均匀权重与多少个等权 observations 有相同的均值方差。

## 3.4 Tail-mass horizon

年龄大于等于 \(L\) 的总权重为

$$
\sum_{j=L}^{\infty}(1-\beta)\beta^j=\beta^L.
$$

若只允许尾部剩余质量 \(\varepsilon\)，则

$$
L_{\varepsilon}
=
\left\lceil\frac{\log\varepsilon}{\log\beta}\right\rceil.
$$

所以一个 decay 最好同时报告：half-life、mean age、ESS 和例如 95%/99% mass horizon。

# 4. EMA 作为一阶线性滤波器

对零初值递推取 \(z\)-transform：

$$
H(z)=\frac{M(z)}{X(z)}
=
\frac{1-\beta}{1-\beta z^{-1}}.
$$

它只有一个极点 \(z=\beta\)，是稳定的一阶 low-pass filter。频率响应为

$$
|H(e^{i\omega})|^2
=
\frac{(1-\beta)^2}
     {1+\beta^2-2\beta\cos\omega}.
$$

在 \(\omega=0\) 时 gain 为 1；频率越高，衰减越强。较大的 \(\beta\) 带来更强平滑，也带来更大的动态延迟。

## 4.1 对 iid noise 的方差降低

若 \(x_t\) iid，方差为 \(\sigma^2\)，则稳态 EMA 方差为

$$
\operatorname{Var}(m_t)
=
\sigma^2\sum_{j=0}^{\infty}w_j^2
=
\sigma^2\frac{1-\beta}{1+\beta}
=
\frac{\sigma^2}{N_{\mathrm{eff}}}.
$$

这把滤波理论与 ESS 完全连接起来。

## 4.2 EMA 会制造 autocorrelation

即使输入 \(x_t\) 是 white noise，输出仍满足

$$
m_t=\beta m_{t-1}+(1-\beta)x_t,
$$

所以

$$
\operatorname{Corr}(m_t,m_{t-k})=\beta^k.
$$

因此“EMA feature 很平滑、很持久”不能被当作原始信号可预测性的证据；其中一部分 persistence 是滤波器机械产生的。

## 4.3 Step response 与 ramp lag

若真实 level 在时刻 0 从 0 跳到 1，EMA 响应为

$$
m_t=1-\beta^{t+1}.
$$

若输入是线性趋势 \(x_t=ct\)，稳态 tracking lag 约为

$$
x_t-m_t\approx c\frac{\beta}{1-\beta}.
$$

也就是 slope 乘以 mean age。这个结果解释了为什么慢 EMA 在 trend regime 中系统性落后。

# 5. 什么时候 EMA 是统计意义上的最优滤波？

考虑 local-level model：

$$
\ell_t=\ell_{t-1}+\eta_t,
\qquad
x_t=\ell_t+\varepsilon_t,
$$

其中 \(\eta_t\sim(0,Q)\)、\(\varepsilon_t\sim(0,R)\)。Kalman 更新为

$$
\hat\ell_{t|t}
=
\hat\ell_{t|t-1}
+K_t(x_t-\hat\ell_{t|t-1}).
$$

在稳定参数下 \(K_t\to K\)，这就是 SES/EMA：

$$
\hat\ell_t=(1-K)\hat\ell_{t-1}+Kx_t.
$$

令 signal-to-noise ratio \(q=Q/R\)，稳态 Riccati 方程给出

$$
q=\frac{\alpha^2}{1-\alpha},
$$

等价地

$$
\alpha^*=
\frac{\sqrt{q^2+4q}-q}{2}.
$$

解释非常直接：latent level 变化快，即 \(Q/R\) 大，则 \(\alpha\) 大、half-life 短；measurement noise 大，即 \(Q/R\) 小，则 \(\alpha\) 小、half-life 长。Muth 的经典工作正是把 exponential forecasting 的最优性放在特定随机过程下讨论 [@muth1960]；现代 state-space 框架则把 likelihood、initial state 和 forecast distribution 一并纳入 [@hyndman2002]。

## 5.1 固定 gain 的 tracking MSE

若误差 \(e_t=m_t-\ell_t\)，可推出稳态

$$
\frac{\operatorname{Var}(e_t)}{R}
=
\frac{(1-\alpha)^2q+\alpha^2}
     {\alpha(2-\alpha)}.
$$

最小化它得到上面的 Kalman gain。这说明 tuning \(\alpha\) 本质上是在 process drift 与 observation noise 之间做 bias–variance / tracking–smoothing 权衡。

## 5.2 SES 与 ARIMA\((0,1,1)\)

令 one-step innovation \(e_t=x_t-m_{t-1}\)，SES 更新为

$$
m_t=m_{t-1}+\alpha e_t.
$$

又因为 \(x_t=m_{t-1}+e_t\)，可得

$$
\Delta x_t=e_t-(1-\alpha)e_{t-1}.
$$

因此 SES 对应 ARIMA\((0,1,1)\)，其中 MA 系数在该符号约定下为 \(1-\alpha\)。这不是说任何 EMA feature 都隐含正确的 ARIMA 模型，而是说当 innovations 表示成立时，两种表述给出同一预测结构。

# 6. 有 trend 或 seasonality 时，单一 EMA 不够

Holt 方法增加 level 与 trend：

$$
\ell_t
=
\alpha x_t+(1-\alpha)(\ell_{t-1}+b_{t-1}),
$$

$$
b_t
=
\gamma(\ell_t-\ell_{t-1})+(1-\gamma)b_{t-1},
$$

$$
\hat x_{t+h|t}=\ell_t+h b_t.
$$

Winters 再加入 seasonal state [@holt2004; @winters1960]。这里的原则比公式更重要：若数据的动态包含 level、slope、seasonality 或多时间尺度，就应该给这些结构独立状态，而不是要求一个 half-life 同时完成所有任务。Gardner 的两次综述总结了 exponential smoothing 方法及其状态空间发展 [@gardner1985; @gardner2006]。

# 7. Exponentially weighted moments、方差与协方差

## 7.1 描述性加权样本协方差

给定归一化权重 \(w_i\)，加权均值和 raw second central moment 为

$$
\mu_w=\sum_iw_ix_i,
\qquad
S_w=\sum_iw_i(x_i-\mu_w)(x_i-\mu_w)^\top.
$$

若 observations iid、权重预先固定，则

$$
E[S_w]
=
\left(1-\sum_iw_i^2\right)\Sigma,
$$

所以无偏修正为

$$
\widehat\Sigma_{\mathrm{unbiased}}
=
\frac{S_w}{1-\sum_iw_i^2}.
$$

这是一种加权**样本统计量**。

## 7.2 RiskMetrics conditional covariance

对零均值收益 \(r_t\)，RiskMetrics 更新

$$
\Sigma_t
=
\beta\Sigma_{t-1}
+(1-\beta)r_{t-1}r_{t-1}^{\top}.
$$

它估计的是时变条件协方差 state，不以 iid sample covariance 的无偏性为首要目标。RiskMetrics 1996 给出的经典校准包括 daily \(\beta=0.94\) 与 monthly \(\beta=0.97\) [@riskmetrics1996]。

若 \(\Sigma_0\succeq0\)、\(0\le\beta\le1\)，由于每一步都是 PSD matrices 的 convex combination，\(\Sigma_t\succeq0\)。但如果每个 covariance element 使用不同 decay，或缺失处理不一致，PSD 不再自动保证。

## 7.3 EWMA 是 IGARCH 边界，而不是完整的 mean-reverting volatility model

单变量递推

$$
h_t=\beta h_{t-1}+(1-\beta)r_{t-1}^2
$$

可看成 GARCH\((1,1)\) 的零截距且 ARCH+GARCH 系数为 1 的边界情形 [@bollerslev1986]。冲击不会向一个由正截距决定的长期方差回归。因此它适合快速、稳健的在线 proxy，但不应被误称为具有有限 long-run variance 的一般 GARCH 模型。

实际市场往往同时存在快、慢 volatility components。可使用多个 exponential states：

$$
v_t=\sum_{m=1}^M a_m v_t^{(m)},
\qquad
v_t^{(m)}=\beta_m v_{t-1}^{(m)}+(1-\beta_m)r_{t-1}^2,
$$

这仍保持常数内存，并可近似长记忆 decay [@zumbach2004]。

# 8. Irregular sampling：先问你在衰减什么

设 observations 在 \(t_1<t_2<\cdots\) 到达，\(\Delta_i=t_i-t_{i-1}\)。regular-grid 的固定 \(\beta\) 会把“上一笔 event”和“固定 clock duration”混为一谈。

## 8.1 Event-time EMA

$$
m_i=\beta m_{i-1}+(1-\beta)x_i.
$$

这里 half-life 的单位是 observation count：例如“5 笔 trade”。

## 8.2 Clock-time normalized observation average

若 clock-time decay rate 为 \(\lambda=\log2/H\)，定义

$$
A_i=e^{-\lambda\Delta_i}A_{i-1}+x_i,
\qquad
W_i=e^{-\lambda\Delta_i}W_{i-1}+1,
$$

$$
\bar x_i=A_i/W_i.
$$

这估计的是所有已观察样本的 clock-decayed normalized average。长时间没有 observation 时，旧 observations 的总权重会衰减，但状态不会凭空向某个 baseline 运动。

## 8.3 Dynamic-state recursion

若要让旧状态本身按 clock time 失效，可写

$$
a_i=e^{-\lambda\Delta_i},
$$

$$
m_i=a_i m_{i-1}+(1-a_i)x_i.
$$

它不是上一节 normalized observation average，而是 variable-gain dynamic-state recursion。两者在 finite history 下通常不同。Wright 以及 Cipra–Hanzák 的工作专门研究 irregularly spaced exponential smoothing 与 discount least squares [@wright1986; @cipra2006; @cipra2008]。

## 8.4 更原则化的 irregular Kalman filter

如果 latent level 是连续时间 random walk，process variance 应随间隔增长：

$$
Q_i=q_c\Delta_i.
$$

于是

$$
P_{i|i-1}=P_{i-1|i-1}+q_c\Delta_i,
$$

$$
K_i=\frac{P_{i|i-1}}{P_{i|i-1}+R}.
$$

长 gap 后 uncertainty 增大，Kalman gain 自动增大。这和“旧 observation 按 \(e^{-\lambda\Delta}\) 衰减”不是同一个模型：一个描述 state uncertainty accumulation，另一个直接指定 memory kernel。

# 9. Half-life 应怎样选择？

参数不应根据 full-sample 图形“看起来更平滑”来选。推荐顺序：

1. 明确 estimand 与 clock：event time、wall clock、business time 或 label-availability time；
2. 建立候选 half-life grid，并同时记录 mean age、ESS 和 tail horizon；
3. 用 rolling / expanding point-in-time validation；
4. 只在 observation 或 label 真正可用时更新；
5. 以最终任务 loss 选择参数，而不是以平滑曲线的视觉效果选择；
6. 检查 regime、liquidity、group support 和 missingness 下的稳定性。

若接受 local-level model，可通过 likelihood 估计 \(Q,R\)，再由 \(q=Q/R\) 映射到 \(\alpha\)。若不相信模型，则把 half-life 当作 hyperparameter，用严格时序 validation。对 volatility/covariance，应另外比较 one-step log score、QLIKE、VaR coverage、portfolio loss，而不只是 squared error。

# 10. EMA、EWMM 与 online exponential weights 的关系

Luxenberg–Boyd 将 moving average 推广为 exponentially weighted moving models：在每个时点求解带指数折扣的 convex objective，因此 moving regression、quantile、covariance 和 regularized models 都能获得统一表述 [@luxenberg2024]。

另一方面，online learning 中的 exponential weights 给 experts 加权：

$$
p_{t,k}
\propto
\exp\{-\eta L_{t-1,k}\},
$$

其中 \(L_{t-1,k}\) 是 expert \(k\) 的累计损失。这里指数作用在**累计 loss**上，而不是 observation age 上；理论目标是 regret bound，而不是平滑 latent state [@freund1997; @cesabianchi2006; @dalalyan2008]。两者可以组合，但不能因为公式里都有 exponent 就视为同一理论。

# 11. 对 corporate-bond trade / quote / residual 数据的建议

## 11.1 先区分四类状态

1. **Level state：** spread、dealer consensus、quote imbalance；
2. **Trend state：** 最近 widening/narrowing 的速度；
3. **Uncertainty state：** prediction errors 或 spread innovations 的 EWMA variance；
4. **Support/age state：** 距离最近更新的时间、有效权重、ESS、active dealer count。

不要让一个 EMA 同时承担 level、trend、volatility 和 data freshness。

## 11.2 同时维护 event clock 与 wall clock

对每个 CUSIP，可同时维护：

- 最近 \(k\) 次 trade/quote updates 的 event-time state；
- 5m、30m、2h 等 wall-clock half-life states；
- time since last update、current denominator/effective mass、dealer support。

Event-time EMA 衡量“最近几笔”的方向；clock-time state 衡量“信息经过多久仍有价值”。在 illiquid bond 上，两者差别可能很大。

## 11.3 Point-in-time 更新纪律

- quote/trade 到达后才能更新对应 feature；
- residual 只有在 target 实现后才能进入 residual EMA；
- next-trade、1-hour、1-day targets 必须按各自 label availability time 更新；
- benchmark rollover、session boundary、CUSIP universe entry 需要明确初始化或 reset 规则；
- target overlap 会机械制造 residual correlation，必须用 non-overlapping target 或显式 overlap audit 检查。

EMA feature 的评估应和 last value、finite rolling window、Kalman/ETS、multi-scale states 以及无 EMA baseline 比较。最重要的不是 feature importance，而是它是否在 point-in-time out-of-sample loss 上带来稳定增益。

# 12. 可复用代码与阅读顺序

同目录 `exponential_weighting.py` 实现：

- regular normalized EWA 与 recursive EMA；
- irregular clock-decayed observation average 与 dynamic-state recursion；
- weighted sample covariance 与 RiskMetrics covariance；
- half-life、mean age、ESS、tail horizon 转换；
- local-level optimal Kalman gain；
- expert-level exponential weights。

`test_exponential_weighting.py` 验证参数恒等式、pandas-style recursion、irregular weighting、covariance bias correction、PSD 与 Kalman gain 最优性。

建议文献顺序：

1. Muth：先看 exponential forecasting 为什么可能最优 [@muth1960]；
2. Holt、Winters、Gardner：理解 level/trend/seasonality 方法族 [@holt2004; @winters1960; @gardner1985; @gardner2006]；
3. Hyndman 等：把方法放入统一 state-space likelihood [@hyndman2002; @hyndman2008]；
4. RiskMetrics 与 Bollerslev：区分 conditional EWMA covariance 与 GARCH mean reversion [@riskmetrics1996; @bollerslev1986]；
5. Wright、Cipra–Hanzák：处理 irregular observation times [@wright1986; @cipra2008]；
6. Luxenberg–Boyd：从 moving average 推广到 moving convex models [@luxenberg2024]。

<div class="takeaway">

### 最终判断

EMA 的理论并不薄弱；薄弱的通常是使用方式。一个可解释、可验证的指数加权 feature 至少应该报告：

- estimand；
- update clock 与 label availability；
- half-life、mean age、ESS、tail horizon；
- initialization/reset；
- 与 probabilistic model 或 validation loss 的关系；
- 它机械引入的 lag、autocorrelation 和 target-overlap 风险。

做到这些以后，EMA 才从“随手加的一列 rolling feature”变成一个清楚定义的在线统计状态。

</div>

# References
