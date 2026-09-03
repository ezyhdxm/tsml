# 3A. 为什么要有 vol surface？为什么不是直接修模型？ {#why-surface}

<div class="callout success">
<strong>先回答你的质疑：</strong>如果把不同 strike 的 implied volatility 当成同一个 underlying 的不同“真实常数波动率”，那么确实没有一个统一的 Black–Scholes 模型支持这件事。正确的解释是：<strong>放弃常数波动率模型对全部期权的联合定价假设，保留 Black 公式作为价格与 IV 之间的可逆换算。</strong>修模型与构建 surface 并不是二选一；前者解释动态，后者整理需要被解释的报价。
</div>

本节先固定今天为 0；$T$ 表示到期，等于前文的剩余期限 $\tau$。为把注意力放在逻辑上，推导例子暂设利率和分红为零，因此 $F=S_0$、discount factor 为 1；涉及实际 carry 时再恢复前文的 forward 形式。

## 3A.1 Black–Scholes 的数学推导没有允许你偷偷更换 underlying

常数波动率模型在风险中性测度下假设

$$
dS_t=rS_t\,dt+\sigma_0 S_t\,dW_t.
$$

同一个 underlying 只有一个 $\sigma_0$。利用 Itô 公式对 $\log S_t$ 展开并积分：

$$
\log S_T=\log S_0+(r-\tfrac12\sigma_0^2)T+\sigma_0W_T.
$$

给定今天的信息，$S_T$ 因而服从一个确定的 lognormal distribution。每个 strike 的 call 都对**这同一个分布**取期望：

$$
C(K,T)=e^{-rT}\mathbb E^Q[(S_T-K)^+]
=C_{\mathrm{BS}}(S_0,K,T;\sigma_0).
$$

所以在该模型内部，反解任何 strike、任何 maturity 的 implied volatility，都应得到同一个 $\sigma_0$。市场存在 smile，表明这个联合分布假设不能同时解释所有价格，<strong>不是 Black–Scholes 的代数推导突然允许了多个不同的 $\sigma_0$</strong>。Derman–Kani 的原始论文正是从这一矛盾出发，扩展 underlying dynamics。[原始讨论：Derman–Kani (1994)](https://emanuelderman.com/the-volatility-smile-and-its-implied-tree/)

### 只让波动率随日历时间变化，够不够？

把常数改成已知的确定性函数 $a(t)$：

$$
\frac{dS_t}{S_t}=r\,dt+a(t)\,dW_t.
$$

此时

$$
\log(S_T/S_0)=rT-\frac12\int_0^T a(u)^2du
+\int_0^T a(u)dW_u.
$$

最后一个积分仍是 Gaussian，方差是 $\int_0^T a(u)^2du$。所以所有同一到期的 call 仍由同一个 lognormal law 定价，其 Black-equivalent volatility 为

$$
\sigma_{\mathrm{imp}}(K,T)
=\sqrt{\frac1T\int_0^T a(u)^2du}.
$$

它可以随 $T$ 变，但不随 $K$ 变。<strong>确定性时间变动能解释 term structure，不能单独解释同一期限的 smile。</strong>要产生 strike dependence，需要改变终值分布形状，例如状态依赖的 diffusion、随机波动率或跳跃；不是简单地“换成明天另一个常数”。

## 3A.2 必须区分：模型、换算公式、统计曲面

| 对象 | 它在回答什么？ | 它有没有指定标的资产如何演化？ |
|---|---|---|
| Black–Scholes 模型 | 在 GBM 假设下，这些期权应该值多少？ | 有 |
| Black/BS 换算公式 | 给定一个价格，相当于公式中的哪个 IV 数字？ | 反解时没有增加新的动态假设 |
| 今天的 implied-vol surface | 不同合约今天的价格如何组织成一个连续查询对象？ | 没有完整指定 |
| Heston / local-vol / LSV 模型 | 什么动态过程能解释这些价格，并用于路径定价与对冲？ | 有，但还需校准和适用条件 |
| 跨日 surface 统计模型 | 历史曲面怎样共动、缺失区域怎样推断、未来报价如何变化？ | 不一定能推出一致的风险中性 underlying dynamics |

把 Black 公式在 $\sigma$ 方向看成一个严格递增函数。对满足价格上下界内部条件的市场价格，定义

$$
\sigma_{\mathrm{imp}}(K,T)
=\operatorname{BlackInverse}\bigl(C_{\mathrm{mkt}}(K,T);F,K,T,D\bigr).
$$

于是

$$
C_{\mathrm{mkt}}(K,T)
=C_{\mathrm{Black}}\bigl(F,K,T,D;\sigma_{\mathrm{imp}}(K,T)\bigr)
$$

此时是一个**定义导致的等式**，不是“常数波动率模型预测市场成功”的检验结果。对每个数据点反解参数，总能在该点拟合；模型价值应由跨合约约束、未观测价格、动态或对冲来检验。Dupire 在原始文章开头也明确区分了模型给出的价格与由市场价格反推出的 IV。[Dupire 原文重刊](https://www.risk.net/derivatives/equity-derivatives/1500211/pricing-with-a-smile)

可以类比债券的 yield：两只不同现金流的债券有不同 yield，并不意味着对每只债券指定一个不同的真实短利率过程。Yield 是价格的等价报价坐标；term-structure dynamics 是另一个模型问题。这个类比只用于区分角色，不表示 yield curve 与 IV surface 具有相同的无套利约束。

<div class="callout warning">
<strong>这不是说“surface 没有模型假设”。</strong>单个价格换成 IV 的动作很弱；但把稀疏报价补成整张 surface 时，你必须加入插值、平滑、形状或历史先验。缺失区域的价格不由换算公式凭空决定。真正的建模发生在这里。
</div>

## 3A.3 一个可计算反例：同一个分布，却有不同 IV

不妨设今天 forward 为 100，到期一年。终值分布是两个具有同样均值 100 的 lognormal 分布的混合：

$$
S_T\mid B=1\sim100\exp(-0.1^2/2+0.1Z),
\qquad
S_T\mid B=2\sim100\exp(-0.4^2/2+0.4Z),
$$

其中 $\mathbb P^Q(B=1)=\mathbb P^Q(B=2)=1/2$，$Z$ 为标准正态，独立于 $B$。这里 $B$ 只是定义这个教学用终值分布；我们没有声称它已经唯一确定连续时间动态。

先算均值：

$$
\mathbb E^Q[S_T]
=\tfrac12\,100+\tfrac12\,100=100.
$$

再按条件期望定价：

$$
\begin{aligned}
C(K,T)
&=\mathbb E^Q\!\left[\mathbb E^Q[(S_T-K)^+\mid B]\right]\\
&=\tfrac12 C_{\mathrm{BS}}(K,T;10\%)
+\tfrac12 C_{\mathrm{BS}}(K,T;40\%).
\end{aligned}
$$

<strong>对所有 strikes，我们用的始终是这同一个混合分布。</strong>因为它给出非负概率、正确均值及 payoff expectation，同一期限的价格天然满足单调性与凸性。但反解 Black IV 后得到：

| Strike | 统一混合分布给出的 call price | 等价 Black IV |
|---|---:|---:|
| 80 | 23.215549 | 28.7951% |
| 100 | 9.919852 | 24.9298% |
| 120 | 4.667713 | 27.8192% |
| 140 | 2.613010 | 31.1646% |

<figure>
<img src="classical_examples/mixture_smile.png" alt="One mixture distribution gives a non-flat implied-volatility smile">
<figcaption>新增图 A：本次实际计算的教学例子，非市场数据。同一个非 lognormal 终值分布映射到 Black 坐标后出现 smile；不是不同 strikes 对应不同标的资产。</figcaption>
</figure>

为什么 IV 不是简单的 25%？因为 call price 对波动率不是线性函数，因此“先算两种情景的价格再平均”不等于“先平均 volatility 再定价”。它也不是 $\sqrt{(0.1^2+0.4^2)/2}=29.1548\%$。不同 strike 的 payoff 对分布不同区域敏感，匹配一个 lognormal family 时需要不同的等价参数。

这已经回答了最根本的逻辑问题：<strong>一个合理的统一模型完全可以产生非平坦的 implied-vol surface；矛盾的只是“一个统一的常数波动率 GBM”与非平坦 surface 同时为真。</strong>

## 3A.4 那为什么不直接换成正确模型？其实一直在换

一种工作流是先选择 Heston 等动态，再从它计算所有期权价格，最后需要报价时再反解 IV。另一种是先把市场报价整理成静态 surface，再把它作为动态模型校准目标。还可以直接对原始 bid/ask 校准动态模型，并把模型输出作为 surface。<strong>并不存在“必须先建一张非参数曲面”的定理。</strong>

Heston 在风险中性测度下的基本结构是

$$
\begin{aligned}
dS_t&=(r-q)S_tdt+\sqrt{v_t}\,S_t\,dW_t^S,\\
dv_t&=\kappa(\bar v-v_t)dt+\xi\sqrt{v_t}\,dW_t^v,\\
d\langle W^S,W^v\rangle_t&=\rho\,dt.
\end{aligned}
$$

本小节的 $v_t$ 是瞬时方差，$\bar v$ 是长期方差水平，$\kappa$ 是回复速度，$\xi$ 是方差的波动强度，$\rho$ 是两个 Brownian shocks 的相关系数。它们不是五个新的 surface 坐标，而是同一个动态模型的一组参数。相关性和随机方差允许非 lognormal 终值分布。[Heston (1993)](https://doi.org/10.1093/rfs/6.2.327)

取同一组参数，对每个 $(K,T)$ 计算

$$
C_{\mathrm{Heston}}(K,T)
=D(T)\mathbb E^Q[(S_T-K)^+],
$$

再反解，就得到它的 model-implied surface。<strong>修模型并不会让 surface 消失，而是让 surface 有了一个生成机制。</strong>

为何不只保留 Heston 参数？因为一个低维模型族一般只能产生全部可能报价曲面中的一部分。市场报价噪声、模型误设、跨期限形状与局部供需都会造成 residual。实务设计可选择接受误差，扩展动态模型，或者加一层市场校正；但任何校正都要重新检查约束和对冲的一致性，不能仅因为 vanilla 拟合更好就宣称模型更正确。

## 3A.5 一张 surface 包含什么？又缺什么？

在零利率、零分红、足够光滑且覆盖全部 strikes 的理想情况下，固定 $T$：

$$
C(K,T)=\int_K^\infty(s-K)f_T(s)ds.
$$

逐次对 $K$ 求导，积分下限项因 $(K-K)=0$ 消失：

$$
C_K(K,T)=-\int_K^\infty f_T(s)ds,
\qquad
C_{KK}(K,T)=f_T(K).
$$

所以一整条 call-price curve 确定该到期的风险中性边际分布；跨全部 $T$ 的 surface 确定一族边际分布。对足够光滑的终值 payoff，还可展开

$$
h(S_T)=h(0)+h'(0)S_T+\int_0^\infty h''(K)(S_T-K)^+dK.
$$

取期望就能用 cash、underlying 和一条 call strip 定价这类终值 payoff。这说明 surface 不是无用的画图对象：它组织了相当丰富的终值定价信息。

但是边际分布不是联合分布。它不能单独告诉你

$$
\mathcal L^Q(S_{T_2}\mid S_{T_1}),
$$

所以通常不能唯一决定 barrier、Asian、forward-start 等路径相关产品。连续时间中，marginal-mimicking / Markovian-projection 结果展示了具有相同单时点分布、却有不同动态结构的过程。[Atlan (2006)](https://arxiv.org/abs/math/0604316)；[Lacker–Shkolnikov–Zhang (2019)](https://arxiv.org/abs/1905.06213)

### 一个不用随机微积分的两期反例

取 $S_0=3$，第一期 $S_1\in\{2.5,3.5\}$，各概率 $1/2$；第二期 $S_2\in\{1,3,5\}$。下面每行是“给定第一期状态，第二期三个状态的条件概率”：

| 模型 | 给定 $S_1$ | 到 1 的概率 | 到 3 的概率 | 到 5 的概率 |
|---|---:|---:|---:|---:|
| A | 2.5 | 1/4 | 3/4 | 0 |
| A | 3.5 | 1/4 | 1/4 | 1/2 |
| B | 2.5 | 1/2 | 1/4 | 1/4 |
| B | 3.5 | 0 | 3/4 | 1/4 |

逐行计算可得，两模型均满足 $\mathbb E^Q[S_2\mid S_1]=S_1$；再对两行平均，两模型的第二期边际均为 $(1/4,1/2,1/4)$。所以它们都是这两个交易日期上的 martingale models，并对两个到期、所有 strike 的 European calls 给出相同价格。

但 payoff 为 $\mathbf 1\{S_1<3,\ S_2>4\}$ 的路径数字期权，在 A 下值 0，在 B 下值 $\tfrac12\times\tfrac14=1/8$。<strong>相同 vanilla surfaces，并不意味着相同 path-dependent prices。</strong>本例只涉及两个到期，不冒充对所有连续期限的构造。

## 3A.6 Local volatility：surface 和“修模型”如何接上？

考虑零利率、零分红的 local-vol model：

$$
dS_t=a(t,S_t)S_t\,dW_t.
$$

这里 $a(t,s)$ 是“到了时间 $t$、资产价格为 $s$ 时，瞬时扩散有多大”，与“今天给 strike $K$、到期 $T$ 的合约反解到的 $\sigma_{\mathrm{imp}}(K,T)$”不是同一件东西。

在密度、导数、边界衰减与 martingale 性等正则条件下，Dupire 关系为

$$
a(T,K)^2=\frac{2C_T(K,T)}{K^2C_{KK}(K,T)}.
$$

它提供了从静态 vanilla prices 到一个匹配边际分布的动态模型的桥梁。[Dupire (1994，重刊)](https://www.risk.net/derivatives/equity-derivatives/1500211/pricing-with-a-smile)

<details>
<summary>逐步推导：不用把 Dupire 当成需要背诵的公式</summary>

从一个光滑测试函数 $h$ 出发，用 Itô 公式并取期望，随机积分的期望为零：

$$
\frac{d}{dt}\mathbb E[h(S_t)]
=\frac12\mathbb E[a(t,S_t)^2S_t^2h''(S_t)].
$$

若 $S_t$ 的密度为 $f(t,s)$，等式两边写成积分并对右边做两次分部积分，在边界项消失时得到

$$
\partial_t f(t,s)
=\tfrac12\partial_{ss}\!\left[a(t,s)^2s^2f(t,s)\right].
$$

在固定到期 $T$，临时把括号里的函数记为 $A(s)$。于是

$$
\begin{aligned}
C_T(K,T)
&=\frac12\int_K^\infty(s-K)A''(s)ds\\
&=\frac12\left([(s-K)A'(s)]_K^\infty-\int_K^\infty A'(s)ds\right)\\
&=\frac12 A(K)
=\frac12 a(T,K)^2K^2f(T,K).
\end{aligned}
$$

由前面已经推导的 $C_{KK}(K,T)=f(T,K)$，只要分母非零即可移项得到 Dupire。这里的 $T$ 导数是<strong>今天观察的价格曲面沿到期轴的导数</strong>，不是预测明天市场报价如何移动。

当利率 $r(T)$ 与连续分红率 $q(T)$ 是确定性函数，未归一化的 call price 对应

$$
a(T,K)^2=
\frac{2\left[C_T+(r-q)KC_K+qC\right]}{K^2C_{KK}}.
$$

随机利率、离散分红或 American exercise 不能无条件套这个简式。
</details>

<strong>即使能拟合今天的全部 vanilla prices，也不能据此验证 local-vol dynamics 就是实际动态。</strong>密度的数值二阶导还容易放大报价噪声；而 surface 随 spot 变化的方式会影响 hedge。SABR 原始工作正是强调“拟合当前 smile”与“产生合适的 smile dynamics / hedge”之间的差别。[Hagan et al., Managing Smile Risk](https://www.wilmott.com/managing-smile-risk/)

## 3A.7 把 surface 塞回 BS，为什么不能直接照搬 delta？

假设为讨论一个确定的 spot shock 规则，写

$$
C(S)=C_{\mathrm{BS}}(S,K,T;\sigma_{\mathrm{imp}}(S,K,T)).
$$

链式法则给出

$$
\frac{dC}{dS}
=\Delta_{\mathrm{BS}}+\mathrm{Vega}_{\mathrm{BS}}
\frac{\partial\sigma_{\mathrm{imp}}}{\partial S}.
$$

BS delta 是保持输入 volatility 不动的偏导；市场 delta 还依赖你假定 smile 在 spot 变化时如何响应。比如零 carry、固定 $K,T$ 下：

- **Sticky strike**：保持各固定 strike 的 IV 不动，第二项为零。
- **保持 log-moneyness 曲线形状不变**：若 $\sigma_{\mathrm{imp}}(S,K,T)=s(\log(K/S),T)$，则 $\partial_S\sigma_{\mathrm{imp}}=-s_k/S$。

这是两种不同的情景约定，不是由今天的 surface 唯一推出来的动态真相。“保持 log-moneyness”也不等于不加说明地说“sticky delta”，因为 delta 还依赖 volatility 和具体市场 quote convention。

真实随机波动率还可能存在不能由 spot alone hedge 的独立 shock；完整 Itô 展开会出现 vega、vanna、volga 等项。所以上式只说明<strong>为何必须补充动态规则</strong>，不是声称用一项 chain-rule correction 就能实现完美对冲。

## 3A.8 为什么工程上常把 surface 单独保留？

这是一个设计判断，不是“所有机构都必须如此”的行业定律：<strong>让共同的报价表示与下游动态模型分离，可以减少把单一模型偏差写进整个系统的风险。</strong>

一张经清洗、注明 bid/ask 与插值可靠度的 surface，既可服务 vanilla quoting，也可用于比较不同模型的校准 residual、组织风险敞口、生成统计特征。换一个 exotic pricer，不必重新定义市场报价语言。反过来，如果任务只涉及少量合约，一个合适的动态模型已足够，就完全可以直接校准它，不额外建设复杂 surface-learning 系统。

**到这里应记住：surface 回答“市场在给各类 payoff 什么价格”；动态模型回答“什么路径机制与这些价格兼容”。从报价反解 IV 不修复 BS 模型；选择 local-vol、Heston 或更丰富动态才是在修模型。**

