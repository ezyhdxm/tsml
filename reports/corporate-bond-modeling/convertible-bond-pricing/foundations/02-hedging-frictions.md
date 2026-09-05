## BS3 · 不能连续对冲：失去的是逐路径复制 {#bs-discrete}

### BS3.1 先固定损益符号

从这里开始，除另有说明外，为了专注对冲问题令 $q=0$。设交易者**卖出一份 call**，收取 $C(0,S_0)$，用股票和现金建立复制组合 $X$。到期现金支付 $H=(S_T-K)^+$，卖方的对冲损益定义为

$$
E_T=X_T-H.
$$

$E_T>0$ 是卖方赚到钱；$E_T<0$ 是复制资金不足。下文数值表全部报告折现到 0 时刻的 $e^{-rT}E_T$。买入期权并反向做股票 hedge 的、已融资净损益，在无费用且完全反向的同一设置中符号相反。

### BS3.2 离散策略仍然可以严格自融资

只能在 $0=t_0<t_1<\cdots<t_N=T$ 交易，就在 $[t_i,t_{i+1})$ 固定持股

$$
\theta_t=\Delta_i=C_S(t_i,S_{t_i}).
$$

在正确的 Black–Scholes 模型下，从自融资收益方程减去期权价值过程，有**精确恒等式**

$$
\boxed{e^{-rT}E_T=\sum_{i=0}^{N-1}\int_{t_i}^{t_{i+1}}e^{-rt}\big[\Delta_i-C_S(t,S_t)\big]\big(dS_t-rS_tdt\big).}
$$

这里没有外部补钱。误差来自区间内实际持仓是旧 delta，而不是即时 delta。在 $\mathbb Q$ 下，$dS-rSdt=\sigma S\,dW^{\mathbb Q}$；在适当可积条件下，平均误差为 0，但单条路径的误差不为 0。在真实 $\mathbb P$ 下还含有 $(\mu-r)Sdt$ 项，一般不能再声称真实预期损益严格为 0。

**有一个无偏的价格或平均对冲误差，不等于已经复制了终值。** 当有限交易日期之间的股票终值仍是连续分布，股票与现金通常不足以复制任意非线性 payoff。两分支二叉树每步能精确复制，是那个离散两状态模型的性质，不是实盘每天交易一次就自动 complete。

### BS3.3 一步误差为何和 gamma 有关？

在长度 $h$ 的小区间冻结局部系数，$\Delta S\approx\sigma S\sqrt h Z$。期权增量中的非线性部分是 $\Gamma(\Delta S)^2/2$；PDE 中的 theta/carry 抵消的是 $\Gamma\sigma^2S^2h/2$。因此卖方一步剩余误差的主项为

$$
\boxed{\varepsilon_i\approx\frac12\Gamma_i\big[\sigma^2S_i^2h-(\Delta S_i)^2\big]
=-\frac12\Gamma_i\sigma^2S_i^2h(Z_i^2-1).}
$$

因为 $\mathbb E[Z^2]=1$、$\operatorname{Var}(Z^2)=2$，

$$
\mathbb E_i[\varepsilon_i]\approx0,\qquad
\operatorname{Var}_i(\varepsilon_i)\approx\frac12\Gamma_i^2\sigma^4S_i^4h^2.
$$

把许多小区间的中心化误差累积起来，在等距网格、连续扩散及适当正则性/可积条件下，典型结果是

$$
\operatorname{Var}^{\mathbb Q}(e^{-rT}E_T)
\approx\frac h2\mathbb E^{\mathbb Q}\left[\int_0^T e^{-2rt}\Gamma_t^2\sigma^4S_t^4dt\right],
\qquad \operatorname{SD}(E_T)=O(\sqrt h).
$$

因此把交易频率提高四倍，纯离散误差的标准差大约减半。这个尺度不是对 digital、barrier、到期 kink、非均匀网格和跳跃模型的无条件保证；应分别检验数值收敛。右边的折现积分说明：误差不仅由终值 volatility 决定，还由**路径上 gamma 在什么时间、什么股价附近变大**决定。

### BS3.4 连续对冲也不能消除错误波动率

假设真实股票仍连续，但瞬时波动率是 $a_t$；交易者使用一个 $C^{\mathrm{model}}(t,S)$，其 PDE 中波动率为 $\sigma_m(t,S)$，并连续持有该模型的 delta。初始资金等于模型价，其他 carry 一致，则直接用 Itô 相减可得

$$
\boxed{e^{-rT}E_T=\frac12\int_0^T e^{-rt}\Gamma_t^{\mathrm{model}}S_t^2\big[\sigma_m^2(t,S_t)-a_t^2\big]dt.}
$$

这是对指定光滑模型标价及连续股价的恒等关系，不需要先把真实波动率设成常数。对持有凸期权的反向 hedge，符号相反。

它揭示两个不同误差：**离散化误差**可随网格加密而减小；**模型波动率错配**一般不会。正确的比较是 gamma-weighted realized variance 与模型方差，而不是不加权的全年 realized volatility 和一个初始 IV。即使全年平均方差相同，高波动发生在 gamma 大的时段还是小的时段，结果也会不同。相关稳健性研究见 [El Karoui–Jeanblanc-Picqué–Shreve (1998)，出版方摘要与书目信息](https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-9965.00047)；这里的恒等式由上述收益方程独立重推，不把摘要当作完整定理条件。

若出现股价跳跃 $J=S_t-S_{t-}$，卖方的瞬时残差为

$$
\Delta E=\theta_{t-}J-\big[C(t,S_{t-}+J)-C(t,S_{t-})\big].
$$

取 $\theta=C_S$，对凸的 $C$ 该残差非正；有限跳跃不能靠让交易网格更密来消掉。可转债的违约还改变合约本身的回收现金流，应使用后文的 $G-V+\theta\eta S$，而不是把存续价值函数在违约后的股价上机械延拓。

## BS4 · 加入 bid–ask：更频繁交易不再免费 {#bs-costs}

### BS4.1 先分清两层 spread

股票 bid–ask 是**执行 hedge 的成本**。可转债或期权自身的 bid–ask 是**买卖衍生品的报价差**，还可能含库存、借券、融资、对手方及风险补偿。把衍生品 mid price 加上“一次股票 half-spread”，既没有计算未来调仓，也不能得到通用的衍生品 ask。

设股票 mid 为 $S$，报价为

$$
S^{\mathrm{ask}}=(1+\kappa)S,\qquad S^{\mathrm{bid}}=(1-\kappa)S.
$$

$\kappa$ 是比例 **half-spread**，full spread 是 $2\kappa$。例如 $\kappa=0.0005$，在 $S=100$ 时 bid/ask 为 99.95/100.05；买入或卖出一股相对 mid 的执行成本均为 0.05。

从 $\theta_{i-1}$ 调到 $\theta_i$，不论方向，都支付

$$
\mathrm{TC}_i=\kappa S_i|\theta_i-\theta_{i-1}|.
$$

这个模型假设可按给定 spread 成交、没有固定费和市场冲击；真实成本可依时间、数量、方向和状态改变。成本参数不是公司债的信用 spread，也不是 yield basis points。

### BS4.2 把开仓、调仓、平仓全部记到账上

用 $B_i^+$ 表示 $t_i$ 调仓后的现金余额。令 $\theta_{-1}=0$，期权卖出所得为 $p$，则

$$
B_0^+=p-\theta_0S_0-\kappa S_0|\theta_0|.
$$

对 $i=1,\ldots,N-1$：

$$
B_i^-=B_{i-1}^+e^{r(t_i-t_{i-1})},
$$

$$
\boxed{B_i^+=B_i^--S_i(\theta_i-\theta_{i-1})-\kappa S_i|\theta_i-\theta_{i-1}|.}
$$

若期权现金结算，到期要平掉股票，卖方净损益是

$$
E_T=B_{N-1}^+e^{r(T-t_{N-1})}+\theta_{N-1}S_T-\kappa S_T|\theta_{N-1}|-(S_T-K)^+.
$$

若实物交割，可以用部分持仓直接交付，末端交易数量应据合约重写；不能同时算“全部股票平仓”又算“股票交割”而重复收费。可转债转股得到股票时也有同样的库存净额问题。

对**固定的一条 hedge 策略**，若费用从现金账户支付且不改变持股规则，有

$$
e^{-rT}E_T^{\mathrm{net}}=e^{-rT}E_T^{\mathrm{gross}}-
\sum_{i=0}^{N}e^{-rt_i}\mathrm{TC}_i,
$$

其中末端交易定义为将持股调到 0。后面的实验用两套独立账本逐路径核对这一等式。所有费用折现后再相加，避免把今天和未来的美元混在一起。

### BS4.3 为什么 continuous delta hedge 的成本会发散？

对普通光滑期权，短时间内 delta 的主要变动是

$$
\Delta\theta\approx\Gamma\Delta S\approx\Gamma\sigma S\sqrt h Z.
$$

因为 $\mathbb E|Z|=\sqrt{2/\pi}$，一次调仓的平均成本近似为

$$
\mathbb E_i[\mathrm{TC}_i]\approx\kappa\sigma S_i^2|\Gamma_i|\sqrt{\frac{2h}{\pi}}.
$$

每单位时间约交易 $1/h$ 次，所以平均运行成本率是

$$
\boxed{c(t,S)\approx\kappa\sigma S^2|\Gamma|\sqrt{\frac{2}{\pi h}}.}
$$

因此，在相同 delta 策略、非退化 gamma 的连续扩散区间及固定 $\kappa>0$ 下：

$$
\text{纯离散误差 SD}\sim\sqrt h,\qquad
\text{累计调仓成本}\sim\frac{\kappa}{\sqrt h}.
$$

前者趋零，后者却变大。这不是“Brownian 每次波动很小所以交易免费”；无限多次小交易的绝对成交量不能相互抵消。对 gamma 恒为零的线性 payoff，不适用同样的发散论证。[Leland (1985)，出版方摘要](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1985.tb02383.x) 正是从交易费用和修订频率出发修正复制策略；本节的系数采用明确的 half-spread 定义自行推算。

### BS4.4 Leland 调整：有用的成本近似，不是修好了一切

如果把上述平均运行成本当作每单位时间必须融资的支出，那么自融资方程多一个 $-c\,dt$。对一个拟议的卖方资金价值 $C^L$，同样匹配漂移得到近似方程

$$
C_t^L+(r-q)SC_S^L+\tfrac12\sigma^2S^2C_{SS}^L-rC^L
+\kappa\sigma S^2|C_{SS}^L|\sqrt{\frac{2}{\pi h}}=0.
$$

对 $\Gamma\ge0$ 的 vanilla call，可以把 gamma 系数合并为

$$
\boxed{\sigma_L^2=\sigma^2+2\kappa\sigma\sqrt{\frac{2}{\pi h}}.}
$$

这说明为什么有人会用“更大的 volatility”补偿卖方调仓成本。这里因子 2 与 $\kappa$ 是 half-spread 直接相关；不同文献将 $k$ 定义成 full spread 或往返费率时，表面系数会不同。

必须保留四个限定：它把随机交易成本近似成条件均值；不消除有限频率的随机复制误差；上述运行项没有自动包办初始建仓和最终平仓费用；在固定 spread 下令 $h\to0$ 并不能由此宣称无误差、有限资金的精确复制。若组合 gamma 换号，$|\Gamma|$ 使问题非线性，也不能给全部头寸塞入同一个增大的常数 volatility。

### BS4.5 怎样讨论“合适的对冲频率”？

先给一个**明确但人为选择的准则**。若运行成本近似 $A\kappa h^{-1/2}$、对冲误差方差近似 $Bh$，最小化

$$
J(h)=A\kappa h^{-1/2}+\frac\gamma2 Bh
$$

就有

$$
h^*=\left(\frac{A\kappa}{\gamma B}\right)^{2/3}.
$$

这里 $\gamma$ 的单位要使两项可比较；$A,B$ 依产品、路径和参数而变。若改成“平均成本 + $\alpha$ 倍误差标准差”，则

$$
J(h)=A\kappa h^{-1/2}+\alpha\sqrt{Bh},\qquad
h^*=\frac{A\kappa}{\alpha\sqrt B}.
$$

**所谓最优频率依赖你惩罚方差、标准差、尾部损失还是资本消耗，不能脱离目标给一个统一答案。** 上式是冻结尺度近似，不是直接拿来设实盘调仓日程的通用公式。

进一步可以不用固定时间触发，而是设置 no-trade band：当前持仓与目标持仓的偏离在容忍区内就不动，超出后交易到边界。其状态至少包括 $t,S$ 和已有持仓 $\theta$；两个人在同一股价持有不同股票数量，下一步动作可能不同。小比例交易费下的渐近分析可得到特定假设下 $\kappa^{1/3}$ 量级的 band 宽度，但常数及目标中心依风险偏好、gamma 和投资机会而定。[Whalley–Wilmott (1997)，作者公开论文](https://users.ox.ac.uk/~ofrcinfo/file_links/mf_papers/1999mf08.pdf)。本报告没有把该控制问题或最优 band 数值求解伪装成已实现。

### BS4.6 有摩擦以后，无套利还剩什么？

无套利要求没有**按可成交价格和允许策略计算的**确定净盈利，并没有失效。失效的是“股票 mid 上的精确复制给出唯一衍生品价格”这一捷径。买入复制组合和卖出反向复制组合的执行成本不同，通常只能得到与策略集合相联系的价格界。

例如等间距执行价的 call butterfly：买低、高执行价各一份必须付各自 ask，卖中间执行价两份只能收 bid。该组合终值非负；只有建仓净收入在计入全部融资、结算及其他费用后仍构成确定优势，才是可执行套利。仅用 mid price 发现曲率有点负，并不能省略这些交易方向和费用。

实际报价还需要一个残余风险准则。用 $H_d=e^{-rT}H$ 表示折现负债，$G(\theta)$ 表示零初始财富策略的折现交易收益，$\mathrm{TC}_d(\theta)$ 为折现费用，$\rho$ 为对**损失**的现金平移不变风险度量。固定真实概率模型、初始库存及允许策略集合后，可定义

$$
\mathcal R(H_d)=\inf_{\theta}\rho\big(H_d-G(\theta)+\mathrm{TC}_d(\theta)\big).
$$

在该优化有意义的条件下，一种无差异报价约定是

$$
p^{\mathrm{ask}}=\mathcal R(H_d)-\mathcal R(0),\qquad
p^{\mathrm{bid}}=\mathcal R(0)-\mathcal R(-H_d).
$$

减去 $\mathcal R(0)$ 是为了扣除“不做期权也能进行最优投资”的基准。它们是指定风险偏好和约束下的报价，不是市场必然给出的 bid/ask，也不是一般都关于 BS mid 对称。指数效用/entropic 风险、CVaR 等选择会改变结果。可以用神经网络近似策略，但网络不替你定义风险偏好或真实概率。[Bühler et al., Deep Hedging](https://arxiv.org/abs/1802.03042) 将费用、限制和风险准则直接放进对冲优化；本报告只介绍其建模入口，没有训练该模型。
