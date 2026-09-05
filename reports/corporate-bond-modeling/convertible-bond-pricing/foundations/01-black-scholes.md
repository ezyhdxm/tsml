## BS1 · Black–Scholes：先复制现金流，再谈风险中性 {#bs-replication}

::: {.callout}
**新增基础专题的主线**：单一股票扩散 → 自融资复制 → 定价 PDE → 正态积分；然后分别拿掉连续交易、零交易成本和固定波动率。原报告第 5 节的可转债行权问题仍在后面，章节编号不变。此专题的欧式股票 call 只是可转债期权部分的教学基准，不是把整张可转债当成普通 call。
:::

### BS1.1 模型承诺了什么？

先不考虑信用、提前行权和股票跳跃。股票支付连续股息率 $q$，无风险现金账户为 $A_t$。在真实概率 $\mathbb P$ 下写

$$
dS_t=(\mu-q)S_t\,dt+\sigma S_t\,dW_t^{\mathbb P},\qquad dA_t=rA_t\,dt.
$$

$\mu$ 是股票的预期**总回报率**；因此除息股价漂移是 $\mu-q$。先令 $r,q,\sigma$ 为已知常数，$\sigma>0$。欧式 call 在 $T$ 支付 $H=(S_T-K)^+$，价值记为 $C(t,S)$。

复制还需要：能够连续交易股票及现金；可以买卖任意小数量并按同一价格成交；没有交易费、冲击、借券限制；借贷按相同 $r$ 进行；没有未建模的跳跃。数学上还要求策略适应已有信息、适当可积，并排除无限借钱的 doubling 策略。**这些是下面等式成立的条件，不是对真实市场的描述。** 基础来源为 [Black–Scholes (1973)](https://www.journals.uchicago.edu/doi/10.1086/260062)；以下采用带连续股息的记号逐步重推。

### BS1.2 为什么 Taylor 展开需要保留二阶项？

小时间 $h$ 内，Brownian 增量尺度是 $\sqrt h$，所以股票随机增量尺度也是 $\sqrt h$。普通 Taylor 展开中的 $C_{SS}(\Delta S)^2/2$ 因而是 $h$ 量级，不能与高阶项一起丢掉。在细分网格极限中，Brownian 二次变差满足

$$
\sum_i(\Delta W_i)^2\longrightarrow t,\qquad
\sum_i\big[(\Delta W_i)^2-\Delta t_i\big]\longrightarrow0.
$$

这不是说每一个有限区间都有 $(\Delta W_i)^2=\Delta t_i$。若 $\Delta W=\sqrt h Z$，$Z\sim N(0,1)$，单步仍有 $(\Delta W)^2=hZ^2$；其均值是 $h$，方差是 $2h^2$。这个差别正是后面离散对冲误差的来源。

对光滑的 $C$ 应用 Itô 公式：

$$
dC=\left[C_t+(\mu-q)SC_S+\frac12\sigma^2S^2C_{SS}\right]dt+\sigma SC_S\,dW^{\mathbb P}.
$$

到期 payoff 在 $S=K$ 不光滑，但 $t<T$ 时的标准 call 价值光滑；可先在 $[0,T-\varepsilon]$ 推导，再在适当可积条件下取极限。不能把到期处的 gamma 当作一个处处有界的普通函数。

### BS1.3 最容易省略、也最重要的一行：自融资

设复制组合的总财富为 $X_t$，持有 $\theta_t$ 股，剩余现金为 $X_t-\theta_tS_t$。自融资意味着调仓的钱只能来自组合内部，而不是每次缺钱就从外面补：

$$
\boxed{dX_t=\theta_t\,dS_t+q\theta_tS_t\,dt+r(X_t-\theta_tS_t)\,dt.}
$$

三个收益来源依次是股价变动、股息和现金利息。若改变 $\theta$，买入新股票的支出会由现金账户支付，因而不是额外利润。

一个常见但危险的写法是先定义 $\Pi=C-C_SS$，然后直接写 $d\Pi=dC-C_SdS$。因为 $C_S$ 也变化，乘积微分本来还有 $S\,dC_S+d\langle C_S,S\rangle$；只有把相应调仓融资放进现金账户，才有正确的**收益过程**。上面的 $X$ 方程从一开始就把账记对了。

将股票 SDE 代入自融资方程，得到

$$
dX=\big[\theta\mu S+r(X-\theta S)\big]dt+\theta\sigma S\,dW^{\mathbb P}.
$$

为了使 $X=C$，先匹配随机项，唯一自然选择是

$$
\theta=C_S.
$$

再匹配漂移：

$$
C_t+(\mu-q)SC_S+\tfrac12\sigma^2S^2C_{SS}
=\mu SC_S+r(C-SC_S).
$$

$\mu SC_S$ 在两边消掉，整理为

$$
\boxed{C_t+\frac12\sigma^2S^2C_{SS}+(r-q)SC_S-rC=0,\qquad C(T,S)=(S-K)^+.}
$$

**$\mu$ 消失不是因为假定投资者风险中性，而是因为同一份股票随机风险已用股票复制。** 复制所需的初始资金是期权价格；如果相同终值的可交易组合价格不同，在这些无摩擦假设下就可以一买一卖。

### BS1.4 从 PDE 到风险中性期望，不把 $\mathbb Q$ 当作魔法

引入一个概率测度，使股票总回报漂移变成 $r$：

$$
dS=(r-q)S\,dt+\sigma S\,dW^{\mathbb Q}.
$$

在常系数情形，可令 $\lambda=(\mu-r)/\sigma$，并用密度

$$
\left.\frac{d\mathbb Q}{d\mathbb P}\right|_{\mathcal F_T}
=\exp\left(-\lambda W_T^{\mathbb P}-\tfrac12\lambda^2T\right),
\qquad W_t^{\mathbb Q}=W_t^{\mathbb P}+\lambda t.
$$

指数密度具有均值 1；在这些条件下 $W^{\mathbb Q}$ 为新测度下的 Brownian motion。这个换测度只改漂移，不改二次变差，所以 $\sigma$ 仍是同一个扩散系数。

现在不是直接宣布“价格等于期望”，而是将 PDE 代入 Itô 公式：

$$
d\left(e^{-rt}C(t,S_t)\right)=e^{-rt}\sigma S_t C_S(t,S_t)\,dW_t^{\mathbb Q}.
$$

在使随机积分为真 martingale 的可积条件下，取条件期望：

$$
\boxed{C(t,S_t)=e^{-r(T-t)}\mathbb E^{\mathbb Q}[(S_T-K)^+\mid\mathcal F_t].}
$$

所以风险中性期望是**同一个复制 PDE 的概率表示**，不是一个与复制无关、对任何不完全市场都自动给出唯一价格的法则。带股息时，折现的除息股价本身不是 martingale；加上已收股息的折现总收益才是。

::: {.takeaway}
**本节只记一条链**：自融资收益方程 → 令 $\theta=C_S$ 匹配扩散 → 匹配漂移得到 PDE → 折现价值为 martingale → 条件期望表示。每次加摩擦或额外风险，都应回到这条链，明确是哪一步失效。
:::

## BS2 · 把 Black–Scholes 的正态积分真正算完 {#bs-integral}

令 $\tau=T-t$。在 $\mathbb Q$ 下，对 $\log S$ 应用 Itô：

$$
d\log S=(r-q-\tfrac12\sigma^2)dt+\sigma\,dW^{\mathbb Q}.
$$

因此给定当前 $S$，

$$
S_T=S\exp\left[(r-q-\tfrac12\sigma^2)\tau+\sigma\sqrt\tau Z\right],\quad Z\sim N(0,1).
$$

写 $a=\sigma\sqrt\tau$，并定义

$$
d_2=\frac{\log(S/K)+(r-q-\tfrac12\sigma^2)\tau}{\sigma\sqrt\tau},\qquad d_1=d_2+a.
$$

$S_T>K$ 等价于 $Z>-d_2$，所以

$$
\mathbb Q(S_T>K\mid S_t=S)=\Phi(d_2).
$$

把 call 拆为“收到股票”和“付出执行价”两项：

$$
C=e^{-r\tau}\mathbb E^{\mathbb Q}[S_T\mathbf1_{S_T>K}]-Ke^{-r\tau}\Phi(d_2).
$$

第一项中唯一需要处理的积分是

$$
\int_{-d_2}^{\infty}e^{az}\phi(z)\,dz.
$$

直接完成平方：

$$
az-\frac{z^2}{2}=-\frac{(z-a)^2}{2}+\frac{a^2}{2},
\qquad e^{az}\phi(z)=e^{a^2/2}\phi(z-a).
$$

令 $u=z-a$，则

$$
\int_{-d_2}^{\infty}e^{az}\phi(z)dz
=e^{a^2/2}\int_{-d_2-a}^{\infty}\phi(u)du
=e^{a^2/2}\Phi(d_1).
$$

乘回外面的系数，$-\sigma^2\tau/2$ 与 $a^2/2$ 抵消：

$$
e^{-r\tau}\mathbb E^{\mathbb Q}[S_T\mathbf1_{S_T>K}]
=S e^{-q\tau}\Phi(d_1).
$$

最终得到

$$
\boxed{C_{\mathrm{BS}}=S e^{-q\tau}\Phi(d_1)-K e^{-r\tau}\Phi(d_2).}
$$

第二项中的 $\Phi(d_2)$ 才是该测度下价内概率。第一项中的 $\Phi(d_1)$ 来自以 $S_T$ 加权后的截断期望，不能把两个 $\Phi$ 都解释成同一个“涨过执行价的概率”。更不能把它们当作真实世界的上涨概率。[原论文公开副本](https://www.cs.princeton.edu/courses/archive/fall09/cos323/papers/black_scholes73.pdf)。

### BS2.1 Delta、gamma、vega 从哪里来？

微分公式时，利用 $S e^{-q\tau}\phi(d_1)=K e^{-r\tau}\phi(d_2)$，含 $\partial d_i/\partial S$ 的项抵消：

$$
\Delta=C_S=e^{-q\tau}\Phi(d_1),\qquad
\Gamma=C_{SS}=\frac{e^{-q\tau}\phi(d_1)}{S\sigma\sqrt\tau},
$$

$$
\mathcal V=C_\sigma=S e^{-q\tau}\phi(d_1)\sqrt\tau.
$$

$\Delta$ 的单位是“股 / 一份期权”；$\Gamma$ 是每一股价单位引起的 delta 变化。$\mathcal V$ 对应 $\sigma$ 增加 1.00，而 1 vol point 的一阶价格变化是 $0.01\mathcal V$。$C_t$ 的方向是日历时间向前，和 $\partial C/\partial\tau$ 相反。

对于原文无违约、零息、只在到期转股的精确分解 $V=Fe^{-r\tau}+mC_{\mathrm{BS}}$，有 $V_S=m\Delta$、$V_{SS}=m\Gamma$、$V_\sigma=m\mathcal V$。**一旦条款允许提前终止或信用与股票耦合，就应对完整 $V$ 求导，不能一直借用 $m$ 倍 vanilla Greeks。**

### BS2.2 为什么还要使用一个明显不完美的模型？

Black–Scholes 同时给出三种不同东西：一个明确假设下的复制定理、一个方便计算的价格函数，以及把期权市场价格转换为 implied volatility 的报价坐标。前者的假设失效，不代表后两者毫无用处；但也不能因为它是方便的报价坐标，就继续把它当成真实动态和无误差对冲保证。

后面依次保留前面的其他假设、只改变一件事，以免把交易成本、离散误差和未被覆盖的风险混成一个“波动率修正”。
