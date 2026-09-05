## 01 · 先把问题摆正：这是一份会改变身份的合约 {#overview}

一张普通公司债承诺支付现金；一张普通股票期权赋予股票相关的选择权。**可转债把两者放进同一份会提前终止、会被赎回、也可能违约的合约里。** 关键不只是“债券加期权”，而是：做出一个选择后，其他现金流与选择权是否还存在？

本文从一个完全可核算的例子出发，再逐层加入提前转股、发行人赎回、投资者回售及信用风险。数值部分全部使用合成参数，不使用真实交易数据；它是定价教程与模型检查，不是任何证券的估值意见。

::: {.callout}
**贯穿全文的四个问题**：现在转股能拿到什么？继续持有还保留什么？发行人能否提前结束合约？若先违约，股票和债券分别变成什么？
:::

### 阅读路线

第 2–4 节建立 payoff 和无违约基准；第 5–8 节把最优决策与违约写成递推式及 PDE；第 9–11 节落到树、有限差分和实算结果；第 12–15 节处理真实条款、校准、对冲及新文献。附录提供可直接运行的最小 Python 基准。

**三个必须区分的对象**：$V$ 是可转债整体价值；$mS$ 是转股所得股票的市值；$B_{\mathrm{straight}}$ 是删去转股等选择权后、在明确的信用及回收模型下计算的普通债券价值。它们不是同一个价格，也不是三种可以随意互换的“fair value”。

本文使用普通、投资者自愿转股的可转债作为对象，不把强制转股债、监管触发的 CoCo、可交换为其他公司股票的 exchangeable bond 混成同一个合约。不同的触发者和标的必须另写 payoff。

## 02 · 把 term sheet 翻译成数学 {#contract}

### 2.1 先只保留九个参数

| 记号 | 含义 | 本文合成基准 |
|:--|:--|--:|
| $F$ | 每张债券面值 | 100 |
| $S$ | 当前每股股价 | 50 |
| $m$ | 每张债券可转换的股数 | 2 |
| $T$ | 距到期的年数 | 3 |
| $r$ | 连续复利无风险利率 | 4% |
| $q$ | 连续股息率 | 1% |
| $\sigma$ | 存续状态下股价扩散波动率 | 30% |
| $h$ | 风险中性违约强度 | 2.5% / 年 |
| $R$ | 违约时按面值回收的比例 | 40% |

前半部分先令 $h=0$。基准是零息债；加入票息时另行说明。价格统一为**每 100 面值的货币金额**，不是 yield，也不是债券 spread。$m$ 的单位是“股 / 每张债券”，绝不能把每 1,000 面值的 conversion ratio 直接塞进每 100 面值的模型。

转股价只是换算关系：

$$
K=\frac{F}{m}=50,\qquad \text{conversion parity}=mS=100.
$$

若市场价格是 $P_{\mathrm{mkt}}$，常见的转股溢价定义为

$$
\frac{P_{\mathrm{mkt}}}{mS}-1.
$$

这是相对于当前转股价值的价格比例，不等于隐含波动率、信用利差，也不等于某个固定执行价期权的 premium。

### 2.2 真正需要逐项读的条款

| 条款 | 要翻译成的对象 | 常见错误 |
|:--|:--|:--|
| 转股窗口与转股比例 | 允许转股的日期/状态，以及 $m(t,\text{history})$ | 默认从发行日起随时转股 |
| Coupon、到期偿还额 | 金额、支付日、累计/除息规则 | 提前转股后仍保留未来全部票息 |
| Issuer call | 可赎回日期、价格、触发条件、通知期 | 把 call price 当成债券价值的绝对上限 |
| Holder put | 回售日期、金额、公司事件条件 | 与 issuer call 搞反方向 |
| 转股结算 | 实物股票、现金或混合；观察/平均窗口 | 始终使用即时 $mS$ |
| 违约与回收 | 违约定义、顺位、回收基数、结算延迟 | 写了 recovery rate 却没有说明乘什么 |
| 调整与特殊事件 | 反稀释、make-whole、重设转股价 | 把所有调整当成免费增加 $m$ |

这些不是“模型做完后的补丁”。例如 Rubrik 2025 年发行文件规定了有条件转股、20/30 个交易日的股价触发、现金/股票/混合结算选择以及赎回前保护期；因此一张美国可转债也不能自动当成从今天起可随时实物转股的 American option。[公开条款实例：Rubrik 8-K](https://www.sec.gov/Archives/edgar/data/1943896/000119312525140729/d920850d8k.htm)。这只是一个已核对的实例，不代表所有市场的统一条款。

## 03 · 什么时候“债券 + call”是精确的？ {#decomposition}

### 3.1 从到期 payoff 开始，而不是从产品口号开始

先假设：无违约、零票息、只在到期转股、没有 call/put、固定 $m$，且立即以股票结算。到期时，在收回本金和接收股票之间选择：

$$
V(T,S_T)=\max(F,mS_T).
$$

逐状态恒等变形：

$$
\max(F,mS_T)=F+(mS_T-F)^+=F+m(S_T-K)^+.
$$

因此线性定价给出

$$
\boxed{V(t,S)=F e^{-r(T-t)}+m C_{\mathrm{BS}}(S,K,T-t;r,q,\sigma).}
$$

这是**在这些假设下精确成立的复制关系**。不是“某一部分看上去像债券、剩下看上去像期权”的近似说法。若有确定票息，且只能在付清所有票息后的到期决策时转股，可另加票息现值；若允许提前转股并因此丧失后续票息，不能这样无条件相加。

### 3.2 再把 Black–Scholes 展开一次

令 $\tau=T-t$，$\Phi$ 为标准正态分布函数：

$$
C_{\mathrm{BS}}=S e^{-q\tau}\Phi(d_1)-K e^{-r\tau}\Phi(d_2),
$$

$$
d_1=\frac{\log(S/K)+(r-q+\sigma^2/2)\tau}{\sigma\sqrt{\tau}},\qquad d_2=d_1-\sigma\sqrt{\tau}.
$$

它来自 $\mathbb Q$ 下的分布

$$
\log S_T=\log S+(r-q-\sigma^2/2)\tau+\sigma\sqrt{\tau}Z,\quad Z\sim N(0,1).
$$

将 call 的期望拆成两个截断项：

$$
e^{-r\tau}\mathbb E[(S_T-K)^+]
=e^{-r\tau}\mathbb E[S_T\mathbf1_{S_T>K}]-K e^{-r\tau}\mathbb Q(S_T>K).
$$

第二项概率是 $\Phi(d_2)$；第一项用正态密度完成平方后得到 $S e^{-q\tau}\Phi(d_1)$。所以它使用的是风险中性的远期漂移，不是你对未来股价涨幅的预测。

### 3.3 同一 payoff，还有另一个有用分解

$$
\max(F,mS_T)=mS_T+m(K-S_T)^+.
$$

所以

$$
V(t,S)=mS e^{-q\tau}+mP_{\mathrm{BS}}(S,K,\tau).
$$

第一种分解突出“现金本金 + 上行”；第二种突出“股票敞口 + 下行保护”。两者相同，不代表真实可转债存在无条件本金保护：这里暂时假定发行人绝不违约。

### 3.4 用数字固定直觉

本例无违约零息欧式价值为 **112.1834**。其中本金现值是 $100e^{-0.04\times3}=88.6920$，转股选择权价值为 **23.4914**。现在的 parity 是 100，但总价值不是 100，因为持有人保留了到期再决定的权利。

::: {.takeaway}
**本节模板**：先逐状态写出到期现金流，再做恒等分解，最后才分别定价。某个条款使其中一项现金流随其他决策消失时，静态分解就需要重新检查。
:::

## 04 · Bond floor 与价格形状：保护有条件 {#floor}

在第 3 节的模型中，$V\ge F e^{-r\tau}$。若无 issuer call，且持有人总可以选择从不转股，那么在同一信用、回收与票息假设下，可转债不应低于对应 straight bond 的模型价值。

但“floor”不是承诺。信用恶化会降低 $B_{\mathrm{straight}}$；违约可能使现金回收远低于面值。若添加发行人赎回，比较对象应保留相同赎回条款，不能再把不可赎回直债的现值不加说明地当下界。

此外，只有**现在确实允许转股、即时结算且忽略交易摩擦**时，才可以用 $V\ge mS$。只能到期转股时，正确的股票相关静态下界是 $mS e^{-q\tau}$，而不是 $mS$；高股息下这个区别很重要。

<div class="figure"><canvas id="shapeChart" aria-label="股价与可转债价格、转股价值和直债价值的关系" role="img"></canvas><div class="legend">实线：显式违约欧式可转债　虚线：当前 parity　点线：同模型直债</div><p class="caption">图 1 · 全部为合成参数；仅到期转股、零票息、无赎回。由第 8 节闭式公式计算。深度价内时并不强制贴合当前 parity，因为持有人尚不能即时转股且股票支付股息。</p></div>

**债性、平衡态、股性**是有用的描述，不是三种不同证券。股价低时，赎回本金或违约回收的状态相对重要；股价在转股价附近时，终值在现金与股票之间切换的概率对波动率敏感；股价很高时，股票交付主导。加入 call、put、重设与信用耦合后，分界和形状都可能改变。

## 05 · 提前转股、回售与赎回：一个节点的决策 {#exercise}

### 5.1 先定义 continuation value

记 $V_{\mathrm{cont}}(t,S)$ 为“这个时点暂不结束合约，向前走一个小时间步，再最优决策”的价值。允许即时转股时：

$$
V=\max\{V_{\mathrm{cont}},mS\}.
$$

若同时允许投资者以 $P(t)$ 回售，持有人的即时选择下界为

$$
L(t,S)=\max\{mS,P(t)\},\qquad V=\max\{V_{\mathrm{cont}},L\}.
$$

只在指定日有 put 权利时，只在那些日期加入 $P(t)$。未开放转股的时点也不能随意加入 $mS$。

### 5.2 Issuer call 为什么不是 $V\le C(t)$？

用 $C(t)$ 表示现金赎回额，避免与第 3 节的 $C_{\mathrm{BS}}$ 混淆。假设发行人一旦 call，持有人仍可立刻选择转股，则 call 时持有人拿到

$$
U(t,S)=\max\{C(t),mS\}.
$$

若股价 60、$m=2$、赎回额 105，持有人会选择 120 的股票，不会被迫只拿 105。上界是 120，不是 105。

在**无通知期、call 即刻结算、转股在 call 时可用、回售额不高于赎回额**的简化节点上：

$$
\boxed{V=\max\bigl\{L,\min\{V_{\mathrm{cont}},U\}\bigr\}.}
$$

这是本文数值例子的事件约定；一般合约若出现不同权利优先级、重叠窗口或结算延迟，必须重新写节点规则。经典文献将这类约束表述为最优停止/双方博弈及互补条件。[AFV，§2](https://cs.uwaterloo.ca/~paforsyt/convert.pdf)。

### 5.3 三个不用解 PDE 也能检查的节点

| 假设 | 数值 | 正确动作与价值 |
|:--|:--|:--|
| continuation 112，parity 100，没有 call | $\max(112,100)$ | 继续持有，112 |
| continuation 118，parity 110，可立即 call 105 | $\min(118,\max(105,110))$ | 发行人 call，持有人转股，110 |
| continuation 92，parity 60，可 put 100 | $\max(92,60,100)$ | 回售，100 |

**股价超过转股价并不自动意味着应转股。** 现在转换会失去剩余可选择性，也可能失去票息；应比较 $mS$ 与 continuation，而不是只比较 $S$ 与 $K$。

### 5.4 票息事件必须与行权排序一致

本文的合成票息规则是：到 coupon date，仍存续的持有人先收到本期票息，再决定是否转股、回售或接受赎回；提前离开的持有人不再收未来票息。令 $V^+$ 为本期票息已付后的价值，则

$$
V(t_i^-,S)=c_i+V(t_i^+,S).
$$

实际合约可能要求转股时补偿应计利息、放弃部分利息或采用不同 record date。上述简单加法仅适用于本文约定；不能不看条款直接搬用。

## 06 · 从短时间条件期望推到无违约 PDE {#pde}

在 $\mathbb Q$ 下先采用

$$
dS=(r-q)S\,dt+\sigma S\,dW^{\mathbb Q}.
$$

在没有票息事件、也没有立即行权的 continuation region，定价条件为

$$
V(t,S)=e^{-r\Delta t}\mathbb E^{\mathbb Q}[V(t+\Delta t,S+\Delta S)\mid S_t=S].
$$

二阶展开，并使用 $\mathbb E[\Delta S]=(r-q)S\Delta t+o(\Delta t)$、$\mathbb E[(\Delta S)^2]=\sigma^2S^2\Delta t+o(\Delta t)$：

$$
\mathbb E[V(t+\Delta t,S+\Delta S)]
=V+\left(V_t+(r-q)SV_S+\frac12\sigma^2S^2V_{SS}\right)\Delta t+o(\Delta t).
$$

再用 $e^{-r\Delta t}=1-r\Delta t+o(\Delta t)$，减去 $V$、除以 $\Delta t$、令步长趋零：

$$
\boxed{V_t+\frac12\sigma^2 S^2V_{SS}+(r-q)SV_S-rV=0.}
$$

若连续支付票息流 $c(t)$，左侧加 $c(t)$；若是离散票息，时间间隔内部不加该项，而是在支付日应用第 5.4 节的跳跃关系。两种方法不能同时计入同一笔票息。

有行权权利时，PDE 只在 continuation region 成立。令

$$
A[V]=V_t+\tfrac12\sigma^2S^2V_{SS}+(r-q)SV_S-rV.
$$

对于 $L<U$ 的区域，条件可写为

$$
\begin{cases}
A[V]=0,& L<V<U,\\
A[V]\le0,& V=L<U,\\
A[V]\ge0,& V=U>L.
\end{cases}
$$

符号可从“继续一小步的价值 $\approx V+A[V]\Delta t$”检查：在持有人行权下界，继续不应更好；在发行人赎回上界，继续不应对发行人更有利。上下界相等时价值直接被固定，不应机械套用严格区间的条件。

::: {.takeaway}
**本节模板**：先写 discounted conditional expectation，再展开得到 continuation PDE；最后单独加入 terminal payoff、事件跳跃与行权障碍。不要把所有条款都硬塞进漂移项。
:::

## 07 · TF 分拆：现金部分加信用 spread {#tf}

### 7.1 拆的不是两只互不相关的证券

Tsiveriotis–Fernandes（1998，以下简称 TF）将总价值分为未来以现金结算的部分与股票交付相关部分。记

$$
V=E+B,
$$

其中 $B$ 是该合约在最优转股/赎回/回售策略下的 **cash-only component**，不是另买一张普通债券的价格。以下给出该分拆的教学形式；原论文可通过[期刊入口](https://www.pm-research.com/content/iijfixinc/8/2/95)定位，官方数值软件也明确区分 spread 输入与显式违约输入。[MathWorks 文档](https://www.mathworks.com/help/fininst/cbondbyitt.html)。

给现金部分信用折现 spread $s_c$，在事件之间有

$$
B_t+\frac12\sigma^2S^2B_{SS}+(r-q)SB_S-(r+s_c)B=0,
$$

$$
E_t+\frac12\sigma^2S^2E_{SS}+(r-q)SE_S-rE=0.
$$

相加得

$$
\boxed{V_t+\frac12\sigma^2S^2V_{SS}+(r-q)SV_S-rV-s_cB=0.}
$$

“股票部分以 $r$ 折现”不等于“股票不可能下跌”，也不等于违约不影响股票。它是该近似体系的折现分配规则，不是一份已经完全指定好的股价—违约联合过程。

### 7.2 分拆如何跟随行权状态改变？

忽略末期 coupon，若到期 $mS>F$，则 $E=mS,B=0$；若选择本金，则 $E=0,B=F$。提前转股同样将 $B$ 设为零；现金回售把 $E$ 设为零；支付 coupon 则将现金金额加到 $B$。

因此 $B$ 的终值与边界由 $V=E+B$ 的决策共同决定。**不能先独立算一张 straight bond，再声称剩余价值就是 TF 的 $E$。** 两个分量在区间内的方程简单，但通过行权条件耦合。

### 7.3 TF 能回答什么，不能自动回答什么？

它提供了一个容易解释、可实施的现金/股票风险分配方式。问题是仅有 $s_c$ 还没有指定：违约时股票跌多少？债券收回什么？持有人在违约发生时还可否转股？只用股票 delta 能否对冲那个跳跃？AFV 的工作明确强调了这种缺失。[AFV，§4–5](https://cs.uwaterloo.ca/~paforsyt/convert.pdf)。

所以 TF 可以作为基准或对照模型，但把它的 spread sensitivity 直接解释为结构一致的违约风险、或把所有 TF 树的 Greeks 视为同一种对象，都需要额外验证。本文不会把下一节的显式违约树称为 QuantLib 的 TF 引擎。

## 08 · 显式违约：为什么股价漂移也要改？ {#default}

### 8.1 把 default time 的条件分布写清楚

令 $\tau_D$ 为违约时间。恒定风险中性强度 $h$ 意味着，在今天仍未违约的条件下，未来存活概率是

$$
\mathbb Q(\tau_D>t+u\mid \tau_D>t)=e^{-hu}.
$$

因此

$$
\mathbb Q(t<\tau_D\le t+\Delta t\mid\tau_D>t)
=1-\frac{e^{-h(t+\Delta t)}}{e^{-ht}}
=1-e^{-h\Delta t}=h\Delta t+o(\Delta t).
$$

$h$ 是定价测度 $\mathbb Q$ 下的强度；历史违约频率属于实际测度 $\mathbb P$，不能未经风险溢价处理就替换它。仅靠股票与无风险资产通常不足以唯一确定违约风险价格；选定 $h$ 与回收规则也是模型及校准的一部分。

若强度依赖股价，$h=h(t,S_t)$，则需要沿随机路径累计强度；不再能把 $e^{-h(t,S_t)u}$ 当成一般精确存活概率。

### 8.2 违约时股票下跌比例为 $\eta$

假设股票在 default 时从 $S$ 跳到 $(1-\eta)S$，$0\le\eta\le1$。在违约前，让其连续部分漂移为 $\mu_{\mathrm{pre}}$。一个短时间步内的期望总回报约为

$$
\mu_{\mathrm{pre}}\,dt-\eta h\,dt+q\,dt.
$$

风险中性定价要求含股息回报为 $r\,dt$，所以

$$
\boxed{\mu_{\mathrm{pre}}=r-q+\eta h.}
$$

写成带一次违约跳跃的形式，在违约前：

$$
\frac{dS_t}{S_{t-}}=(r-q+\eta h)dt+\sigma dW_t^{\mathbb Q}-\eta\,dH_t,
$$

其中 $H_t=\mathbf1_{\{\tau_D\le t\}}$。当 $\eta<1$ 时，还需要为存活的违约后股票另行指定动态；本文只使用 pre-default 定价方程，并在违约时终止可转债。

::: {.callout}
$+\eta h$ 不是“信用越差，真实世界预期股价涨得越快”。它是在风险中性测度下补偿跳跃损失的存续态漂移。只把 $r$ 换成 risky discount rate、却让股票动态完全不变，会定义出另一套模型。
:::

### 8.3 明确违约时持有人拿到 $G$

本文数值模型选择 $\eta=1$，即股票在违约时归零；债券立即按面值回收 $G=RF$，违约后没有 coupon，也没有剩余转股权。

若允许违约时在回收和残余股票之间选择，可研究 $G=\max\{RF,m(1-\eta)S\}$；**这是一项额外合约/处置假设，不是债券持有人的通用法定权利**。AFV 提供了显式指定股票跳跃及回收的建模框架；本文下面的短步推导是在选定假设下独立展开。[AFV，§4](https://cs.uwaterloo.ca/~paforsyt/convert.pdf)。

### 8.4 逐项展开 default / survival 两种情况

在不立即行权、不付票息的时段：

$$
V=e^{-r\Delta t}\left[(1-h\Delta t)\mathbb E[V(t+\Delta t,S+\Delta S)\mid\text{survival}]+h\Delta t\,G\right]+o(\Delta t).
$$

条件于本步存续时，使用漂移 $r-q+\eta h$。像第 6 节一样展开并整理：

$$
\boxed{V_t+\tfrac12\sigma^2S^2V_{SS}+(r-q+\eta h)SV_S-(r+h)V+hG=0.}
$$

四项分别是：时间/扩散效应、存续态股价漂移、由于折现和违约终止产生的扣减、违约现金流。$h$ 出现两处不叫重复计数：$+\eta h$ 补偿股票跳跃，$-hV+hG$ 处理该债券自身的终止损益。

**自洽性检查**：把一个在违约时变为 $(1-\eta)S$、无股息的股票本身代入，即 $V=S,q=0,G=(1-\eta)S$，方程恰好为零。这是检查跳跃漂移符号的简单方式。

### 8.5 普通债券是这个方程的退化情形

无转股、零 coupon、恒定 $r,h,R$，股票导数消失。用 $\tau=T-t$：

$$
B_{\mathrm{straight}}=F e^{-(r+h)\tau}+RF\int_0^{\tau}h e^{-(r+h)u}\,du
=F e^{-(r+h)\tau}+RF\frac{h}{r+h}(1-e^{-(r+h)\tau}).
$$

第一项是存续至到期的本金，第二项是在不同违约时刻支付回收的现值。离散 coupon 只需另加 $\sum_i c_i e^{-(r+h)(t_i-t)}$。

在本例中，零息直债价值 **85.0091**。常见近似 $s_c\approx h(1-R)$ 此处给出 150bp，但将本金统一按 $r+s_c$ 折现并不精确等于上述 recovery-of-par 模型。按违约前市场价值回收与按面值回收会给出不同方程；必须说明回收基数。

### 8.6 一个可以严格校验数值程序的闭式解

继续取 $\eta=1$、$G=RF$、常数系数、只在到期转股。违约过程与扩散布朗运动独立。在存续条件下，$S_T$ 是漂移 $r+h-q$ 的 lognormal。于是

$$
V=e^{-(r+h)\tau}\mathbb E[\max(F,mS_T)\mid\tau_D>T]+RF\frac{h}{r+h}(1-e^{-(r+h)\tau}).
$$

令

$$
d_1^h=\frac{\log(S/K)+(r+h-q+\sigma^2/2)\tau}{\sigma\sqrt\tau},\qquad d_2^h=d_1^h-\sigma\sqrt\tau,
$$

得到

$$
\boxed{V=mS e^{-q\tau}\Phi(d_1^h)+F e^{-(r+h)\tau}\Phi(-d_2^h)+RF\frac{h}{r+h}(1-e^{-(r+h)\tau}).}
$$

当 $r+h=0$ 时回收积分按连续极限处理；在本文非负 $r,h$ 的交互范围里，它只有二者皆零的退化情形。将 $h=0$ 代回得到第 3 节；令 $m=0$ 则直接使用 straight-bond 公式，不计算 $K=F/m$。

本例可转债闭式价值为 **111.6656**。把 **85.0091** 的 risky bond 加上第 3 节 **23.4914** 的无违约 call，会得到 **108.5005**，不是该显式违约模型的正确答案。差异来自期权也必须服从同一套跳跃—存续动态。这不证明任一模型是真实市场的唯一正确模型，而是说明不能混拼两套假设。

## 09 · 一个可运行的 survival/default 树 {#tree}

### 9.1 股票的两个存续分支，加一个终止分支

取 $\Delta t=T/N$，$u=e^{\sigma\sqrt{\Delta t}}$、$d=1/u$。在本文 jump-to-zero 模型中，本步存活概率是 $a=e^{-h\Delta t}$。在**条件存续**下使用

$$
p=\frac{e^{(r+h-q)\Delta t}-d}{u-d}.
$$

这保证 $a[pu+(1-p)d]=e^{(r-q)\Delta t}$，即无条件股票除息价值增长与风险中性要求一致。实际三个分支的概率是 $ap$、$a(1-p)$、$1-a$；不能把条件概率 $p$ 误当成无条件 up 概率。

必须检查 $0\le p\le1$。强度或漂移大、步长粗时可能不满足；应细化网格或换离散方案，不能直接 clip 成合法概率然后仍声称同一模型。[相关树离散研究：Milanov–Kounchev (2012)](https://arxiv.org/abs/1206.1400)。

### 9.2 把一步内的回收现金流精确积分

对常数系数及固定 $RF$ 回收，节点 continuation 为

$$
V_{\mathrm{cont}}=e^{-(r+h)\Delta t}[pV_{\mathrm{up}}+(1-p)V_{\mathrm{down}}]
+RF\frac{h}{r+h}(1-e^{-(r+h)\Delta t}).
$$

第二项相对于“都在步末回收”的近似更干净。它只对本文的固定强度、固定面值回收规则精确；若 $h,G$ 依赖路径或股价，不能直接照搬。

到期初始化 $\max(F,mS_T)$，必要时加末期 coupon。倒推时先计算 continuation，再按日期应用 call、put、conversion，最后按第 5.4 节约定加本时点 coupon。每一步只保留当前向量，内存 $O(N)$，总运算 $O(N^2)$。

### 9.3 为什么不只给一个价格？

增加 $N$ 不代表所有 Greeks 都单调变好。终值 kink 与行权边界相对格点的位置会改变。应分别查看价格收敛、跨股价光滑性和有限差分 bump 稳定性，尤其不要用“价格到小数点两位不变”替代 gamma 检查。

<div class="figure"><canvas id="convergenceChart" aria-label="不同树步数的欧式价格绝对误差" role="img"></canvas><p class="caption">图 2 · 第 8.6 节闭式解为基准；实线为显式违约，虚线为无违约；横轴为步数，纵轴为每 100 面值的绝对误差。两轴取对数。这里只展示对齐的偶数网格，不据此断言任意网格误差都单调。</p></div>

## 10 · 有限差分、多因子与 Monte Carlo 怎么选？ {#numerics}

### 10.1 一维 PDE：先在 log-stock 上看清结构

令 $x=\log S$，则 $SV_S=V_x$、$S^2V_{SS}=V_{xx}-V_x$。显式违约 PDE 变为

$$
V_t+\tfrac12\sigma^2V_{xx}+(r-q+\eta h-\tfrac12\sigma^2)V_x-(r+h)V+hG=0.
$$

在均匀 $x$ 网格上，用中心差分离散导数。令 $a=\sigma^2/2$、$b=r-q+\eta h-\sigma^2/2$、$k=r+h$，continuation 的三对角系数为

$$
\ell_- =\frac{a}{\Delta x^2}-\frac{b}{2\Delta x},\quad
\ell_0=-\frac{2a}{\Delta x^2}-k,\quad
\ell_+=\frac{a}{\Delta x^2}+\frac{b}{2\Delta x}.
$$

用距到期时间 $\tau$ 向前推进：

$$
(I-\theta\Delta\tau L_h)V^{n+1}
=(I+(1-\theta)\Delta\tau L_h)V^n+\Delta\tau f^{n+\theta},\quad f=hG.
$$

$\theta=1$ 为全隐式，$\theta=1/2$ 为 Crank–Nicolson。终值有 kink 时可先做几个隐式半步再切换 CN；漂移强时中心差分未必保持单调，需要检查 off-diagonal 系数并考虑 upwinding/拟合网格。稳定性、单调性和高阶精度不是同一件事。

每步带上下障碍时，要解离散互补问题，可用 policy iteration、penalty 或 projected iteration。只做一次“无约束求解 + clip”是一种分裂近似，不应无条件宣称与完整障碍求解等价。[AFV，数值附录](https://cs.uwaterloo.ca/~paforsyt/convert.pdf)。

低股价边界应与同模型债券/回售价值一致；高股价边界应由允许转股、股息、call 及剩余时间决定。只在适用时才设 $V=mS$。边界位置要扩张测试，不能让截断误差冒充网格收敛。

### 10.2 随机利率会多出哪些项？

若 $dr=a_r(t,r)dt+b_r(t,r)dW_r$，$d\langle W_S,W_r\rangle=\rho\,dt$，则在原 PDE 上加入

$$
a_rV_r+\tfrac12b_r^2V_{rr}+\rho\sigma S b_rV_{Sr},
$$

并将短端利率作为状态及折现变量。这里的 mixed derivative 来自股票与利率增量的协方差。若违约强度也随机，继续增加强度状态及交叉项；这能更丰富地表达联合风险，但增加参数可识别性和计算成本。

### 10.3 Monte Carlo 并不是模拟 payoff 后简单平均

只有到期 payoff 的产品可以直接折现平均。存在提前行权时，需要每个决策时点的 conditional continuation estimate，再比较 stop/continue。LSMC 可用回归或机器学习拟合该条件期望；发行人 call 又增加对手方的最小化问题。

拟合决策策略与评估该策略应使用独立模拟路径或交叉拟合，防止前视拟合偏差。普通单方 American option 的固定策略可给一个下界；同时存在 issuer call 时，未经博弈分析，不能把同一句下界保证原封不动搬过来。

| 方法 | 本文建议的适用起点 | 主要审计点 |
|:--|:--|:--|
| 闭式解 | 简化欧式、常数系数 | 用作退化和单位测试，不硬套复杂条款 |
| 树 | 少数状态、明确日程 | 转移概率、日期对齐、价格与 Greeks 的格点效应 |
| 有限差分 | 一到两个连续因子、复杂事件 | 边界、障碍求解、单调性、事件前后值 |
| LSMC / 神经条件期望 | 较高维或复杂历史状态 | 样本外策略评估、回归误差、双方决策 |
| CTMC 离散 | 离散状态近似及矩阵递推 | 状态截断、生成矩阵、跨因子耦合 |

## 11 · 合成实验：实际跑了什么，得到什么？ {#experiments}

### 11.1 欧式闭式解 vs 树

以下表格来自本次实际执行的 Python 计算，不是抄录论文，也不是对市场价格的拟合。对两个模型分别使用对应的风险中性树概率。

| 树步数 $N$ | 无违约欧式价格 | 显式违约欧式价格 | 对显式违约闭式的绝对误差 |
|--:|--:|--:|--:|
| 60 | 112.103268 | 111.586047 | 0.079535 |
| 120 | 112.143303 | 111.625773 | 0.039810 |
| 300 | 112.167366 | 111.649649 | 0.015934 |
| 600 | 112.175394 | 111.657614 | 0.007969 |
| 1200 | 112.179409 | 111.661598 | 0.003985 |
| 2400 | 112.181416 | 111.663590 | 0.001992 |
| 闭式 | 112.183424 | 111.665583 | — |


树的 2,400 步显式违约价格与闭式解相差约 **0.00199 / 每 100 面值**。这说明该测试点的价格离散误差较小，不证明一般条款、一般网格或 delta/gamma 都已达同样精度。

### 11.2 逐项打开条款

保持 $h=2.5\%,R=40\%,\eta=1$。American 表示在所有网格时点允许转股；票息为年率 2%，每半年 1 元；call 从 1.5 年起允许、现金额 105、**没有通知期和 soft-call 股价触发**；put 只在 1.5 年时以 100 行使。金额均为除本时点已付票息后的行权额。

| 合约版本 | $N=600$ | $N=1{,}200$ | 两网格价差绝对值 |
|:--|--:|--:|--:|
| 欧式，零 coupon | 111.6576 | 111.6616 | 0.0040 |
| American，零 coupon | 111.8586 | 111.8621 | 0.0036 |
| American，2% coupon | 117.0254 | 117.0294 | 0.0040 |
| American，coupon + call | 112.4146 | 112.3936 | 0.0210 |
| American，coupon + put | 117.5182 | 117.5217 | 0.0034 |
| American，coupon + call + put | 113.2901 | 113.2774 | 0.0127 |


打开回售提高价值，打开发行人赎回降低价值；同时添加二者的净影响不能通过独立 option premium 简单相加，因为最优策略发生改变。含 call 的 600/1,200 步差异约 0.021，明显大于本例欧式的差异，因此只报一个带很多小数的 call 价格会给人虚假精确感。

### 11.3 交互实验：改变一个参数再比较条款

<div class="lab" id="pricingLab">
<div class="controls">
<label>股价 S <output id="spotOut">50</output><input id="spotInput" type="range" min="15" max="100" step="1" value="50"></label>
<label>波动率 σ <output id="volOut">30%</output><input id="volInput" type="range" min="10" max="60" step="1" value="30"></label>
<label>风险中性强度 h <output id="hazOut">2.5%</output><input id="hazInput" type="range" min="0" max="10" step="0.1" value="2.5"></label>
<label><span>合约版本</span><select id="contractInput"><option value="european">欧式零息：闭式 / 树对照</option><option value="american_zero">American 零息</option><option value="american">American + 2% coupon</option><option value="call">American + coupon + call</option><option value="put">American + coupon + put</option><option value="both">American + coupon + call / put</option></select></label>
</div>
<div class="metrics"><div><span>模型价值 · N=300</span><strong id="labPrice">计算中</strong></div><div><span>当前 parity</span><strong id="labParity">100.0000</strong></div><div><span>欧式零息闭式参考</span><strong id="labExact">111.6656</strong></div></div>
<p class="caption" id="labNote">网页直接计算，无网络请求。300 步是交互速度与误差的折中；它不是生产级 pricer。</p>
<canvas id="labChart" aria-label="不同股价下当前合约、欧式基准及parity" role="img"></canvas>
<p class="caption">图 3 · 实线：当前选择的条款；虚线：同模型欧式零息；点线：parity。上方闭式参考仅适用于欧式零息，不是其他版本的解析答案。所有合约均采用本文人工约定，不代表真实证券。</p>
</div>

### 11.4 本次完成的退化与方向检查

| 检查 | 结果 |
|:--|:--|
| 2,400 步欧式树 vs 闭式：绝对误差 < 0.003 | 通过 |
| 关闭转股后，树等于同模型直债闭式（1e−10） | 通过 |
| American 价值不低于欧式 | 通过 |
| 增加 call 不提高持有人价值 | 通过 |
| 增加 put 不降低持有人价值 | 通过 |
| call + put 价值落在对应单边权利界内 | 通过 |
| 可立即转股时价值不低于 parity（多股价点） | 通过 |
| 关闭转股的 coupon bond 匹配现金流现值（1e−10） | 通过 |
| 无违约、无股息、无票息时提前转股溢价为零 | 通过 |
| 非法转移概率被显式拒绝 | 通过 |


没有进行实盘回测，没有读取商业数据，没有验证发行人真实行为，也没有运行 QuantLib 来宣称跨库一致。附录 Python 与浏览器 JavaScript 实现独立运行，并对同一基准价格交叉检查；两者都仍属于同一组教学假设。

## 12 · 真正让问题变难的是条款状态 {#path-dependence}

### 12.1 Soft call 的 20/30 不是今天 $S>1.3K$

令日度触发指示为 $I_j=\mathbf1_{\{S_j>1.3K_j\}}$。最近 30 个交易日里至少 20 天触发的条件是

$$
\sum_{j=n-29}^{n}I_j\ge20.
$$

同一个今天的股价，配上不同历史，赎回资格可以不同。因此 $V(t,S)$ 一般不够，需要历史状态。更微妙的是，**只存当前 count 也通常不够**：明天滚动窗口退出的是哪一个旧 $I_j$，取决于它的顺序。

例如两个窗口当前都计数 20，但一个明天移除 1，另一个移除 0。给定相同明天的新指示，它们的计数会不同。精确状态需要保留足够的队列/bit pattern，或采用经验证的状态压缩。使用 compressed count approximation 时，应把近似误差单独量化。上文 Rubrik 发行文件中的历史触发就是一个具体条款例子，而不是只看 spot 的 barrier。[发行文件](https://www.sec.gov/Archives/edgar/data/1943896/000119312525140729/d920850d8k.htm)。

### 12.2 Call notice period 是一个新状态，不是一条线

若发出 call notice 后需等一段时间，持有人仍可在期间作出选择，股价和信用也继续变化。发行人今天发通知的成本是一个“notice 已启动”子合约的价值，通常不是即时 $\max(C,mS)$。状态需至少包含是否已发通知及剩余通知天数。

### 12.3 Reset 与 call 阈值会相互作用

下调转股价意味着提高 $m=F/K$，看起来对持有人有利；但若 call threshold 同时按新 $K$ 下调，发行人可能更早有资格赎回。两个条款一起变动时，不能用“增加单一投资者权利必定增值”的论证。

注意逻辑边界：**在发行人的机会集合和所有其他条款都不改变的前提下**，扩大持有人的可选动作不应降低价值。若 reset 同时改变 issuer call 的触发，前提已经失效。2026 年 Zhu–Chen–Langrené 预印本将这类路径依赖 reset/call 放进神经条件期望递推，并报告了相关实例；本文只核对其摘要，不把结论推广成所有市场的经验规律，也未复现该文。[arXiv:2605.12189](https://arxiv.org/abs/2605.12189)。

### 12.4 Dilution 与发行人的 capped call 不要混为一谈

如果直接用已交易股票 $S$ 作状态，转股交付价值依具体股数与结算规则计算；如果从 firm value 出发同时定价股本、其他债务和新发行股份，则需要显式处理资本结构与稀释。两种体系不能在中间随意切换。

发行人为管理稀释而另外买入的 capped call，通常是另一份交易。除非持有人所持债券条款明确赋予相应权益，否则不能自动把发行人的 hedge payoff 加到可转债投资者的 payoff 上。Rubrik 的 8-K 将 notes 与 capped-call transactions 分开描述，可用来练习区分交易主体。[同一发行文件的 Capped Call Transactions 部分](https://www.sec.gov/Archives/edgar/data/1943896/000119312525140729/d920850d8k.htm)。

## 13 · 校准：不是找到一条 bond spread 就完成了 {#calibration}

### 13.1 输入应当对应模型中的随机对象

$r(t)$ 从一致的贴现/融资基准构建；股息、股票借贷与公司事件需对应股价动态；信用曲线需匹配发行人、顺位、回收及流动性差异；股权期权数据约束波动率/尾部；完整条款决定停止与结算规则。这里没有理由让一个从其他债券来的 spread 自动吸收所有误差。

尤其是第 8 节的 $\sigma$ 是**违约前扩散波动率**。观察到的 vanilla equity option implied volatility 已反映市场期权价格；若模型另外加入 default jump，再把整条 Black–Scholes implied vol 当作完全相同的扩散参数，可能重复或错误分配尾部风险。

Andersen–Buffum 的校准研究正是强调 debt 与 equity option 市场的联合约束及数值校准偏差；本报告核对了出版方摘要，不声称读到了该站点受限全文或复现其结果。[出版方摘要](https://www.risk.net/journal-of-computational-finance/2160572/calibration-and-implementation-of-convertible-bond-models)。

### 13.2 可以写成一个联合拟合问题

用 $\theta$ 表示波动率、强度函数和必要相关参数。一个示意目标是

$$
\min_\theta\sum_i w_i^{\mathrm{eq}}\bigl(C_i^{\mathrm{model}}(\theta)-C_i^{\mathrm{mkt}}\bigr)^2
+\sum_j w_j^{\mathrm{cr}}\bigl(B_j^{\mathrm{model}}(\theta)-B_j^{\mathrm{mkt}}\bigr)^2
+\lambda\,\operatorname{Penalty}(\theta).
$$

权重可依据报价不确定性与 bid–ask 尺度选择；这只是一个设计框架，不是唯一准则。不要把 illiquid quote 的最后一位小数当作必须拟合的真值。使用可转债本身参与校准时，还应留出其他证券/日期检查模型泛化，而非用同一个价格既拟合又宣称验证。

### 13.3 有股价依赖的 hazard 时，delta 也变了

例如人为指定

$$
h(S)=h_0\left(\frac{S_0}{S}\right)^\beta,\quad \beta\ge0.
$$

这是一个可研究的模型族，不是统计事实。固定 hazard 的 partial delta 与让 hazard 随股价调整的 total delta 不同。若价格写作 $\widetilde V(S,h)$，有

$$
\frac{dV}{dS}=\frac{\partial\widetilde V}{\partial S}+\frac{\partial\widetilde V}{\partial h}\frac{dh}{dS}.
$$

实际 state-dependent PDE 的依赖涉及未来整条路径，不能总把它简化为一个当前点的链式法则；上式用于解释风险口径差别。做 bump-and-revalue 前，必须声明 bump 哪些市场报价、是否重校准，以及保持哪些参数固定。

### 13.4 OAS、implied vol 与模型误差

固定其他假设后，可以反解使模型匹配市场价的 volatility 或额外 spread。它们是**在某模型及条款下的隐含参数**，不是从市场价格唯一识别出的“真实信用”或“真实波动率”。若条款解析错了，隐含参数可能只是在补偿错误。

对于纯现金 bond 可以讨论单一 yield；可转债的未来支付形态取决于股票和决策，用一个 yield 概括全部风险尤其容易误导。模型价格、issuer credit、equity exposure、funding/liquidity residual 最好分开记录。

## 14 · Greeks 与 convertible arbitrage：delta-neutral 不等于无风险 {#risk}

### 14.1 先把单位说清楚

在第 8.6 节的欧式基准，固定 $h,R,r,q,\sigma$，$\Delta=V_S$ 的单位是“股 / 每张债券”。归一化 delta $\Delta/m$ 才接近常用的 0–1 股性刻度。Gamma 是每一股价单位变化引起的 delta 变化；vega 应说明是一单位绝对波动率还是 1 vol point。

在这个特定闭式模型中，直接微分可得

$$
\Delta=m e^{-qT}\Phi(d_1^h),\qquad \Gamma=\frac{m e^{-qT}\phi(d_1^h)}{S\sigma\sqrt T},
$$

其中 $\phi$ 为标准正态密度。本次另用闭式公式做 central bump，结果如下，均为 $S=50$ 的合成例子。

| 风险量 | 数值 | 计算口径 |
|:--|--:|:--|
| Delta | 1.393848 | 股 / 每张债券；股价 bump ±0.05 |
| Gamma | 0.025228 | 每一股价单位；bump ±0.05 |
| Vega：+1 vol point | 0.567583 | 价格变化；σ 中心 bump ±0.01 |
| Rate：+1bp | -0.012170 | 价格变化；r 中心 bump ±0.0001 |
| Hazard：+1bp | -0.001267 | 价格变化；h 中心 bump ±0.0001 |


Rate 1bp 是 bump $r$、保持 $h$ 与扩散波动率不变的价格变化；hazard 1bp 是 bump $h$，**不等于 CDS spread 1bp**。后者还要通过信用曲线的校准映射。通常债性更强时对 credit 更敏感，但在含固定回收和风险中性跳跃补偿的模型里，不能不加条件地断言所有状态下 hazard sensitivity 都严格为负。

### 14.2 股票 hedge 为什么消不掉违约跳跃？

持有一张可转债，并做空 $\Delta$ 股。违约前组合为 $V-\Delta S$。违约发生时，债券变成 $G$，股票变成 $(1-\eta)S$，瞬时组合损益为

$$
\boxed{\Delta\Pi_{\mathrm{default}}=G-V+\Delta\eta S.}
$$

令 $\Delta=V_S$ 可以消除连续小波动的一阶暴露，却不保证上式为零。本例该跳跃损益约为 **−1.9732 / 每 100 面值**；这只是特定参数点下的瞬时模型损益，不含融资、借券、执行或违约结算摩擦。

因此 “long convertible + short stock” 不等于无风险套利。即使 delta 接近零，仍有 gamma、volatility、credit/recovery、funding、股票借贷与召回、流动性和跳跃风险。把交易称为 convertible arbitrage，并不意味着所有收益来自可锁定的 no-arbitrage discrepancy。

### 14.3 一个有用但有前提的 gamma–theta 近似

在单一连续扩散、其他参数不变、已扣除一致 carry 且 hedge 足够频繁的理想模型里，delta-hedged option-like 部分有近似波动率损益

$$
d\Pi\approx\tfrac12\Gamma S^2\bigl(\sigma_{\mathrm{realized}}^2-\sigma_{\mathrm{model}}^2\bigr)dt.
$$

对可转债还要加上信用、票息、行权、融资与 default jump 项；存在 call 时 gamma 的形状也可能被改变。不能从这个局部近似直接推导“实际波动率高于隐含波动率就必赚”。

::: {.takeaway}
**本节模板**：先声明价格单位和重校准规则，再给 delta/gamma/vega/credit sensitivity；最后单独计算有限幅度跳跃损益。连续风险对冲与违约风险对冲不是同一个问题。
:::

## 15 · 阅读地图与实现顺序 {#reading}

### 15.1 这些文献各自解决哪层问题？

| 来源 | 适合带着什么问题读 | 本报告核验边界 |
|:--|:--|:--|
| [Tsiveriotis–Fernandes, 1998](https://www.pm-research.com/content/iijfixinc/8/2/95) | 为什么区分现金与股票交付部分？ | 核对书目信息；数学框架与公开 AFV 讨论、官方软件文档交叉核对；原付费全文未获取 |
| [Ayache–Forsyth–Vetzal, 2003](https://cs.uwaterloo.ca/~paforsyt/convert.pdf) | 违约时股票、回收与 hedge 怎样一致？ | 阅读作者公开论文并核对方程所在页面；本文未复现其全部数值实验 |
| [Andersen–Buffum, 2004](https://www.risk.net/journal-of-computational-finance/2160572/calibration-and-implementation-of-convertible-bond-models) | 股权期权与债务怎样联合校准？ | 已核对出版方摘要；未把受限正文当已阅读 |
| [Milanov–Kounchev, 2012](https://arxiv.org/abs/1206.1400) | 股权—信用树的条件转移概率为何重要？ | 核对公开摘要；本文的常数强度树是自建教学实现，不声称逐行复现 |
| [Vachon–Mackay, 2024 / revised 2025](https://arxiv.org/abs/2403.06303) | 能否用 CTMC 统一债券、call/put 与可转债数值方法？ | 核对摘要与版本；未复现 |
| [Zhu–Chen–Langrené, 2026 preprint](https://arxiv.org/abs/2605.12189) | 路径依赖 reset 与 call 怎样用神经条件期望处理？ | 核对摘要；未做全文精读与实证复现；不将个案结论一般化 |
| [QuantLib 官方源码](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/bond/binomialconvertibleengine.hpp) | 实际引擎接收了什么，又把哪些 term structure 简化了？ | 查阅 2026-09-05 可见源码；未运行 QuantLib |

在已核对的 QuantLib `BinomialConvertibleEngine` 实现中，输入最终用于构造平坦利率、股息和常数波动率的树；engine 名称本身也指明 TF。**给它一个完整曲面对象，并不意味着该树保留了全部 smile 与期限结构。** 应阅读具体版本实现，而不只读构造函数参数名称。[对应官方文件](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/bond/binomialconvertibleengine.hpp)。

### 15.2 实现时先攻哪几步？

先让无信用、零息、欧式树通过 Black–Scholes 对照；再加入 jump-to-zero 与 recovery-of-par，通过第 8.6 节闭式解；随后逐项加入离散 coupon、American conversion、指定日 put 与无通知期 call。最后才增加真实日历、notice、rolling trigger、结算窗口和联合校准。

每增加一层，都保留上一层的退化测试。尤其先写清 contract/event engine，再加复杂 stochastic dynamics。把错误 term sheet 交给更强的神经网络，只会更快地算出另一张证券的价格。

**模型开发的验收问题**：是否还原了所有现金流？是否有可验证的简单极限？是否把风险中性与实际概率分开？是否对价格和 Greeks 分别检验？是否能解释风险来自模型、条款、校准还是数值离散？

## 附录 A · 最小 Python 基准与可复算数字 {#python}

下面代码只实现第 8–9 节的零息、欧式、常数系数、jump-to-zero、立即面值回收模型。这样保留一个不掩盖假设的最小基准。第 11 节扩展条款结果与参数完整保存在同目录 `results.json`；网页交互代码位于 `interactive.js`，由渲染程序内联进 HTML。

```python
from math import erf, exp, expm1, isfinite, log, sqrt
import numpy as np


def european_convertible(
    S: float = 50.0, F: float = 100.0, m: float = 2.0,
    T: float = 3.0, r: float = 0.04, q: float = 0.01,
    sigma: float = 0.30, h: float = 0.025, R: float = 0.40,
    steps: int = 1200,
) -> tuple[float, float]:
    """Return (tree_price, analytic_price), per F face amount.

    European conversion only; no coupons/call/put; stock jumps to
    zero at default; immediate recovery R*F; constant coefficients.
    """
    inputs = (S, F, m, T, r, q, sigma, h, R)
    if not all(isfinite(x) for x in inputs):
        raise ValueError("Inputs must be finite")
    if min(S, F, T, sigma) <= 0 or m < 0 or h < 0 or not 0 <= R <= 1:
        raise ValueError("Invalid model inputs")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    p = (exp((r + h - q) * dt) - d) / (u - d)
    if not 0 <= p <= 1:
        raise ValueError("Invalid transition probability; refine time grid")
    a = r + h

    def recovery_pv(t: float) -> float:
        # Continuous limit avoids cancellation or division by zero.
        integral = -expm1(-a * t) / a if abs(a) > 1e-12 else t
        return R * F * h * integral

    stock = S * np.exp((2 * np.arange(steps + 1) - steps) * log(u))
    value = np.maximum(F, m * stock)
    discount = exp(-a * dt)
    recovery = recovery_pv(dt)
    for _ in range(steps):
        value = discount * (p * value[1:] + (1 - p) * value[:-1]) + recovery
    if m == 0:
        exact = F * exp(-a * T) + recovery_pv(T)
    else:
        normal = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
        d1 = (log(S * m / F) + (a - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        exact = m * S * exp(-q * T) * normal(d1)
        exact += F * exp(-a * T) * normal(-d2) + recovery_pv(T)
    return float(value[0]), exact


if __name__ == "__main__":
    tree, exact = european_convertible()
    print(f"tree={tree:.8f}, exact={exact:.8f}, error={tree-exact:.8f}")
    assert abs(european_convertible(steps=2400)[0] - exact) < 0.003
    bond_tree, bond_exact = european_convertible(m=0)
    assert abs(bond_tree - bond_exact) < 1e-9
```

运行基准输出：`tree=111.66159789, exact=111.66558266, error=-0.00398477`。数值积分/树结果可能随运行环境在最后几位有浮点差异。代码刻意不接受任意真实 term sheet，避免用一个看似通用的函数隐藏尚未实现的条款。

## 附录 B · 六个最容易踩的坑 {#pitfalls}

| 看起来合理的说法 | 更精确的版本 |
|:--|:--|
| 可转债就是债券加普通 call | 在限制条件下可以精确成立；提前结束与信用耦合一般破坏静态拼接 |
| 股价高于转股价就该转换 | 应比较 parity 与继续持有的全部权利价值 |
| callable convertible 不能高于 call price | 持有人可能在被 call 时转股；无通知期简化上界是 $\max(C,mS)$ |
| 有 bond floor 就不会亏本金 | floor 是有信用与条款假设的模型价值，不是保本承诺 |
| 默认 hazard 只进入折现 | 有股票跳跃时，风险中性存续漂移也要补偿跳跃损失 |
| delta-neutral 就没有 default risk | 连续 delta 对冲不消除 $G-V+\Delta\eta S$ 的跳跃损益 |

**最后一句**：可转债定价可以理解成“一个存续状态的条件期望递推 + 一组精确的合约事件 + 一个明确的违约处置规则”。先把这三件事对齐，再讨论更复杂的模型。

---

资料核对与计算日期：2026-09-05。本文为公开资料基础上的独立教学推导；不含商业行情、公司内部数据或私有实现。来源链接标在对应论述及阅读表内。数学使用原生 MathML，图表和计算脚本内联，可在支持 MathML 的现代浏览器中离线阅读；离线时外部参考链接不可访问。
