## BS5 · 波动率随时间变化：不一定需要放弃复制 {#bs-time-vol}

**“波动率不是常数”至少有三种含义**：今天已经知道的时间函数 $\sigma(t)$；当前股价决定的局部函数 $\sigma_{\mathrm{loc}}(t,S_t)$；另有独立随机冲击的状态变量 $v_t$。三者对定价、校准和市场完整性的影响不一样。

### BS5.1 已知的时间变化只改变累计方差

先保持 $r,q$ 为常数，仅令 $\sigma(t)$ 为确定函数。在 $\mathbb Q$ 下

$$
\frac{dS_t}{S_t}=(r-q)dt+\sigma(t)dW_t^{\mathbb Q}.
$$

积分得

$$
\log\frac{S_T}{S_t}=(r-q)(T-t)-\frac12\int_t^T\sigma^2(u)du+\int_t^T\sigma(u)dW_u^{\mathbb Q}.
$$

因为积分核是确定的，最后一个随机积分是均值 0、方差

$$
w(t,T)=\int_t^T\sigma^2(u)du
$$

的高斯变量。重做 BS2 的完成平方，只要令

$$
d_1=\frac{\log(S/K)+(r-q)\tau+w(t,T)/2}{\sqrt{w(t,T)}},\qquad d_2=d_1-\sqrt{w(t,T)},
$$

就仍有

$$
C=S e^{-q\tau}\Phi(d_1)-K e^{-r\tau}\Phi(d_2).
$$

因此有效波动率是

$$
\boxed{\sigma_{\mathrm{eff}}^2(t,T)=\frac{1}{T-t}\int_t^T\sigma^2(u)du,}
$$

即**方差的时间平均再开方**，而不是 volatility 的算术平均。连续无摩擦下仍只有一条 Brownian 风险，正确的 delta 可以复制；模型改变了，不代表复制论证自动失效。

### BS5.2 相同初始价格，不等于中途价格和 hedge 相同

设 $S_0=K=100,T=1,r=3\%,q=0$。前半年波动率 15%，后半年 40%，则

$$
w(0,1)=0.5(0.15)^2+0.5(0.40)^2=0.09125,
\qquad\sigma_{\mathrm{eff}}=30.2076\%.
$$

实际计算的初始 call 价格为 **13.363583**。错误地使用算术平均 $27.5\%$，价格只有 **12.316196**。

如果把两个半年的波动率顺序颠倒，初始欧式 call 价格不变，因为累计方差相同。但在 $t=0.5$、条件于相同 $S=100$ 时，剩余半年 volatility 为 40% 的价格是 **11.922566**，为 15% 的价格是 **4.984228**。未来 hedge 也不同。

这说明：初始终值分布足以决定该欧式 payoff 的价格，却不足以替代完整的中途动态。对于提前转股、call notice 或路径触发的可转债，“累计方差相同”更不保证价格相同。

只随时间变化的确定波动率允许 IV 随期限变化，但同一到期日内，标准欧式期权的 BS implied vol 不随执行价变化。所以它可以解释 term structure，不能单独解释 strike smile。

## BS6 · Local volatility：波动率路径是随机的，但风险源仍可只有一个 {#bs-local-vol}

### BS6.1 不要把“函数是确定的”误读成“实现路径是确定的”

局部波动率模型写作

$$
dS_t=(r-q)S_tdt+\sigma_{\mathrm{loc}}(t,S_t)S_t\,dW_t^{\mathbb Q}.
$$

函数 $\sigma_{\mathrm{loc}}(t,s)$ 在校准后固定，实际的 $\sigma_{\mathrm{loc}}(t,S_t)$ 却因股价随机而随机。重要的是没有另外独立的 Brownian 波动率因子。在一因子过滤、非退化扩散、连续无摩擦交易及适当正则条件下，仍然能以股票和现金复制欧式合约。

直接重做 BS1：随机项仍是 $C_S\sigma_{\mathrm{loc}}S\,dW$，令 $\theta=C_S$ 即可消除。PDE 变成

$$
\boxed{C_t+(r-q)SC_S+\frac12\sigma_{\mathrm{loc}}^2(t,S)S^2C_{SS}-rC=0.}
$$

因此不能说“只要 volatility 随机，Black–Scholes 的复制思路就完全不能用了”。真正需要问的是：随机性是否引入了**现有交易资产无法覆盖的独立风险**。

### BS6.2 Dupire 公式从 vanilla 价格怎样推出来？

固定今天为 0。以下 $C(K,T)$ 表示今天观察到的、执行价为 $K$、到期为 $T$ 的欧式 call 价格；它不是前面的 $C(t,S)$ 函数。$p(s,T)$ 是局部波动率扩散的风险中性股价密度。假设足够光滑、尾部项可消失，并且没有股票跳跃，则

$$
C(K,T)=e^{-rT}\int_K^\infty(s-K)p(s,T)\,ds.
$$

先对执行价做普通微分：

$$
C_K=-e^{-rT}\int_K^\infty p(s,T)ds,\qquad
\boxed{C_{KK}=e^{-rT}p(K,T).}
$$

再对到期时间微分。由 Itô 对任意光滑测试函数求期望，并对空间变量分部积分，可得到密度的 forward equation：

$$
p_T=-\partial_s[(r-q)sp]+\frac12\partial_{ss}[\sigma_{\mathrm{loc}}^2(T,s)s^2p].
$$

把它放进 $C_T$ 的积分。漂移项分部积分一次，扩散项分部积分两次，得到

$$
C_T=-rC+(r-q)e^{-rT}\int_K^\infty sp(s,T)ds
+\frac12\sigma_{\mathrm{loc}}^2(T,K)K^2 e^{-rT}p(K,T).
$$

而

$$
e^{-rT}\int_K^\infty sp(s,T)ds=C-KC_K.
$$

代回并整理：

$$
C_T=-qC-(r-q)KC_K+\frac12\sigma_{\mathrm{loc}}^2(T,K)K^2C_{KK}.
$$

所以

$$
\boxed{\sigma_{\mathrm{loc}}^2(T,K)=
\frac{2\left[C_T+qC+(r-q)KC_K\right]}{K^2C_{KK}}.}
$$

分母要求 $C_{KK}>0$；密度为零或曲面不光滑处不能机械地相除。对确定但非常数的 $r(T),q(T)$，使用一致贴现后可得到对应的时变系数版本。引入离散现金股息、股票跳跃或违约后，forward equation 有额外项，上述纯扩散公式不应原样套用。

[Dupire, Pricing with a Smile](https://www.risk.net/derivatives/equity-derivatives/1500211/pricing-with-a-smile) 的出版方页面说明该方法来自 1994 年原文，网页为后来的重刊入口。本报告核对该入口与方法背景；这里给出的带 $r,q$ 公式由密度积分独立推导，并未声称获取了该站受限全文。

### BS6.3 为什么有了公式还不能直接对行情做差分？

$C_{KK}$ 是二阶导数，会放大稀疏、过时和有 bid–ask 噪声的报价误差。于是“先插值一条随意的 IV 曲面，再用差分求 local vol”可能制造负方差或尖峰。应先构造与贴现、股息及静态约束一致的平滑价格曲面，再检查密度、方差正性、网格和尾部外推。

这与交易摩擦并不矛盾：可成交套利要用 bid/ask 检验；但如果你选择用一个风险中性概率模型作为**内部一致的估值基准**，就不能让同一模型在相同到期日隐含负的概率密度。内部模型约束与市场上是否能净赚，是两层不同的问题。

对有报价区间的校准，一种简单准则是只惩罚模型价离开区间的部分：

$$
\sum_j w_j\left[\max(C_j^{\mathrm{bid}}-C_j^{\mathrm{model}},0)^2
+\max(C_j^{\mathrm{model}}-C_j^{\mathrm{ask}},0)^2\right]
+\lambda\,\mathrm{SmoothnessPenalty}.
$$

这是建模选择，不是唯一统计准则。它避免要求模型精确穿过每个不确定的 mid，也不能保证任意输入报价集合都存在一个同时满足的平滑扩散模型。

### BS6.4 Implied vol surface 不是另一个随手替换的扩散系数

$\sigma_{\mathrm{imp}}(K,T)$ 定义为使 BS 公式等于一个市场价格的反解。$\sigma_{\mathrm{loc}}(t,S)$ 则定义股票的瞬时二次变差。两者的自变量、含义和用途不同，一般并不相等。

所以正确流程是：**将报价组织为价格/IV 曲面 → 选择并校准一个动态模型 → 用该动态定价和 hedge**。给每个 $(K,T)$ 填一个 IV 是合理的报价表示；把每份期权各自的 IV 都当作同一股票在未来的真实常数 volatility，就不是一个统一的联合动态了。

即使 local vol 完美拟合今天全部欧式价格，也只是匹配一系列终值边际分布。它没有据此唯一识别未来 conditional smile、跨时间联合分布或触发概率。对未来 forward-start、平均结算及提前行权相关的可转债风险，还必须评估动态模型本身。
