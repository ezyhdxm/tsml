## BS9 · 合成实验：把三种误差真正分开 {#bs-experiments}

### BS9.1 实验协议与单位

以下数字来自本次实际执行的 `hedging_experiments.py`，完整结果存于 `hedging_results.json`。不是市场样本、论文结果转录或 Heston 校准复现。

共同参数为 $S_0=K=100,T=1,r=3\%,q=0$。卖出一份现金结算的欧式 call；股票可以买卖任意数量、按相同利率融资；不含信用、跳跃、借券限制、固定费用或冲击。每个场景生成 **32,768 条独立路径**，seed 为 **20260905**。模拟使用明确指定的 $\mathbb Q$，即股票总回报漂移为 $r$，不将它当作历史概率模型。

使用 1,024 个最细时间区间的精确 GBM 转移；16、64、256、1,024 个区间的 hedge 都读取同一批嵌套路径。因每段 volatility 已知，股票采样没有 Euler 近似误差；这不意味着 hedge 没有离散误差。$N$ 是一年内区间数，不冒充某个真实市场的交易日日历。

所有 P&L 和费用都折现至 $t=0$，单位为每份 payoff 的货币金额。交易费按股票 half-spread **5bp**、full spread 10bp 计算，包含开仓、区间调仓及现金结算后的平仓。即使策略亏损，也不能额外注资修饰结果。

### BS9.2 常数 volatility：风险与成本的方向相反

真实和 hedge 模型都使用 $\sigma=25\%$。初始 BS premium 为 **11.348477**。

| 年内区间数 $N$ | 无费用 hedge P&L 标准差 | 平均全部费用现值 | 扣费用后平均 P&L |
|--:|--:|--:|--:|
| 16 | 2.057089 | 0.119009 | −0.117553 |
| 64 | 1.052123 | 0.182187 | −0.182438 |
| 256 | 0.533287 | 0.306358 | −0.309313 |
| 1024 | 0.264403 | 0.553267 | −0.553994 |

无费用平均 P&L 分别为 0.001456、−0.000250、−0.002955、−0.000727，对应均值 Monte Carlo 标准误约 0.011364、0.005812、0.002946、0.001461。它们与该 $\mathbb Q$ 下均值为零的理论关系一致。**表中的大幅标准差不是“均值估计不够准”，而是单条交易路径真实存在的风险。**

以 $\log N$ 回归 $\log\mathrm{SD}$，斜率为 **−0.4930**，接近理论的 −1/2；仅对中途调仓费用、不含开平仓，$\log$–$\log$ 斜率为 **0.5091**，接近 +1/2。它们只是这组参数和四个网格的实算尺度检查，不是实盘定律或普遍收敛证明。

如果人为选择“平均费用 + 0.5 × 无费用误差标准差”为比较分数，四个网格依次为 **1.147553、0.708249、0.573001、0.685468**；在这四个固定时间策略中，256 最低。更换风险权重会改变结果。这里没有优化连续的交易频率，更没有证明该策略胜过 no-trade band 或其他自适应 hedge。

<div class="lab" id="bsHedgeLab">
<div class="controls">
<label>股票 half-spread（bp）<output id="bsSpreadOut">5</output><input id="bsSpread" type="range" min="0" max="20" step="1" value="5"></label>
<label>标准差权重 α <output id="bsAlphaOut">0.5</output><input id="bsAlpha" type="range" min="0" max="2" step="0.1" value="0.5"></label>
<label><span>真实 / hedge 动态</span><select id="bsMode"><option value="constant">正确的常数 volatility</option><option value="revelation">正确的 volatility 揭示模型</option></select></label>
</div>
<div class="metrics"><div><span>四个网格中分数最低的 N</span><strong id="bsBestGrid">256</strong></div><div><span>平均费用 + α × gross SD</span><strong id="bsBestScore">0.5730</strong></div></div>
<canvas id="bsCostChart" role="img" aria-label="对冲频率增加时误差标准差、交易费用和风险成本分数的变化"></canvas>
<div class="legend">实线：无费用 P&L 标准差　虚线：费用现值　点线：费用 + α × 标准差</div>
<p class="caption">图 BS1 · 同一批 Monte Carlo 结果，网页不重新抽样。改变 spread 只对固定策略的成交费用线性缩放；策略本身没有随费用重新优化。α 是人为风险权重，分数不是期权报价。</p>
</div>

### BS9.3 一个可精确核算的 stochastic-volatility 反例

为了不把 Heston 数值误差与不可复制风险混在一起，另设一个**波动率消息揭示模型**：前半年 volatility 为 25%；在 $t_*=0.5$ 宣布后半年 volatility 为 10% 或 40%，在选定 $\mathbb Q$ 下概率各为 1/2。消息独立于股票 Brownian motion，宣布时股票价格连续，但期权价格因剩余方差改变而跳跃。

宣布前，给定当前股价，期权价值是两种未来场景 BS 价格的概率混合：

$$
C_-(t,S)=\tfrac12 C_{\mathrm{BS}}(S,K,T-t;r,0,\sigma_{\mathrm{eff},L})
+\tfrac12 C_{\mathrm{BS}}(S,K,T-t;r,0,\sigma_{\mathrm{eff},H}),
$$

其中

$$
\sigma_{\mathrm{eff},j}^2(T-t)=0.25^2(t_*-t)+\sigma_j^2(T-t_*),\quad j=L,H.
$$

宣布后，使用已知的剩余 volatility 计算 BS 价格及 delta。初始模型价格为 **11.810661**；hedger 使用这个正确价格和正确的条件 delta，宣布以前不能使用已经在模拟器内部抽好的未来 regime。

令 $C_L,C_H$ 表示宣布时两种可能的剩余 call 价值。宣布前 $C_-=(C_L+C_H)/2$，宣布后 $C_+$ 是其中一个。股票及现金财富在宣布瞬间都不会因消息跳跃，所以即使连续 hedge，也有卖方误差

$$
\boxed{E_{\infty,d}=e^{-rt_*}(C_--C_+).}
$$

宣布后该差额随现金利率累积，折现回 0 恰好得到上式。它的条件均值为零，但条件方差为

$$
\operatorname{Var}(E_{\infty,d}\mid S_{t_*})
=\frac14e^{-2rt_*}(C_H-C_L)^2.
$$

这是一个**无需 Heston 求解器就能证明的风险下限例子**。没有股票价格跳跃不代表所有衍生品状态风险都可被股票复制；这里缺少的是交易该独立消息风险的工具。

| 网格 $N$ | 常数 vol：gross SD | 消息揭示：gross SD | 与连续极限误差的 RMSE |
|--:|--:|--:|--:|
| 16 | 2.057089 | 3.882241 | 2.478031 |
| 64 | 1.052123 | 3.244114 | 1.268688 |
| 256 | 0.533287 | 3.045905 | 0.639029 |
| 1024 | 0.264403 | 2.993660 | 0.316637 |

对 $S_{t_*}$ 的已知 lognormal 分布做 160 点 Gauss–Hermite 积分，连续极限误差的标准差为 **2.982801**；同批 Monte Carlo 的直接极限样本标准差为 **2.976344**。1,024 步 gross error 与该样本极限的相关性为 **0.99439**。

因此离散策略确实收敛到了正确的**非零误差极限**，并不是模型还没收敛就草率宣布有风险。常数 vol 的误差趋零，而这个模型只会把离散部分越做越小，独立消息风险留下来。

<div class="figure"><canvas id="bsVolRiskChart" role="img" aria-label="常数波动率与波动率消息模型的对冲误差标准差及非零极限"></canvas><div class="legend">实线：常数 volatility　虚线：消息揭示模型　点线：揭示模型的连续 hedge 极限</div><p class="caption">图 BS2 · 全部是不含交易费的误差，排除成本造成的混淆。这个模型具有波动率消息跳跃，不是 Heston 的连续方差扩散；用于严格隔离未被股票覆盖的风险。</p></div>

这里将高 volatility 的 $\mathbb Q$ 概率设成 1/2，是一项定价假设。只交易股票和现金时，股票无套利本身不能决定该概率；在适当等价性下，多种概率都可使股票折现收益保持 martingale。可以用期权价格或风险偏好确定它，但不能用“股票 drift 已经改成 $r$”跳过这一步。

### BS9.4 什么已经验证，什么尚未实现？

已实际计算：BS 闭式基准、确定时间变化的累计方差例子、四种固定频率的自融资账本、开仓/调仓/平仓费用、P&L 均值与标准差、95% loss VaR/ES、消息揭示的正确条件 delta、非零连续误差极限及解析积分对照。脚本包含九项检查；这些检查验证声明模型下的实现和数值行为，不是模型真实有效性的证据。

未实现：Heston Fourier/PDE 数值引擎、local-vol 曲面校准、LSV、Leland 策略回测、no-trade band 最优化、效用无差异报价、deep hedging、带费用的真实可转债对冲。它们在文中有推导或建模说明，不把理论章节伪装成已跑实验。原报告的可转债树及十项检查保持独立，不能借它们替这些新模型背书。

复现命令：

```bash
python -m pip install numpy scipy
python hedging_experiments.py --paths 32768 --seed 20260905
python render.py
```

更改实验参数后，图表使用新 JSON，但本节静态数字需要同步核对；它们不应自动被当成新参数结果。不同 NumPy/SciPy 环境的随机实现和末位浮点数可能不同。本次计算版本保存在 JSON；采样均值标准误、数值离散误差、模型错误和真实交易风险应分别报告。

### BS9.5 原始文献怎么接着读？

| 文献 | 对应本专题的问题 | 阅读/实现边界 |
|:--|:--|:--|
| [Black–Scholes (1973)](https://www.journals.uchicago.edu/doi/10.1086/260062) | 为什么复制价格不需要股票预期回报？ | 核对原论文；正文以自融资账户和带股息记号独立推导 |
| [Hayashi–Mykland (2005)](https://galton.uchicago.edu/~mykland/paperlinks/hedgeerrors.pdf) | BS3 的离散误差尺度与极限分布 | 核对第 3 节；本报告给局部 Taylor 推导，不复制其完整弱收敛证明 |
| [Leland (1985)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1985.tb02383.x) | 交易频率与成本如何进入复制近似？ | 核对出版方摘要；本报告明确 half-spread 后重推成本系数；未回测其策略 |
| [Whalley–Wilmott (1997)](https://users.ox.ac.uk/~ofrcinfo/file_links/mf_papers/1999mf08.pdf) | 为什么已有库存和 no-trade band 是状态？ | 核对作者公开版本的控制结构与小费用尺度；未解控制问题 |
| [El Karoui et al. (1998)](https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-9965.00047) | 波动率错配与 hedge 的稳健性 | 核对出版方摘要；正文路径误差由收益方程独立推导 |
| [Dupire，1994 原文的重刊入口](https://www.risk.net/derivatives/equity-derivatives/1500211/pricing-with-a-smile) | vanilla marginals 怎样联系 local vol？ | 核对出版方入口；带股息公式由密度积分推导，未获取受限全文 |
| [Heston (1993)](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf) | 方差风险价格、PDE 和仿射特征函数 | 核对原论文方程；正文重推，未运行其数值引擎 |
| [Brunick–Shreve (2013)](https://arxiv.org/abs/1011.0111) | BS7.6 中条件方差与边际分布匹配 | 核对作者摘要；完整 mimicking 定理有额外条件，未做 LSV 校准 |
| [Denis–Martini (2006)](https://arxiv.org/abs/math/0607111) | 区分随机 vol 概率模型与模型不确定性超复制 | 核对作者摘要；作为 BS7.6 稳健方差界的进一步阅读，未复现 |
| [Bühler et al., Deep Hedging](https://arxiv.org/abs/1802.03042) | 直接优化有摩擦的终端风险 | 核对作者摘要；没有训练模型或宣称算法胜过本报告基准 |

::: {.takeaway}
**新增专题的结论**：Black–Scholes 先给出一个有条件的精确复制基准。离散交易让复制变成随机误差；bid–ask 让无限频繁调仓代价高昂；独立 stochastic volatility 让股票 hedge 留下无法消掉的风险。确定时间变化和一因子 local volatility 则说明，“volatility 可变”本身不是不完全性的充分条件。修正动态模型、选择对冲策略和形成买卖报价，是三个关联但不同的任务。
:::

[回到原报告第 5 节：可转债的提前转股、回售与赎回](#exercise)。
