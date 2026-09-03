# 3B. 经典方法怎么做？为什么要 VAE，收益到底在哪里？ {#classical-methods}

<div class="callout">
<strong>先给一个不替 VAE 辩护的结论：</strong>如果任务只是把今天已经比较密集、干净的 vanilla quotes 插成平滑曲面，没有理由默认从 VAE 开始。经典方法提供更小、更透明的建模问题；VAE 需要证明的，是<strong>历史统计结构在稀疏补全、非线性表示或条件分布上带来的样本外增量</strong>，而不是“也能画出一张曲面”。
</div>

## 3B.1 经典方法不是一个 baseline：先分三条路线

| 路线 | 主要输入 | 输出 | 代表方法 |
|---|---|---|---|
| 当日横截面构造 | 今天不同 strike / maturity 的 quotes | 连续可查询的 price / IV surface | 分段插值、受约束 spline、SVI/SSVI |
| 动态模型校准 | 今天 quotes，外加动态假设与参数约束 | 一个风险中性的 underlying path model | local vol、Heston、SABR、LSV / jumps |
| 历史统计建模 | 多天或多资产的 surface panels | factors、补全器、预测分布或情景分布 | PCA / functional PCA、因子状态空间、GP、参数时间序列 |

它们可以组成同一条 pipeline：先清洗价格、拟合 SSVI，再对 SSVI 参数或其 residual 做时间序列，必要时校准动态模型。VAE 通常是第三条路线的非线性替代或增强，也可以替代第一条中的补全映射；它并不自动完成第二条。

## 3B.2 从零开始：报价 → 价格一致性 → 网格 → 插值

假设今天拿到一组 $(K_i,T_i,C_i)$，其中 $C_i$ 为 mid，同时保留 bid/ask。一个完整的经典流程首先不是“fit 神经网络”，而是确认：合约是 European 还是 American、carry 与 forward 是否一致、put/call 是否可用 parity 归一、不同报价是否同步、IV inversion 是否接近边界。

再决定拟合空间。若用每个点的 IV，其 raw error 容易夸大低 vega 区域的价格噪声。由一阶 Taylor 展开，

$$
\Delta C_i\approx\mathrm{Vega}_i\,\Delta\sigma_i.
$$

因此 price MSE 近似对应 $\sum_i\mathrm{Vega}_i^2(\Delta\sigma_i)^2$，而等权 IV MSE 近似对应以 $1/\mathrm{Vega}_i^2$ 加权的 price errors。没有哪种永远正确：你应先决定模型到底在优化报价误差、波动率状态误差还是交易风险。Bid/ask-normalized price error 通常是值得同时报告的另一把尺子。

### 最简单的 maturity interpolation

在相同 log-forward-moneyness $k$，两个期限 $T_1<T_2$ 已有 total variance。对中间 $T$ 取

$$
u=\frac{T-T_1}{T_2-T_1},
\qquad
w(k,T)=(1-u)w(k,T_1)+u\,w(k,T_2),
$$

然后令 $\sigma(k,T)=\sqrt{w(k,T)/T}$。例如 3 个月 IV 为 30%、一年 IV 为 20%，则两端 total variance 为 0.0225 与 0.04。6 个月取 $u=1/3$，得到

$$
w(k,0.5)=0.0283333,
\qquad
\sigma(k,0.5)=23.8048\%.
$$

直接线性插 IV 会给出 26.6667%，是另一个插值模型。选择 total variance 是为了与累计不确定性、价格和 calendar 结构更好对齐，不代表它在所有情形都统计最优。

<strong>只插 total variance 不能自动保证 butterfly 无套利。</strong>即使各固定 $k$ 上 $w$ 随期限递增，跨 strike 的曲率仍需检查。并且 $w(k,T)$ 的 calendar 单调关系需要相应的 deterministic carry / proportional-dividend 设定与正确归一化；不应机械地要求有离散分红的未归一化 call price 对到期单调。相关标准条件见 [Gatheral–Jacquier](https://arxiv.org/html/1204.0646)。

## 3B.3 在 price space 直接构造：受约束最小二乘与 spline

这是回答“为什么非得搞 IV surface”的另一个关键：<strong>完全可以先拟合 price surface，最后有需要再转成 IV。</strong>Fengler 的工作就是从受形状约束的 smoothing splines 出发处理曲面；经典方法并不等于任意 cubic interpolation。[作者机构版本](https://edoc.hu-berlin.de/items/0abf3928-5a9f-4788-b094-ae39f3089a78)

先只看一个到期、$r=q=0$。在固定 strikes 上，把待估 call prices 记为 $c_i$；令 $b_i$ 为报价 half-spread，$s_i=(c_{i+1}-c_i)/(K_{i+1}-K_i)$。一个透明的离散优化是

$$
\min_{c_1,\ldots,c_n}
\frac12\sum_{i\in O}\left(\frac{c_i-C_i^{\mathrm{mid}}}{b_i}\right)^2
+\lambda\sum_i(s_{i+1}-s_i)^2,
$$

满足

$$
(S_0-K_i)^+\le c_i\le S_0,
\qquad -1\le s_i\le0,
\qquad s_{i+1}\ge s_i.
$$

其中 $O$ 是今天真正有报价的 indices，$b_i$ 要有数值下限防止除零，$\lambda$ 控制平滑程度。若 bid/ask 区间可行，还可以加

$$
C_i^{\mathrm{bid}}\le c_i\le C_i^{\mathrm{ask}},\quad i\in O.
$$

目标是二次函数，约束是线性的，所以这是一个 convex quadratic program。它不是某篇 spline 论文的逐行复刻，而是把同样的“拟合 + 形状约束”思想压成最容易理解的有限维版本。

求得节点后，分段线性 price interpolation 在 strike 区间内保持递减与凸性；但它的 density 可能有离散质量，未必足够光滑用于 Dupire。真正需要二阶导时，可改用带相应形状约束的 spline，并处理边界与 tails。<strong>光滑程度与无套利是两项不同设计目标。</strong>

### 一个实际运行的修复例子

配套脚本使用第 3A 节的混合分布作为隐藏真值。在七个 strikes 上造一组报价，故意把 ATM midpoint 抬高 2 个价格单位，并给该点较宽的 bid/ask。其他点的 half-spread 为 0.08，ATM 为 2.10，保证至少真值落在全部报价区间内。以 $\lambda=0$ 解上面的 quadratic objective。

结果：midpoints 的最小相邻 slope 增量为 **−0.165183**，存在离散 convexity violation；投影后的最小增量为约 **−2.2×10⁻¹⁶**，即浮点误差范围内的零，全部 fitted prices 同时位于 bid/ask 区间内。

<figure>
<img src="classical_examples/price_projection.png" alt="Bid-ask-aware convex price projection">
<figcaption>新增图 B：受约束 price projection，不需要历史训练集或 VAE。合成真值仅用于展示，优化目标不使用真值。检查只针对该到期和指定 strikes，并非全连续域证书。</figcaption>
</figure>

这也回应了交易摩擦的疑问：我们没有要求 raw midpoints 绝对一致，而是在它们的报价区间内找一个可供定价系统使用的一致 shadow surface。如果区间本身不可行，应报告 infeasibility、检查 stale quotes，或加入有记录的 slack；不能悄悄改掉原始 quotes 后宣称市场不存在问题。

## 3B.4 SVI / SSVI：把一张自由曲面压成少数形状参数

对某个固定到期，raw SVI 写成

$$
w(k)=a+b\left[\rho(k-c)+\sqrt{(k-c)^2+h^2}\right].
$$

本小节暂用 $c$ 表示中心位移、$h>0$ 表示中心平滑尺度，避免与全文 mask $m$、implied volatility $\sigma$ 混淆。$a$ 控制水平，$b$ 控制翼部斜率尺度，$\rho$ 控制左右不对称。

关键导数可直接算出：

$$
w'(k)=b\left[\rho+\frac{k-c}{\sqrt{(k-c)^2+h^2}}\right],
\qquad
w''(k)=\frac{bh^2}{((k-c)^2+h^2)^{3/2}}.
$$

所以 $b\ge0$ 时 total variance 是凸的，左右翼渐近斜率分别为 $b(\rho-1)$ 和 $b(\rho+1)$。但这不是 call-price convexity；只保证 $w''\ge0$ 仍可能产生 butterfly arbitrage。基本的 $w\ge0$ 条件也不够，必须检查第 3.4 节的 $g(k)$、tails 和跨期限一致性。[SVI/SSVI 原文](https://arxiv.org/html/1204.0646)

一个明确的校准流程是：先把当日 quotes 转成 $w_i=T\sigma_i^2$；用多初值 weighted least squares 拟合五个参数；沿较密 strike 网格检查价格约束；跨期限联合约束或校正；最后做经验证的插值与外推。SSVI 进一步用 $\theta(T)$ 和共享形状函数连接期限，并允许使用足够的参数条件控制静态套利。

它的核心收益是**低参数量、可解释、直接利用当日报价，不需要数年历史数据**。代价是形状族可能限制拟合；极稀疏时参数可能不识别；逐期限独立拟合后再随意插参数，也可能破坏整体约束。历史参数平滑、hierarchical pooling 和 regime models 都能作为经典增强，不属于 VAE 独有能力。

## 3B.5 Heston、SABR、local vol：不是另一种随意插值

第 3A 节已经展示 Heston 的动态。经典 calibration 是在允许的参数范围内求

$$
\min_{\text{parameters}}
\sum_{i\in O}
\left(\frac{C_{\mathrm{model}}(K_i,T_i)-C_i^{\mathrm{mid}}}{b_i}\right)^2,
$$

也可以使用 bid/ask interval loss，让区间内部的模型价不再因偏离 midpoint 而受罚。对受支持合约，模型输出同时提供价格、模拟路径和风险敏感度；但少量参数未必能精确匹配整个市场面板。

SABR 则以某一到期对应的 forward 为对象，在相应 forward measure 下写

$$
dF_t=\alpha_t F_t^\beta dW_t,
\qquad d\alpha_t=\nu\alpha_t dZ_t,
\qquad d\langle W,Z\rangle_t=\rho dt.
$$

这里的 $\beta$ 是 forward 的弹性参数，<strong>不是 VAE 的 KL 权重</strong>。常用的一种校准安排是固定 $\beta$，用 ATM 与 smile quotes 拟合 $\alpha_0,\rho,\nu$，再用模型价格或近似 IV 公式查询。SABR 的原始动机包括 smile dynamics 与 hedge，而不只是减少横截面 RMSE。[Hagan et al.](https://www.wilmott.com/managing-smile-risk/)

要分别注意三个边界。第一，SABR 的近似 Black-IV 公式不是精确 transition law，其误差可能在 wings 或长端造成不合适的价格形状。第二，每个 expiry 独立拟合一组 SABR 参数，不自动构成一个跨所有 expiries、tenors 的统一动态模型。第三，rates 的 normal、shifted-lognormal 与 Black conventions 必须分清，不能把 equity 的 positive-forward 假设无条件搬过来。

Local vol 通过 surface derivatives 恢复扩散系数；LSV 把状态依赖项与随机方差结合，试图兼顾 vanilla marginal calibration 和 richer dynamics。它们说明：<strong>市场已经有大量“修模型”的路线，VAE 不是因为传统金融从未修过 BS 才出现的。</strong>[Local-stochastic volatility 的校准与投影关系](https://arxiv.org/abs/1905.06213)

## 3B.6 PCA / 因子模型：经典方法也能学习历史和生成曲面

把每天曲面展平为 $x_t$，经典线性因子模型写成

$$
x_t=\bar x+Bf_t+\epsilon_t.
$$

$B$ 是共享 loading，$f_t$ 是当天少量 factors。PCA 从训练期求 $B$；今天只见到坐标 $O$ 时，可以求

$$
\widehat f_t=(B_O^\top B_O+\lambda I)^{-1}B_O^\top(x_{t,O}-\bar x_O),
$$

然后恢复未报价位置。这就是第 8 节保留的强 PCA baseline。

给 factors 加 VAR / state-space dynamics，就能预测；给它们指定 Gaussian mixture 或经验重采样，就能生成 scenarios；使用 probabilistic PCA、Kalman filter 或 Gaussian process，则能提供 conditional uncertainty。比如联合 Gaussian 曲面分成已知与缺失坐标，条件均值就是

$$
\mathbb E[x_M\mid x_O]
=\mu_M+\Sigma_{MO}\Sigma_{OO}^{-1}(x_O-\mu_O),
$$

条件方差为

$$
\operatorname{Var}(x_M\mid x_O)
=\Sigma_{MM}-\Sigma_{MO}\Sigma_{OO}^{-1}\Sigma_{OM}.
$$

这些是 Gaussian conditioning 的直接结果；协方差仍须从训练期估计并 regularize。它说明<strong>“能生成”“能补全”“能给 uncertainty”都不是 VAE 独占的卖点</strong>。真正要比较的是分布假设、非线性程度、计算代价与样本外效果。

## 3B.7 VAE 增加了什么，而不是替代了什么？

线性因子映射是 $x\approx\bar x+Bf$，VAE 则换成 nonlinear decoder $x\approx D(z)$，并训练一个给定观测后的概率 encoder。这里可能有四类收益，但每一类都需要证据。

**第一，跨曲面借信息。** 当今天整条 tenor 没有报价，单日 spline 的信息主要来自邻近期限与平滑假设；历史模型还可以使用以往相似 level/skew regimes 下的联合形状。这是 learned prior 的收益，但 PCA、GP 或参数层的 hierarchical model 也能借历史信息。不能给 VAE 全部历史、却给 classic baseline 只有今天几个点，再把优势全部归功于 neural architecture。

**第二，表达弯曲或多模态的低维结构。** 若 level 升高时 skew、curvature 和 term structure 的关系随 regime 改变，一个固定 loading matrix 可能不够。Nonlinear decoder 有机会用更少 factors 描述这种变化。代价是更多参数、潜变量不识别与 extrapolation 风险；在近似线性的低噪声数据中，PCA 可能更准。

**第三，降低重复条件推断成本。** Masked encoder 把“每来一张曲面就解一次 latent optimization”摊到训练阶段，部署时做一次 forward pass。但原始 VAE completion 若仍逐张优化 latent code，就没有获得这部分 amortized speedup。SVI 的小规模 warm-start fit 本来就可能很快；速度必须实际测。

**第四，学习非 Gaussian 的条件分布。** 当缺失区域确实存在多种合理形状，单个均值不够，可以采样多张 completion surfaces。不过 VAE 的 posterior variance 不自动等于可信的市场 uncertainty，更不自动包含参数不确定性；要检验 coverage、proper scores 和 rare regimes。

Gopal 的论文直接展示了这一领域的 baseline 问题：更强的 Heston-with-jumps baseline 能胜过早期 VAE，随后 residual / uncertainty architecture 改进才带来显著提升。它支持“认真比较、分解增益”，不支持“VAE 一定比金融模型好”。[Gopal (2024)](https://arxiv.org/html/2411.05998v1)

## 3B.8 在本报告里，VAE 的收益其实证明到哪一步？

本次原有 SSVI 实验在随机隐藏 50% 时，hidden-cell RMSE 为：PCA-8 **0.200 vol points**，MLP-VAE **3.031**，ConvVAE **1.982**。这些数字来自冻结的 `reproduction/results.json`，详见第 8 节。

因此应该说：<strong>在已测试的 VAE 中，二维卷积比 MLP 有优势；但 VAE 本身没有胜过 PCA。</strong>前者是网络 inductive bias 的证据，不是 variational inference 或 KL regularization 的独立贡献。数据是低噪声、四因子的 SSVI family，因此这不是所有真实市场任务的结论，但必须保留为明确的负面对照。

本实验没有比较以下对照，因此不能宣称已证明相应收益：同结构 deterministic AE；$\beta=0$ 或其他 KL 权重；历史正则化 SSVI；GP conditional completion；真正的 bid/ask pricing 与 hedging P&amp;L；真实市场 future-surface prediction。

要回答“为什么是 VAE，而不是 AE”，至少在匹配 encoder/decoder 容量的条件下，比较 deterministic AE、带 decoder regularization 的 AE、VAE，以及采用经验 latent distribution 的 AE。VAE 的 KL 会牺牲一部分 reconstruction 来约束 latent distribution；这是一项 trade-off，不是免费精度提升。

## 3B.9 “收益”怎样落到可测的业务量？

### 统计收益不等于交易 alpha

给某个未报价 option 补出更准的 IV，首先只是减少 fair-value estimation error。其价格影响一阶近似为

$$
|\Delta C|\approx\mathrm{Vega}\times|\Delta\sigma|.
$$

再除以 half-spread，才知道误差是报价区间的几倍。低 vega 的 deep-wing IV 改善很多，价格可能只动一点；高 vega 的 ATM IV 小误差可能对应更大资金影响。要声称交易收益，还需要 fill、inventory、hedging、transaction costs 与 adverse selection 的独立评估。论文的 reconstruction RMSE 不能直接换算成收益率。

### 系统收益也需要算训练和维护成本

设每天要处理 $N$ 次更新，经典拟合平均耗时 $t_{\mathrm{fit}}$，VAE inference 平均耗时 $t_{\mathrm{infer}}$，每天摊销的训练、再校准与监控成本为 $T_{\mathrm{maint}}$。粗略的计算节省条件是

$$
N\bigl(t_{\mathrm{fit}}-t_{\mathrm{infer}}\bigr)>T_{\mathrm{maint}}.
$$

这只是 decision accounting，不是实测速度结论。若 $N$ 很小、SVI warm start 已经很快，VAE 不一定划算；若 $N$ 很大、需要复杂 probabilistic completion，amortization 才可能更有价值。

### 定价收益、压缩收益、风险收益分别验收

| 你声称的增量 | 应报告什么 | 不能用什么替代 |
|---|---|---|
| 未报价点更准 | 严格样本外 masked price/IV MAE，按 vega、期限和 wings 分组 | 训练重建曲线更低 |
| 更符合报价区间 | bid/ask 内比例、越界幅度、quote reliability 分层 | 只报告平滑度 |
| 推断更快 | 包括预处理与 repair 的 p50/p99 latency、摊销成本 | GPU 单次 forward 理论 FLOPs |
| 生成分布更好 | 联合/尾部分布、约束幅度、coverage、stress tests | 只给几张好看的 samples |
| hedge / risk 更稳 | Greek 稳定性、样本外 hedge error、成本后指标 | 同日 IV RMSE |

## 3B.10 一项重要区别：市场曲面分布不是自动的风险中性路径分布

每天的 option prices 可以由某个风险中性测度 $Q$ 表示；但按历史日期收集到的 surface panels，描述的是现实测度 $P$ 下市场状态如何变化。这是两个不同层次。

所以从历史训练的 VAE 采出一张合理 surface，或用 CVAE 预测明天 surface，<strong>不等于已经定义出一个能定价 exotic 的 $Q$-path model</strong>。即使每个时刻单独通过静态 no-arbitrage checks，整条 option-price process 在合适 numeraire 下仍需满足动态约束。

同样，condition on interest rate 让模型学到相关性，不等于证明一个外生 rate shock 的因果反应。用于 stress/scenario 时，应说明是 conditional historical scenario、人工 shape shock，还是来自结构动态模型的 shock。

## 3B.11 对这个项目，我会怎样选方法？

这是根据任务作出的建模建议，而不是推荐某个方法永远胜出。

| 主要目标 | 我会先做 | VAE 进入比较的条件 |
|---|---|---|
| 今天密集 vanilla quotes 的插值 | bid/ask-aware price fit、SVI/SSVI 或受约束 spline | 经典方法留下稳定、可解释且有经济量级的样本外误差 |
| 整条 tenor / wing 稀疏补全 | pooled SSVI、PCA/因子条件补全、GP | 历史跨区域关系明显非线性，且在相同信息集下胜过这些基线 |
| 路径依赖期权定价与 hedge | calibrated local vol、stochastic vol、LSV 等合适动态 | VAE 用作辅助表示、参数先验或加速器，不能用静态 decoder 取代路径模型 |
| 多日 risk scenarios | 因子 dynamics / 参数动态与经验 resampling | 需要复杂联合尾部或多模态，并能通过分布与动态一致性检验 |

<strong>最值得验证的组合不一定是“经典方法 vs VAE”，而可能是“经典结构 + 小型 learned residual”。</strong>先用 SSVI 或受约束价格表示承担大部分形状与约束，再检验 neural residual 是否在相同数据、相同时间切分下增加价值。即使如此，residual 加回后也必须重新检查 admissibility。

## 3B.12 新增示例怎样重跑？

```bash
python classical_demo.py --output-dir classical_examples
```

它不训练模型，不调用市场数据，也不修改原有 VAE 实验结果。输出包括混合分布 IV 表、bid/ask 内价格投影表、两期 martingale 的验证结果与两张图。示例只验证本节的数学和实现逻辑，<strong>不是一项新市场 benchmark，也不是对 VAE 交易收益的证明</strong>。

本节新增来源：Fengler 的受约束 smoothing；Gatheral–Jacquier 的 SVI/SSVI；Heston 的 stochastic volatility；Hagan et al. 的 SABR / smile-risk 动机；Gopal 的经典基线与 VAE 改进。链接均已放在相关论断处，新增数值表由随附脚本独立计算。

