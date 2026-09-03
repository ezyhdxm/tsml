# 6. 把文献放在同一坐标系里：真正的分歧在哪里？

## 6.1 第一问不是“用什么网络”，而是“生成什么对象”

同一张市场曲面可以有多种表示：

1. 直接生成 implied volatility $\sigma$；
2. 生成 $\log\sigma$，自动保证正值；
3. 生成 total variance $w=\tau\sigma^2$；
4. 生成 normalized call prices；
5. 生成 SVI/SSVI、SABR、Heston 或更一般 arbitrage-free model 的参数；
6. 生成一组 latent factors，再由一个 constrained surface constructor 输出曲面。

这里有一个非常实用的偏序：

$$
\text{更接近市场报价}
\quad\longleftrightarrow\quad
\text{更容易训练，但约束更难保证},
$$

$$
\text{更接近 admissible parameterization}
\quad\longleftrightarrow\quad
\text{保证更强，但表示能力与 calibration 误差更受模型族限制}.
$$

直接输出 IV 的 VAE 最灵活，但 decoder 可以在任意 cell 上制造局部凹陷。Ning et al. 让 VAE 输出 SDE 参数，牺牲一些 model-free flexibility，换来 by-construction admissibility。Zhang et al. 则折中：第一步自由学习表示，第二步用 constrained DNN construction。

<div class="callout">
<strong>研究设计原则：</strong>先明确需要的是“报价精确拟合”“有限网格可行”“连续域静态无套利”还是“动态无套利”。这四个目标逐级更强，不能用同一个 vague 的 arbitrage-free 标签代替。
</div>

## 6.2 第二问：completion、generation 与 forecasting 是三种不同统计问题

给定部分曲面 $x_{\mathrm{obs}}$，completion 想要的是条件分布

$$
p(x_{\mathrm{miss}}\mid x_{\mathrm{obs}}).
$$

无条件 scenario generation 想要的是整个横截面分布

$$
p(x).
$$

未来预测还要加入过去信息 $h_t$：

$$
p(x_{t+1:t+H}\mid h_t).
$$

这三个分布之间没有自动包含关系：

- reconstruction loss 小，只检验 $x\to z\to x$；
- completion 还要求模型在各种 mask 下做正确条件推断；
- prior generation 还要求 $p(z)$ 与训练后 aggregate posterior 匹配；
- forecasting 还要求 latent dynamics 与 regime shift 正确。

因此，Chen et al. 的 CVAE+memory 不能只与同日 VAE reconstruction 比；Richert & Buch 的 pseudo-Gibbs completion 也不能用无条件样本的视觉效果代替 conditional calibration。

## 6.3 第三问：无套利一共有五种实现方式

### 路线 A：hard parameterization

令 decoder 输出参数 $a$，再由一个满足约束的确定性映射 $G$ 得到曲面：

$$
z\longmapsto a=D(z),
\qquad
x=G(a).
$$

只要 $a$ 落在 admissible parameter set，$G(a)$ 就无套利。优点是保证最强；缺点是生成分布受 $G$ 的模型族限制，且参数 admissibility 本身可能复杂。

### 路线 B：constrained constructor

VAE 只输出 features 或 latent state，最终 surface 由一个专门的 constrained network/optimization layer 构建。这是 Zhang et al. 的两步思想。优点是把表示与约束解耦；缺点是第二阶段可能成为主要误差来源。

### 路线 C：soft penalty

在重建损失上增加

$$
\lambda_{\mathrm{cal}}\,P_{\mathrm{cal}}(x)
+
\lambda_{\mathrm{bf}}\,P_{\mathrm{bf}}(x).
$$

通常 $P$ 是负 margin 的 ReLU 平方。优点是实现简单、可微；缺点是只在训练访问到的输出上被优化，无法保证 latent tails。

### 路线 D：post-generation repair

先生成，再求最近的可行曲面或最近的可行 latent code：

$$
\min_{\tilde z}\|\tilde z-z\|^2
\quad\text{s.t.}\quad M(\tilde z)\ge0.
$$

优点是保留原模型；缺点是 repair 会改变生成分布，非凸可行集还可能有多个投影。

### 路线 E：学习真实 latent distribution

先训练 autoencoder/VAE，再用 flow 或 diffusion 学 aggregate posterior：

$$
\epsilon\sim N(0,I)
\longmapsto z\sim q_{\mathrm{agg}}(z)
\longmapsto x=D(z).
$$

它不自动提供 hard guarantee，但能显著减少“高斯 prior 采到了 decoder 未见区域”的问题。2026 latent-flow 工作属于这一路线。

## 6.4 为什么 PCA 是必须严肃对待的基线？

设展平曲面为 $x\in\mathbb R^p$。PCA 假设

$$
x\approx \bar x+Bf,
$$

其中 $B$ 是固定 loading matrix，$f$ 是低维 factors。VAE 则把线性映射换成非线性 decoder：

$$
x\approx D(z).
$$

所以 VAE 可以看作**带概率正则化的 nonlinear factor model**。但“非线性更强”不等于“小样本下误差更小”：

- 若数据由少数平滑参数生成，线性切平面已经很好；
- VAE 的 KL 与 stochastic sampling 会牺牲一部分 reconstruction precision；
- neural decoder 参数更多，容易把噪声当作 curvature；
- 固定网格、低维、日频曲面往往只有数千样本，远小于图像数据规模。

因此本报告的 SSVI 实验中，PCA-8 解释了 99.99579% variation，并在所有 completion masks 上明显胜过 VAE。这个结果不是“实验失败”，而是在提醒：要证明 VAE 的价值，数据必须包含 PCA 不能表达的 nonlinear manifold、multi-modality、heteroscedastic uncertainty 或复杂 mask conditioning。

## 6.5 VAE 和 tensor factor model 到底是什么关系？

对规则三维对象，例如 date × maturity × moneyness，可写一个线性 tensor factor model：

$$
X_t(i,j)
\approx
\sum_{r=1}^{R} f_{t,r}a_r(i)b_r(j).
$$

它把 maturity loading 与 moneyness loading 分离，参数少、解释性强。VAE decoder 则允许

$$
X_t(i,j)\approx D(z_t)_{ij},
$$

其中 maturity 与 moneyness 可以发生任意非线性交互。两者并非互斥：

- encoder 可以先提取 tensor factors，再对 residual 用 VAE；
- decoder 可以是低秩 tensor basis 加 nonlinear correction；
- latent dynamics 可以是 state-space model，而不是把每天视为独立样本；
- constraint layer 可以作用于最终 total-variance surface。

一个很自然的研究框架是

$$
X_t=X^{\mathrm{structured}}(f_t)
+R(z_t),
$$

其中第一项承担可解释的 level/skew/curvature/tensor structure，第二项只拟合非线性 residual，并对两项之和施加 no-arbitrage。这样比“纯 tensor”更灵活，也比“纯黑箱 VAE”更可识别。

## 6.6 如何比较论文：至少要同时报六类指标

| 维度 | 最低要求 | 常见但不充分的替代品 |
|---|---|---|
| 同日重建 | pointwise RMSE/MAE，按 grid 区域分解 | 只放几张看起来平滑的图 |
| 条件补全 | 多种随机和结构化 masks，置信区间 | 只随机去掉少量 cells |
| 分布拟合 | marginal + joint/surface distance + tail quantiles | 只比较均值曲面 |
| 静态无套利 | violation rate、数量、幅度、修复成本 | 只说训练数据无套利 |
| 时间预测 | chronological split、persistence、proper scores、coverage | 随机 train/test split |
| 下游价值 | calibration、hedging、P&amp;L/risk sensitivity | reconstruction loss |

特别要避免把“每个 cell 的 Wasserstein 很小”解释成 joint surface distribution 正确。所有 cells 的边际分布都可能匹配，但跨 tenor、跨 strike 的共动结构仍然错误。

## 6.7 数据处理往往比模型差异更大

横跨这些论文，影响最大的设计常常不是 latent dimension，而是：

- delta grid 还是 log-forward-moneyness grid；
- 先拟合 SVI/SABR 再采样，还是直接用 raw quotes；
- missing cells 是删除、插值、nearest-neighbor 还是 mask channel；
- train/test 是否严格按时间；
- 利率、dividend、forward 与 spot 的口径；
- bid/ask/mid、同步时间与 quote filtering；
- 0DTE 是否混入，短端是否单独建模；
- 标准化是否只用 training statistics。

一旦 preprocessing 已把每一天投影到某个低维平滑 family，后面的 VAE 可能主要是在学习 preprocessing 的输出分布。因此，报告模型结果时必须连同 surface-construction recipe 一起报告。

# 7. 复现性审计：怎样判断“这篇论文能不能重跑”？

## 7.1 本报告的 10 分 rubric

每篇论文按以下信息是否完整给分：

1. 原始数据源与许可；
2. 日期范围与资产 universe；
3. quote filters 与坐标定义；
4. grid/interpolation/missing-data 规则；
5. chronological split；
6. 完整网络结构；
7. loss 权重、optimizer、learning-rate schedule；
8. batch、epochs、early stopping、seed；
9. evaluation code 与 metric definition；
10. 可运行代码、processed data 或 checkpoints。

分数不是论文质量评分，而是**从公开信息重建数字的难度评分**。

<div class="table-wrap"><table id="audit-table" class="data-table"><thead><tr><th>论文</th><th>审计分</th><th>主要阻塞点</th></tr></thead><tbody><tr><td>Bergeron et al.</td><td>4/10</td><td>EDI FX 不公开；模型宽度给出，但学习率、batch、epoch、seed 与完整代码不足。</td></tr><tr><td>Ning et al.</td><td>7/10</td><td>数据不公开；校准与网络细节充分；公开 demo/预计算输出，但完整 CTMC 校准链较重。</td></tr><tr><td>Zhang et al.</td><td>5/10</td><td>OptionMetrics 不公开；网格与预测网络较清楚；VAE 训练与超参选择/验证流程不完整。</td></tr><tr><td>Dierckx et al.</td><td>5/10</td><td>方法与用例清楚，但可直接获得的数据、预处理与完整训练配置不足。</td></tr><tr><td>Richert &amp; Buch</td><td>3/10</td><td>FENICS cube 不公开、仅约 120 个真实完整 cube；无公开端到端代码。</td></tr><tr><td>Gong et al.</td><td>4/10</td><td>数据与主要损失权重给出；网络宽度、优化器细节、seed 与代码不足。</td></tr><tr><td>Gopal</td><td>6/10</td><td>时间切分、搜索空间和优化 schedule 很详细；FX 数据不公开，代码/权重未见公开。</td></tr><tr><td>Feugang Nteumagné et al.</td><td>5/10</td><td>Heston 数据可原则上重造；结构和范围给出，但学习率、精确 batch、seed、代码与汇总表不足。</td></tr><tr><td>Wang et al.</td><td>7/10</td><td>合成数据范围、网格、结构、优化器较详细；β 的逐实验取值与代码/seed 仍不足。</td></tr><tr><td>Chen et al.</td><td>8/10</td><td>OptionMetrics 不公开，但预处理、切分、结构、seed 和公共代码较完整。</td></tr><tr><td>Singh et al.</td><td>9/10</td><td>原始 Binance archive、处理数据、配置、checkpoint 和代码公开；主要风险是原始供应商字段口径与版本。</td></tr><tr><td>Brooks et al.</td><td>8/10</td><td>代码与绝大多数超参公开；OptionsDX 数据受许可，且预处理/repair 需精确版本锁定。</td></tr><tr><td>Buchegger &amp; Gonon</td><td>6/10（初评）</td><td>论文很新；公开摘要与方法足够定位，但本报告未对完整代码/数据链作最终审计。</td></tr><tr><td>Wang, Liu &amp; Vuik（latent geometry）</td><td>7/10（初评）</td><td>30k Heston、28×28、2D latent、5 seeds 与边界计算细节清楚；发布于 2026-08-31，代码与高维可扩展性尚待审计。</td></tr></tbody></table></div>

## 7.2 三种“复现”必须分开说

### Exact replication

使用相同数据版本、同一 preprocessing、代码 commit、seed 与环境，目标是重现论文表格。金融市场数据常受许可证限制，所以这一级很少能完全实现。

### Computational reproduction

使用作者公开代码和 processed data，允许硬件或 library 小差异，验证主表结论与数量级。这是 Singh et al. 公共归档最接近支持的层级。

### Structural replication

重新实现核心机制，在公开或合成数据上检验方向性命题，例如：

- ConvVAE 是否比 MLP 更适合二维局部缺失？
- mask channel 是否必要？
- no-arbitrage penalty 是否改善重建与生成？
- PCA 是否在低维光滑数据上更强？

本报告实际完成的是第三类。它不能证明原论文的市场 benchmark 数字，却能检验模型机制是否自洽。

## 7.3 为什么不把“仓库存在”当作可复现？

一个 GitHub 链接仍可能缺少：

- 原始数据字段与 vendor version；
- 云盘中的大文件或过期 checkpoint；
- 从 raw quotes 到 grid 的不可见手工步骤；
- exact commit 与依赖锁；
- 重跑论文表格的 single command；
- train/validation/test 的日期边界；
- 随机种子或多 seed 统计。

因此，本报告在逐篇卡片中同时记录“代码是否公开”和“真正阻塞点”。

