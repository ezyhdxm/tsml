---
title: "Time Series Cross-Validation"
subtitle: "从预测任务、信息可用性到可信 OOS：教程式综述"
date: "2026-09-04"
lang: zh-CN
---

<div class="lead">
<p class="eyebrow">TSML · METHODS & EVALUATION · TUTORIAL SURVEY</p>
<p><strong>时间序列验证不是一种 splitter，而是一份可回放的预测协议。</strong>先说清楚何时预测、预测什么、届时知道什么、多久重训、对谁平均，再讨论 rolling window、gap、purging 和 K-fold。</p>
<p>本文以一般 forecasting 为主线，另用不规则事件、未来窗口标签与面板数据说明边界情况。公式从条件风险与简单协方差展开；数值来自独立运行的合成实验，不使用任何真实工作数据。</p>
</div>

::: {.callout}
**阅读路线。** 首读第 1–6 节，建立任务与切分的对应；第 7–12 节把它变成可以实施的验证流程；第 13–15 节看实际实验、文献地图与研究边界。附录给出 splitter、复现实验和自测题。文中“本文推导”“合成实验”“文献结论”分别标明，不把工程建议写成普遍定理。
:::

# 1. 先问：你的 OOS 数字究竟在估计什么？ {#risk}

## 1.1 从一次真正的预测开始

假设在时刻 $t$，你拥有信息集 $\mathcal F_t$，构造特征 $X_t$，预测未来目标 $Y_t$。下标 $t$ 表示**预测发起时刻**，不表示目标已经在 $t$ 实现。例如，$Y_t=S_{t+24h}-S_t$；也可以是未来某个窗口内观测的函数。

令 $c(t)\le t$ 为最近一次模型训练的数据截止时刻。用截止该时刻允许使用的训练样本 $D_{c(t)}$，以及预先规定的训练算法 $A$，得到

$$
\widehat f_{c(t)}=A(D_{c(t)}),\qquad
\widehat Y_t=\widehat f_{c(t)}(X_t).
$$

这里 $A$ 不只是模型类：它还包括缺失值处理、标准化、特征筛选、调参、early stopping、训练窗口与重训规则。一个按月重训的系统，与一个每条记录到来就重训的系统，即使都叫 LightGBM，也不是同一个 forecasting procedure。[R1](#r1) [R2](#r2)

给定损失 $\ell$，在预测时刻的条件风险是

$$
r_t=\mathbb E\!\left[\ell(Y_t,\widehat Y_t)\mid\mathcal F_t\right].
$$

实际评估把多个发起时刻的损失聚合：

$$
\widehat R=\frac{\sum_{i\in\mathcal V}w_i\,
\ell(Y_i,\widehat f_{c(t_i)}(X_i))}{\sum_{i\in\mathcal V}w_i}.
$$

**这只是一个已经实现的历史平均。**要把它解读成下一阶段的预测风险，需要历史评估时段与未来部署环境之间存在可辩护的联系。切分本身不会创造这种联系。

## 1.2 三种经常混为一谈的目标

| 目标 | 真正的问题 | 常见证据 |
|---|---|---|
| 评估学习程序 | 在不同历史起点按同样规则训练，表现如何？ | 多个 rolling origins、完整嵌套流程 |
| 评估最终已训练模型 | 今天这一个具体模型，在接下来一段时间表现如何？ | 最靠近部署的未触碰测试段、上线记录 |
| 选择配置 | 哪个窗口、特征集、超参数更值得使用？ | 仅在开发区间内的时间序列验证 |

三个问题相互关联，但答案不相同。普通 CV 也并不自动精确估计“手头这个最终拟合模型”的条件误差。Bates、Hastie 与 Tibshirani 对特定线性模型明确分析了这一差别；不能把其 IID 结论原封不动搬成时间序列定理。[R3](#r3)

**一个有用的协议表达：**先写清“对象 × 发起时刻 × horizon × 信息集 × 更新规则 × 损失 × 权重”。其中任意一项变化，都可能让两个 OOS 分数失去可比性。

::: {.takeaway}
**本节落点：**先固定 forecasting procedure，再选择 splitter。我们想验证的是“过去若真的部署这套程序，会发生什么”，不是“从 DataFrame 中拿走几行以后还能拟合多好”。
:::

# 2. 四种不同的问题：不能统称为 leakage {#four-problems}

## 2.1 信息泄漏：用了当时不可能拥有的东西

例如预测周五价格变化，却用到周五收盘后才能算出的全天成交量；或者周一重训时，纳入周二才会成熟的训练标签。错误在于**信息可用时刻**，不在于是否调用了随机 shuffle。

## 2.2 依赖导致评估不稳定：没偷看，但有效信息少

相邻 24 小时收益共享大量增量，因此相邻预测损失可能相关。一个完全遵守时间顺序的测试集，仍可能只有很少的有效独立信息。平均 MAE 可以有意义，朴素标准误却可能太小。

## 2.3 分布或部署协议不匹配：测的不是未来要做的任务

在一个稳定机制下，跨时间随机抽样可以测量该机制中的预测误差；在制度突变后，它可能主要反映旧机制的混合平均。另一个例子是每天重训的验证，拿来承诺每月重训的上线表现。都不是简单加一个 gap 可以解决的。

## 2.4 选择偏差：试得太多，最后挑中了运气

你反复看同一个验证集，改特征、调窗口、换标签定义，最后挑最小 MAE。即使每次切分都完全合法，最终赢家的展示分数仍可能乐观。Cawley 与 Talbot 专门研究了模型选择过程的过拟合；它不是“训练集拟合过度”的同义词。[R4](#r4)

| 症状 | 优先处理 | 不能指望它解决 |
|---|---|---|
| 训练标签在训练时尚未可用 | label-availability 截止、purging | 未来 regime 改变 |
| 相邻损失高度相关 | 配对比较、时间聚合、HAC/block bootstrap | 已发生的信息泄漏 |
| 旧市场表现不能代表新市场 | 按时间走的回放、近期分层报告、窗口敏感性 | 无条件保证未来稳定 |
| 验证集看了几十次 | nested selection、锁定最终测试段、记录试验 | 通过增加 folds 自动消除选择偏差 |

这些问题可以同时出现。Hewamalage、Ackermann 与 Bergmeir 的综述覆盖了划分、误差度量、统计检验等环节，适合作为本文的总背景阅读。[R5](#r5)

# 3. 时间戳不是一个字段：建立 point-in-time 数据契约 {#availability}

## 3.1 每个样本至少记录什么？

对第 $i$ 个样本，用以下记号：

| 记号 / 字段 | 含义 | 示例 |
|---|---|---|
| $t_i$ / `prediction_time` | 作出预测的时刻 | 周一 10:00 |
| $X_i$ | 该时刻真实可获取的特征 | 截至 10:00 已发布并接收的数据 |
| $I_i=[s_i,e_i]$ | 标签依赖的结果区间的保守包络 | 周一 10:00 至周二 11:00 |
| $a_i$ / `label_available_at` | 标签已完整构造、发布并进入系统的时刻 | 周二 11:15 |

还可以记录各原始源的 event、publication、ingestion timestamp，特征流水线版本和 label vintage。区间 $I_i$ 是依赖记录的概括，不意味着标签一定对区间内每一个点都有依赖。对稀疏事件，记录实际依赖的原始记录 ID 会更精确。

最基本的可用性约束为

$$
X_i\text{ 只由 }\mathcal F_{t_i}\text{ 中的信息构造},
\qquad D_c=\{(X_i,Y_i):t_i<c,\ a_i<c\}.
$$

本文采用严格不等号，保守排除恰好与截止时刻同刻才到达的标签。如果真实系统有可靠的同刻事件序列和处理顺序，可以调整边界，但必须固定约定。

**训练计算耗时也要考虑。**若 09:00 开始训练、09:10 才上线，数据 cutoff 是 09:00，模型最早可用于 09:10 的预测。不能因为上线在 09:10，就让训练数据包含这十分钟内才到达的信息。

## 3.2 一个“已经按日期切分”仍然泄漏的例子

模型于周二 09:00 开始训练。某条训练记录的预测时刻是周一 10:00，目标用未来 23–25 小时窗口内的观测。窗口到周二 11:00 才结束，再加 15 分钟发布延迟。

$$
t_i<c,\qquad e_i>c,\qquad a_i>c.
$$

按 `prediction_time < cutoff` 会留下它；按 `label_available_at < cutoff` 则会删除它。**23 小时旧的记录，不等于 23 小时前已经知道的训练标签。**

对固定 horizon $H$、固定延迟 $D$，若 $a_i=t_i+H+D$，训练条件简化成

$$
t_i<c-H-D.
$$

因此固定时间 gap 可以是 availability 的实现方式。但当 horizon、窗口选择、发布延迟随样本变化时，一个统一 gap 只是粗略上界，不如逐样本判断准确。

## 3.3 不要把“历史最终版”误当“当时可见版”

宏观数据修订、迟报成交、撤销与更正、事后清洗标志，都可能使今天查询的过去数据不同于当时实际可见的数据。特征必须回放当时可见版本；训练标签也只能使用截止时刻已可用的版本。最终评分可以使用事先指定的更成熟标签版本，但应把该口径与训练 vintage 分开记录。

全样本 standardization、PCA、winsorization 阈值和 feature selection 也属于训练。把它们放在 CV 外面，会让验证数据影响拟合过程。官方 scikit-learn 文档把这种问题归入预处理泄漏，并建议用 pipeline 保持训练/验证隔离。[R6](#r6)

::: {.takeaway}
**本节落点：**对于未来窗口标签，最重要的元数据往往不是 `target_date`，而是 `label_available_at`。先把这个字段定义对，很多关于 gap 的争论就会变成可以检查的布尔条件。
:::

# 4. 各种切分到底模拟了什么？ {#splitters}

<div class="figure">
<div class="legend"><span class="key train">训练</span><span class="key gap">排除 / gap</span><span class="key test">验证</span><span class="key unused">未使用</span></div>
<div class="split-row"><b>Expanding · 1</b><div class="track"><i class="train" style="flex:4">过去</i><i class="gap">间隔</i><i class="test" style="flex:2">验证</i><i class="unused" style="flex:5"></i></div></div>
<div class="split-row"><b>Expanding · 2</b><div class="track"><i class="train" style="flex:6">过去</i><i class="gap">间隔</i><i class="test" style="flex:2">验证</i><i class="unused" style="flex:3"></i></div></div>
<div class="split-row"><b>Rolling · 2</b><div class="track"><i class="unused" style="flex:2"></i><i class="train" style="flex:4">固定窗口</i><i class="gap">间隔</i><i class="test" style="flex:2">验证</i><i class="unused" style="flex:3"></i></div></div>
<div class="split-row"><b>Blocked K-fold</b><div class="track"><i class="train" style="flex:4">之前的数据</i><i class="test" style="flex:2">验证</i><i class="train" style="flex:6">之后的数据也训练</i></div></div>
<p class="caption">图 1 · 顺序块只是验证集的形状，不保证训练集只来自过去。横轴为时间；示意块宽不是实际样本数量。</p>
</div>

## 4.1 单次 chronological holdout

先训练，再在其后的时间段验证。实现简单、接近一次真实上线，但结论依赖这个测试段的市场状态、季节与难度。它适合做终局测试，不适合被无限次重复查看和调参。

## 4.2 Expanding-window / rolling-origin evaluation

给定训练截止 $c_1<c_2<\cdots<c_K$，第 $k$ 次使用所有满足 availability 的过去样本，并预测后面的区间。

$$
\mathcal T_k=\{i:t_i<c_k,\ a_i<c_k\},\qquad
\mathcal V_k=\{i:v_k\le t_i<v_k+B\},\quad v_k\ge c_k.
$$

$B$ 是评估块长度，不是 forecast horizon。随着 cutoff 向前移动，训练集不断扩大。这是 FPP 中 time series cross-validation 的核心思路，并可扩展到多步预测。[R1](#r1)

## 4.3 Sliding / rolling training window

只用最近 $W$ 时间长度的数据：

$$
\mathcal T_k(W)=\{i:c_k-W\le t_i<c_k,\ a_i<c_k\}.
$$

它与 expanding window 的区别不只是计算量：它让学习程序主动遗忘旧机制。短窗口更容易适应变化，但估计噪声更大；长窗口样本多，却可能混入过时关系。因此 $W$ 是需要在开发区间选择的模型配置，而不是预处理中的一个无关常数。

本文以预测发起时间定义 $W$。以标签可用时间定义窗口是另一种合法政策，但两者保留的样本不同，不能偷偷混用。

## 4.4 Blocked K-fold、h-block 与 hv-block

Blocked K-fold 先把记录分成连续时间块，然后轮流拿一个块验证，其余块训练。**“连续分块”不等于“只用过去”。**验证块之后的数据通常仍在训练集里。

h-block 在测试点附近删除邻居；hv-block 使用测试块并在两侧留出缓冲，试图降低训练与测试间的局部依赖。它们服务于依赖数据下的模型选择或风险估计，但不是天然的实时部署回放。Racine 的工作是这条文献线的重要入口。[R7](#r7)

理论上要特别谨慎：Zheng 的说明指出，Racine 文中借助 balanced incomplete block design 的一个论证不成立。因此不要未经核查就宣传“hv-block 已在一般依赖数据下保证模型选择一致性”。这项批评针对特定论证，不等于否定每一种具体模型下可能成立的结果。[R8](#r8)

## 4.5 Prequential evaluation

按时间依次执行“预测 → 等标签到达 → 评分 → 按既定规则更新”。如果标签延迟，必须使用待评分队列，不能在发出预测后立刻把尚未实现的目标用于更新。Dawid 的 prequential 思想提供了这一顺序视角。[R2](#r2)

它特别适合在线学习，但也适用于按日、周、月重训的离线系统：更新时刻不同，原则相同。

| 方法 | 训练可能使用验证之后的数据？ | 最自然的用途 | 主要局限 |
|---|---|---|---|
| 单次时间 holdout | 否 | 最终测试 / 一次上线模拟 | 单一时期，方差与 regime 敏感 |
| Expanding walk-forward | 否 | 累积训练的历史部署回放 | 旧数据可能过时；早期模型训练量小 |
| Rolling-window walk-forward | 否 | 有遗忘机制的部署回放 | 窗口选择本身要验证 |
| Blocked / hv-block | 是，通常如此 | 稳定机制下的依赖数据评估 | 不是严格时间外推 |
| Purged K-fold / CPCV | 是，通常如此 | 降低已知重叠、历史子样本稳健性 | 不创造独立历史，也不模拟真实时间箭头 |
| Prequential | 否 | 在线或周期更新系统 | 必须正确处理延迟反馈与状态更新 |

# 5. 随机 K-fold 在时间序列上一定错吗？ {#random-cv}

## 5.1 有时间依赖，不等于一切随机 CV 都不成立

考虑稳定的 AR(1)：

$$
S_{t+1}=\phi S_t+\varepsilon_{t+1},\qquad |\phi|<1,
$$

其中 innovations 独立同分布，均值为零，且与过去独立。用 $X_t=S_t$ 预测 $Y_t=S_{t+1}$。虽然 $S_t$ 很相关，但正确模型的预测误差是下一期 innovation；它与原始序列的自相关性质不同。

Bergmeir、Hyndman 与 Koo 研究了纯自回归预测，给出特定假设下标准 K-fold 可用的理论与实验依据。他们强调误差不相关及模型适当设定；该结果不是“任意平稳过程 + 任意学习器 + 任意损失”的通用许可证。[R9](#r9)

这里尤其要区分两个命题：

**统计命题：**在稳定机制和合适正则性条件下，随机 CV 可能很好地估计某类平均预测风险。

**操作命题：**用一个测试点之后的数据训练的模型，不是该测试点发生时真实能部署的模型。

二者可以同时为真。风险估计的渐近合理性，并不会改变历史信息的可用性。

## 5.2 三个不能跳过的限定

“ADF 拒绝单位根”不等于所有联合分布稳定；“残差 ACF 不显著”也不是独立性的证明，更不是整个建模流程不存在 leakage 的证书。均值误差不相关，不能自动推出分位数 hit process、绝对误差或平方误差不相关。

还要区分正确的一步自回归输入，与包含未来窗口、全样本统计、迟报修订的特征。后者即使随机 CV 取得漂亮分数，也不在上述简单情形内。

最后，真实任务可能就是外推到最新时期，而不是从同一个稳定分布中再抽一个样本。这时 chronological evaluation 更直接地对应目标。

## 5.3 实证文献为什么看起来“互相矛盾”？

Cerqueira、Torgo 与 Mozetič 比较了多种性能估计方法。他们的实验发现，在平稳情形中交叉验证方法可以有竞争力；面对非平稳真实序列，保持时间顺序、跨多个测试时期重复的 OOS 方法更可靠。[R10](#r10)

这不与上一节冲突：数据机制、学习器、风险定义和比较基准都不同。本文的实际默认是：**以时间向前的验证作为部署证据；把随机或双向 blocked CV 限定为有明确统计目的的补充实验。**

# 6. Purging、embargo 与 gap：三个层次 {#purging}

## 6.1 Purging：针对已知依赖关系删样本

假设标签 $Y_i$ 使用区间 $I_i=[s_i,e_i]$ 上的原始结果。对于验证样本集合 $\mathcal V$，一种常见的区间 purging 规则是删除满足

$$
I_i\cap I_j\ne\varnothing\quad\text{for some }j\in\mathcal V
$$

的训练样本。这是保守的重叠检测；若只记录连续包络，可能删掉实际并不共享原始数据的样本。对多对象数据，还应标记依赖对象：不同实体的同一时段并不必然共享同一条结果，但可能受到共同冲击。

在严格 forward 训练中，**先检查 $a_i<c$**。对于未来起始、未来结束的标签，这通常已移除了跨越训练时点的未成熟记录。为了实现这一点，可以使用 purging，但不能只检查标签与验证区间是否相交、却忽略另一些虽不相交但在 cutoff 尚未发布的标签。

## 6.2 Embargo：在测试区间之后再留一个缓冲

在允许从验证之后取训练数据的双向切分中，后侧训练记录可能通过 backward-looking features 读到测试期的信息。金融 ML 中常用 embargo 删除测试之后的一段训练记录；purging 与 embargo 在 López de Prado 的书中有专门讨论。[R11](#r11)

本文采用一种明确的保守约定：先以验证目标的最晚结束时刻 $E_{\mathcal V}=\max_{j\in\mathcal V}e_j$ 为右边界，再删除预测时刻位于 $(E_{\mathcal V},E_{\mathcal V}+g]$ 的后侧训练样本。不同实现可能从验证 origin block 的末端计数，因此必须查看具体语义，不能只看到一个 `embargo=0.01` 就认为定义相同。

即便标签区间不重叠，训练特征仍可能直接包含测试结果。更一般的审核要检查实际特征/标签的数据来源。一个统一 embargo 只是对这种关系的近似保护，**不是独立性证明**，更无法把未来训练变成历史上合法的训练。

## 6.3 Gap：一个实现参数，可能服务于不同目的

Gap 可以为标签成熟留时间，可以模拟实际训练延迟，也可以故意降低边界附近的依赖。应先说明目的，再确定时间长度。标签可用性给出的是硬约束；“再隔多远可以弱化依赖”需要另外的假设与敏感性分析。

scikit-learn 的 `TimeSeriesSplit(gap=...)` 中，gap 是**样本条数**，不是小时；它不会读取每条记录的 label end 或发布时间。该工具按行索引切分，文档也提示等间隔数据与跨 folds 时间跨度可比性的问题。[R12](#r12)

<div class="interactive" id="gap-demo">
<h3>动手改一改：24 条记录的边界审计</h3>
<p>假设每小时一条记录，训练截止在第 16 小时；验证为第 16–23 小时。标签可用时刻 = 发起时刻 + horizon + 发布延迟。颜色只表示本演示的可用性，不表示独立性。</p>
<label>Horizon <input id="horizon" type="range" min="1" max="10" value="4"> <output id="horizon-out">4</output> 小时</label>
<label>发布延迟 <input id="delay" type="range" min="0" max="4" value="1"> <output id="delay-out">1</output> 小时</label>
<div id="demo-cells" class="demo-cells" aria-label="训练与标签成熟时间示意"></div>
<p id="demo-summary" aria-live="polite">默认 H=4、D=1：只有 t&lt;11 的历史记录满足 a&lt;16；t=11–15 的历史记录仍不能训练。</p>
<noscript><p>本示例默认下，t=0–10 可训练，t=11–15 被排除，t=16–23 用于验证。</p></noscript>
</div>

## 6.4 为什么不应机械使用“lookback + horizon”？

预测周二 10:00 时，使用过去 7 天的历史观测是正常的。这段 feature lookback 与训练期原始数据重叠，并不构成 leakage，因为它们本来就是当时已知的历史。

因此，对严格向前回放，训练窗口与验证特征 lookback 没有必要完全隔离。硬性 gap 通常首先由训练标签成熟与实际处理延迟决定，而不是自动等于 $L+H$。反过来，在**后侧训练**的双向 CV 中，长度为 $L$ 的回看特征可能读到测试结果，$L$ 才会进入相关的依赖分析。

::: {.takeaway}
**本节落点：**purging 删除明确不该共享的依赖；embargo 增加边界缓冲；gap 是切分参数。它们不能替代 point-in-time 特征构造，也不能保证历史样本成为独立样本。
:::

# 7. CPCV：有价值，但不要把它神化 {#cpcv}

把历史划成 $N$ 个连续组，每次拿 $k$ 组测试，其余训练，并对每个组合执行 purging/embargo。组合数是

$$
\binom Nk.
$$

例如 $N=6,k=2$，需要 15 次组合切分。每个组出现在 $\binom{5}{1}=5$ 个测试组合里；在标准路径分配约定下，可整理为

$$
\phi=\frac{k}{N}\binom Nk=\binom{N-1}{k-1}=5
$$

条覆盖全历史各组的预测路径。这是组合计数，不是五段相互独立的真实历史。路径之间共享原始数据与训练样本；多数拟合还会使用被预测组之后的数据。[R11](#r11)

CPCV 能回答“这个配置在不同历史子样本组合中是否很脆弱”，也有助于观察选择结果对分段方式的敏感性。它不能直接回答“在当时只知道过去的条件下，下一天到底能预测多好”。

也不要把 CPCV 与用于估计 backtest overfitting probability 的 **CSCV** 混为一谈。Bailey 等人的 PBO/CSCV 框架研究的是策略选择与回测过拟合；它与 purged forecasting validation 有联系，但评估对象不同。[R13](#r13)

**本文建议的角色分工：**walk-forward 是部署证据；CPCV 是明确标注假设的稳健性补充。增加组合数可以复用已有历史，不会增加新的经济状态信息。

# 8. Nested validation：把选择过程也放回过去 {#nested}

## 8.1 为什么“最低 CV 分数”通常过于乐观？

设配置 $\lambda$ 的验证估计为 $\widehat R_\lambda=R_\lambda+\epsilon_\lambda$。先忽略时间变化，并假设每个估计都无偏。选择

$$
\widehat\lambda=\arg\min_\lambda\widehat R_\lambda.
$$

由于最小值会选择较有利的噪声，至少有

$$
\mathbb E\!\left[\min_\lambda\widehat R_\lambda\right]
\le \min_\lambda\mathbb E[\widehat R_\lambda]
=\min_\lambda R_\lambda.
$$

**这一步只是 min 的基本不等式，已足以说明问题。**“先选最优、再汇报同一个最优分数”并不是对所选模型的独立测试。现实中还包含人为反复查看、不同标签和数据清洗版本，选择范围通常比日志中的超参数表更大。[R4](#r4)

## 8.2 外层与内层分别负责什么？

在外层 cutoff $c_k$，只拿当时已经可用的历史。内层以更早的多个 cutoff 比较配置，选择 $\widehat\lambda_k$；然后在外层可用训练集重新拟合，预测外层未来块。

$$
\widehat\lambda_k
=\arg\min_\lambda\widehat R^{\mathrm{inner}}_k(\lambda),
\qquad
\widehat f_k=A_{\widehat\lambda_k}(D_{c_k}).
$$

外层测试的是**整套选择与重训程序**，不是一个预知未来赢家的固定超参数。内层每一个 cutoff 也要检查 label availability，而不是先把外层数据整理好就默认内层合法。

有时真实部署政策是“一次选定配置，此后只定期重训参数”。那就应在开发区间固定配置，再在所有后续外层块保持不变。**每个外层重新调参**和**只在开始调一次**是两种不同的合法协议。

## 8.3 Early stopping 最常见的漏洞

不能使用外层测试标签选择 boosting round，再把该块分数称作纯 OOS。可以在内层选 stopping rule / iteration count，然后按预先定义的汇总规则，例如内层选择出的轮数中位数，在外层训练集重拟合。也可以在外层历史内部再划一段 validation，但它的标签也必须在 cutoff 已成熟。

模型融合权重、残差修正、quantile calibration、feature selection 同样属于拟合，必须在正确层级完成。

## 8.4 最终 holdout 是否永远不能用于更新？

不是。若上线协议本来就是按周重训，那么在最终测试期内，过去已经成熟的测试标签可以按**预先冻结的规则**进入后续训练。这是在评估自适应程序。

不能做的是：看到终局测试结果以后，人工改变窗口、特征或模型，再仍把修改后的结果当作未触碰测试。规则性更新与事后研究决策必须分开记录。

::: {.takeaway}
**本节落点：**需要被隔离的不是所有后续学习，而是未在评估前定义、却受测试结果启发的选择行为。每个预测都应能追溯到当时已确定的训练与选择规则。
:::

# 9. Horizon、重训频率、测试块：不要混用三个时间尺度 {#horizons}

## 9.1 测试一个月，不等于预测一个月

你可以在整个月的每个交易日都预测未来一天，并每天接收新特征，但只在月初拟合一次参数。这种测试的 horizon 始终是一天。

另一种任务是在月初一次性预测未来整个月的路径。这时第 20 天的预测不能使用第 19 天已经实现的真实值。把它塞进 lag feature，就是用一步滚动任务替代了固定起点多步任务。[R14](#r14)

**参数更新**、**状态更新**与**特征更新**也不同。冻结模型参数不等于冻结当时可观测的特征；Kalman filtering 可以顺序吸收截至当前时刻的观测，但用后续观测做 smoothing 后的历史 state 不可直接充当实时特征。

## 9.2 多 horizon 数据必须按 origin 切分

如果把 $(X_t,h,Y_{t,h})$ 展开成多行，同一个 $t$ 的 1 小时、24 小时、3 天标签不应随意分散到不同 folds。首先按预测 origin 的日历区间划分，再分别检查每个 horizon 标签的 availability。

一个样本的短期标签已成熟、长期标签还没成熟，有两种明确做法：

**保守整行规则：**所有输出都成熟后才训练该 origin。简单，但牺牲短 horizon 数据。

**逐输出 mask：**只让已成熟目标进入对应损失，未成熟目标不参与梯度与调参。实现更复杂，必须确保 shared representation、归一化和缺失处理都没有读取未成熟标签。

报告各 horizon 的独立 OOS 分数，再给预先定义的加权总分。不能靠大量容易的短 horizon 把真正关心的 next-day 改善稀释掉。

# 10. 不规则事件与 panel：splitter 不会替你定义总体 {#irregular}

## 10.1 “后 1000 条”与“后一天”是不同任务

交易活跃时 1000 条记录可能只覆盖几分钟，稀疏时可能跨越多天。按行数切出的等大 folds，会把交易活跃度混入时间跨度。对日历 horizon 任务，应先在 calendar time 上确定训练和验证边界，再取其中的事件记录。

记录数不同的 calendar blocks 完全可以评估；但汇总权重决定结果偏向高活跃还是低活跃时期。等时间长度也不保证难度相同，季节、节假日和 regime 仍需分层看。

## 10.2 未来窗口没有事件怎么办？

设 $M_t=1$ 表示预先规定的未来窗口里有至少一次事件。如果只保留这些行，那么测试指标趋向的是

$$
\mathbb E\!\left[\ell(Y_t,\widehat Y_t)\mid M_t=1\right],
$$

而不是对所有发起时刻的风险。只在有事件时才定义标签未必错误，但必须说明业务目标就是这一条件总体，并同时报告覆盖率与等待时间分布。

“24 小时附近最近的一个有事件窗口”也不是固定 24 小时标签。窗口的位置本身由未来事件决定，实际 horizon 是随机的。设它为 $H_t$；预测目标通常是尚未知道 $H_t$ 时的混合分布，而不是事后知道 $H_t$ 后的条件分布：

$$
F(y\mid\mathcal F_t)
=\int F(y\mid\mathcal F_t,H_t=h)\,
\mathrm dP(h\mid\mathcal F_t).
$$

这是全概率公式，而不是在建议必须显式估计这个积分。它说明：**用实际未来 horizon 当训练与测试输入，评估的是有额外信息的任务。**此外，混合分布的分位数一般不等于各条件分位数的加权平均。

## 10.3 观察结束不等于“没有事件”

数据只截至某个日期，而最后几天的目标窗口还没完整覆盖，这属于右端未成熟/未观察完整。应把这些 origins 标记为不可评分，或者延后评估，而不能把它们误标成“无事件”或随意删除后不报告。

推荐按 origin cohort 汇报：总预测次数、完整观察窗口数、有定义标签数、已成熟标签数、评分数、实际 horizon 分布。这样才能区分模型问题与标签可获得性变化。

## 10.4 多对象面板：时间与 entity 是两条不同泛化轴

| 部署问题 | 主要划分要求 | 允许什么？ |
|---|---|---|
| 已知对象的未来 | 所有对象按共同日历 cutoff 切分 | 使用同一对象过去的数据 |
| 未见对象的未来 | 时间切分 + entity / issuer holdout | 使用其他对象的已知历史 |
| 同时刻横截面插值 | 按任务定义横截面可观测信息 | 不能把它冒充未来预测 |

如果同一天不同债券共享市场冲击，随机把某些债券放训练、另一些放验证，通常不是对“下一天”任务的干净测试。反过来，预测已知对象未来时，要求训练和测试从不出现相同对象也没有必要，它把任务改成了 cold start。

同一时刻已经公布的其它对象信息可作为特征；同一天尚未发生的结果不可。关键仍是当时的信息集，而不是简单的“跨对象信息一律禁止”。

::: {.takeaway}
**本节落点：**不规则数据至少要同时固定 calendar horizon、event-selection rule 和 aggregation measure。切分正确，只能保证你正确验证了某个任务，不能保证这个任务就是最初想预测的那个。
:::

# 11. MAE、quantile 与加权：把目标和 score 对齐 {#scores}

## 11.1 MAE 为什么对应条件中位数？

固定 $X=x$，设条件分布函数为 $F_x$。在连续分布、可交换微分与积分的条件下，令

$$
L(q)=\mathbb E[|Y-q|\mid X=x].
$$

把 $Y\le q$ 与 $Y>q$ 两部分分开求导：

$$
L'(q)=\mathbb P(Y<q\mid x)-\mathbb P(Y>q\mid x)
=2F_x(q)-1.
$$

最优点满足 $F_x(q)=1/2$。有原子时用次梯度，条件变成 $F_x(q-)\le1/2\le F_x(q)$。因此 MAE 检验的是对所定义标签的中位数预测表现；它不是直接测量某个不可观测 latent fair value 的误差。

## 11.2 一般分位数使用 pinball loss

对 $\tau\in(0,1)$，定义

$$
\rho_\tau(u)=u\bigl(\tau-\mathbf1\{u<0\}\bigr),\qquad u=Y-q.
$$

同样求导得到

$$
\frac{\partial}{\partial q}\mathbb E[\rho_\tau(Y-q)\mid X=x]
=F_x(q)-\tau.
$$

最优解就是 $\tau$ 条件分位数。特别地，$\rho_{0.5}(u)=|u|/2$，因此中位数 pinball 与 MAE 对模型的排序相同。分布预测的评分思想可进一步参见 proper scoring rules 文献。[R15](#r15)

**只看 coverage 不够。**例如 90% 分位数可以在全样本达到约 90% 命中率，却在某些状态持续偏高、另一些状态持续偏低。应同时看 pinball、分时段/状态的 hit rate，以及极端损失。

## 11.3 一个 noisy label 能评估 quantile 吗？

可以。无需每个 $X$ 都有很多次重复目标观测；把恰当 scoring loss 在足够多的可比、依赖受控的 OOS 观测上平均，就能比较预测规则。但单点的 $|Y-q|$ 不是“预测分位数与真实分位数之差”。

对条件真分位数 $q_\tau$，由上一式积分，得到本文使用的一个有用恒等式：

$$
\mathbb E[\rho_\tau(Y-q)-\rho_\tau(Y-q_\tau)\mid x]
=\int_{q_\tau}^{q}\bigl(F_x(u)-\tau\bigr)\,\mathrm du\ge0.
$$

若密度在 $q_\tau$ 附近连续且正，则小偏差时约为

$$
\frac12 f_x(q_\tau)(q-q_\tau)^2.
$$

这解释了为什么 pinball excess risk 能评估 quantile，但其数值与 quantile 位置误差之间还取决于局部密度。本文不把该局部近似当作有限样本置信区间。

## 11.4 Fold average、row average、day average 不相等

一天有 100 条预测，绝对误差均为 1；另一天只有一条，绝对误差为 10。则

$$
\mathrm{MAE}_{\mathrm{row}}=\frac{100+10}{101}\approx1.089,
\qquad
\mathrm{MAE}_{\mathrm{day}}=\frac{1+10}{2}=5.5.
$$

两个数字都计算正确，只是总体不同。Fold 分数等权相加，只有在合适条件下才等于把所有 OOS 行汇总后的 MAE；当 fold 的评分样本量不同，应明确选择哪一种。

对 panel 可增加 entity-macro 指标；对事件数据可同时报告 row-weighted 与 day-weighted。主指标必须事先确定，不应在结果出来后挑最好看的口径。

## 11.5 给 micro / noisy observations 降权，会改变什么？

如果权重只是 $w(X)>0$，在固定 $X=x$ 下，它只把 conditional risk 乘以常数，不改变理想条件分位数。但在受限模型类、有限样本和整体训练中，它仍会改变拟合的关注区域。

如果权重还依赖 $Y$ 或与 $Y$ 有关的未来特征 $Z$，则加权最优解对应的是 tilted conditional distribution。例如

$$
F_w(y\mid x)
=\frac{\mathbb E[w(X,Y,Z)\mathbf1\{Y\le y\}\mid X=x]}
{\mathbb E[w(X,Y,Z)\mid X=x]}.
$$

因此“按未来成交量加权”可以是一个明确的业务评价选择，但不是不改变目标的纯降噪操作；更不能因此把未来成交量直接当作当前可知输入。

# 12. OOS 改善到底可靠吗？ {#uncertainty}

## 12.1 对每个相同事件做配对比较

比较模型 A 与基准 B，在相同 origins、相同标签版本、相同 horizon、相同权重上计算

$$
d_i=\ell(Y_i,\widehat Y_i^A)-\ell(Y_i,\widehat Y_i^B).
$$

$\overline d<0$ 表示 A 更好。比起分别对两个 MAE 做区间，直接分析配对的 $d_i$ 可以消除一部分共同难度变化。不同模型缺失预测时，应同时报告完整 coverage 与共同样本上的配对结果，避免只在容易样本上比较。

Diebold–Mariano 的核心就是用 loss differential 比较预测准确性，而不是要求使用平方损失。[R16](#r16)

## 12.2 自相关为什么使朴素标准误失真？

先考虑等权、弱平稳的损失差序列，记 $\gamma_k=\operatorname{Cov}(d_t,d_{t-k})$。直接展开方差：

$$
\begin{aligned}
\operatorname{Var}(\overline d)
&=\frac1{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\operatorname{Cov}(d_i,d_j)\\
&=\frac1n\left[\gamma_0+2\sum_{k=1}^{n-1}
\left(1-\frac{k}{n}\right)\gamma_k\right].
\end{aligned}
$$

朴素 IID 标准误只保留 $\gamma_0/n$。当自协方差累计为正，区间会偏窄；负相关也可能使朴素区间偏宽，因此不能不加条件地说依赖一定降低有效样本量。

若相关可求和，一个解释性近似是

$$
n_{\mathrm{eff}}\approx
\frac{n}{1+2\sum_{k\ge1}\rho_k},\qquad \rho_k=\gamma_k/\gamma_0.
$$

这针对当前的 loss / loss-difference 序列，不是从原始价格的 ACF 直接算出来的万能样本量。

## 12.3 HAC 与 block bootstrap 的正确位置

Newey–West 型估计用有限滞后及 Bartlett 权重估计长期方差：

$$
\widehat\Omega=\widehat\gamma_0+
2\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)\widehat\gamma_k,
\qquad
\operatorname{SE}(\overline d)\approx\sqrt{\widehat\Omega/n}.
$$

这里 $\widehat\gamma_k=n^{-1}\sum_{t=k+1}^{n}(d_t-\overline d)(d_{t-k}-\overline d)$。一致性需要适当弱依赖、矩条件和 bandwidth 条件；任意选个 $L$ 不会自动得到正确 coverage。[R17](#r17)

Block bootstrap 则成段重抽样，以保留部分局部依赖；stationary bootstrap 使用随机块长度。[R18](#r18) 对固定 OOS 预测表做 bootstrap，主要描述这段评估记录上的抽样不确定性，**不是完整训练与超参数选择的不确定性**。后者需要另行设计重拟合或程序级重采样。

对不规则 panel，一个实用起点是先构造固定日历频率的配对损失汇总，再按连续时间块抽样，保留同块中的横截面共同冲击。需要明确无事件日期、非交易日和权重的处理；不能把事件间隔不等的第 1、2、3 条记录直接解释为等长时间 lag。

## 12.4 为什么不能把 K 个 fold 当 K 次独立实验？

Expanding folds 的训练集高度重叠；测试期可能共享目标增量和状态；配置也是共同选择的。因此“fold 标准差除以 $\sqrt K$”不是可靠的通用标准误。

同样，CPCV 的数百组合也不是数百次独立市场实验。依赖校正并不能撤销多次研究选择带来的 winner's curse。

## 12.5 还有哪些边界？

在强非平稳、小样本、嵌套模型、频繁调参或学习程序变化下，标准渐近检验可能不适用。Giacomini–White 提供了以 forecasting method 和条件预测能力为中心的另一条文献线，但也有自身假设，不能当作所有场景的自动修正器。[R19](#r19)

实际报告至少应给出总体配对改善、按时间分层的改善、时间相关性诊断、多个合理 bandwidth/block length 的敏感性，以及经济上是否足够重要。统计显著与业务有用不是一回事。

# 13. 三个已经运行的合成实验 {#experiments}

::: {.callout}
**实验边界。**以下为本文原创合成演示，每个实验运行 200 个固定种子的重复；不是对任何真实市场或论文结果的复现。环境为 NumPy 2.3.5；生成与估计代码见附录 B。表中的“±”是跨重复均值的 Monte Carlo 标准误，不是单次 CV 的置信区间。未根据最终随机结果更改 seed。
:::

## 13.1 实验 A：稳定 AR(1)，随机 CV 并没有必然崩溃

生成 $S_{t+1}=0.8S_t+\varepsilon_{t+1}$，$\varepsilon\sim N(0,1)$；以平稳分布初始化并 burn-in 300 步。使用 1200 个历史预测对，未来 400 对留作验证；用带截距 OLS 从当前值预测下一步。

随机 5-fold 使用 960 条训练记录；expanding 在 origins 400、600、800、1000 各预测之后 200 条。未来块用全部 1200 条训练参数，逐次使用当时已知的 lag，不做块内重拟合。离散步内约定先接收当前 $S_c$（使前一预测对的标签成熟），再无计算延迟地拟合并预测下一步；训练 cutoff 位于这一接收动作之后。这不是把尚未接收的同刻标签偷偷纳入附录的严格截止规则。

| 评估方式 | MAE 均值 ± MCSE |
|---|---:|
| 随机 5-fold | **0.7995 ± 0.0013** |
| Expanding CV | **0.7998 ± 0.0015** |
| 随后的 400 个 origins | **0.7966 ± 0.0021** |
| 已知真实参数的理论 MAE | $\sqrt{2/\pi}=0.7979$ |

<figure class="bar-chart">
<h3>实验 A · 不同评估的 MAE</h3>
<div class="bar-line"><span>随机 5-fold</span><div class="bar-rail"><div class="bar-fill" style="--w:94.06%"></div></div><b>0.7995</b></div>
<div class="bar-line alt"><span>Expanding CV</span><div class="bar-rail"><div class="bar-fill" style="--w:94.09%"></div></div><b>0.7998</b></div>
<div class="bar-line"><span>随后未来块</span><div class="bar-rail"><div class="bar-fill" style="--w:93.72%"></div></div><b>0.7966</b></div>
<figcaption>各条均从 0 起；横轴共同上限为 0.85。样本量和起点并非完全匹配，不能据此给 splitter 排名。</figcaption>
</figure>

这里随机 CV 与未来误差接近，符合稳定、正确设定的小模型直觉。注意训练样本量不同，所以这不是“严格相同训练预算下 splitter 优劣”的实验，也不验证任意学习器的定理。

## 13.2 实验 B：机制突变时，随机混合平均回答了另一个问题

生成可在 origin 观察的 $X_t\sim N(0,1)$，以及

$$
Y_t=\beta_t X_t+\varepsilon_t,\qquad
\beta_t=\begin{cases}1,&t<800,\\-1,&t\ge800,\end{cases}
\quad \varepsilon_t\sim N(0,0.3^2).
$$

这里每个 $Y_t$ 在下一离散步前可用于更新；没有额外报告延迟。仍用前 1200 条做开发，后 400 条做未来块。近期 forward CV 取 origins 900、1000、1100，每次评估后 100 条；rolling 政策仅训练最近 200 条。

| 协议 | MAE 均值 ± MCSE |
|---|---:|
| 全历史池上的随机 5-fold | **0.7539 ± 0.0016** |
| 近期 forward / expanding | **1.3109 ± 0.0037** |
| 近期 forward / rolling 200 | **0.4392 ± 0.0025** |
| 最终未来块 / expanding | **1.0830 ± 0.0035** |
| 最终未来块 / rolling 200 | **0.2397 ± 0.0007** |

<figure class="bar-chart">
<h3>实验 B · 机制突变后的错位</h3>
<div class="bar-line"><span>随机 5-fold</span><div class="bar-rail"><div class="bar-fill" style="--w:53.85%"></div></div><b>0.7539</b></div>
<div class="bar-line alt"><span>近期 expanding</span><div class="bar-rail"><div class="bar-fill" style="--w:93.64%"></div></div><b>1.3109</b></div>
<div class="bar-line"><span>近期 rolling 200</span><div class="bar-rail"><div class="bar-fill" style="--w:31.37%"></div></div><b>0.4392</b></div>
<div class="bar-line alt"><span>未来 expanding</span><div class="bar-rail"><div class="bar-fill" style="--w:77.36%"></div></div><b>1.0830</b></div>
<div class="bar-line"><span>未来 rolling 200</span><div class="bar-rail"><div class="bar-fill" style="--w:17.12%"></div></div><b>0.2397</b></div>
<figcaption>各条均从 0 起；横轴共同上限为 1.4。不同训练政策必须分别对照自己的未来块，而不是混为一个模型。</figcaption>
</figure>

随机 CV 的约 0.75 不是对最新 regime 中 expanding 模型约 1.08 的良好估计。近期 forward 能暴露这一问题，并显示遗忘旧数据的价值。

**但不要只摘取自己喜欢的结论：**近期 rolling CV 的约 0.44 也明显高于其最终约 0.24，因为第一个近期训练窗口仍混入旧 regime。正确时间顺序并不消除时间位置和训练组成差异。此例只说明这个 DGP 下 rolling 更合适，不说明 rolling 总优于 expanding。

## 13.3 实验 C：重叠标签没有让平均 MAE 自动失效，却让 IID SE 严重偏小

设独立 $\varepsilon_t\sim N(0,1)$，目标为相邻 24 个未来增量之和：

$$
Y_t=\sum_{j=1}^{24}\varepsilon_{t+j},\qquad\widehat Y_t=0.
$$

没有训练步骤，因此这里**不存在训练泄漏**。相邻目标满足

$$
\operatorname{Corr}(Y_t,Y_{t+k})=\frac{24-k}{24},\quad 0\le k<24.
$$

绝对误差的相关函数不等于上式，但仍有明显依赖。每次生成 2000 个 loss observations，比较 200 次重复中 MAE 的真实波动与各种 SE 估计。

| 数量 | 数值 |
|---|---:|
| 理论 MAE | **3.9088** |
| 合成平均 MAE | **3.8802 ± 0.0189** |
| 200 次 MAE 的经验标准差 | **0.2668** |
| 平均 IID SE | **0.0652** |
| 平均 HAC SE，L=24 | **0.2191** |
| 平均 HAC SE，L=48 | **0.2340** |
| 平均 HAC SE，L=96 | **0.2382** |

<figure class="bar-chart">
<h3>实验 C · 平均标准误与跨重复经验波动</h3>
<div class="bar-line"><span>IID SE</span><div class="bar-rail"><div class="bar-fill" style="--w:23.29%"></div></div><b>0.0652</b></div>
<div class="bar-line alt"><span>HAC L=24</span><div class="bar-rail"><div class="bar-fill" style="--w:78.25%"></div></div><b>0.2191</b></div>
<div class="bar-line"><span>HAC L=48</span><div class="bar-rail"><div class="bar-fill" style="--w:83.57%"></div></div><b>0.2340</b></div>
<div class="bar-line alt"><span>HAC L=96</span><div class="bar-rail"><div class="bar-fill" style="--w:85.07%"></div></div><b>0.2382</b></div>
<div class="bar-line"><span>MAE 的经验 SD</span><div class="bar-rail"><div class="bar-fill" style="--w:95.29%"></div></div><b>0.2668</b></div>
<figcaption>各条均从 0 起；横轴共同上限为 0.28。经验 SD 也由有限的 200 次重复估计，不是精确已知真值。</figcaption>
</figure>

IID SE 只有经验波动的约四分之一。HAC 明显更接近，但这些有限 bandwidth 的平均估计仍偏低；不能据此宣称其区间已经精确校准。本实验也没运行 coverage study。正确结论是：**重叠需要处理不确定性，不代表应该一律删掉所有相邻 OOS 预测。**

# 14. 文献地图：经典结论与新问题各自负责什么？ {#literature}

## 14.1 不是所有论文都在研究同一个 estimand

| 文献线 | 首要阅读 | 本文取走的内容 | 不应过度解释成 |
|---|---|---|---|
| 顺序预测与 rolling origins | Dawid；FPP [R2](#r2) [R1](#r1) | 先预测、后观察；滚动起点 | 不需要关心 label delay |
| CV 与模型选择理论 | Arlot–Celisse [R20](#r20) | 风险估计、选择目标与假设区分 | 任何 K 都给无偏最终风险 |
| 自回归上的随机 CV | Bergmeir–Hyndman–Koo [R9](#r9) | 在受限条件下并非禁用随机 CV | 所有金融序列都可 shuffle |
| 依赖数据 block CV | Racine；Zheng [R7](#r7) [R8](#r8) | 缓冲与测试块的统计动机 | 已证明一般意义的独立 / 一致 |
| 非平稳实证与评估实践 | Cerqueira；Hewamalage [R10](#r10) [R5](#r5) | 时间外推、指标和统计测试的整体流程 | 某个方法在所有 DGP 必胜 |
| 金融标签重叠与选择 | López de Prado；Bailey [R11](#r11) [R13](#r13) | Purging、组合回测、选择偏差 | 多路径等于多份独立未来 |
| 模型比较与不确定性 | DM、NW、Politis–Romano、GW [R16](#r16)–[R19](#r19) | 配对损失与时间相关 | 任意数据都可套标准 p 值 |
| 现代 forecasting benchmark | TFB、GIFT-Eval、fev-bench [R21](#r21)–[R23](#r23) | 统一任务协议、协变量和跨数据集评估 | leaderboard 排名就是本地未来收益 |

Arlot 与 Celisse 的 survey 强调区分理论与实证、以及不同模型选择目标。它不是时间序列工程手册，但有助于避免把“预测效果好”与“选中最小真实模型”混为同一个一致性结论。[R20](#r20)

## 14.2 2024–2026：foundation models 把泄漏边界推到了预训练

TFB（2024）关注数据域覆盖、基线多样性以及公平统一的评估 pipeline。其主要价值不是一个新的 splitter，而是避免不同方法在不同流程下比较。[R21](#r21)

GIFT-Eval（2024）面向通用、尤其 zero-shot 的时间序列预测评估，并提供预训练/评估数据隔离的设计。对于预训练模型，**你本地测试集没有用于 fine-tuning，并不等于它没进入过预训练**。[R22](#r22)

fev-bench（2025，本文查阅至 2026-06 的 v4）扩展了对实际预测任务和协变量的关注。评价模型时，必须区分未来已知协变量、仅过去可观测协变量和待预测量，而不能只比较一列 target 的误差。[R23](#r23)

Meyer 等人关于 time-series foundation model evaluation 的预印本进一步讨论未知预训练语料及其 information leakage 风险；本文采用 2026-02 的 v3 标题与版本，不把预印本的全部主张当作共识定理。[R24](#r24)

对研究流水线的直接含义是：记录 checkpoint、预训练数据声明和可核查时间覆盖，区分 zero-shot / fine-tuned 设置，并对潜在数据重叠给出限制说明。新增一个本地 `TimeSeriesSplit` 无法审计黑箱预训练。

# 15. 一套能执行的默认方案，以及它不能保证的事 {#protocol}

## 15.1 面向 next-day、未来窗口、MAE/quantile 的起始协议

以下是一个**可调整的设计模板，不是普遍最优参数**。数据总长度、机制变化和真实部署频率决定具体数值。

| 决策 | 起始做法 | 必须预先记录 |
|---|---|---|
| 目标 | 固定 origin、固定窗口/事件选择规则 | 无事件与右端未成熟如何处理 |
| 信息 | point-in-time 特征与逐样本 availability | vintage、发布/摄入延迟、同刻边界 |
| 开发验证 | 多个 calendar-time forward blocks | 模型上线时刻、训练 cutoff、块长度 |
| 训练历史 | expanding 与少数 rolling windows 比较 | 窗口候选只在开发区间选择 |
| 更新 | 按真实线上频率回放 | 参数、状态、特征各自何时更新 |
| 选择 | 内层 forward 调参 / early stopping | 外层不参与选择 |
| 主指标 | 中位数 MAE；其它分位数 pinball | row/day/entity 权重 |
| 基准 | 简单、可部署、同信息集的基准 | delta target 的零变化基准仅在适用时使用 |
| 比较 | 同 origin 的 loss differential | coverage 与共同样本数量 |
| 不确定性 | 时间分层 + 配对块方法 / HAC 敏感性 | 不把 folds 当独立重复 |
| 最终测试 | 留下最新一段未用于研究决策的时期 | 允许的规则性重训事先冻结 |
| 上线后 | 保存 immutable prediction log | 数据版本、model ID、截止时刻、预测时间 |

一个例子可以是按月 outer test blocks、每周重训、每个 origin 预测未来一天；内部再用若干较早月份选择训练窗口和超参数。这里“月、周、天”分别代表评估块、重训频率、forecast horizon，不能用一个 `test_size` 代替三者。

## 15.2 保存的不应只是折均值，而是预测账本

每一条 OOS 记录建议包括 `origin_id`、entity、prediction time、forecast horizon / 实际标签窗口、model ID、训练 cutoff、label available time、预测、目标、loss、权重、fold ID、标签与特征版本。公开报告只需展示字段契约；真实数据及内部路径留在私有系统。

随后任何 MAE、按日曲线、paired improvement、coverage 与统计推断，都从同一份不可变预测账本派生。这样才能发现“两份图实际上使用了不同评分样本”的问题。

## 15.3 不存在能无条件保证未来的 CV

本文给出一个简单的不可区分论证：设两个世界在截至今天 $T$ 的全部联合分布完全相同，但 $T$ 之后一个保持关系 $Y=X+\varepsilon$，另一个变成 $Y=-X+\varepsilon$。任何只读取历史的 CV 程序在这两个世界给出相同结果，却不可能同时正确预测两者未来风险。

所以任何未来泛化结论都需要某种稳定性、可转移性、机制约束或未来情景假设。Nested、purging、block bootstrap 都有价值，但它们解决的是特定问题，不是对不可知未来的万能保险。

## 15.4 哪些问题值得继续研究？

**信息契约可验证性。**把特征、标签、发布延迟和修订版本组成可审计的依赖图，自动判断每个历史模型是否可部署；研究固定 gap 相比精确 availability 的样本效率损失。

**随机事件时间下的 quantile evaluation。**区分 fixed-calendar 风险、event-conditioned 风险与随机 horizon 混合目标，再研究延迟反馈、窗口缺失与适当评分的联合影响。这一段是研究问题的组织方式，不是宣称尚无相关文献或已经证明新定理。

**非平稳下的评估与窗口选择。**不仅比较模型，还比较评估协议对未来风险的估计误差、选择 regret、适应延迟与不确定性校准。要用可控 DGP 与多个公开数据机制支撑，不能只在一个数据集上找最漂亮的切分。

::: {.takeaway}
**全文收束：**一个可信的 OOS 结论，应同时回答：任务是否一致、当时信息是否合法、选择过程是否隔离、平均对象是否明确、依赖与漂移是否被诚实呈现。正确的 splitter 只是其中一个部件。
:::

# 附录 A. 按真实时间和标签成熟度切分 {#code}

下面的实现只负责**索引与样本 eligibility**。它不会神奇验证原始特征的 PIT 正确性，不会执行嵌套调参，也不会决定无标签样本对应的统计总体。评估块内默认冻结参数；要模拟周内重训，应在真实重训时刻再生成 blocks。

```python
from dataclasses import dataclass
from typing import Iterator
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class CalendarFold:
    fit_time: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def availability_splits(
    frame: pd.DataFrame,
    folds: list[CalendarFold],
    *,
    train_window: str | None = None,
    extra_gap: str = "0h",
    min_train: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Return positional indices; use frame.iloc, not frame.loc.

    Required columns: prediction_time, label_available_at.
    All timestamps and fold boundaries must be timezone-aware UTC.
    Missing label_available_at is allowed; that row is not trainable.
    Validation includes all origins, including not-yet-scoreable labels.
    extra_gap is an additional origin-time restriction, not a replacement
    for the sample-specific label availability check.
    """
    required = {"prediction_time", "label_available_at"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    if min_train < 1:
        raise ValueError("min_train must be positive")
    t = frame["prediction_time"]
    a = frame["label_available_at"]
    for name, col in [("prediction_time", t), ("label_available_at", a)]:
        if not isinstance(col.dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{name} must be timezone-aware datetime")
        if str(col.dt.tz) != "UTC":
            raise ValueError(f"{name} must use UTC")
    if t.isna().any() or not t.is_monotonic_increasing:
        raise ValueError("prediction_time must be nonmissing and sorted")
    if (a.notna() & (a < t)).any():
        raise ValueError("This future-target splitter requires a >= t")
    gap = pd.Timedelta(extra_gap)
    window = None if train_window is None else pd.Timedelta(train_window)
    if pd.isna(gap) or gap < pd.Timedelta(0):
        raise ValueError("extra_gap must be finite and nonnegative")
    if window is not None and (pd.isna(window) or window <= pd.Timedelta(0)):
        raise ValueError("train_window must be finite and positive")
    previous_end = None
    for fold in folds:
        c, v, e = map(pd.Timestamp,
                      (fold.fit_time, fold.valid_start, fold.valid_end))
        if any(pd.isna(z) or z.tzinfo is None or str(z.tzinfo) != "UTC"
               for z in (c, v, e)):
            raise ValueError("Fold times must be nonmissing UTC timestamps")
        if not c <= v < e:
            raise ValueError("Require fit_time <= valid_start < valid_end")
        if previous_end is not None and v < previous_end:
            raise ValueError("Validation origin blocks must not overlap")
        previous_end = e
        train = (t < c-gap) & a.notna() & (a < c)
        if window is not None:
            train &= t >= c-window
        valid = (t >= v) & (t < e)
        tr = np.flatnonzero(train.to_numpy())
        va = np.flatnonzero(valid.to_numpy())
        if len(tr) < min_train or len(va) == 0:
            raise ValueError("Insufficient training rows or empty validation")
        assert not np.intersect1d(tr, va).size
        assert (a.iloc[tr] < c).all()
        assert (t.iloc[tr] < c).all()
        yield tr, va
```

**边界自测。**以下例子特意让一个很早的 origin 拥有超长标签延迟，并让一个验证标签未知。后者仍应被预测，不能被 splitter 悄悄过滤。

```python
t = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
a = pd.Series(t + pd.Timedelta("2h"))
a.iloc[1] = t[10]             # old origin, delayed label
a.iloc[9] = pd.NaT            # prediction still belongs in validation
frame = pd.DataFrame({"prediction_time": t, "label_available_at": a})
fold = CalendarFold(t[8], t[8], t[11])
tr, va = next(availability_splits(frame, [fold]))
assert tr.tolist() == [0, 2, 3, 4, 5]
assert va.tolist() == [8, 9, 10]
# origin 6 has a == cutoff, excluded by the strict boundary convention.
# The validation block is half-open, so origin 11 is excluded.
```

**评分必须另做。**模型对 `va` 中所有 origins 发出预测后，再按评估数据截止时刻决定哪些标签已经可评分。报告所有被排除的原因；若想给完整 cohort 结论，应等待整个规定窗口成熟，而不是只算到达最快的标签。

# 附录 B. 合成实验完整复现 {#reproduce}

把以下代码保存为 `experiments.py`，安装 NumPy 后运行。输出 `results.json`。本节代码仅含独立生成的随机序列和基础 OLS；所有 forecast pairs 的生成方式、origin 范围、seed 与 bandwidth 都固定。它不声称是生产 CV 系统。

```python
"""Public synthetic demonstrations; no real or connected financial data.
Run: python experiments.py. Requires only NumPy. Seeds and DGP are fixed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPEATS = 200

def ols_predict(x, y, train, test):
    xtr = np.column_stack([np.ones(len(train)), x[train]])
    beta = np.linalg.lstsq(xtr, y[train], rcond=None)[0]
    return np.column_stack([np.ones(len(test)), x[test]]) @ beta

def mae(x, y, train, test):
    return float(np.mean(np.abs(y[test] - ols_predict(x, y, train, test))))

def random_cv(x, y, n, rng):
    chunks = np.array_split(rng.permutation(n), 5)
    all_rows = np.arange(n)
    return np.mean([mae(x, y, np.setdiff1d(all_rows, te), te) for te in chunks])

def forward_cv(x, y, origins, width, window=None):
    return np.mean([mae(x, y, np.arange(0 if window is None else max(0, c-window), c),
                          np.arange(c, c+width)) for c in origins])

def mean_se(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "mcse": float(a.std(ddof=1) / np.sqrt(len(a)))}

def hac_mean_se(z, bandwidth):
    u = z - z.mean()
    n = len(u)
    lrv = np.dot(u, u) / n
    for k in range(1, bandwidth + 1):
        lrv += 2 * (1-k/(bandwidth+1)) * np.dot(u[k:], u[:-k]) / n
    return float(np.sqrt(max(lrv, 0) / n))

def run():
    stationary = {k: [] for k in ['random_cv', 'expanding_cv', 'future']}
    drift = {k: [] for k in ['random_cv', 'recent_expanding_cv', 'recent_rolling_cv',
                             'future_expanding', 'future_rolling']}
    overlap = {k: [] for k in ['mae', 'iid_se', 'hac24', 'hac48', 'hac96']}
    for seed in range(REPEATS):
        rng = np.random.default_rng(10000 + seed)
        innovations = rng.normal(size=1901)
        series = np.empty(1901)
        series[0] = rng.normal(scale=1 / np.sqrt(1-0.8**2))
        for t in range(1, 1901):
            series[t] = 0.8 * series[t-1] + innovations[t]
        series = series[300:]
        x, y = series[:-1], series[1:]
        stationary['random_cv'].append(random_cv(x, y, 1200, rng))
        stationary['expanding_cv'].append(forward_cv(x, y, [400,600,800,1000], 200))
        stationary['future'].append(mae(x, y, np.arange(1200), np.arange(1200,1600)))
        rng = np.random.default_rng(20000 + seed)
        x = rng.normal(size=1600)
        slope = np.where(np.arange(1600) < 800, 1.0, -1.0)
        y = slope * x + rng.normal(scale=0.3, size=1600)
        drift['random_cv'].append(random_cv(x, y, 1200, rng))
        drift['recent_expanding_cv'].append(forward_cv(x, y, [900,1000,1100],100))
        drift['recent_rolling_cv'].append(forward_cv(x, y, [900,1000,1100],100,200))
        drift['future_expanding'].append(mae(x,y,np.arange(1200),np.arange(1200,1600)))
        drift['future_rolling'].append(mae(x,y,np.arange(1000,1200),np.arange(1200,1600)))
        rng = np.random.default_rng(30000 + seed)
        eps = rng.normal(size=2023)
        target = np.convolve(eps, np.ones(24), mode='valid')
        loss = np.abs(target)
        assert len(loss) == 2000
        overlap['mae'].append(float(loss.mean()))
        overlap['iid_se'].append(float(loss.std(ddof=1)/np.sqrt(2000)))
        for lag in (24,48,96):
            overlap[f'hac{lag}'].append(hac_mean_se(loss,lag))
    result = {
        'repeats': REPEATS, 'numpy_version': np.__version__,
        'stationary': {k:mean_se(v) for k,v in stationary.items()},
        'drift': {k:mean_se(v) for k,v in drift.items()},
        'overlap': {k:mean_se(v) for k,v in overlap.items()},
        'overlap_empirical_sd': float(np.std(overlap['mae'],ddof=1)),
        'oracle_stationary_mae': float(np.sqrt(2/np.pi)),
        'oracle_overlap_mae': float(np.sqrt(24)*np.sqrt(2/np.pi)),
    }
    Path(__file__).with_name('results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    run()
```

# 附录 C. 五个自测问题 {#exercises}

<details><summary>1. 验证特征回看 30 天，是否必须把训练和验证隔开 30 天？</summary>
<p>不必须。严格 forward 回放中，验证时读取已知历史是合法的。先检查训练标签成熟与特征可用性。如果用验证之后的记录训练，再检查后侧特征是否读到了验证结果；这是另一种问题。</p>
</details>

<details><summary>2. 24 小时目标、最多 2 小时发布延迟，gap=24 就够了吗？</summary>
<p>如果 gap 按小时计，仍可能不够；如果按记录条数计，更无法直接判断。固定上界例子需要结合 26 小时与严格边界约定，最准确的判断仍是每条记录的 label_available_at 是否早于 cutoff。</p>
</details>

<details><summary>3. 两个模型 MAE 相差 0.02，但测试集有一百万行，是否一定显著？</summary>
<p>不是。需要相同预测任务上的配对损失差、依赖结构和覆盖信息。一百万条高度聚集在少数时间段的事件记录，并不等于一百万次独立实验。还要考虑 0.02 在业务上是否有意义。</p>
</details>

<details><summary>4. 在最终测试月内，第二周训练使用第一周已经成熟的标签，是不是泄漏？</summary>
<p>若每周更新规则在测试前已固定，且标签在训练截止前真实可用，这是合法的 prequential 更新。根据第一周表现临时换特征、窗口或模型，则属于利用测试反馈做研究选择。</p>
</details>

<details><summary>5. 所有 folds 都按时间切了，能否使用实际未来事件间隔作为输入？</summary>
<p>不能把它称作原信息集下的预测。如果发起时刻不知道该间隔，加入它就是扩大条件信息集。可以把这种结果作为明确标注的事后条件 / oracle 实验，但必须另给真实可部署的信息集下的结果。</p>
</details>

# 参考文献与带问题的阅读顺序 {#references}

<div class="references">

<p id="r1"><strong>[R1]</strong> Hyndman, R. J. & Athanasopoulos, G. <em>Forecasting: Principles and Practice</em>, 3rd ed., §5.10, Time series cross-validation. <a href="https://otexts.com/fpp3/tscv.html">作者开放教材</a>。入门先读：rolling origin 的预测时点如何随训练集移动？</p>

<p id="r2"><strong>[R2]</strong> Dawid, A. P. (1984). Statistical theory: the prequential approach. <em>Journal of the Royal Statistical Society A</em>, 147(2), 278–290. <a href="https://academic.oup.com/jrsssa/article/147/2/278/7106293">期刊页面</a>。思考：评估对象为何可以是顺序预测，而不是参数估计？</p>

<p id="r3"><strong>[R3]</strong> Bates, S., Hastie, T. & Tibshirani, R. (2024). Cross-validation: What does it estimate and how well does it do it? <em>JASA</em>, 119(546), 1434–1445. <a href="https://doi.org/10.1080/01621459.2023.2197686">DOI</a> · <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/">开放全文</a>。在线发表年份为 2023；此处采用卷期年份。重点：最终模型的误差与学习程序的平均误差。</p>

<p id="r4"><strong>[R4]</strong> Cawley, G. C. & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. <em>JMLR</em>, 11, 2079–2107. <a href="https://jmlr.org/papers/v11/cawley10a.html">开放全文</a>。重点：验证过程本身如何被过拟合？</p>

<p id="r5"><strong>[R5]</strong> Hewamalage, H., Ackermann, K. & Bergmeir, C. (2023). Forecast evaluation for data scientists: Common pitfalls and best practices. <em>Data Mining and Knowledge Discovery</em>, 37, 788–832. <a href="https://doi.org/10.1007/s10618-022-00894-5">DOI</a> · <a href="https://arxiv.org/abs/2203.10716">作者预印本</a>。最接近本文主题的整体 survey；先读 partitioning 与 error measures，再读 statistical testing。</p>

<p id="r6"><strong>[R6]</strong> scikit-learn developers. Common pitfalls and recommended practices: Data leakage. <a href="https://scikit-learn.org/stable/common_pitfalls.html">官方文档</a>，访问于 2026-09-04。软件文档，不是理论保证。</p>

<p id="r7"><strong>[R7]</strong> Racine, J. (2000). Consistent cross-validatory model-selection for dependent data: hv-block cross-validation. <em>Journal of Econometrics</em>, 99(1), 39–61. <a href="https://doi.org/10.1016/S0304-4076(00)00030-0">DOI</a>。与 R8 一起读，区分方法、模型选择目标与理论论证。</p>

<p id="r8"><strong>[R8]</strong> Zheng, W. (2019). hv-Block cross validation is not a BIBD: a note on the paper by Jeff Racine (2000). <a href="https://arxiv.org/abs/1910.08904">arXiv:1910.08904</a>。预印本说明；不要把对一个证明环节的反例扩大成对所有 block 方法的否定。</p>

<p id="r9"><strong>[R9]</strong> Bergmeir, C., Hyndman, R. J. & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. <em>Computational Statistics & Data Analysis</em>, 120, 70–83. <a href="https://doi.org/10.1016/j.csda.2017.11.003">DOI</a> · <a href="https://robjhyndman.com/papers/cv-wp.pdf">作者 PDF</a>。重点核对纯自回归设定、误差与正则性假设，勿直接推广到任意 quantile/窗口目标。</p>

<p id="r10"><strong>[R10]</strong> Cerqueira, V., Torgo, L. & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. <em>Machine Learning</em>, 109, 1997–2028. <a href="https://doi.org/10.1007/s10994-020-05910-7">DOI</a> · <a href="https://arxiv.org/abs/1905.11744">作者预印本</a>。重点：平稳合成数据与非平稳实际序列为什么得到不同结论？</p>

<p id="r11"><strong>[R11]</strong> López de Prado, M. (2018). <em>Advances in Financial Machine Learning</em>. Wiley，Ch. 7 Cross-Validation in Finance；Ch. 12 Backtesting through Cross-Validation. <a href="https://www.wiley-vch.de/de/fachgebiete/finanzen-wirtschaft-recht/advances-in-financial-machine-learning-978-1-119-48208-6">出版社页面</a>。方法来源；本文区间与边界约定为明确化后的工程表述，不声称逐字复刻某个软件实现。</p>

<p id="r12"><strong>[R12]</strong> scikit-learn developers. TimeSeriesSplit. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html">官方 API</a>，访问于 2026-09-04。检查 gap、test_size、max_train_size 的单位，而不是只看类名。</p>

<p id="r13"><strong>[R13]</strong> Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2017). The probability of backtest overfitting. <em>Journal of Computational Finance</em>, 20(4), 39–69. <a href="https://scholarworks.wmich.edu/math_pubs/42/">作者机构记录</a> · <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253">早期作者版本</a>。重点区分 CSCV、CPCV、策略选择与 forecast error。</p>

<p id="r14"><strong>[R14]</strong> Hyndman, R. J. & Athanasopoulos, G. <em>FPP3</em>, §13.8, Forecasting on training and test sets. <a href="https://otexts.com/fpp3/training-test.html">作者开放教材</a>。对照 one-step 更新信息与 fixed-origin multi-step。
</p>

<p id="r15"><strong>[R15]</strong> Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. <em>JASA</em>, 102(477), 359–378. <a href="https://doi.org/10.1198/016214506000001437">DOI</a> · <a href="https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf">作者 PDF</a>。本文 pinball 导数与 excess-risk 积分在正文独立展开。
</p>

<p id="r16"><strong>[R16]</strong> Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. <em>Journal of Business & Economic Statistics</em>, 13(3), 253–263. <a href="https://doi.org/10.1080/07350015.1995.10524599">DOI</a>。从 loss differential 出发读，而不是只记住检验名称。</p>

<p id="r17"><strong>[R17]</strong> Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. <em>Econometrica</em>, 55(3), 703–708. <a href="https://www.jstor.org/stable/1913610">期刊记录</a> · <a href="https://www.nber.org/papers/t0055">1986 工作论文</a>。重点：长期方差、bandwidth 与一致性条件。</p>

<p id="r18"><strong>[R18]</strong> Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. <em>JASA</em>, 89(428), 1303–1313. <a href="https://doi.org/10.1080/01621459.1994.10476870">DOI</a>。思考：重抽的是损失序列，还是整个重新训练程序？</p>

<p id="r19"><strong>[R19]</strong> Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. <em>Econometrica</em>, 74(6), 1545–1578. <a href="https://doi.org/10.1111/j.1468-0262.2006.00718.x">DOI</a>。适合在配对平均损失比较之后继续读；注意估计窗口与检验框架假设。</p>

<p id="r20"><strong>[R20]</strong> Arlot, S. & Celisse, A. (2010). A survey of cross-validation procedures for model selection. <em>Statistics Surveys</em>, 4, 40–79. <a href="https://doi.org/10.1214/09-SS054">DOI</a> · <a href="https://arxiv.org/abs/0907.4728">作者预印本</a>。最后回到理论：你追求 risk minimization，还是 model identification？</p>

<p id="r21"><strong>[R21]</strong> Qiu, X. et al. (2024). TFB: Towards comprehensive and fair benchmarking of time series forecasting methods. <em>PVLDB</em>, 17(9), 2363–2377. <a href="https://arxiv.org/abs/2403.20150">论文</a>。关注统一评估 pipeline，不把 benchmark 排名外推为本地保证。</p>

<p id="r22"><strong>[R22]</strong> Aksu, T. et al. (2024). GIFT-Eval: A benchmark for general time series forecasting model evaluation. <a href="https://arxiv.org/abs/2410.10393">论文</a>。关注 zero-shot 和预训练数据隔离；此处采用首次预印本年份。</p>

<p id="r23"><strong>[R23]</strong> Shchur, O. et al. (2025/2026). fev-bench: A realistic benchmark for time series forecasting. <a href="https://arxiv.org/abs/2509.26468v4">arXiv:2509.26468v4</a>，2026-06-30 修订。关注真实任务、协变量与评价口径。</p>

<p id="r24"><strong>[R24]</strong> Meyer, M. et al. (2025/2026). Rethinking evaluation in the era of time series foundation models: (Un)known information leakage challenges. <a href="https://arxiv.org/abs/2510.13654v3">arXiv:2510.13654v3</a>，2026-02-25 修订。预印本；关注难以核实的预训练重叠，而不是将它当作新 CV 定理。</p>

</div>

<p class="footer-note">范围说明：本报告是围绕预测验证的教程式综述，而非 PRISMA 式系统综述；不声称穷尽所有 dependent-data CV 文献。检索与软件文档核查截至 2026-09-04。原始数据、标签定义及业务部署约束仍需要在各自项目中单独审计。</p>
