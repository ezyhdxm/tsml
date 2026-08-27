# 生产化实现与验证路线

# 最小数据契约

每个 training observation 至少需要以下逻辑字段。字段名可按实际 source 映射，但语义必须固定。

| 组别 | 必要信息 | 关键审计 |
|---|---|---|
| 标识 | bond、issuer、dealer、event/snapshot id | sibling rows 是否能被完整 group |
| 时间 | quote event time、observation time、label-availability time | timezone、latency、PIT cutoff |
| side | dealer bid/offer 或 client buy/sell | signed concession 方向是否统一 |
| quantity | 数值、单位、semantic type、minimum/maximum/incremental flag | missing 不等于 zero；up-to 不等于 point |
| quote | spread/price、firmness、validity interval | benchmark timestamp；stale quote |
| reference mid | leave-one-dealer-out 或 cross-fitted mid | 当前 quote 是否泄漏进 anchor |
| liquidity state | trailing volume/size、amount outstanding、active dealers、freshness | rolling window 是否 PIT |
| risk state | CS01/DV01、spread/rate volatility、jump/event state | unit 与 target space 一致 |
| dealer state | signed flow/inventory proxy、quote skew、capacity proxy | side convention、dealer anonymization stability |
| protocol | run/RFQ/voice/electronic、pre-event breadth | post-event fields 禁止用于 arrival prediction |
| labels | observed concession、availability/nonresponse | completed-trade selection |

<div class="note">
<strong>公开报告边界：</strong>实现时可以使用内部字段，但 public repo 只保留抽象 schema、数学和合成示例，不提交真实 dealer identifiers、公司数据、内部 host/path 或 proprietary model outputs。
</div>

# 目标构造

## Step 1：统一 spread convention

对所有 sides 定义 $Y>0$ 为客户更差：

$$
Y=a_s(S^{\mathrm{quote}}-M^{(-d)}).
$$

对 bid/offer 分别做 unit tests，使用合成例子确认 tighter/worse 的方向。

## Step 2：构造 PIT reference mid

优先级：

1. 同 snapshot leave-one-dealer-out robust consensus；
2. 若 dealer coverage 不足，使用只含过去信息的 fair-mid model；
3. 对 anchor uncertainty 产生权重或第二阶段 measurement-error sensitivity；
4. 对缺少可靠 mid 的 observation 不强行训练 quantity curve。

## Step 3：标注 quantity semantics

建立枚举：

```text
REQUESTED_POINT
FIRM_POINT
GOOD_UP_TO
MINIMUM_SIZE
INCREMENTAL_LADDER
CUMULATIVE_LADDER
UNSPECIFIED
MISSING_SOURCE
```

第一版 curve model只使用可解释为 point quantity 的 observations；`GOOD_UP_TO` 用于 availability/censoring model；ladder 单独转换成 cumulative execution curve。

## Step 4：availability target

对 grid node $q_k$ 定义

$$
A_k=1\{\text{quote/depth available at least to }q_k\}.
$$

训练 side-specific probability：

$$
\widehat p_k(X)
=P(A_k=1\mid X).
$$

若使用多个 nodes，应施加或后处理成

$$
\widehat p_{100k}
\ge
\widehat p_{250k}
\ge\cdots\ge
\widehat p_{5mm}.
$$

# Feature architecture

# Quantity-independent baseline $m(X)$

应主要描述：

- quantity-neutral fair-mid uncertainty；
- current market liquidity level；
- dealer side-specific aggressiveness；
- recent quote dynamics；
- peer consensus/dispersion；
- benchmark-adjusted spread state；
- time-of-day、freshness、event state。

baseline 不应通过明显的 quantity proxies 偷渡 curve，例如把 displayed max size、quantity bucket statistics 或 post-RFQ response count 放入 $m(X)$ 却声称 quantity 已被隔离。

# Liquidity scale $Q^\star(X)$

第一批候选：

$$
\begin{aligned}
&\log(\text{trailing median trade size}),\\
&\log(1+\text{trailing volume}),\\
&\log(\text{amount outstanding}),\\
&\log(1+\text{active dealer count}),\\
&\text{quote/trade freshness},\\
&\text{issue age and structural complexity}.
\end{aligned}
$$

应报告每个 scale proxy 对 conditional-curve alignment 的贡献，而不只看 final MAE。

# Risk amplitude $A(X)$

第一批候选：

$$
\begin{aligned}
&CS01\times \sigma_{\Delta s},\\
&DV01\times \sigma_{\Delta r},\\
&\sqrt{\tau_{\mathrm{unwind}}},\\
&\text{CDX/market stress},\\
&\text{jump/event indicator},\\
&\text{hedgeability/substitutability},\\
&\text{dealer capacity stress}.
\end{aligned}
$$

若 target 是 spread bps，应避免把 spread level 本身当作 risk amplitude 的唯一 proxy。

# Sparse shape modifiers

只允许机制明确、PIT 可得的 variables：

- information intensity；
- side × inventory pressure × capacity regime；
- protocol × expected dealer participation；
- stress tail indicator；
- hedge-capacity exhaustion indicator。

static sector/rating/tenor 可用于分层审计，但不应自动生成大量 free-form interaction curves。

# QuantityCurveEngine 的输出契约

对每个 state $X$，在 grid

$$
\mathcal G=
\{100k,250k,500k,1mm,2mm,5mm\}
$$

输出：

```text
fair_bid_spread_100k ... fair_bid_spread_5mm
fair_offer_spread_100k ... fair_offer_spread_5mm
availability_bid_100k ... availability_bid_5mm
availability_offer_100k ... availability_offer_5mm
uncertainty_100k ... uncertainty_5mm
support_score_100k ... support_score_5mm
interpolation_flag / extrapolation_flag
```

并产生 summary features：

```text
normalized_bid_spread_1mm
normalized_offer_spread_1mm
size_adjustment_observed_to_1mm
small_size_slope
institutional_size_slope
block_tail_slope
curve_curvature
capacity_scale_q_star
risk_amplitude_a
quantity_semantic_quality
```

# normalization 公式

设实际 quote quantity 为 $q$，reference quantity 为 $q_0$。对 signed concession：

$$
\widehat Y^{\mathrm{norm}}(q_0)
=
Y(q)-
\left\{
\widehat\mu(X,q)-\widehat\mu(X,q_0)
\right\}.
$$

转换回 quote spread：

$$
\widehat S^{\mathrm{norm}}(q_0)
=
M+
a_s\widehat Y^{\mathrm{norm}}(q_0).
$$

这个写法适用于 additive、scale–amplitude 和 sparse-interaction models；区别只在 conditional contrast 的估计方式。

# 基于 normalized quotes 的下游 features

所有 dealers 被映射到同一 $q_0$ 后，再计算：

- normalized median/trimmed mean consensus；
- normalized best bid/offer；
- normalized dispersion；
- dealer 相对 normalized consensus 的 deviation；
- 有多少比例 dealers 比 quantity-matched CPP/reference 更 aggressive；
- dealer breadth at requested quantity；
- curve slope/curvature disagreement across dealers；
- dealer max-size contraction / expansion signal。

原始 quantity 与 max-size dynamics 不应被删除：它们是 liquidity/capacity signal，只是不应继续污染 price consensus。

# 验证矩阵

## 预测层

| 维度 | 必报指标 |
|---|---|
| overall | OOT MAE/Huber、bias、calibration slope |
| quantity | 每个 grid node 与 observed bucket 的 bias/loss |
| side | bid/offer 或 client buy/sell 分开 |
| support | common support、tail extrapolation 分开 |
| liquidity | liquid/medium/illiquid buckets |
| risk | IG/HY、volatility、CS01 buckets |
| stress | normal/stress、month/quarter-end |
| dealer | active/cold-start、capacity proxy buckets |
| protocol | run/RFQ/electronic/voice，严格按可用 data |
| time | rolling weekly/monthly stability |

## 结构层

- $Q^\star$ 归一化后，liquidity-group curves 是否对齐；
- $A(X)$ 归一化后，risk-group amplitude 是否收敛；
- retained $r(X,Q)$ 是否通过 purified interaction constraints；
- turning point 与 tail slope 是否跨 folds 稳定；
- curve 是否出现不合理 crossing/oscillation；
- availability probability 是否随 quantity 单调下降。

## identification 层

- same-snapshot ladder difference；
- same dealer–bond–side short-window pair；
- client/dealer fixed-effect sensitivity；
- completed trade vs all inquiry/quote sample；
- overlap trimming sensitivity；
- quote semantic type sensitivity。

## 下游价值层

quantity normalization 的最终目的不是单独降低 quote-target MAE，而是改善后续决策。应比较：

- normalized dealer consensus 对 future trade/RFQ execution 的 calibration；
- CPP-relative dealer breadth feature 的稳定性；
- fair-value model 的 OOT residual bias by observed quantity；
- dealer ranking 在不同 displayed sizes 下是否更一致；
- quantity curve features 是否对 future liquidity/trade probability 提供增量信息。

# Failure modes 与诊断

## Failure 1：curve 只复制 quantity selection

**症状：** large size 在 liquid state 中预测极好，但 matched pairs 中差；client/dealer fixed effects 后 slope 大幅变化。

**处理：** 强化 overlap、fixed effects、same-state contrasts；将结果标为 predictive rather than counterfactual。

## Failure 2：reference mid 吸收当前 quote

**症状：** residual variance 异常小；dealer-specific curve shrink；换 leave-one-dealer-out mid 后结果大变。

**处理：** event-level cross-fitting；anchor uncertainty analysis。

## Failure 3：up-to size 被当 point size

**症状：** quote quantity 与 price 几乎无关系，但 availability 很强；displayed sizes 在少数 round lots 堆积。

**处理：** 转为 censoring/depth model，不进入 point-concession regression。

## Failure 4：black-box interaction 只来自 tail sparsity

**症状：** $5\text{mm}$ SHAP interaction 很大，common-support OOT improvement 为零或反向。

**处理：** support-aware weighting、extrapolation flag、低自由度 tail、simultaneous bands。

## Failure 5：quantity-neutral baseline 不 neutral

**症状：** $m(X)$ 使用 quantity proxies 或 post-event fields，additive curve 被压平。

**处理：** feature contract；ablation；orthogonalized basis；model audit。

## Failure 6：bid/offer side convention 错误

**症状：** 两边 curve 符号相反或 normalization widening/narrowing 方向不合理。

**处理：** 合成单元测试；在 signed concession space 统一，再映射回 spread。

## Failure 7：grid 节点被误当独立 targets

**症状：** 相邻 nodes 跳跃、crossing、2mm/5mm 不稳定。

**处理：** pooled basis model、log-Q smoothness、shared scale/amplitude；不要训练六个完全独立模型。

# 最小可行研究（MVP）

## Phase A：数据语义与 support

1. 给每条 quantity 标 semantic type；
2. 画 overall 与 conditional quantity distributions；
3. 统计每个 grid node 的 effective support；
4. 构造 PIT leave-one-dealer-out mid；
5. 建立 event groups 和 time folds。

## Phase B：additive curve

1. 用 log-Q piecewise-linear basis；
2. cross-fit $E[Y\mid X]$ 与 $E[B(Q)\mid X]$；
3. 估计 $h(Q)$ 与 simultaneous bands；
4. 检查 normalized residual by quantity；
5. 用 matched pairs 验证 quantity difference。

## Phase C：scale 与 amplitude

1. 比较三至五个 $Q^\star$ candidates；
2. 选择最能对齐 liquidity-group curves 的 scale；
3. 加入低维 $A(X)$；
4. 比较 $\mathcal M_0,\mathcal M_1,\mathcal M_2$ 的 OOT 与 structural metrics。

## Phase D：interaction challenge

1. 逐组检验 inventory、information、stress、competition、hedgeability；
2. 通过 global family test 后建立 $\mathcal M_3$；
3. 用 $\mathcal M_4$ 发现遗漏，但不直接部署；
4. common-support 与 matched-pair confirmation；
5. 确定最终 retained interactions。

## Phase E：productionization

1. curve + availability + support score API；
2. fixed reference quantity normalization；
3. normalized dealer consensus features；
4. weekly/monthly drift monitoring；
5. node-level and curve-level fallback；
6. model card 记录 semantic scope、extrapolation boundary 与 known failure modes。

# 推荐的优先结论

基于经济机制与现有文献，研究应以以下先验开始，但允许数据推翻：

1. **raw $Q$ 不可全局分离；relative $Q/Q^\star$ 更接近可分离。**
2. **大多数 rating/age/sector/maturity interaction 是 liquidity/risk proxy。**
3. **risk 与 stress 主要改变 amplitude；liquidity capacity 主要改变 horizontal scale。**
4. **正常 inventory state 主要改变 intercept，接近 capacity limit 才明显改变 quantity slope。**
5. **information intensity 与 quantity-dependent dealer participation 是最可能的 genuine shape interactions。**
6. **client/dealer composition 是 pooled size curve 的最大识别威胁之一。**
7. **displayed quote size 的语义可能比 functional form 更先决定结论。**

# 文献证据矩阵

| 文献 | 数据/对象 | 与本报告最相关的结论 | 对模型的含义 |
|---|---|---|---|
| Edwards, Harris & Piwowar [@edwards2007corporate] | TRACE corporate bonds | pooled transaction costs 随 trade size 下降 | 建立 size discount benchmark，但不能单独识别 counterfactual curve |
| Pinter, Wang & Zou [@pinter2024size] | nonanonymous government/corporate bond data | 控制 client identity 后 size discount 反转为 size penalty；information-intensive periods 更强 | 必须控制 client/dealer composition，优先检验 information interaction |
| Reichenbacher & Schuster [@reichenbacher2022size] | U.S. corporate-bond liquidity measures | liquidity measure 需要适配 trade size distribution | 支持 $Q/Q^\star$ 与 size-adjusted diagnostics |
| Bessembinder et al. [@bessembinder2018capital] | corporate-bond dealer intermediation | capital commitment 与 illiquidity/market structure 紧密相关 | dealer capacity 进入 $Q^\star$ 或 $A(X)$ |
| Goldberg & Nozawa [@goldberg2021liquidity] | yields + dealer positions | dealer inventory capacity 是 liquidity supply 与 asset prices 的驱动因素 | inventory/capital state 不能只做 additive feature |
| Goldstein & Hotchkiss [@goldstein2020providing] | dealer behavior in illiquid bonds | dealer activity、inventory 与 liquidity provision 异质 | dealer-specific state 与 side 需要审计 |
| Choi, Huh & Shin [@choi2024customer] | corporate-bond customer liquidity provision | 客户也可提供 liquidity；large trades 与 relationships/capacity 重要 | dealer-vs-customer liquidity role 与 relationship 影响 size curve |
| O’Hara & Zhou [@ohara2021electronic] | electronic corporate-bond trading | protocol 改变 execution、dealer risk-bearing 与 liquidity | protocol/competition 可能与 quantity 交互 |
| Hendershott et al. [@hendershott2026quote] | BondCliQ dealer runs + TRACE | indicative quotes 影响 order flow；quote competition 不完美；size 字段可能 nonbinding | 先审计 displayed-size semantic；不要只依赖 quote quantity |
| Kargar et al. [@kargar2026sequential] | platform inquiries/offers | OTC trade 是 sequential search，失败询价与后续 offers 有信息 | repeated inquiry 与 stopping selection 进入 identification |
| Jacobsen & Venkataraman [@jacobsen2025receiving] | large corporate-bond blocks | block distribution、receiver terms 与 information/inventory risk 相关 | large-size tail 需要单独的 information/capacity hypothesis |
| ICE size-adjusted pricing [@ice2026size] | industry evaluated pricing | 产业界明确提供 side- and size-adjusted values | 证明业务需求存在，但不是 specific estimator 的学术验证 |
| Robinson [@robinson1988root] | semiparametric regression | double residualization 识别 partially linear effect | residual correction 必须同时 orthogonalize quantity basis |
| Chernozhukov et al. [@chernozhukov2018double] | double/debiased ML | orthogonal score + cross-fitting 降低 nuisance bias | 适合 $m(X)$ 使用高容量 ML 的场景 |
| Hastie & Tibshirani [@hastie1993varying] | varying-coefficient models | coefficient/function 可随 modifier 变化 | 用低维 $Z_j f_j(Q)$ 建模 sparse interactions |
| Härdle & Mammen [@hardle1993comparing] | specification testing | 比较受限与 nonparametric regression fit | 用于 additive/scale model vs flexible challenger 的 global test |
| Hooker [@hooker2007generalized] | dependent-feature functional ANOVA | 依赖 features 下定义低阶 effects | quantity 与 liquidity 相关时避免 naive ANOVA/attribution |
| Lengerich et al. [@lengerich2020purifying] | piecewise-constant/tree models | main/interaction effects 需 purification 才可识别 | tree challenger 的 interaction 报告应先净化 |

# 建议阅读顺序

## 第一组：先理解 empirical size puzzle

1. Edwards, Harris & Piwowar [@edwards2007corporate]；
2. Pinter, Wang & Zou [@pinter2024size]；
3. Reichenbacher & Schuster [@reichenbacher2022size]。

读完后应能回答：pooled size discount 为什么不等于同一 client/state 下的 quantity effect？

## 第二组：理解 dealer capacity、inventory 与 block tail

1. Bessembinder et al. [@bessembinder2018capital]；
2. Goldberg & Nozawa [@goldberg2021liquidity]；
3. Goldstein & Hotchkiss [@goldstein2020providing]；
4. Jacobsen & Venkataraman [@jacobsen2025receiving]。

读完后应能回答：哪些 state 更像改变 $Q^\star$，哪些更像改变 $A(X)$，哪些会改变 tail shape？

## 第三组：理解 quote/protocol/search selection

1. O’Hara & Zhou [@ohara2021electronic]；
2. Hendershott et al. [@hendershott2026quote]；
3. Kargar et al. [@kargar2026sequential]。

读完后应能回答：displayed size、dealer participation、failed inquiry 与 completed trade sample 分别选择了什么？

## 第四组：建立统计检验

1. Robinson [@robinson1988root]；
2. Chernozhukov et al. [@chernozhukov2018double]；
3. Hastie & Tibshirani [@hastie1993varying]；
4. Hooker [@hooker2007generalized]；
5. Lengerich et al. [@lengerich2020purifying]；
6. Härdle & Mammen [@hardle1993comparing]。

读完后应能独立写出 additive null、orthogonalized basis estimator、pure interaction constraints 与 OOT global specification test。

# 最终建议

不要把研究设计简化为：

```text
fit fair value without quantity
→ fit residual on quantity
→ interpolate six nodes
```

更严谨的版本是：

```text
quantity semantics + PIT target
→ support / selection audit
→ cross-fitted additive quantity curve
→ liquidity-axis normalization Q / Q*
→ risk-amplitude normalization A(X)
→ sparse, purified shape interactions
→ unrestricted challenger
→ OOT + matched-pair + conditional-calibration decision
```

最终最可能的生产结构不是完全 additive，也不是完全自由的 black box，而是

$$
\boxed{
Y
=
m(X)
+A(X)h\!\left(
\log\frac{Q}{Q^\star(X)}
\right)
+\sum_{j\in\mathcal J_{\mathrm{retained}}}
Z_j f_j\!\left(
\log\frac{Q}{Q^\star(X)}
\right)
+\varepsilon.
}
$$

这保留了 quantity normalization 的可解释性，同时允许数据证明在 information、stress、inventory-limit 或 competition-collapse 状态下，quantity curve 确实需要改变形状。

# References
