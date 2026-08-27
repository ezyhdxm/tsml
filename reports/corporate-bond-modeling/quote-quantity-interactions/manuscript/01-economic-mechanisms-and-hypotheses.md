# 经济机制：什么应当与 quantity 发生 interaction

# 文献中的 size puzzle

最早的 corporate-bond transaction-cost evidence 普遍发现，平均交易成本随 trade size 增大而下降，并且高评级、较新发行、临近到期的 bond 成本更低 [@edwards2007corporate]。这类 pooled “size discount” 很容易被读成一个全市场通用的 decreasing $h(Q)$。

然而，nonanonymous bond data 揭示了关键的 composition effect。Pinter、Wang 与 Zou 将同一客户内部的 size variation 与不同客户之间的 size variation 分开后发现：larger clients 通常获得更好价格，造成 pooled size discount；但控制 client identity、dealer、client-dealer relationship 与时间固定效应后，同一客户的更大交易反而承担更高成本，即 size penalty [@pinter2024size]。这一结果至少有三层含义：

1. **raw quantity slope 不是纯技术性 execution curve**；
2. client bargaining power、sophistication 与 relationship 可以同时影响 $Q$ 和 price；
3. 在用于 quote normalization 的模型中，client/dealer composition 至少要作为 nuisance structure 控制，即使 deployment 时拿不到完整 client identity。

dealer-run 数据又带来另一种识别风险。Hendershott 等研究发现，只有一部分 quotes 带 size；在带 size 的 quotes 中，大部分 displayed size 超过 $1\text{mm}$，但实际 client-dealer trades 中超过 $1\text{mm}$ 的比例小得多。因此 displayed quote size 很可能经常不 binding，研究者也主要依赖 quote price、quote incidence 与 quote quality [@hendershott2026quote]。这意味着 dealer-run quantity 不能未经语义审计就当作精确的 $P(Q=q)$ observation。

与此同时，size-adapted liquidity literature 显示，若 liquidity measure 忽略 trade-size distribution，时间变化或 cross-sectional difference 可能只是 average size composition 变化，而不是真正 liquidity 变化 [@reichenbacher2022size]. 这与本报告的问题完全同构：要估计 quantity correction，必须把“在什么 state 下出现什么 quantity”与“同一 state 下 quantity 改变的价格效应”分开。

# 一个可解释的 dealer cost decomposition

考虑 dealer 对一笔 quantity 为 $q$ 的交易要求的单位 concession。一个简化分解是

$$
C(q,X)
\approx
\frac{F(X)}{q}
+b(X)
+\lambda(X)sI
+\frac{1}{2}\lambda(X)q
+AS(q,X)
-BG(q,X),
$$

其中：

- $F(X)$：固定 quotation、search、operational、onboarding 或 ticket cost；
- $F(X)/q$：固定成本在小单上无法充分摊薄；
- $b(X)$：普通 per-unit intermediation cost；
- $I$：dealer 当前 inventory exposure；
- $s\in\{-1,+1\}$：交易使 inventory 增加还是减少；
- $\lambda(X)$：inventory risk、funding、capital 与 liquidation difficulty；
- $AS(q,X)$：adverse-selection / information risk；
- $BG(q,X)$：客户 bargaining、relationship 或 competition 带来的折扣。

这个式子不是 structural pricing model，而是一张机制地图。它说明 pooled quantity curve 可以同时包含方向相反的成分：

- 小 size：$F/q$ 较大，但 sophisticated large clients 也可能有更强 bargaining；
- 中等 institutional size：固定成本已摊薄，curve 可能平坦；
- 大 block：inventory、capital、search-for-receivers 和 information risk 开始主导，curve 变陡。

因此全局 curve 可能是 decreasing、flat、increasing 或浅 U-shape。没有充分证据时，不应对 $100\text{k}$ 到 $5\text{mm}$ 整段施加单调性。

## quadratic inventory cost 的一个重要推论

若 dealer inventory disutility 为

$$
V(I)=\frac{\lambda}{2}I^2,
$$

接受方向 $s$、quantity $q$ 的订单后，incremental cost 为

$$
V(I+sq)-V(I)
=\lambda sIq+\frac{\lambda}{2}q^2.
$$

换算成每单位 quantity：

$$
\frac{V(I+sq)-V(I)}{q}
=\lambda sI+\frac{\lambda}{2}q.
$$

这带来一个容易忽略的 hypothesis：在纯 quadratic、无 hard limit 的局部模型中，当前 inventory $sI$ 首先改变的是 **intercept**，quantity slope 由 $\lambda$ 决定。只有在以下情形中，`inventory level × quantity` 才应显著改变 slope/curvature：

- inventory risk function 高于二次；
- position 接近 hard limit、concentration limit 或 capital threshold；
- expected liquidation horizon 随 position size 非线性增加；
- hedge capacity 在某个 size 后耗尽；
- dealer 的 risk aversion / funding shadow price 随 market stress 跳变。

因此看到 `side × inventory proxy` 很重要，并不等于必须让它在全 quantity range 自由改变 curve。一个更稀疏的结构是：正常状态影响 $m(X)$，capacity-stress indicator 才进入 $A(X)$ 或 $r(X,Q)$。

# 结构化 quantity model

经济机制自然导向

$$
\mu(X,Q)
=
m(X)
+A(X)h(u)
+r(X,Q),
\qquad
u:=\log Q-\log Q^\star(X).
$$

这里 $u$ 是 liquidity-normalized log quantity。

- $Q^\star(X)$ 把不同 bonds/states 的 quantity axis 对齐；
- $A(X)$ 把不同 risk/capital states 的 concession amplitude 对齐；
- $r(X,Q)$ 只保留无法由前两者解释的 shape difference。

这比把所有 `quantity × feature` 交给 LightGBM 有三个优势：

1. 更容易判断 interaction 的经济来源；
2. 在 $2\text{mm}$、$5\text{mm}$ 稀疏区更稳定；
3. normalization 可被明确写成 reference-quantity contrast，而非依赖 local SHAP attribution。

# Hypothesis 1：relative liquidity capacity 决定横向尺度

## 命题

raw dollar quantity 没有跨 bond 的稳定含义。更稳定的坐标应是

$$
\frac{Q}{Q^\star_{i,t}},
$$

其中 $Q^\star_{i,t}$ 表示 quote 时点的可用 liquidity capacity。

候选 proxy 包括：

$$
\begin{aligned}
Q^\star_{i,t}=f(&\text{trailing median trade size},
\text{trailing volume},
\text{amount outstanding},\\
&\text{active dealer count},
\text{quote frequency},
\text{quote freshness},\\
&\text{time since trade},
\text{issue age},
\text{issue complexity}).
\end{aligned}
$$

### 可检验预测

控制 $u=\log Q-\log Q^\star(X)$ 后，以下表面 interaction 应显著减弱：

- `quantity × amount outstanding`；
- `quantity × trailing volume`；
- `quantity × median trade size`；
- `quantity × active dealers`；
- `quantity × issue age`；
- `quantity × rating` 中由 liquidity composition 产生的部分。

若 $Q^\star$ 抓住了主要机制，不同 liquidity buckets 的 conditional contrast curves

$$
\Delta(q,q_0\mid Z_{\mathrm{liq}})
$$

在改用 $q/Q^\star$ 后应大致重合。

### 文献连接

dealer capital commitment、turnover、block frequency 与 average trade size 的变化共同反映 liquidity capacity [@bessembinder2018capital]；size-adapted liquidity measures 的表现也说明，trade-size composition 不能与 liquidity state 混为一谈 [@reichenbacher2022size]。ICE 的 size-adjusted pricing 是产业界将 security、side 与 size 一起映射到可执行估值的直接先例，但其公开说明不足以识别具体 functional form [@ice2026size].

# Hypothesis 2：risk exposure 决定纵向 amplitude

## 命题

相同 $Q/Q^\star$ 在不同风险单位下的经济成本不同。一个粗略的 spread-risk exposure 是

$$
R_s(Q,X)
=
Q\times CS01(X)\times \sigma_{\Delta s}(X)
\times \sqrt{\tau_{\mathrm{unwind}}(X)}.
$$

rate-risk component 可写成

$$
R_r(Q,X)
=
Q\times DV01(X)\times \sigma_{\Delta r}(X)
\times \sqrt{\tau_{\mathrm{hedge}}(X)}.
$$

因此

$$
A(X)
=f(CS01, DV01, \sigma_s, \sigma_r,
\tau_{\mathrm{unwind}}, \text{jump risk},
\text{hedgeability}, \text{optionality}).
$$

### 可检验预测

- 在 price-concession target 中，duration/CS01 interaction 应明显；
- 在 spread-bp target 中，机械 duration effect 减弱，但 dollar VaR、capital 和 hedging cost 仍可使 $A(X)$ 上升；
- illiquid callable/HY bond 的 $2\text{mm}-1\text{mm}$ contrast 应大于 liquid bullet IG bond，即使 raw spread level 相近；
- 用 risk-normalized quantity 替代 raw quantity 后，tenor、rating 与 volatility buckets 的 slope heterogeneity 应收缩。

dealer inventory capacity 被压缩时，corporate-bond liquidity 与 prices 可以显著偏离正常状态 [@goldberg2021liquidity; @chikis2021dealer]。这支持让 stress/risk state 进入 amplitude，而不是仅作为 additive spread-level feature。

# Hypothesis 3：side × inventory pressure 正常时影响 intercept，临界时影响 slope

## 候选 features

真实 inventory 往往不可得，可用 point-in-time proxy：

- dealer 最近同 CUSIP 的 signed flow；
- dealer 最近同 issuer 的 signed flow；
- sector/rating bucket signed flow；
- bid/offer quote skew；
- dealer 相对 peers 的 side-specific aggressiveness；
- max displayed quantity 的近期收缩；
- quote 更新方向与 frequency asymmetry。

### 可检验预测

令 $Z_{\mathrm{inv}}$ 为 signed inventory pressure：

1. 在 normal-capacity bucket，$Z_{\mathrm{inv}}$ 对 $m(X)$ 的作用强于对 $h'(Q)$ 的作用；
2. 在高 utilization / market stress bucket，`quantity × side × Z_inv` 显著，尤其在 $2\text{mm}$ 与 $5\text{mm}$；
3. 帮助 dealer 减仓的方向可能出现 flat 甚至 locally favorable large-size curve；
4. 增加不良 inventory 的方向出现更高 convexity。

BondCliQ dealer-run evidence 显示 dealers 会用 quote incidence、quality 与 side 来管理 inventory、吸引 order flow，但 quote competition 并不完美 [@hendershott2026quote]。Goldstein 与 Hotchkiss 对 illiquid bond dealer behavior 的研究也强调 active versus passive dealer behavior 与 inventory management [@goldstein2020providing]。

# Hypothesis 4：information intensity 产生 genuine shape interaction

## 机制

大单既增加 inventory，也可能传递更多 private information。经典 microstructure 中 trade size 可以与 information content 联动；在 bond data 中，within-client size penalty 在 macro surprises、COVID 等 information-intensive periods 更强 [@pinter2024size]。block receivers 也会担心 dealer 转售的 block 含有不利信息；透明度和 informed-trading proxies 会影响 receiving investors 的 terms [@jacobsen2025receiving]。

## 候选 state

- earnings、rating action、M&A 或 issuer news window；
- recent equity/CDS jump；
- CDX/sector shock；
- first trade after long inactivity；
- unusual direction relative to recent flow；
- client sophistication / information-sensitive client proxy；
- order splitting 或 repeated inquiry pattern。

### 可检验预测

information intensity 不只是平移 curve，而应使大 size tail disproportionately 变差：

$$
\left[
\Delta(5\text{mm},1\text{mm})
\right]_{Z_{\mathrm{info}}=1}
>
\left[
\Delta(5\text{mm},1\text{mm})
\right]_{Z_{\mathrm{info}}=0},
$$

而 $250\text{k}-100\text{k}$ contrast 的变化较小。这种“tail-specific steepening”若在 scale/amplitude normalization 后仍存在，才应归入 $r(X,Q)$。

# Hypothesis 5：market stress 同时压缩 scale、放大 amplitude

## 命题

stress state 不是简单的 parallel shift，而可能同时满足

$$
Q^\star(X)\downarrow,
\qquad
A(X)\uparrow.
$$

候选 signals：

- CDX bid-ask / realized volatility；
- Treasury or rate volatility；
- market-wide dealer breadth；
- cross-dealer quote dispersion；
- fund outflow pressure；
- downgrade wave；
- month/quarter-end balance-sheet pressure；
- 临近收盘与 overnight risk。

### 可检验预测

- $100\text{k}$、$250\text{k}$ quote 变化有限；
- $1\text{mm}$ 后 slope 明显变陡；
- $2\text{mm}$、$5\text{mm}$ availability probability 下降；
- stress-normalized curve 若仍不能重合，才需要 stress-specific shape function。

COVID episode 的研究表明，liquidity supply contraction 与 liquidity demand shock 都重要，dealer inventory constraints 会影响价格与可承接数量 [@ohara2021anatomy; @chikis2021dealer]。

# Hypothesis 6：competition、protocol 与 relationship 改变 quantity curve

## 机制

dealer competition 通常降低 quote concession，但参与 dealer 数量本身可能随 quantity 下降：

$$
N_{\mathrm{respond}}=N(Q,X),
\qquad
\frac{\partial N_{\mathrm{respond}}}{\partial Q}<0.
$$

于是 large-size region 既承担 inventory risk，也失去 competition benefit。

候选 features：

- pre-event active dealer count；
- electronic / voice / bilateral / RFQ protocol；
- pre-RFQ response propensity；
- dealer specialization；
- client-dealer relationship；
- quote dispersion 与 quote freshness；
- recent successful/unsuccessful inquiries。

### 可检验预测

- competition 主要降低 $m(X)$，但若 response propensity 随 $Q$ 急降，也会增加 large-size slope；
- relationship effect 在 large size 与 stress 时更强；
- electronic advantage 可能集中在可标准化、较小或较 liquid 的 trades，stress/large-size 时转向 voice/bilateral [@ohara2021electronic];
- sequential search/repeated inquiry state 应改变客户最终愿意接受的 concession [@kargar2026sequential].

<div class="warning">
<strong>防止 post-event leakage。</strong> 若目标是在 RFQ arrival 时预测，最终 response count、winning quote、completed trade、post-RFQ quote updates 都不能作为 feature。只能使用 arrival 时点之前的 dealer breadth、response propensity 与 relationship history。
</div>

# Hypothesis 7：hedgeability 与 substitutability 降低 large-size slope

候选 features：

- liquid single-name CDS；
- CDX/sector-index hedge quality；
- ETF membership；
- 同 issuer 活跃 bonds 的数量与 curve richness；
- same-sector/rating liquid substitutes；
- Treasury hedge quality；
- callable / structural complexity。

若 dealer 可立即 hedge 大部分 risk，则同样的 $Q/Q^\star$ 应产生更小 $A(X)$，且 large-size tail 更平。这里 static sector、rating、tenor 更适合被视为上述机制的 proxy，而非默认保留的直接 interaction。

# Hypothesis 8：client/dealer composition 与 endogenous quantity selection

## quantity 不是随机 treatment

客户会根据 urgency、information、portfolio constraint 与 relationship 选择 quantity；dealer 也会根据 inventory、market state 与信心选择 displayed quantity。于是

$$
Q\not\perp X,
$$

甚至在控制 observables 后仍有

$$
Q\not\perp \varepsilon.
$$

典型 selection：

- liquid bond / calm market 更容易显示大 size；
- strong-relationship clients 同时获得大 capacity 与好价格；
- informed/urgent clients 选择大单但愿意支付更多；
- dealer 只在 inventory favorable 时显示 $5\text{mm}$；
- unsuccessful large RFQ 根本不进入 completed-trade sample。

### 可检验预测

- pooled curve 与 client/dealer fixed-effect curve 显著不同；
- quote sample 与 completed-trade sample 的 curve 不同；
- 只在 common-support 区域估计时，large-size advantage 收缩；
- same dealer–bond–side–short-window contrasts 与全截面 regression 的 slope 不同。

这个 hypothesis 是对所有其他 interaction 解释的前置审计：看到 `Q × liquidity` 并不自动说明 liquidity 改变 counterfactual quantity effect，也可能只是 liquidity 改变观察到 large $Q$ 的概率。

# 哪些变量不应优先直接做 quantity interaction

| 变量 | 更合理的首要角色 | 直接 interaction 的风险 |
|---|---|---|
| credit spread level | quantity-neutral mid / event-risk proxy | 混入 fundamental level；应优先用 spread volatility、jump risk |
| rating | liquidity、risk、clientele 的粗 proxy | 控制更直接机制后可能不稳定 |
| tenor / maturity | price-space 的 duration conversion；risk exposure | spread-bp target 中简单 bucket interaction 缺乏机制 |
| sector | hedgeability、flow、inventory concentration 的 proxy | 高维、样本稀疏，容易过拟合 |
| dealer average aggressiveness | baseline intercept $m(X)$ | 只有 capacity/size specialization 才应改变 slope |
| issue age | liquidity scale $Q^\star$ | 直接 shape interaction 可能只是 volume/freshness proxy |

# quantity 字段的语义决定模型类型

## Requested quantity

客户明确请求 $Q=q$，dealer 对这一 size 报价。这最接近一个 point observation：

$$
Y(q).
$$

## Maximum / good-up-to quantity

“price $p$, up to $q_{\max}$”更像 interval constraint：

$$
Y(q)=Y_0,
\qquad 0<q\le q_{\max},
$$

或至少表示 availability：

$$
P(Q_{\max}\ge q\mid X).
$$

把 $q_{\max}$ 当作恰好发生在该点的 continuous regressor 会错误解释 capacity。

## Incremental quantity ladder

若 ladder level 是 incremental size，客户执行 cumulative size $q$ 时应计算 cumulative VWAP，不应对 level price 直接 interpolation。例如

$$
500\text{k}@99.20,
\qquad
\text{next }1\text{mm}@99.10,
$$

则 $1.5\text{mm}$ cumulative price 为

$$
P_{\mathrm{VWAP}}(1.5\text{mm})
=
\frac{0.5\times 99.20+1.0\times 99.10}{1.5}.
$$

## Missing quantity

missing 可能表示：

- dealer 未显示 size；
- source 不采集 size；
- default institutional capacity；
- quote run template 省略；
- truly unavailable。

必须用 source/dealer/protocol/time 的 missingness model 审计，不能填零或统一填 $1\text{mm}$。
