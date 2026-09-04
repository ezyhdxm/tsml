# 信用产品定价与 Risk Management

> Public-only sanitized edition. 本文刻意排除任何公司的内部项目、人员、系统、模型编号、生产配置、非公开数据或 proprietary implementation。

## 0. Scope

这份教程用同一条逻辑学习主要信用产品：

**合同现金流 → 风险中性定价 → 市场校准 → Greeks / scenario risk → hedge → residual risk → FRTB 映射**。

这里的公式是产品经济结构和常见市场框架，不代表任何机构特定的生产级 pricer。真实估值还需要处理 day-count、accrual-on-default、coupon conventions、stubs、settlement、holiday calendars、default state、collateral、curve construction 等细节。

## 1. 共通定价地基

绝大多数信用产品都可以抽象成

\[
V_0=\mathbb E^{\mathbb Q}\left[\sum_k D(0,t_k)CF_k(X_{t_k})\right].
\]

差异主要来自 cashflow function 与 state variables：CDS 依赖 survival/default；CDX 再加入 constituents 与 index factor；option 加 volatility 与 nonlinear payoff；tranche 加 portfolio-loss distribution；TRS 加 financing leg；ETF 加 holdings 与 creation/redemption；callable bond 加 issuer exercise。

统一记号：

- \(D(0,t)\)：discount factor；
- \(Q(t)=\mathbb P^{\mathbb Q}(\tau>t)\)：risk-neutral survival probability；
- \(\lambda(t)\)：hazard rate；
- \(R\)：recovery；
- \(N\)：notional；
- \(\alpha_i\)：accrual year fraction。

## 2. CDS

### 2.1 两条腿

Protection leg：

\[
PV_{prot}=N(1-R)\int_0^T D(0,t)\lambda(t)Q(t)dt.
\]

Premium leg：

\[
PV_{prem}=sN\sum_i\alpha_iD(0,t_i)Q(t_i)+AoD.
\]

其中 risky annuity / RPV01 是 premium leg 对 1bp coupon 的现值系数。Par CDS 满足两腿相等，因此

\[
s_{par}=\frac{(1-R)\int_0^T D(0,t)\lambda(t)Q(t)dt}{RPV01}.
\]

如果 hazard 在每个 maturity bucket 内 piecewise constant，就可以从短 maturity 开始逐段 bootstrap hazard curve。常见近似 \(s\approx\lambda(1-R)\) 只在简化条件下成立。

### 2.2 Risk

- **CS01**：credit curve bump 1bp 的 PV 变化；
- **JTD**：真正 default 的离散价值跳跃；
- **Recovery risk**：既影响 LGD，又影响由 market spreads 反推出的 hazard curve；
- **Rates risk**：premium/protection cashflows 仍需 discount；
- **Bond-CDS hedge**：一阶 notional 常按

\[
N_{CDS}^{hedge}\approx\frac{CS01_{bond}}{CS01_{CDS,unit}}
\]

匹配，但仍有 bond-CDS basis、liquidity、funding、curve-shape 与 default-settlement residual risk。

## 3. CDX

CDX 可以先近似成 weighted single-name CDS portfolio：

\[
V_{index}\approx N\sum_iw_iV_i.
\]

真实 index 使用标准 coupon，因此若市场 fair spread \(s^*\) 与 contract coupon \(c\) 不同，会产生 upfront。一阶近似：

\[
Upfront\approx(s^*-c)RPV01_{index}N.
\]

发生 constituent default 后，surviving index factor 会改变 premium base，因此 default state 同时影响 PV、P&L、option 和 tranche。

主要风险：

- index spread；
- single-name dispersion / index basis；
- constituent default 与 surviving factor；
- series / roll risk；
- constituent look-through。

即使 index parallel CS01 可以与各 constituent sensitivities reconcile，也不能把复杂状态下的 risk 简单按 notional weights 精确拆分。

## 4. CDX Option

Payer option 通常在 spreads widening 时受益。一个常用的 Black-style 骨架是

\[
V_{payer}=A[F\Phi(d_1)-K\Phi(d_2)],
\qquad
d_{1,2}=\frac{\ln(F/K)\pm\tfrac12\sigma^2T}{\sigma\sqrt T}.
\]

其中 \(F\) 为 forward index spread，\(K\) 为 strike，\(A\) 为 forward risky annuity。

Greeks：

\[
\Delta_F=A\Phi(d_1),
\qquad
\Gamma_F=\frac{A\phi(d_1)}{F\sigma\sqrt T},
\qquad
Vega=AF\phi(d_1)\sqrt T.
\]

ATM 附近 gamma/vega 通常最显著。Expiry 前 constituent default 还会引入 front-end protection：

\[
V_{option}=V_{surviving\ index\ option}+PV(FEP/default\ adjustment).
\]

如果 \(V=f(s_1,\ldots,s_m)\)、index spread \(S=g(s_1,\ldots,s_m)\)，一般

\[
\frac{\partial^2V}{\partial s_i^2}
\neq
w_i\frac{\partial^2V}{\partial S^2},
\]

因为还存在 chain-rule square terms、cross-gamma、annuity 与 default-state effects。这也是 CDX option 在 FRTB curvature 中最重要的非线性来源之一。

风险管理同时关注 index delta、vol surface、gamma、single-name dispersion 与 realized defaults。

## 5. CDX Tranche

Attachment \(A\) 与 detachment \(D\) 让 tranche 只承担 portfolio loss 的一个区间。若 portfolio cumulative loss fraction 为 \(L_t\)，tranche cumulative loss 为

\[
\ell_{A,D}(L_t)=\min\{\max(L_t-A,0),D-A\}.
\]

归一化 tranche loss fraction：

\[
TL_t=\frac{\ell_{A,D}(L_t)}{D-A}.
\]

两条腿：

\[
PV_{prem}=sN_{tr}\sum_i\alpha_iD_i\mathbb E[1-TL_{t_i}],
\]

\[
PV_{prot}=N_{tr}\int_0^TD(0,t)d\mathbb E[TL_t].
\]

所以定价核心不再是“平均 spread”，而是整个 portfolio loss distribution。Default correlation 会改变 loss distribution 的形状，因此不同 attachment/detachment tranche 对 correlation 的反应可能很不同。

主要风险：index spread/hazard、correlation/base correlation、recovery、JTD/name concentration、liquidity 与 model risk。

## 6. Bond / Loan TRS

TRS 把 reference asset 的 total return 与 financing leg 交换。某 period 的 total-return cashflow 可抽象为

\[
CF_i^{TR}=q[P_i-P_{i-1}+Coupon_i+Principal_i+Recovery_i].
\]

Financing leg 常类似

\[
qN_i(L_i+m)\alpha_i.
\]

如果 inception PV=0，fair financing spread 可写成

\[
m^*=\frac{PV(total\ return\ leg)-PV(reference\ floating\ leg)}{A_{fund}},
\qquad
A_{fund}=\sum_iD_i\alpha_iN_i.
\]

风险至少包括：

- **DV01**：underlying 与 financing leg 的 rates exposure；
- **CS01**：underlying credit spread exposure；
- **default/recovery**：reference asset default 会改变 price、recovery cashflow 与 remaining notional；
- **funding basis**；
- **amortization/prepayment**，尤其 loan TRS；
- **counterparty / collateral / XVA**。

TRS 的实现难点往往不是公式本身，而是 coupon、reset、principal、default、recovery、business-day 与 amendment 等事件顺序必须完全一致。

## 7. Bond ETF

ETF NAV：

\[
NAV_t=\frac{\sum_iq_iP_i+Cash-Liabilities}{Shares\ outstanding}.
\]

Authorized Participants 通过 creation/redemption 把 ETF market price 与 underlying basket 连接，但在 fixed income 中，underlying liquidity 与 stale marks 会让 price-NAV basis 暂时存在。

Constituent risk 来自

\[
dNAV\approx\frac1{Shares}\sum_iq_i\,dP_i.
\]

若

\[
dP_i\approx-DV01_i\,dy_i-CS01_i\,ds_i,
\]

则可以得到 ETF 对 rates 与 credit 的 look-through exposure。

实际难点包括 non-trading constituents、bid/ask、sampling basket、issuer/sector concentration、ETF-NAV basis、tracking difference 与 rebalance risk。

## 8. CRT / SRT

Credit Risk Transfer / Synthetic Risk Transfer 通常把 loan portfolio 的一段信用损失转给 protection seller 或 investors，而基础贷款未必出售。

可继续使用 tranche loss framework。Premium leg 对 outstanding protected notional 支付：

\[
PV_{prem}=s\sum_iD_i\alpha_i\mathbb E[N^{prot}_{t_i}],
\]

protection leg 对 tranche loss increment 支付：

\[
PV_{prot}=\int_0^TD(0,t)d\mathbb E[Loss_t^{tranche}].
\]

Fair premium 令两腿相等。若 pool amortizes/prepays，protected notional、WAL 与 expected-loss timing 都改变，par spread 也会随之变化。

主要风险：

- PD / hazard deterioration；
- default correlation 与 concentration；
- recovery / LGD；
- amortization / prepayment；
- proxy-hedge basis；
- legal / structural / regulatory-recognition risk。

## 9. ABX / CMBX

ABX 参考 RMBS tranches，CMBX 参考 CMBS tranches。它们表面像 CDS indices，但 underlying 是 securitization tranches，因此 state variables 还包括 prepayment、writedown、interest shortfall、extension 与 waterfall。

统一 cashflow 视角：

\[
PV_{prot}=\mathbb E\left[\sum_kD(0,\tau_k)LossEvent_k\right],
\]

\[
PV_{prem}=s\sum_i\alpha_iD_i\mathbb E[N^{eff}_{t_i}].
\]

Mortgage prepayment 之所以会影响 credit index pricing，是因为它改变 collateral principal path、WAL、subordination 与 future loss absorption。

主要风险：credit/collateral quality、prepayment/extension、loss severity、waterfall/subordination、liquidity 与 index-cash basis，以及监管 risk-factor representation。

## 10. Callable / Sinkable Bonds

Callable bond 可以理解为

\[
P_{callable}=P_{straight}-V_{issuer\ call}.
\]

Rates 下行时 straight bond 上涨，但 issuer call option 更值钱，抵消一部分上涨，因此 callable bond 可能出现低 convexity 甚至局部 negative convexity。

利率树中先计算 continuation value：

\[
V_t^{cont}=D_t\mathbb E_t[V_{t+\Delta t}+CF_{t+\Delta t}],
\]

如果 issuer 可 call：

\[
V_t=\min\{V_t^{cont},CallPrice_t+Accrued_t\}.
\]

调 OAS 直到模型回溯价格等于市场 dirty price。

主要风险：rate duration、rate vol vega、credit/OAS、exercise behavior/model risk。

对于确定性 sinking schedule：

\[
P=\sum_iD_i(Coupon_i+Principal_i).
\]

本金更早回收通常降低 duration/CS01，并增加 reinvestment risk。如果 sinking mechanism 本身允许 issuer discretion，则还会产生 optionality。

## 11. FRTB SA

FRTB Standardised Approach 将市场风险资本拆为

\[
K_{SA}=K_{SBM}+K_{DRC}+K_{RRAO}.
\]

其中 SBM 覆盖 delta、vega 与 curvature。

### 11.1 CSR Delta

对 credit spread risk factor \(x_k\) 计算

\[
s_k=\frac{\partial V}{\partial x_k},
\]

再乘 prescribed risk weight，并按 obligor、sector、credit quality、tenor、curve type 的 bucket 与 correlation 规则聚合。

### 11.2 Curvature

Curvature 不是简单报告二阶导，而是按 prescribed up/down shocks full revaluation，再扣除 delta 已解释的一阶部分：

\[
Curvature\ component\approx Full\ shocked\ loss-Delta\ explained\ loss.
\]

这对 CDX options、callable bonds 等非线性产品尤为重要。

### 11.3 JTD / DRC

连续 spread widening 与真正 default 是不同风险。JTD 捕捉 default 发生时的离散 PV jump；DRC 再按 gross JTD、允许净额、bucket 与 prescribed weights 聚合 capital。对 index 与 index options，constituent look-through 往往是关键。

> 经济 hedge 追求 P&L stability；FRTB mapping 追求可审计、可复现、符合 prescribed risk buckets 的表达。两者相关，但不完全相同。

## 12. Acronym glossary

| Acronym | Full form | 含义 |
|---|---|---|
| ABX | Asset-Backed Securities Index | 参考 subprime RMBS tranches 的信用指数族 |
| AP | Authorized Participant | ETF creation/redemption 参与机构 |
| CDS | Credit Default Swap | 单名信用保护合约 |
| CDX | Credit Default Swap Index | 北美常见 CDS index family |
| CMBX | Commercial Mortgage-Backed Securities Index | 参考 CMBS tranches 的信用指数族 |
| CMBS | Commercial Mortgage-Backed Securities | 商业地产抵押贷款证券化 |
| CRT | Credit Risk Transfer | 信用风险转移 |
| CSR | Credit Spread Risk | FRTB 信用利差风险类别 |
| CS01 | Credit Spread 01 | credit spread 变动 1bp 的 PV sensitivity |
| DRC | Default Risk Charge | FRTB default jump capital charge |
| DV01 | Dollar Value of 1 Basis Point | rates 变动 1bp 的 dollar sensitivity |
| ETF | Exchange-Traded Fund | 交易所交易基金 |
| FEP | Front-End Protection | CDX option expiry 前 defaults 的保护价值调整 |
| FRTB | Fundamental Review of the Trading Book | Basel trading-book market-risk framework |
| JTD | Jump-to-Default | default 时 PV 的离散跳跃 |
| LGD | Loss Given Default | 违约损失率，约为 1 − recovery |
| NAV | Net Asset Value | 基金资产净值 |
| OAS | Option-Adjusted Spread | embedded option 调整后的 spread |
| RMBS | Residential Mortgage-Backed Securities | 住宅抵押贷款证券化 |
| RPV01 | Risky PV01 | 带 survival/default 权重的 risky annuity |
| SA | Standardised Approach | FRTB 标准法 |
| SBM | Sensitivities-Based Method | FRTB SA sensitivity aggregation 模块 |
| SRT | Synthetic Risk Transfer | 通过合成结构转移信用风险 |
| TRS | Total Return Swap | total return leg 对 financing leg |

## 13. Public references

- BIS Basel Framework, MAR20: FRTB Standardised Approach structure  
  https://www.bis.org/committees/bcbs/basel-framework/standard/mar/20/inforce/2023-01-01/published/2020-03-27
- BIS Basel Framework, MAR21: Sensitivities-Based Method  
  https://www.bis.org/committees/bcbs/basel-framework/standard/mar/21/inforce/2023-01-01/published/2024-07-05
- BIS Basel Framework, MAR22: Default Risk Charge / JTD  
  https://www.bis.org/committees/bcbs/basel-framework/standard/mar/22/inforce/2023-01-01/published/2020-03-27
- ISDA 2014 Credit Derivatives Definitions  
  https://www.isda.org/book/2014-isda-credit-derivative-definitions
- ISDA CDS Standard Model documentation  
  https://www.cdsmodel.com/documentation.html
- S&P Dow Jones Indices: CDX  
  https://www.spglobal.com/spdji/en/indices/products/markit-cdx.html
- CFTC: credit index swaption mechanics  
  https://www.cftc.gov/sites/default/files/stellent/groups/public/%40rulesandproducts/documents/ifdocs/rul112613bgc001.pdf
- BIS: The rise and risks of synthetic risk transfers  
  https://www.bis.org/publications/rise-and-risks-synthetic-risk-transfers
- S&P Dow Jones Indices: ABX / CMBX  
  https://www.spglobal.com/spdji/en/landing/topic/abx/  
  https://www.spglobal.com/spdji/en/landing/topic/cmbx/
- Investor.gov: Exchange-Traded Funds  
  https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-24
- FINRA: sinking-fund bond background  
  https://www.finra.org/rules-guidance/notices/05-21
