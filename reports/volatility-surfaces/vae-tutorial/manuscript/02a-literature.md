# 5. 文献地图：2021–2026 年 vol-surface VAE 的演化

## 5.1 一眼看懂时间线

<div class="timeline">
<div class="timeline-item"><strong>2021–2022：completion 与 latent manifold。</strong>Bergeron et al. 证明少量 latent variables 可以生成并补全 FX surfaces。</div>
<div class="timeline-item"><strong>2021–2023：无套利生成与预测。</strong>Ning et al. 让 VAE 生成 arbitrage-free SDE parameters；Zhang et al. 用 constrained DNN 重建未来 SPX surface。</div>
<div class="timeline-item"><strong>2022–2024：高维 cube、解释性与工业用例。</strong>Richert–Buch 用 Gibbs-VAE 补 swaption cube；Gong et al. 用 covariance regularization 对齐 level/skew/term；Dierckx et al. 扩展到生成、completion 和 anomaly detection。</div>
<div class="timeline-item"><strong>2024–2025：不确定性、conditional generation、feature control。</strong>Gopal 建模 imputation uncertainty；Chen et al. 加 recurrent context；Wang et al. 显式控制形状特征。</div>
<div class="timeline-item"><strong>2026：可复现 CNN benchmark 与 latent generative second stage。</strong>Singh et al. 发布 crypto 数据、配置和 checkpoints；Brooks et al. 用 latent flow matching 修复简单先验；最新工作进一步转向 conditional latent diffusion trajectories。</div>
</div>

## 5.2 可搜索的总表

<input id="lit-search" class="search-box" placeholder="搜索作者、资产、任务、模型或年份，例如：FX / ConvVAE / arbitrage / 2026">

<div class="table-wrap"><table id="literature-matrix" class="data-table literature-table"><thead><tr><th>年份</th><th>作者</th><th>简称</th><th>任务</th><th>数据/网格</th><th>核心模型</th><th>无套利方式</th><th>复现性</th></tr></thead><tbody><tr><td>2022</td><td>Bergeron, Fung, Hull, Poulos, Veneris</td><td>Hands-Off Approach</td><td>FX surface completion / scenarios</td><td>OTC FX, 8×5</td><td>latent fitting; vanilla VAE</td><td>soft / no guarantee</td><td>low</td></tr><tr><td>2023</td><td>Ning, Jaimungal, Zhang, Bergeron</td><td>Arbitrage-Free IVS Generation</td><td>unconditional &amp; conditional generation</td><td>AUD/EUR/CAD FX, 8×5</td><td>VAE over calibrated SDE parameters</td><td>hard by construction</td><td>medium-high</td></tr><tr><td>2023</td><td>Zhang, Li, Zhang</td><td>Two-Step Arbitrage-Free Prediction</td><td>one-step surface forecast</td><td>SPX, irregular→154 grid</td><td>VAE/PCA/sampled features + LSTM + constrained DNN</td><td>hard at second stage</td><td>medium</td></tr><tr><td>2023</td><td>Dierckx, Davis, Schoutens</td><td>Towards Data-Driven Volatility Modeling</td><td>reconstruction, completion, generation, anomaly</td><td>SPX surfaces</td><td>VAE + pointwise decoder / gradient boosting</td><td>not guaranteed</td><td>low-medium</td></tr><tr><td>2024</td><td>Richert, Buch</td><td>Missing Swaption Volatility via Gibbs-VAE</td><td>cube imputation</td><td>normal-vol swaption cube, 4,998 cells</td><td>VAE + pseudo-Gibbs</td><td>not guaranteed</td><td>low</td></tr><tr><td>2024</td><td>Gong et al.</td><td>New Encoding / PCA-VAE</td><td>interpretable generation &amp; extrapolation</td><td>44 stocks + STOXX50, 8×7</td><td>covariance-regularized VAE</td><td>not guaranteed</td><td>low-medium</td></tr><tr><td>2024</td><td>Gopal</td><td>Missing FX IV with Uncertainties</td><td>probabilistic imputation</td><td>5 FX pairs, 8×5</td><td>residual heteroscedastic VAE/IWAE</td><td>not guaranteed</td><td>medium</td></tr><tr><td>2025</td><td>Feugang Nteumagné et al.</td><td>VAE Completing Volatility Surfaces</td><td>synthetic completion</td><td>Heston, 15×17</td><td>dense VAE + latent optimization</td><td>source data generated clean</td><td>medium-low</td></tr><tr><td>2025</td><td>Wang, Liu, Vuik</td><td>Controllable Generation</td><td>feature-controlled scenarios</td><td>60k Heston/SABR, 28×28</td><td>control variables + residual latent VAE</td><td>soft + latent repair</td><td>medium-high</td></tr><tr><td>2025</td><td>Chen et al.</td><td>Conditional Future Surface Generation</td><td>conditional multi-day generation</td><td>SPX, 5×5, 2000–2023</td><td>Conv-CVAE + recurrent context</td><td>no strict guarantee</td><td>high except data</td></tr><tr><td>2026</td><td>Singh, Reddy, Chopra</td><td>Beyond the Smile</td><td>masked completion &amp; anomaly</td><td>BTC/ETH, public 6×7</td><td>ConvVAE + deterministic smile router</td><td>grid-level diagnostics</td><td>very high</td></tr><tr><td>2026</td><td>Brooks, Bajalica, Liu, Ben Tahar</td><td>Latent Flow Matching</td><td>unconditional generation</td><td>SPX, 32×16</td><td>arb-regularized VAE + latent flow</td><td>soft; 90.8% reported</td><td>high except data</td></tr><tr><td>2026</td><td>Buchegger, Gonon</td><td>Arbitrage-Aware Multi-Step Forecasting</td><td>30-step trajectory forecast</td><td>SPX</td><td>arb-aware AE + conditional latent diffusion</td><td>soft / model-dependent</td><td>provisional</td></tr><tr><td>2026</td><td>Wang, Liu, Vuik</td><td>Latent-Space No-Arbitrage Geometry</td><td>safe latent set / repair geometry</td><td>30k Heston, 28×28</td><td>fixed generator + latent margin / level set</td><td>diagnostic and repair; not a global guarantee</td><td>medium-high / provisional</td></tr></tbody></table></div>

<div class="callout warning">
表中的误差不能横向直接排名。不同论文使用 IV decimal、volatility point、volatility basis point、price RMSE、Wasserstein distance 或 satisfaction ratio；网格大小、资产、缺失率和测试期也不同。
</div>

## 5.3 逐篇精读与训练审计

<div class="paper-card" id="paper-bergeron">
<h3>1. Bergeron et al. (2022)：Variational Autoencoders: A Hands-Off Approach to Volatility</h3>
<p class="paper-meta">Maxime Bergeron, Nicholas Fung, John Hull, Zissis Poulos, Andreas Veneris · Journal of Financial Data Science · <a href="https://arxiv.org/abs/2102.03945">arXiv 2102.03945</a></p>
<dl>
<dt>问题</dt><dd>学习 FX volatility-surface manifold；从不完整报价反推最贴合的 latent code；生成 stress/scenario surfaces。</dd>
<dt>数据</dt><dd>OTC FX，2012–2020，来源为 EDI。规则网格是 8 个期限（1w, 1m, 2m, 3m, 6m, 9m, 1y, 3y）× 5 个 delta（0.10, 0.25, 0.50, 0.75, 0.90）= 40 cells。最后约 15%、即 2020-03 至 2020-12 作为 validation/test-like period。</dd>
<dt>模型</dt><dd>encoder 与 decoder 各有 2 个 hidden layers，每层 32 units；比较不同 latent dimensions。生成先验为标准 Gaussian；补全时优化 latent code，使 decoder 输出在已知 cells 上最贴合。</dd>
<dt>训练</dt><dd>使用 Adam；论文没有完整给出 learning rate、batch size、epoch、activation、seed 与 early stopping 细节，这阻止严格复现。</dd>
<dt>主要结果</dt><dd>AUD/USD 使用 4 个 latent dimensions 时，全部 40 点已知的重建 MAE 报告为 33.6 volatility bps；只给 5 个已知点时为 61.1 bps。与 Heston calibration 比较，AUD/USD 为 33.6 vs 56.6 bps，GBP/USD 为 34.0 vs 47.6 bps，USD/MXN 为 56.7 vs 92.2 bps。</dd>
<dt>关键发现</dt><dd>两到四个 latent dimensions 已能解释大部分 surface structure；2020 年 3 月 stress period 在 latent space 中成为明显 outlier。</dd>
<dt>无套利</dt><dd>训练历史 surfaces 并不能自动保证 decoder 在所有 latent points 上无套利；本文重点是 completion 与 realism，而非 hard guarantee。</dd>
<dt>审计</dt><dd><span class="badge audit">4/10</span>数据与训练细节不足；文中 currency list 还存在“五个名称”与后续“六个货币对/含 USDJPY 表格”之间的不一致。核心思想清楚，但不宜宣称 exact replication。</dd>
</dl>
</div>

<div class="paper-card" id="paper-ning">
<h3>2. Ning et al. (2023)：Arbitrage-Free Implied Volatility Surface Generation with VAEs</h3>
<p class="paper-meta">Brian Ning, Sebastian Jaimungal, Xiaorong Zhang, Maxime Bergeron · SIAM Journal on Financial Mathematics 14(4), 1004–1027 · <a href="https://doi.org/10.1137/21M1443546">DOI</a> · <a href="https://github.com/BrianNingUT/ArbFreeIV-VAE">public demo code</a></p>
<dl>
<dt>问题</dt><dd>如何让 VAE 生成的不只是“像历史”，而是由 construction 保证 arbitrage-free？</dd>
<dt>核心结构</dt><dd>第一步逐日把市场 surface 校准到 arbitrage-free stochastic model 的参数；第二步在参数空间训练 VAE；第三步 decoder 输出模型参数，再通过定价模型生成 IV surface。只要参数 transformation 保持 admissible，输出具有 hard guarantee。</dd>
<dt>数据</dt><dd>AUD-USD、EUR-USD、CAD-USD，约 1,900 个交易日，2012-09-18 至 2019-12-30；5 delta × 8 maturities（1m, 2m, 3m, 6m, 9m, 1y, 3y, 5y）。前半段至 2016-05-09 训练，后半段测试。</dd>
<dt>第一阶段</dt><dd>重点模型是 3-regime continuous-time Markov chain；另比较 double-exponential、Gaussian-mixture、CGMY 等 additive-process parameterizations。CTMC 每个 maturity calibration 使用 regime ordering 与 parameter transformations。</dd>
<dt>VAE</dt><dd>全连接 hidden widths 报告为 64, 128, 256, 512；latent dimensions ∈ {3,5,10,15}；$\beta\in\{0.01,0.1,1,10\}$；AdamW，lr=0.001，batch=200 days，2,000 epochs。</dd>
<dt>评估</dt><dd>在 40-point grid 上比较 generated 与 test distributions 的 pointwise 1-Wasserstein distance。条件版本加入 VIX。</dd>
<dt>主要结果</dt><dd>CTMC 对市场 surface 的 median IV fitting RMSE（乘 $10^{-5}$）约为 AUD 8.1、EUR 5.0、CAD 6.0。VAE generation 的最优 average Wasserstein scores 依资产和 $\beta,d_z$ 而变；加入 VIX 通常进一步改善，例如 AUD 在 $d_z=3$ 时从约 4.91 降到 3.24。</dd>
<dt>关键洞见</dt><dd>“无套利”来自 decoder 的输出空间，而不是神经网络自己学会 inequality。约 350 个近期交易日有时已足够训练；过长历史可能因 nonstationarity 反而变差。</dd>
<dt>审计</dt><dd><span class="badge audit">7/10</span>论文细节充分，代码仓库有 figure demos 与 precomputed outputs；但原始 EDI 数据不公开，完整 CTMC calibration 依赖较重的数值链。</dd>
</dl>
</div>

<div class="paper-card" id="paper-zhang">
<h3>3. Zhang, Li &amp; Zhang (2023)：A Two-Step Framework for Arbitrage-Free Prediction</h3>
<p class="paper-meta">Wenyong Zhang, Lingfei Li, Gongqiu Zhang · Quantitative Finance 23(1), 21–34 · <a href="https://doi.org/10.1080/14697688.2022.2135454">DOI</a> · <a href="https://arxiv.org/abs/2106.07177">arXiv</a></p>
<dl>
<dt>问题</dt><dd>预测下一期 SPX IV surface，并保证第二阶段输出满足 static no-arbitrage。</dd>
<dt>数据</dt><dd>OptionMetrics/WRDS，2009-01 至 2020-12，共 3,021 days。原始 daily representation 为 17 deltas × 11 maturities = 187 calls 加 187 puts，共 374 points；先用 DFW regression 映射到 14 log-forward-moneyness × 11 maturities = 154 cells。</dd>
<dt>切分</dt><dd>训练 2009-01 至 2018-06-27，约 2,390 days；测试 2018-06-28 至 2020-12-31。论文没有单独 validation period。</dd>
<dt>第一步表示</dt><dd>比较 PCA、VAE 与直接 sampled surface features。VAE encoder/decoder 各 3 hidden layers × 128 units；latent dimensions 2,5,10,15,20，$d_z=10$ 表现最好。</dd>
<dt>时间模型</dt><dd>LSTM：200 epochs，batch=128，hidden size=12，lr=0.01。第二阶段 DNN：3 hidden layers × 50，20 epochs，batch=1024，lr=0.001；Adam、Xavier initialization、batch normalization。</dd>
<dt>无套利</dt><dd>预测 latent/features 后，不直接把 VAE decoder 当最终输出；另训练一个带静态无套利结构的 DNN surface constructor。这是“表示/动态”和“约束重建”分开的两步法。</dd>
<dt>主要结果</dt><dd>测试 RMSE/MAPE：sampled-features + DNN 为 0.0245/9.90%，VAE + DNN 为 0.0248/9.46%，PCA + DNN 为 0.0544/28.93%，经典 DFW 为 0.0366/15.83%。说明 DNN construction 比 VAE representation 本身更关键。</dd>
<dt>审计</dt><dd><span class="badge audit">5/10</span>网格和第二阶段训练相当清楚，但 OptionMetrics 不公开；没有 validation 却比较多组 latent dimensions，容易产生 hyperparameter-selection ambiguity。</dd>
</dl>
</div>

