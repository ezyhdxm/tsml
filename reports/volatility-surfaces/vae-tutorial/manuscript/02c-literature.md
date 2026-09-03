<div class="paper-card" id="paper-chen">
<h3>10. Chen et al. (2025)：Conditional Generation of Possible Future Volatility Surfaces</h3>
<p class="paper-meta">Jacky Chen, John Hull, Zissis Poulos, Haris Rasul, Andreas Veneris, Yuntao Wu · Journal of Financial Data Science · <a href="https://doi.org/10.3905/jfds.2025.1.196">DOI</a> · <a href="https://github.com/rotmanfinhub/vol-surface-vae-pub">公开代码</a></p>
<dl>
<dt>问题</dt><dd>给定最近若干天的 SPX surface 和市场状态，生成未来一天或多天的 surface distribution，而不只是重建同一天。</dd>
<dt>数据</dt><dd>OptionMetrics SPX，2000-01 至 2023-02。过滤无效 IV、open interest 为零、$K/S\notin[0.7,1.3]$、期限不在 1 month–2 years 的报价，再插值为 5×5 moneyness–maturity grid。</dd>
<dt>时间切分</dt><dd>前 4,000 个交易日（2000-01-03 至 2015-11-27）训练；随后 1,000 日 validation 至 2019-11-18；最后 1,000 日 test 至 2023-02-24。该严格 chronological split 比随机切分可信得多。</dd>
<dt>结构</dt><dd>surface encoder 采用 3 层 convolution，输出 5 维 latent；context memory 为 2 层 recurrent network，每层 hidden size 100，dropout 0.2；surface/context embedding 均为 5 维。训练时 context length 在 4–10 日之间变化。</dd>
<dt>训练</dt><dd>$\beta=10^{-5}$，500 epochs；PyTorch 1.13、RTX 3080 Ti、seed 0、float64。公开仓库给出 preprocessing、VAE/CVAE/LSTM/GRU/RNN variants、训练与生成脚本；原始 OptionMetrics 仍需许可证。</dd>
<dt>报告结果</dt><dd>validation 的 surface reconstruction loss 约 $8.97\times10^{-4}$，test 约 $2.43\times10^{-2}$；明显的 validation–test gap 说明 2020–2023 regime shift 很重要。论文还比较 stochastic paths、maximum-likelihood path 与 SABR synthetic experiments。</dd>
<dt>关键风险</dt><dd>目标 surface 是日频插值后的 25 维对象，模型可能同时学习“真实经济动态”和 preprocessing 的平滑规则。多日生成若逐日 autoregressive rollout，还要区分 conditional uncertainty、model drift 与 error accumulation。</dd>
<dt>审计</dt><dd><span class="badge audit">8/10</span>时间切分、结构、seed、软件环境和代码相当完整；主要阻塞是 OptionMetrics license、插值版本与公开云盘中预处理二进制文件的长期可访问性。</dd>
</dl>
</div>

<div class="paper-card" id="paper-singh">
<h3>11. Singh, Reddy &amp; Chopra (2026)：Beyond the Smile — Hybrid ConvVAE for Crypto</h3>
<p class="paper-meta">Sadanand Singh, Allam Reddy, Manan Chopra · arXiv preprint · <a href="https://arxiv.org/abs/2606.16961">arXiv</a> · <a href="https://github.com/jasper-research/beyond-the-smile-paper">代码、处理数据与 checkpoints</a></p>
<dl>
<dt>问题</dt><dd>在高度稀疏且跨 tenor/delta 有局部结构的 crypto IV grid 上，怎样做 completion、anomaly detection，并把 neural reconstruction 与可解释的 per-tenor smile fit 组合成可部署 router？</dd>
<dt>数据</dt><dd>Binance Options，2023-05 至 2023-10，共 6,034 个完整 hourly surfaces；网格为 6 tenors × 7 deltas = 42 cells。BTC 2,821 surfaces、ETH 3,213；文章同时给出各资产原始可用小时数与完整网格筛选后的样本数。</dd>
<dt>切分</dt><dd>按时间 70/15/15：BTC 1,974/423/424，ETH 2,249/481/483。随机 mask 10%–50%，另专门测试整行、整列、双翼与最长 tenor 缺失。</dd>
<dt>损失</dt><dd>隐藏 cells MSE 权重 1，已观测 cells MSE 权重 0.1，KL 权重 $10^{-3}$。mask 本身作为第二输入 channel，避免网络把“缺失填零”误认为真实零。</dd>
<dt>ConvVAE</dt><dd>三层 $3\times3$ Conv2d、64 channels、GELU、latent 16；Adam lr=$10^{-3}$、batch 128、最多 500 epochs、patience 50。</dd>
<dt>报告结果</dt><dd>BTC random 10/30/50% hidden RMSE：MLP 1.40/1.51/1.67 vol points，Conv 0.94/1.07/1.25；整行缺失 3.43 vs 1.88；long-tenor 2.26 vs 1.54。50% random mask 下，hybrid router 0.83，优于 ConvVAE 1.25 与独立 quadratic smiles 7.00。</dd>
<dt>为什么适合复现</dt><dd>仓库包含源码、配置、processed 6×7 data、checkpoint、ablation 与 provenance 文档，是目前审计透明度最高的 vol-surface VAE 工作之一。本报告实际实现的 masked MLP/ConvVAE 结构与损失正是以它为主要参照。</dd>
<dt>审计</dt><dd><span class="badge audit">9/10</span>从 raw archive provenance 到 checkpoint 都有记录。本文执行环境未能从外网传入其 23 MB 二进制归档，因此没有冒充 exact reproduction；后面做的是等网格、等任务的 SSVI 结构复现。</dd>
</dl>
</div>

<div class="paper-card" id="paper-flow">
<h3>12. Brooks, Bajalica, Liu &amp; Ben Tahar (2026)：Latent Flow Matching</h3>
<p class="paper-meta">Oscar Brooks, Dusica Bajalica, Yating Liu, Imen Ben Tahar · arXiv 2608.00616 · <a href="https://arxiv.org/abs/2608.00616">arXiv</a> · <a href="https://github.com/DusBaja/ivs-generative-benchmark">code</a></p>
<dl>
<dt>问题</dt><dd>即使 VAE reconstruction 好，直接采 $z\sim N(0,I)$ 仍可能因为 aggregate posterior mismatch 生成错误尾部。该文先用 VAE 压缩，再学习一个 flow，把简单高斯连续输运到经验 latent distribution。</dd>
<dt>数据</dt><dd>OptionsDX SPX，2020-01 至 2023-12，约 1,000 daily surfaces；原始报价经 SVI、total-variance interpolation、nearest-neighbor fill 与 calendar cumulative-max 处理，得到 32×16 = 512 cells。训练值按第 4/96 百分位稳健归一化。</dd>
<dt>VAE</dt><dd>512→256→128→64→6 latent，对称 decoder；residual MLP、ReLU；$\beta=10^{-2}$，calendar penalty $3\times10^{-2}$，butterfly penalty $2\times10^{-3}$；lr=$2\times10^{-3}$、batch 64、1,200 epochs、KL warm-up 200。</dd>
<dt>Flow</dt><dd>latent vector field MLP 64→128→64，time embedding 32；lr=$10^{-4}$、batch 64、1,200 epochs；采样时用 100 个 Euler steps 解 ODE。</dd>
<dt>评估</dt><dd>5 independent runs，每次生成 5,000 surfaces；同时看 pointwise Wasserstein、sliced Wasserstein、smile/skew、tail quantiles、financial factors 与 static-arbitrage diagnostics。</dd>
<dt>报告结果</dt><dd>Wasserstein-1 / sliced-Wasserstein：latent flow 0.00283/0.00697，diffusion 0.00694/0.00782，VolGAN 0.03447/0.02960，raw Gaussian latent 0.00287/0.00491。全部静态检查通过率：训练数据 51.2%，flow 90.8%±1.5%，diffusion 46.7%，VolGAN 69.7%，raw latent 32.1%。</dd>
<dt>最重要解释</dt><dd>VAE decoder 提供“曲面 manifold”，flow 负责学习“manifold 上的真实概率质量”。这正面解决了本报告复现实验里 no-arbitrage penalty 对 posterior reconstructions 有效、对 prior samples 反而失效的问题。</dd>
<dt>审计</dt><dd><span class="badge audit">8/10</span>模型、训练、评估和公开代码详细；OptionsDX 原始数据受许可，且 SVI/interpolation/repair 的版本需要精确锁定。</dd>
</dl>
</div>

<div class="paper-card" id="paper-buchegger">
<h3>13. Buchegger &amp; Gonon (2026)：Arbitrage-Aware Multi-Step Forecasting with Latent Diffusion</h3>
<p class="paper-meta">Dominik Manuel Buchegger, Lukas Gonon · arXiv 2608.22478，2026-08-23 · <a href="https://arxiv.org/abs/2608.22478">arXiv</a></p>
<dl>
<dt>问题</dt><dd>不是生成一张独立曲面，而是给定过去 21 日，联合生成未来 30 日的 IV-surface trajectory 与 underlying returns。</dd>
<dt>数据</dt><dd>OptionMetrics SPX，2000-01 至 2025-08。每个交易日先通过 pretrained operator-deep-smoothing 模型把 irregular quotes 映射到统一网格；原始 moneyness $-1.5,-1.4,\ldots,0.5$ 与 1–12 months 网格中保留 170 个受支持 coordinates。</dd>
<dt>表示层</dt><dd>使用 arbitrage-aware regularized autoencoder，而非需要高斯 posterior 的标准 VAE。loss 为 reconstruction 加 weight decay、butterfly 与 calendar penalties，权重分别 $10^{-4}$、$10^{-3}$、$10^{-4}$；最终在 train+validation 上训练 300 epochs。</dd>
<dt>动态层</dt><dd>观察 21 日、预测 30 日；future target 用相对共同 anchor 的 latent displacement 与 returns。Transformer-style denoiser 使用 500 diffusion steps、线性 variance schedule $10^{-4}\to10^{-2}$、$v$-parameterization，训练 100 epochs。</dd>
<dt>样本</dt><dd>5,461/221/637 个 train/validation/test trajectory origins；选择模型后在 5,712 个 train+validation trajectories 上重训。每个 test origin 生成 1,000 条 30 日路径。另训练 15 epochs 的 horizon-specific mean-scaling gate，平均缩放约 0.80。</dd>
<dt>报告结果</dt><dd>前三个 realized-increment PCs 解释 91.76% variation；sample path 的全曲面无套利通过率 88.1%，ensemble mean 98.4%。短期一日预测相对 bootstrap persistence 优势有限甚至劣势，但 10–30 日的分布预测明显改善；95% intervals 仍存在 undercoverage。</dd>
<dt>意义</dt><dd>这篇工作把 surface representation、joint return dynamics、trajectory uncertainty 与 admissibility 放进同一 pipeline；同时诚实揭示 underdispersion 与 one-day horizon 的难度。</dd>
<dt>审计</dt><dd><span class="badge audit">8/10（初评）</span>数据构造协议和公开代码链接很完整，但文章在本报告截止日前仅发布十余天，尚需独立重跑与版本稳定性检验。</dd>
</dl>
</div>

<div class="paper-card" id="paper-geometry">
<h3>14. Wang, Liu &amp; Vuik (2026)：Latent-Space No-Arbitrage Geometry</h3>
<p class="paper-meta">Jing Wang, Shuaiqiang Liu, Cornelis Vuik · arXiv 2609.00332，2026-08-31 · <a href="https://arxiv.org/abs/2609.00332">arXiv</a></p>
<dl>
<dt>问题</dt><dd>固定一个已经训练好的 decoder $g$ 后，哪些 latent codes 会产生 admissible surfaces？“通过率”只是采样分布下的概率，不能告诉我们可行区域本身的形状。</dd>
<dt>核心对象</dt><dd>把每张生成曲面的最差 calendar/butterfly constraint value 压成一个标量 margin $M(z)$。可行集为 $\{z:M(z)\ge0\}$，边界通常位于 $M(z)=0$。正 margin 意味着存在一个仍然可行的邻域。</dd>
<dt>实验</dt><dd>30,000 张 Heston surfaces，28×28 grid，$k\in[-0.27,0.27]$、$\tau\in[0.1,0.6]$；70/15/15 split。VAE latent 2；selected encoder widths 784,512,256,128，symmetric decoder，SiLU，sigmoid output，KL weight 0.3 with warm-up；5 random seeds。</dd>
<dt>边界计算</dt><dd>在 $[-4,4]^2$ 的 161×161 latent grid 直接计算 margin，用 200,000 个 fixed-seed Gaussian samples 估计 prior admissibility，再对 sign-changing edges 二分到 $10^{-5}$。</dd>
<dt>报告结果</dt><dd>五个 seeds 的 test IV-RMSE 几乎相同，均值 0.00646；但 latent box 的 admissible area 从 0.334 到 0.421，Gaussian prior admissibility 从 0.921 到 0.981。99.7% 的 refined boundary points 由 strike convexity 激活；local correction 将 40 个 violating codes 中的 33 个推过边界。</dd>
<dt>关键结论</dt><dd>reconstruction error 无法决定 generator 的无套利几何；可行区域面积也不等于某个 prior 下的可行概率。这个区分为 latent repair、safe sampling 和 distribution-aware regularization 提供了更精确的理论语言。</dd>
<dt>审计</dt><dd><span class="badge audit">7/10（初评）</span>实验设定和多 seed 统计相当清楚，但论文发布于本报告截止日前两天；高维 level-set 算法与代码可得性仍需后续审计。</dd>
</dl>
</div>

<div class="callout warning">
<strong>仍在快速演化的相邻工作。</strong>2026 年还有以 Student-$t$ latent、crisis indicators 与 soft arbitrage penalties 做压力情景的 Crisis-VAE 预印本，以及不用 VAE 的 VolGAN、surface diffusion、operator deep smoothing、neural-field smoothing。它们对方法选择很重要，但本报告只在其直接回答“VAE 作为表示或生成层”的地方展开，避免把所有 deep-volatility papers 混为一类。
</div>

