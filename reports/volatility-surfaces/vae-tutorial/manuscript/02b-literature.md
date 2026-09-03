<div class="paper-card" id="paper-dierckx">
<h3>4. Dierckx, Davis &amp; Schoutens (2023)：Towards Data-Driven Volatility Modeling with VAEs</h3>
<p class="paper-meta">Thomas Dierckx, Jesse Davis, Wim Schoutens · ECML PKDD workshop proceedings / CCIS · <a href="https://doi.org/10.1007/978-3-031-23633-4_8">DOI</a></p>
<dl>
<dt>问题</dt><dd>把 VAE 从单纯 grid-wise decoder 扩展到 point-wise reconstruction、completion、generation 与 anomaly detection。</dd>
<dt>数据</dt><dd>S&amp;P 500 index volatility surfaces。公开可访问的 proceedings 版本说明了 SPX 应用与 train/test evaluation，但没有提供可直接下载的完整市场数据链。</dd>
<dt>模型</dt><dd>比较 grid-wise VAE 与 point-wise decoder；一种方案把 VAE latent features 与具体 $(k,\tau)$ 坐标交给另一个模型，另有 gradient-boosted trees 作为 decoder。这样 latent representation 与逐点拟合能力解耦。</dd>
<dt>发现</dt><dd>低维 axes 呈现 term structure、smile 与 level 的可解释变形；重建误差还可作为异常分数，长期 deep OTM 区域更容易产生系统性误差。</dd>
<dt>意义</dt><dd>这是从“把 surface 当固定图片”迈向“把 latent state 与 query coordinates 分开”的重要桥梁，和今天的 neural field/operator 思路很接近。</dd>
<dt>审计</dt><dd><span class="badge audit">5/10</span>方法脉络清楚，但数据、全部 preprocessing 与端到端配置不足以无歧义重跑。本报告把它列为方向性文献，而不引用精确 benchmark 数字。</dd>
</dl>
</div>

<div class="paper-card" id="paper-richert">
<h3>5. Richert &amp; Buch (2024)：Interpolation of Missing Swaption Volatility Data using Gibbs-VAE</h3>
<p class="paper-meta">Ivo Richert, Robert Buch · Financial Markets and Portfolio Management · <a href="https://arxiv.org/abs/2204.10400">arXiv 2204.10400</a> · <a href="https://doi.org/10.1007/s41237-023-00213-2">DOI</a></p>
<dl>
<dt>问题</dt><dd>rates 市场中的 normal-vol swaption cube 高维且大量缺失；如何生成缺失 cells 的条件分布，并让结果可用于 SABR calibration 与 hedging？</dd>
<dt>真实数据</dt><dd>FENICS，约两年、约 120 个完整 daily Bachelier-normal-vol cubes。单个 cube 展平后有 4,998 dimensions。论文中的起止日期顺序有一处疑似反写，应在复现时核对。</dd>
<dt>合成扩充</dt><dd>先对 expiry/tenor smiles 拟合 $\beta=0.5$ 的 SABR，再对转换后的 $\alpha,\nu,\rho$ parameter fields 采样，生成 10,000 个 synthetic complete cubes。</dd>
<dt>VAE</dt><dd>latent dimension 10；encoder hidden widths 250,200,150,100，decoder 对称；ReLU；Adam lr=$10^{-6}$；训练 50,000 epochs；初始化权重方差约 $1/30$。</dd>
<dt>推断</dt><dd>对 79.6% missing 的 cube 运行 pseudo-Gibbs chain 2,000 steps，burn-in 100；每步在 encoder posterior 与 decoder missing likelihood 之间交替。</dd>
<dt>主要结果</dt><dd>缺失 IV 的 MAE 报告约 1.9123 bp。以 imputed cube 拟合 SABR 后，误差约 1.05 bp，而直接稀疏插值约 5.84 bp；hedging regression 的 $R^2$ 报告为 0.9778 vs 0.8532。</dd>
<dt>审计</dt><dd><span class="badge audit">3/10</span>专有数据、少量完整真实 cubes、无端到端公开代码；50,000 epochs 也需要明确“一个 epoch 的 batch 定义”才能比较计算量。</dd>
</dl>
</div>

<div class="paper-card" id="paper-gong">
<h3>6. Gong et al. (2024)：A New Encoding of IV Surfaces for Synthetic Generation</h3>
<p class="paper-meta">Zheng Gong, Wojciech Frys, Renzo Tiranti, Carmine Ventre, John O’Hara, Yingbo Bai · CADE / arXiv · <a href="https://arxiv.org/abs/2211.12892">arXiv 2211.12892</a></p>
<dl>
<dt>问题</dt><dd>怎样让三个 latent axes 更稳定地对应 level、skew 与 term structure，并用于 extrapolation 和 index-to-stock mapping？</dd>
<dt>数据</dt><dd>2016-10-04 至 2021-10-04，44 只欧洲股票与 STOXX50 index 的 daily IV surfaces，加 OHLC。训练期截至 2020-10-04，之后测试；另把 6 只股票整体 hold out。</dd>
<dt>网格</dt><dd>8 terms（3,6,9,12,18,24,36,48 months）× 7 moneyness（0.80,0.90,0.95,1.00,1.05,1.10,1.20）= 56 cells。</dd>
<dt>PCA-VAE</dt><dd>在 VAE objective 上加入 sampled latent covariance penalty，使 axes 更接近主方向；报告 $\lambda_{\mathrm{cov}}=0.1$，3 个 active latent dimensions。classic VAE 训练到约 108 epochs 才出现较好 turning point，PCA-VAE 约 35 epochs，因此实际用 40 epochs。</dd>
<dt>extrapolation</dt><dd>只给短端 3/6/9/12m 与 moneyness 0.95/1/1.05，共 12 个已知 cells，恢复剩余 44 cells。</dd>
<dt>主要结果</dt><dd>classic VAE 的平均 known/unknown MAE 约 0.0063/0.0248，satisfaction 0.5887；PCA-VAE 为 0.0103/0.0301，但 satisfaction 0.6865。即平均误差略大，却有更多 cells 落入按 bid–ask 设置的接受区间。</dd>
<dt>审计</dt><dd><span class="badge audit">4/10</span>数据范围、网格、covariance weight 与 epochs 有说明，但网络 widths、optimizer/lr、batch、seed 与代码不足。还要注意 VAE latent axes 本来存在旋转不识别，单一 covariance penalty 并不自动给出跨 seed 的经济标签稳定性。</dd>
</dl>
</div>

<div class="paper-card" id="paper-gopal">
<h3>7. Gopal (2024)：Filling in Missing FX Implied Volatilities with Uncertainties</h3>
<p class="paper-meta">Achintya Gopal · arXiv preprint · <a href="https://arxiv.org/abs/2411.05998">arXiv 2411.05998</a></p>
<dl>
<dt>问题</dt><dd>早期 VAE completion 只给 point estimate；怎样让模型同时输出不确定性，并提高缺失报价下的精度？</dd>
<dt>数据</dt><dd>AUDUSD、USDCAD、EURUSD、GBPUSD、USDMXN，沿用 8 maturities × 5 deltas = 40 cells。训练 2012-01-01 至 2020-02-29；validation 2020-03-01 至 2020-12-31；test 2021-01-01 至 2022-12-31。</dd>
<dt>模型</dt><dd>比较 vanilla VAE、IWAE、residual MLP、heteroscedastic diagonal $\sigma$-VAE 与 full/structured $\Sigma$-VAE。missing-value encoder 在训练后再针对 imputation objective 调整。</dd>
<dt>搜索空间</dt><dd>64 configurations：hidden width 64/128，2/3/4 layers，dropout 0.1/0.2/0.3/0.4，embedding 32/64。</dd>
<dt>训练</dt><dd>Adam，weight decay $10^{-5}$，batch 64，100k gradient steps；lr 从 $10^{-7}$ warm up 到 $2\times10^{-4}$（前 5k steps），50k 后减半。imputation encoder 再训练 10k steps，batch 32，importance samples $k=50$，评估时 10k samples。</dd>
<dt>主要结果</dt><dd>10% missing 时，residual architecture 把示例 MAE 从约 19.9 bps 降到 8.4 bps。validation negative ELBO 中，residual $\Sigma$-VAE 报告约 9.4，优于不带 residual 的多种版本。</dd>
<dt>贡献</dt><dd>明确把“曲面补全”视为 probabilistic inference，而不是纯插值；也暴露了 vanilla VAE 在低 missingness 下未必胜过增强的 parametric baseline。</dd>
<dt>审计</dt><dd><span class="badge audit">6/10</span>优化 schedule 与 ablation 很详细，但数据不公开，且未见完整公开训练代码。无套利仍不是核心保证。</dd>
</dl>
</div>

<div class="paper-card" id="paper-feugang">
<h3>8. Feugang Nteumagné, Donfack &amp; Wafo Soh (2025)：Variational Autoencoders for Completing the Volatility Surfaces</h3>
<p class="paper-meta">Bienvenue Feugang Nteumagné, Hermann Azemtsa Donfack, Celestin Wafo Soh · Journal of Risk and Financial Management 18(5), 239 · <a href="https://doi.org/10.3390/jrfm18050239">DOI</a></p>
<dl>
<dt>问题</dt><dd>当规则网格中一大块 IV 缺失时，能否先从完整合成曲面学出低维流形，再通过 latent optimization 找到与已知 cells 相符的完整曲面？</dd>
<dt>数据</dt><dd>从 Heston model 生成 20,000 张曲面，清洗后约 13,500 张；80/20 train/test。参数范围报告为 $v_0\in[0.025,0.035]$、$\rho\in[-0.87,-0.067]$、vol-of-vol $\in[0.5,1.5]$、长期方差 $\in[0.08,0.10]$、mean reversion $\in[0.1,2.2]$。</dd>
<dt>网格</dt><dd>15 maturities（0.1 至 1.5 年，步长 0.1）× 17 strikes（0.50 至 1.30，步长 0.05）= 255 cells。</dd>
<dt>VAE</dt><dd>全连接结构 255→128→64→32→16 latent，再对称解码到 255；ELU hidden activations，sigmoid output；100 epochs。文章写“100 batches”，但没有无歧义说明是 batch size 还是每 epoch 的 batch 数。</dd>
<dt>补全</dt><dd>测试时随机去掉 100 个 cells，冻结 decoder，通过优化 latent vector 使已知 155 个 cells 上的平方误差最小；这对应第 4.8 节的方法 A，而不是 masked encoder。</dd>
<dt>贡献</dt><dd>数据是可原则上重新生成的，适合作为教学实验；同时也提醒我们：若所有样本都来自同一个五参数 Heston family，VAE 的优势可能只是学习一个已知低维参数流形，必须与 PCA、直接 Heston calibration 和 SVI baseline 比较。</dd>
<dt>审计</dt><dd><span class="badge audit">5/10</span>参数范围与主结构足够重造近似数据，但学习率、随机种子、精确 batching、清洗拒绝规则、模型选择过程及完整代码不足，无法保证逐表复现。</dd>
</dl>
</div>

<div class="paper-card" id="paper-wang">
<h3>9. Wang, Liu &amp; Vuik (2025)：Controllable Generation of IV Surfaces with VAEs</h3>
<p class="paper-meta">Jing Wang, Shuaiqiang Liu, Cornelis Vuik · arXiv 2509.01743 · <a href="https://arxiv.org/abs/2509.01743">arXiv</a></p>
<dl>
<dt>问题</dt><dd>普通 VAE 只能说“随机生成一张像历史的曲面”，却很难要求“短端 vol 上升 5 points、左 skew 变陡而 term structure 不变”。该文把 level、slope、curvature、term-structure descriptors 作为显式控制变量。</dd>
<dt>数据</dt><dd>60,000 张合成曲面：30,000 Heston 加 30,000 SABR；规则网格为 28 maturities × 28 log-moneyness，$k\in[-0.27,0.27]$、$\tau\in[0.1,0.6]$。</dd>
<dt>表示</dt><dd>先在 anchor point 附近用局部回归把曲面压成四个可解释 shape features；decoder 接收这些控制量和额外 5 维 residual latent。这样把“我要什么形状”与“其余无法解释的细节”分开。</dd>
<dt>网络</dt><dd>encoder hidden widths 256,128；decoder 128,256；ReLU；batch 64；Adam lr=$3\times10^{-4}$；最多 5,000 epochs。控制变量直接作为 decoder input，residual code 承担剩余 variation。</dd>
<dt>无套利</dt><dd>在训练分布凸包内且 residual coordinates 限制在 $[-3,3]$ 时，文中 60,000 个生成样本未检测到 violations；扩大先验范围后约 9% 违反约束。作者再固定控制量，只调整 residual latent，做 post-generation repair。</dd>
<dt>关键洞见</dt><dd>可解释 latent 不是仅靠“观察某一轴像 skew”得到，而是把经济 descriptor 写进模型输入。另一方面，“样本中零 violation”仍是有限域经验结论，不是对所有 latent codes 的数学保证。</dd>
<dt>审计</dt><dd><span class="badge audit">7/10</span>合成数据规模、网格、网络与 optimizer 很清楚，原则上可重造；但各实验的 $\beta$、完整参数采样分布、seed 与代码尚不足以 bit-for-bit 重跑。</dd>
</dl>
</div>

