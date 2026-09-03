# 4. 什么是 VAE？从概率模型一步一步推导

## 4.1 普通 autoencoder 做了什么，缺了什么？

普通 autoencoder 由 encoder 和 decoder 组成：

$$
z=E(x),\qquad \widehat x=D(z).
$$

训练时最小化重建误差，例如

$$
\|x-D(E(x))\|^2.
$$

它可以压缩曲面，却没有规定 latent space 中“没有训练样本的地方”是什么。随便从 $N(0,I)$ 抽一个 $z$，decoder 未必会输出合理曲面；两个 latent codes 之间线性插值，也可能穿过从未训练过的区域。

VAE 的核心变化是：不再把 $z$ 当成一个确定点，而是把它放进一个生成概率模型。

<div class="paper-card">
<h3>VAE 的三个分布</h3>
<dl>
<dt>先验</dt><dd>$p(z)=N(0,I)$，规定生成时从哪里抽 latent code。</dd>
<dt>decoder likelihood</dt><dd>$p(x\mid z)$，描述给定 latent code 时曲面如何生成。</dd>
<dt>encoder posterior</dt><dd>$q(z\mid x)$，用神经网络近似难以直接计算的真实后验 $p(z\mid x)$。</dd>
</dl>
</div>

下面的 SVG 是概念结构，不对应某一篇论文的具体宽度。

<div style="overflow:auto;margin:1.2em 0">
<svg viewBox="0 0 900 250" role="img" aria-label="VAE architecture" style="min-width:700px;width:100%;height:auto;border:1px solid var(--line);border-radius:14px;background:var(--bg)">
  <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor"/></marker></defs>
  <g fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#arrow)">
    <path d="M145 125 H265"/><path d="M415 95 H495"/><path d="M415 155 H495"/><path d="M590 125 H700"/>
  </g>
  <g text-anchor="middle" font-family="sans-serif">
    <rect x="35" y="75" width="110" height="100" rx="14" fill="var(--paper)" stroke="var(--accent)"/><text x="90" y="115" fill="currentColor" font-size="18">surface x</text><text x="90" y="145" fill="var(--muted)" font-size="14">maturity × strike</text>
    <rect x="265" y="60" width="150" height="130" rx="14" fill="var(--paper)" stroke="var(--accent)"/><text x="340" y="105" fill="currentColor" font-size="18">encoder q(z|x)</text><text x="340" y="138" fill="var(--muted)" font-size="14">输出 μ 与 log variance</text>
    <rect x="495" y="75" width="95" height="100" rx="48" fill="var(--paper)" stroke="var(--accent2)"/><text x="542" y="116" fill="currentColor" font-size="18">z</text><text x="542" y="145" fill="var(--muted)" font-size="13">μ+sε</text>
    <rect x="700" y="60" width="155" height="130" rx="14" fill="var(--paper)" stroke="var(--accent)"/><text x="778" y="105" fill="currentColor" font-size="18">decoder p(x|z)</text><text x="778" y="138" fill="var(--muted)" font-size="14">输出重建/生成曲面</text>
    <text x="455" y="76" fill="var(--muted)" font-size="13">μ</text><text x="455" y="180" fill="var(--muted)" font-size="13">s</text>
  </g>
</svg>
</div>

## 4.2 为什么 marginal likelihood 很难？

生成模型希望最大化每张曲面的概率

$$
p(x)=\int p(x\mid z)p(z)\,dz.
$$

decoder 是神经网络时，这个积分通常没有闭式解。真实 posterior

$$
p(z\mid x)=\frac{p(x\mid z)p(z)}{p(x)}
$$

也因为分母 $p(x)$ 难算而难以直接使用。VAE 引入可训练的近似 posterior $q(z\mid x)$，把每个 $x$ 快速映射到一个 Gaussian：

$$
q(z\mid x)=N\bigl(\mu(x),\operatorname{diag}(s(x)^2)\bigr).
$$

“amortized inference”的意思是：不是对每张新曲面重新做一次昂贵 Bayesian optimization，而是训练一个共享 encoder，一次 forward pass 就近似后验。

## 4.3 ELBO 的完整推导

我们从 $q(z\mid x)$ 与真实 posterior 的 KL divergence 开始：

$$
\operatorname{KL}\bigl(q(z\mid x)\,\|\,p(z\mid x)\bigr)
=\mathbb E_q\left[\log\frac{q(z\mid x)}{p(z\mid x)}\right].
$$

利用 Bayes 公式 $p(z\mid x)=p(x,z)/p(x)$：

$$
\begin{aligned}
\operatorname{KL}(q\|p(z\mid x))
&=\mathbb E_q\left[
\log q(z\mid x)-\log p(x,z)+\log p(x)
\right]\\
&=\log p(x)+
\mathbb E_q\left[
\log q(z\mid x)-\log p(x,z)
\right].
\end{aligned}
$$

移项：

$$
\log p(x)
=\mathbb E_q\left[\log p(x,z)-\log q(z\mid x)\right]
+\operatorname{KL}(q\|p(z\mid x)).
$$

因为 KL 非负，第一项是 $\log p(x)$ 的下界。把它定义为 ELBO：

$$
\mathcal L(x)
=\mathbb E_q\left[\log p(x,z)-\log q(z\mid x)\right]
\le \log p(x).
$$

再用 $p(x,z)=p(x\mid z)p(z)$ 展开：

$$
\begin{aligned}
\mathcal L(x)
&=\mathbb E_q[\log p(x\mid z)]
+\mathbb E_q[\log p(z)-\log q(z\mid x)]\\
&=\mathbb E_q[\log p(x\mid z)]
-\operatorname{KL}\bigl(q(z\mid x)\,\|\,p(z)\bigr).
\end{aligned}
$$

因此最大化 ELBO 等价于同时做两件事：

1. 让 decoder 在从 encoder 抽到的 $z$ 上重建好 $x$；
2. 让每个 approximate posterior 不要离简单先验 $N(0,I)$ 太远。

训练时最小化负 ELBO：

$$
\mathcal J(x)
=-\mathbb E_q[\log p(x\mid z)]
+\operatorname{KL}(q(z\mid x)\|p(z)).
$$

<div class="callout">
<strong>理解重点：</strong>KL 项不是为了“让 latent factors 独立”这么简单。它首先是变分下界中自然出现的 regularizer，用来连接 encoder posterior 与生成时所用的先验。即使每个 $q(z\mid x)$ 都被正则，所有样本混合后的 aggregate posterior 仍可能不等于 $N(0,I)$。
</div>

## 4.4 为什么重建项常常就是 MSE？

若假设

$$
p(x\mid z)=N(D(z),s_x^2 I),
$$

则 negative log likelihood 为

$$
-\log p(x\mid z)
=\frac{1}{2s_x^2}\|x-D(z)\|^2+\text{constant}.
$$

因此固定 output variance 时，最大化 likelihood 等价于最小化 MSE。对 vol surface，这个假设未必理想：不同期限、不同 delta 的报价噪声和 bid–ask 宽度不同。更合理的改进包括：

- 对每个 cell 用 vega、bid–ask 或流动性加权；
- decoder 同时输出均值与 cell-wise uncertainty；
- 用 full/structured covariance 描述跨 cell 误差；
- 在 option price 或 total variance 空间定义 likelihood，而不是直接对 IV 做等权 MSE。

Gopal 2024 的工作正是沿着 heteroscedastic uncertainty 方向改进早期 FX VAE。

## 4.5 Gaussian KL 为什么有闭式？

对一维

$$
q(z\mid x)=N(\mu,s^2),\qquad p(z)=N(0,1),
$$

把两个正态密度代入 KL 并对 $q$ 取期望，可得

$$
\operatorname{KL}(q\|p)
=\frac12\left(\mu^2+s^2-\log s^2-1\right).
$$

各维独立时直接相加：

$$
\operatorname{KL}(q\|p)
=\frac12\sum_j
\left(\mu_j^2+s_j^2-\log s_j^2-1\right).
$$

实现中网络常输出 $\ell_j=\log s_j^2$，避免直接预测必须为正的方差：

$$
\operatorname{KL}(q\|p)
=-\frac12\sum_j\left(1+\ell_j-\mu_j^2-e^{\ell_j}\right).
$$

<details>
<summary>展开一维 Gaussian KL 的中间步骤</summary>

正态 log density 为

$$
\log q(z)=-\frac12\log(2\pi s^2)-\frac{(z-\mu)^2}{2s^2},
\qquad
\log p(z)=-\frac12\log(2\pi)-\frac{z^2}{2}.
$$

所以

$$
\mathbb E_q[\log q-\log p]
=-\frac12\log s^2
-\frac12\frac{\mathbb E_q[(z-\mu)^2]}{s^2}
+\frac12\mathbb E_q[z^2].
$$

使用 $\mathbb E_q[(z-\mu)^2]=s^2$ 与 $\mathbb E_q[z^2]=\mu^2+s^2$，立即得到上式。
</details>

## 4.6 Reparameterization trick

直接写 $z\sim N(\mu,s^2)$ 会让随机采样节点看似阻断对 $\mu,s$ 的梯度。把随机性移到一个与网络参数无关的变量：

$$
\varepsilon\sim N(0,I),
\qquad
z=\mu+s\odot\varepsilon.
$$

此时 $z$ 对 $\mu$ 与 $s$ 是普通可微函数，Monte Carlo 样本上的 loss 可以反向传播。这就是 reparameterization trick。

## 4.7 $\beta$-VAE：这里的 $\beta$ 不一定大于 1

实践中常写

$$
\mathcal J_{\beta}
=\mathcal J_{\mathrm{recon}}
+\beta\operatorname{KL}(q(z\mid x)\|p(z)).
$$

经典 disentanglement 文献有时取 $\beta>1$。但 vol-surface 论文经常取很小的 $\beta$，例如 $10^{-5}$ 或 $10^{-3}$，因为 surface reconstruction 的精度很重要，过强 KL 会导致 posterior collapse 或过度平滑。这里 $\beta$ 是 reconstruction–generation trade-off 的旋钮，不应机械地解释为“越大越好”。

## 4.8 缺失曲面怎样进入 VAE？四种做法

### A. 先训练完整曲面，再对 latent code 做优化

给定只在 mask $m$ 上可见的曲面，求

$$
z^*=\arg\min_z
\left\|m\odot(D(z)-x)\right\|^2+\lambda\|z\|^2.
$$

第一项拟合已知报价，第二项防止 $z$ 跑到先验尾部。Bergeron 2022 与若干后续工作采用这一思路。优点是训练简单；缺点是每张新 surface 都要优化，且结果依赖初值与非凸目标。

### B. 训练 masked encoder，直接 amortize completion

把输入写成

$$
\bigl(m\odot x,\;m\bigr),
$$

明确告诉 encoder 哪些零值是真零、哪些是缺失。训练 loss 可写成

$$
\mathcal J
=\underbrace{\frac{\|(1-m)\odot(\widehat x-x)\|^2}{\sum_i(1-m_i)}}_{\text{hidden-cell loss}}
+\lambda_{\mathrm{obs}}
\underbrace{\frac{\|m\odot(\widehat x-x)\|^2}{\sum_i m_i}}_{\text{observed-cell loss}}
+\beta\,\mathrm{KL}.
$$

Singh et al. 2026 的公开实现采用这种方式。本次复现实验也采用它。

### C. Pseudo-Gibbs imputation

初始化缺失值后交替：

$$
z^{(r+1)}\sim q(z\mid x_{\mathrm{obs}},x_{\mathrm{miss}}^{(r)}),
$$

$$
x_{\mathrm{miss}}^{(r+1)}
\sim p(x_{\mathrm{miss}}\mid z^{(r+1)}).
$$

Richert–Buch 用它处理极高维 swaption volatility cube。优点是能给缺失值分布；缺点是近似链的 stationary distribution 与 mixing 需要谨慎解释。

### D. Conditional VAE

给定 context $c$，学习

$$
p(x_{t+1}\mid c_t)
=\int p(x_{t+1}\mid z,c_t)p(z\mid c_t)\,dz.
$$

$c_t$ 可以包含过去曲面、spot return、VIX、宏观状态或资产标签。Chen et al. 2025 使用 recurrent context 生成未来曲面分布；Ning et al. 用 VIX 条件改善 FX surface generation。

## 4.9 为什么 CNN 可能比 MLP 更适合规则网格？

一个 $6\times7$ surface 不是普通图片，但相邻期限与相邻 moneyness 通常高度相关。$3\times3$ convolution 共享局部权重，能利用：

- smile 在 moneyness 方向的局部平滑；
- term structure 在 maturity 方向的局部平滑；
- 行/列缺失时从邻近区域外推的 inductive bias。

但是 CNN 也有局限：不同 maturity 间距不均匀、delta grid 与 log-moneyness grid 几何不同，而且边界/wings 的经济行为并不平移不变。规则 CNN 的优势必须通过 structured missingness 而不是只靠 random masking 检验。

## 4.10 VAE 怎样加入 no-arbitrage？

<div class="two-col">
<div class="paper-card"><h3>Hard guarantee</h3><p>decoder 不直接输出任意 IV，而输出满足约束的 SVI/SSVI、SDE 或正态化 call-price 参数。任何合法参数经 pricing map 后都在可行域内。</p><p><strong>优点：</strong>可靠。<br><strong>代价：</strong>模型偏差、校准昂贵、参数可识别性。</p></div>
<div class="paper-card"><h3>Soft penalty</h3><p>把 decoded surface 转成 prices，惩罚 calendar、vertical 与 butterfly violations。</p><p><strong>优点：</strong>灵活、易和任意网络结合。<br><strong>代价：</strong>只在训练分布附近有效，不是证明。</p></div>
<div class="paper-card"><h3>Repair / projection</h3><p>先生成，再优化 latent code 或 surface，使其回到可行域。</p><p><strong>优点：</strong>可在部署端加安全层。<br><strong>代价：</strong>可能改变目标分布与控制变量，且增加延迟。</p></div>
<div class="paper-card"><h3>Learn a better latent prior</h3><p>先用 VAE 压缩，再用 normalizing flow、flow matching 或 diffusion 学 aggregate posterior。</p><p><strong>优点：</strong>改善先验采样与尾部。<br><strong>代价：</strong>两阶段训练，仍需约束诊断。</p></div>
</div>

## 4.11 最容易被忽略的四个 VAE failure modes

1. **Aggregate posterior mismatch**：

$$
q_{\mathrm{agg}}(z)=\int q(z\mid x)p_{\mathrm{data}}(x)\,dx
$$

未必等于 $N(0,I)$。重建很好，不代表从 $N(0,I)$ 采样好。

2. **Latent non-identifiability**：各向同性 Gaussian prior 对旋转不敏感。某个 latent axis 与 skew 高相关，不代表该 axis 在另一个随机 seed 下仍是“skew factor”。

3. **Random-mask optimism**：随机缺失会在缺口周围留下大量邻居；真实市场往往整条 tenor、wing 或整个 asset regime 缺失。

4. **Time leakage**：surface 的插值、标准化、PCA、网格修复如果在全样本上拟合，就会把 test-period 信息泄漏到训练中。所有 preprocessing statistics 必须只用 training period。

