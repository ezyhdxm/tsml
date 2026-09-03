# 8. 本次实跑：在可控的 SSVI 数据上做结构复现

## 8.1 我实际跑了什么？

<div class="callout success">
<span class="badge executed">本次实跑</span><strong>这不是对某篇市场数据论文的 exact replication。</strong>我重新实现了 6×7 masked MLP-VAE、2D ConvVAE、reconstruction-side no-arbitrage penalty、PCA completion 与七种 missingness tests，并在从零生成的、通过离散静态无套利检查的 SSVI 曲面上训练和评估。所有表格均由本次运行的 <code>results.json</code> 自动生成。
</div>

<div class="table-wrap"><table id="run-summary" class="data-table"><thead><tr><th>项目</th><th>本次实跑</th></tr></thead><tbody><tr><td>实验等级</td><td>结构复现（不是原始市场数据的精确复制）</td></tr><tr><td>数据</td><td>1800 个 SSVI 曲面；1260/270/270 train/validation/test；全部通过离散静态无套利检查</td></tr><tr><td>网格</td><td>6 maturities × 7 log-forward-moneyness = 42 cells</td></tr><tr><td>训练</td><td>Adam lr=0.001; batch=128; max epochs=45; patience=12; beta=0.001; seed=20260903</td></tr><tr><td>环境</td><td>Python 3.13.5; PyTorch 2.10.0+cpu; CPU</td></tr><tr><td>PCA baseline</td><td>PCA-8 explained variance = 99.99579%</td></tr></tbody></table></div>

复现脚本固定 seed=20260903，并保存：原始合成曲面、生成因子、训练曲线、三个 checkpoints、逐 mask 指标、单个补全案例和全部图形。这样读者可以把“报告叙述”与“机器输出”逐项核对。

## 8.2 数据怎样生成？从四个因子到一张 6×7 曲面

### 第一步：固定网格

maturity grid 为

$$
\tau\in\{14,30,60,90,120,180\}/365,
$$

log-forward-moneyness grid 为

$$
k\in\{-0.30,-0.20,-0.10,0,0.10,0.20,0.30\}.
$$

因此一张曲面有 $6\times7=42$ 个 cells，和公开 crypto ConvVAE 论文的维度一致，但坐标从 delta 改成更便于静态套利分析的 $k$。

### 第二步：生成一条递增的 ATM total-variance curve

抽取短端 volatility $s$ 与长期 volatility $\ell$，再定义

$$
\theta(\tau)
=
\ell^2\tau
+
(s^2-\ell^2)\frac{1-e^{-3\tau}}{3}.
$$

这里 $\theta(\tau)=w(0,\tau)$。为什么这个形式方便？对 $\tau$ 求导：

$$
\theta'(\tau)
=
\ell^2+(s^2-\ell^2)e^{-3\tau}.
$$

当 $s\ge\ell$ 时，导数显然为正；当 $s<\ell$ 时，最小值出现在 $\tau=0$，等于 $s^2>0$。所以 ATM total variance 随 maturity 严格增加。

本次采样范围为：

$$
s\in[0.16,0.62],
\qquad
\log(\ell/s)\in[-0.55,0.35].
$$

### 第三步：用 SSVI 把 ATM curve 扩成 smile surface

对每个 maturity，使用

$$
\phi(\theta)=\frac{\eta}{\sqrt{\theta}},
$$

以及 SSVI slice

$$
w(k,\tau)
=
\frac{\theta}{2}
\left[
1+\rho\phi k
+
\sqrt{(\phi k+\rho)^2+1-\rho^2}
\right].
$$

最后转回 implied volatility：

$$
\sigma(k,\tau)=\sqrt{\frac{w(k,\tau)}{\tau}}.
$$

四个真实生成因子分别是：短端 level $s$、term log-ratio、skew parameter $\rho$ 与 smile-intensity parameter $\eta$。采样 $\rho\in[-0.90,-0.05]$，并限制 $\eta$ 的上界，使极端 skew 下不过度弯曲。

<figure>
<img src="reproduction/sample_surface_3d.png" alt="Synthetic SSVI implied volatility surface">
<figcaption>图 6　本次生成的一张 SSVI surface。左翼较高来自负 $\rho$；沿 maturity 的 level 由 ATM total-variance curve 控制。</figcaption>
</figure>

<figure>
<img src="reproduction/sample_smiles.png" alt="Three volatility smiles from one SSVI surface">
<figcaption>图 7　同一张曲面的 14d、60d、180d smiles。VAE 需要同时学会每条 smile 的横截面形状和不同 maturity 之间的共动。</figcaption>
</figure>

### 第四步：只保留通过离散检查的曲面

每张 IV surface 先转成 $F=1$ 下的 normalized Black call prices：

$$
c(k,\tau)=N(d_1)-e^kN(d_2).
$$

然后检查：

1. maturity 增加时 call price 不下降；
2. strike-direction secant slope 位于 $[-1,0]$；
3. secant slope 随 strike 不下降，即离散 convexity；
4. IV 为正且有限。

本次保存的 1800 张源曲面全部通过这些**有限网格**检查。这里刻意不用“全连续域严格无套利”措辞，因为网格间与远端 tail 仍需要额外条件。

## 8.3 数据切分与标准化

独立合成样本按固定生成顺序分为 1260 / 270 / 270 个 train/validation/test。由于这里没有时间序列，切分只用于防止模型直接记忆样本，不代表 out-of-time regime test。

为了保证 IV 为正并减少不同 cells 的尺度差异，先取 log：

$$
y_{ij}=\log\sigma_{ij}.
$$

再只用 training set 计算每个 cell 的均值与标准差：

$$
x_{ij}=\frac{y_{ij}-\bar y_{ij}}{s_{ij}}.
$$

所有网络预测 $x$，最后用

$$
\widehat\sigma_{ij}
=
\exp(\bar y_{ij}+s_{ij}\widehat x_{ij})
$$

转回 IV。这样模型不会生成负 volatility，且不会从 validation/test 泄漏 normalization statistics。

## 8.4 五个比较对象

### Mean baseline

缺失 cells 直接填训练集 cell-wise mean。标准化后就是填 0。它检验模型是否真的利用了同一张曲面剩余 cells，而不是只记住平均 surface。

### PCA-8：不是普通的先填均值再 inverse transform

设 PCA loading matrix 为 $B$，只看已观测坐标集合 $O$。对每张不完整曲面，解一个 ridge least-squares factor：

$$
\widehat f
=
\arg\min_f
\|B_Of-(x_O-\bar x_O)\|^2+10^{-4}\|f\|^2.
$$

闭式解为

$$
\widehat f
=
(B_O^\top B_O+10^{-4}I)^{-1}B_O^\top(x_O-\bar x_O),
$$

再用 $\bar x+B\widehat f$ 恢复所有 cells。这是对 missing observations 合理的 PCA completion，而不是让测试集真值参与 factor estimation。

### MLP-VAE 与 ConvVAE

两者都把 $(x\odot m,m)$ 作为 encoder input。MLP 将 42 cells 展平；ConvVAE 将 value 与 mask 堆成两个 $6\times7$ channels，用 $3\times3$ convolutions 学局部 tenor–moneyness patterns。

<div class="table-wrap"><table id="model-table" class="data-table"><thead><tr><th>模型</th><th>结构</th><th>可训练参数</th></tr></thead><tbody><tr><td>MLP-VAE</td><td>84 → 64 → 64 → (μ, log variance), z=16; decoder symmetric</td><td>19658</td></tr><tr><td>ConvVAE</td><td>2-channel 6×7; 3 Conv2d layers, 16 channels; z=16</td><td>44881</td></tr><tr><td>ConvVAE+NA</td><td>与 ConvVAE 相同；重建项另加离散 calendar/vertical/butterfly penalty</td><td>44881</td></tr></tbody></table></div>

### ConvVAE+NA

结构与 ConvVAE 相同，只在 posterior-conditioned reconstruction 上增加离散 calendar、vertical-spread 与 convexity penalties。它是故意设置的机制实验：检验一个常见的 soft-constraint 做法究竟会改善什么、又不会改善什么。

## 8.5 训练 loss 的每一项

每个 mini-batch 都重新采样缺失率 $q\sim U(0.1,0.5)$，再对每个 cell 生成 mask。隐藏单元损失为

$$
L_{\mathrm{hid}}
=
\frac{\sum_{j}(1-m_j)(\widehat x_j-x_j)^2}
{\sum_j(1-m_j)}.
$$

已知单元也需要保持一致，但权重较小：

$$
L_{\mathrm{obs}}
=
\frac{\sum_jm_j(\widehat x_j-x_j)^2}
{\sum_jm_j}.
$$

基础 VAE loss 是

$$
L
=
L_{\mathrm{hid}}
+0.1L_{\mathrm{obs}}
+10^{-3}\operatorname{KL}(q(z\mid x,m)\|N(0,I)).
$$

带约束版本再加

$$
25\,P_{\mathrm{arb}}(\widehat\sigma).
$$

这里 $P_{\mathrm{arb}}$ 对 call-price calendar violation、slope 超出 $[-1,0]$ 和 slope 非单调的负 margin 做平方 ReLU。注意：penalty 作用于 decoder 对训练曲面的重建，而不是对独立 $z\sim N(0,I)$ 的样本。

Adam learning rate $10^{-3}$，batch 128，最多 45 epochs，validation 使用固定 30% random mask，patience 12，并对 gradient norm 截断到 10。

<figure>
<img src="reproduction/training_curves.png" alt="Validation hidden cell loss by epoch">
<figcaption>图 8　三个 VAE 的 validation hidden-cell MSE。曲线来自实际训练日志；早停选择的是总 validation objective，而非事后挑 test error 最低的 epoch。</figcaption>
</figure>

## 8.6 Completion：随机缺失与结构化缺失

测试七种模式：随机隐藏 10%、30%、50%；随机删除一整条 maturity；随机删除一整条 moneyness column；删除左右双翼；删除最长 maturity。误差单位是 **vol points**，即 IV 小数误差乘 100。方括号给出按 test surfaces bootstrap 500 次的 95% interval。

<div class="table-wrap"><table id="completion-table" class="data-table"><thead><tr><th>缺失机制</th><th>Mean</th><th>PCA-8</th><th>MLP-VAE</th><th>ConvVAE</th><th>ConvVAE+NA</th></tr></thead><tbody><tr><td>随机隐藏 10%</td><td>12.210 [11.356, 13.125]</td><td>0.107 [0.096, 0.118]</td><td>2.355 [2.196, 2.501]</td><td>1.375 [1.273, 1.473]</td><td>1.334 [1.227, 1.438]</td></tr><tr><td>随机隐藏 30%</td><td>12.903 [12.125, 13.788]</td><td>0.123 [0.111, 0.136]</td><td>2.706 [2.551, 2.872]</td><td>1.689 [1.593, 1.782]</td><td>1.548 [1.463, 1.639]</td></tr><tr><td>随机隐藏 50%</td><td>12.856 [12.124, 13.595]</td><td>0.200 [0.167, 0.239]</td><td>3.031 [2.855, 3.206]</td><td>1.982 [1.851, 2.122]</td><td>2.071 [1.923, 2.230]</td></tr><tr><td>整条期限缺失</td><td>12.766 [11.864, 13.575]</td><td>0.115 [0.103, 0.128]</td><td>2.414 [2.241, 2.595]</td><td>1.703 [1.552, 1.849]</td><td>1.428 [1.316, 1.543]</td></tr><tr><td>整条 moneyness 缺失</td><td>12.124 [11.233, 13.003]</td><td>0.158 [0.143, 0.176]</td><td>3.166 [2.936, 3.436]</td><td>1.622 [1.520, 1.734]</td><td>1.651 [1.520, 1.775]</td></tr><tr><td>双翼缺失</td><td>13.363 [12.619, 14.128]</td><td>0.201 [0.185, 0.218]</td><td>3.881 [3.642, 4.094]</td><td>2.235 [2.096, 2.376]</td><td>2.464 [2.319, 2.611]</td></tr><tr><td>最长到期缺失</td><td>12.499 [11.655, 13.472]</td><td>0.174 [0.161, 0.188]</td><td>2.562 [2.425, 2.690]</td><td>1.644 [1.509, 1.778]</td><td>1.749 [1.612, 1.902]</td></tr></tbody></table></div>

<div class="metric-grid">
<div class="metric"><b>3.031 → 1.982</b><small>随机隐藏 50%：MLP-VAE → ConvVAE，降低 34.6%</small></div>
<div class="metric"><b>2.414 → 1.703</b><small>整条期限缺失：MLP-VAE → ConvVAE，降低 29.5%</small></div>
<div class="metric"><b>1.703 → 1.428</b><small>整条期限缺失：ConvVAE → ConvVAE+NA，降低 16.1%</small></div>
</div>

<figure>
<img src="reproduction/completion_results.png" alt="Completion RMSE for all models and masking schemes">
<figcaption>图 9　所有 completion tests。ConvVAE 相对 MLP 的优势在整列与双翼缺失上尤其明显，说明二维局部共享确有作用；但 PCA-8 在这个四因子、无噪声、光滑数据上压倒性更强。</figcaption>
</figure>

<figure>
<img src="reproduction/completion_example.png" alt="One full-row missing completion example">
<figcaption>图 10　一张整条 maturity 缺失的案例。图像只用于直观检查，正式比较仍以上表所有 test surfaces 的误差为准。</figcaption>
</figure>

### 怎样解释 PCA 的巨大优势？

PCA-8 在 training set 上解释 99.99579% 方差，而数据真实只有四个连续生成因子；局部非线性流形在有限范围内几乎被八维线性空间覆盖。因此这个实验支持的是：

- CNN inductive bias 确实优于 flatten MLP；
- 但这并不足以战胜一个正确实现的低秩线性 baseline；
- VAE 的价值更可能出现在 noisy/multimodal surfaces、probabilistic imputation、nonlinear regimes 与 prior generation，而不是所有 completion benchmark。

## 8.7 完整重建与 prior generation 不是一回事

先给 encoder 完整曲面，再用 posterior mean 重建：

<div class="table-wrap"><table id="reconstruction-table" class="data-table"><thead><tr><th>模型</th><th>全曲面重建 RMSE（vol points）</th><th>离散无套利通过率</th></tr></thead><tbody><tr><td>MLP-VAE</td><td>2.784</td><td>87.04%</td></tr><tr><td>ConvVAE</td><td>1.717</td><td>93.70%</td></tr><tr><td>ConvVAE+NA</td><td>1.663</td><td>99.63%</td></tr></tbody></table></div>

ConvVAE+NA 将重建曲面的离散无套利通过率提高到接近 100%，说明 penalty 在**训练流形附近**有效。接着完全绕过 encoder，直接采

$$
z\sim N(0,s^2I),
$$

其中 $s=1$ 或 $1.5$，再由 decoder 生成曲面：

<div class="table-wrap"><table id="generation-table" class="data-table"><thead><tr><th>模型</th><th>先验尺度</th><th>全部通过</th><th>calendar 通过</th><th>butterfly 通过</th><th>IV P1</th><th>IV P50</th><th>IV P99</th></tr></thead><tbody><tr><td>MLP-VAE</td><td>1.0</td><td>97.11%</td><td>99.94%</td><td>97.17%</td><td>24.32%</td><td>38.77%</td><td>65.92%</td></tr><tr><td>MLP-VAE</td><td>1.5</td><td>78.61%</td><td>97.94%</td><td>80.11%</td><td>20.64%</td><td>39.58%</td><td>69.64%</td></tr><tr><td>ConvVAE</td><td>1.0</td><td>98.61%</td><td>99.56%</td><td>99.00%</td><td>21.54%</td><td>37.39%</td><td>65.16%</td></tr><tr><td>ConvVAE</td><td>1.5</td><td>81.22%</td><td>94.33%</td><td>86.06%</td><td>18.67%</td><td>39.43%</td><td>72.34%</td></tr><tr><td>ConvVAE+NA</td><td>1.0</td><td>81.44%</td><td>96.11%</td><td>84.17%</td><td>20.87%</td><td>40.97%</td><td>70.85%</td></tr><tr><td>ConvVAE+NA</td><td>1.5</td><td>39.33%</td><td>78.83%</td><td>47.89%</td><td>19.18%</td><td>45.73%</td><td>79.14%</td></tr></tbody></table></div>

<figure>
<img src="reproduction/generation_validity.png" alt="Arbitrage validity under prior sampling">
<figcaption>图 11　标准先验和放大先验下的离散无套利通过率。latent tails 明显更危险；简单 reconstruction penalty 对 ConvVAE+NA 的 prior samples 产生了反直觉退化。</figcaption>
</figure>

<div class="callout negative">
<strong>负结果：soft penalty 改善 reconstruction，却伤害 prior generation。</strong>标准先验下，普通 ConvVAE 的全部通过率约 98.61%，ConvVAE+NA 只有 81.44%；先验尺度放大到 1.5 后，后者进一步降到约 39.33%。这不是说 no-arbitrage penalty 必然有害，而是说“只在 posterior reconstruction 上施加 penalty”没有约束 decoder 在整个 Gaussian prior support 上的行为。
</div>

可以从三个步骤理解：

1. 训练时，encoder 产生的 $z$ 只覆盖 aggregate posterior 的高密度区域；
2. 较小的 KL 权重允许这个 aggregate posterior 偏离 $N(0,I)$；
3. penalty 迫使 decoder 在已访问区域扭曲到更可行，但没有告诉它如何处理 prior 中未访问的方向。

所以重建通过率高并不推出 prior generation 通过率高。更稳妥的做法包括：提高 posterior–prior matching、在 prior samples 上也训练 constraint loss、学习 latent flow、限制 safe latent set，或采用 hard-admissible decoder。

## 8.8 Latent variables 是否恢复了真实四个因子？

对 test set 用 ConvVAE posterior mean 编码，再计算每个真实生成因子与 16 个 latent coordinates 的 Pearson correlations：

<div class="table-wrap"><table id="latent-table" class="data-table"><thead><tr><th>真实生成因子</th><th>与任一 latent 维的最大 |corr|</th><th>解读</th></tr></thead><tbody><tr><td>short-vol level</td><td>0.9612</td><td>非常强：至少一个 latent coordinate 几乎单调追踪短端波动率水平。</td></tr><tr><td>term log-ratio</td><td>0.5119</td><td>中等：term-structure 因子被多维、非线性地分散编码。</td></tr><tr><td>rho（skew）</td><td>0.7790</td><td>较强：skew 因子在 latent 中有清楚但非完全一一对应的方向。</td></tr><tr><td>eta（smile intensity）</td><td>0.7304</td><td>较强：翼部曲率/强度信息被显著编码。</td></tr></tbody></table></div>

<figure>
<img src="reproduction/latent_factor_correlations.png" alt="Correlations between true SSVI factors and VAE latent coordinates">
<figcaption>图 12　某些 latent coordinates 与短端 level、skew、smile intensity 高度相关；term ratio 较分散。但坐标可旋转、可置换、可变号，不能把“第 3 维就是 skew”当作跨 seed 的识别结论。</figcaption>
</figure>

最大绝对相关只是 descriptive diagnostic。VAE likelihood 对 latent 旋转通常近似不变，因此可解释性需要额外 anchor、supervision、independence/orthogonality constraints，或直接像 Wang et al. 那样把经济 features 输入 decoder。

## 8.9 这次复现支持与不支持什么？

### 支持的机制结论

- mask channel + hidden-cell objective 能直接训练 completion；
- 在规则小网格上，ConvVAE 比同规模 MLP 更能利用局部二维结构；
- soft no-arbitrage penalty 能提高 posterior reconstruction 的可行率；
- prior scale 增大时，生成质量和无套利率会快速恶化；
- PCA 在真正低维的平滑数据上是非常强的对手。

### 不能由本实验推出的结论

- 不能重现任何专有 FX/SPX 论文的绝对误差；
- 不能证明 ConvVAE 在真实 noisy quotes 上一定优于 PCA；
- 不能证明有限网格通过就连续域无套利；
- 不能评估 bid–ask、liquidity、stale quotes、0DTE 或 regime shifts；
- 不能评估 future-surface dynamics，因为样本是独立抽取的；
- 不能把 latent correlation 当作因果或唯一 economic factor identification。

## 8.10 为什么没有把公开 crypto 论文称为 exact reproduction？

论文仓库公开了 23 MB 的二进制研究归档，但本次隔离执行环境无法把该外部二进制传入运行目录；文本源码和配置能够审计，原始/processed arrays 却未能在本次会话内载入。与其复制论文表格并称为“复现”，本报告选择：

1. 明确披露数据传输阻塞；
2. 按公开结构重新实现模型和 loss；
3. 用完全可审计的 SSVI data 做机制实验；
4. 保存脚本、数据、checkpoint 和全部机器输出供再次运行。

这比“代码仓库存在，所以结果已复现”的表述更可信。

