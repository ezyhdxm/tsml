---
title: "Vol Surface × VAE"
subtitle: "从隐含波动率曲面、变分推断到无套利生成：教程、文献地图与可复现实验"
author: "Tutorial-style literature review and reproducibility audit"
date: "资料与检索截至 2026-09-03"
lang: zh-CN
---

<div class="callout success">
<strong>这份报告做了三件彼此分开的事。</strong>第一部分从期权价格开始，逐步推导什么是 implied volatility surface，以及 static no-arbitrage 到底限制什么；第二部分从 latent-variable model 开始推导 VAE 的 ELBO、Gaussian KL 与 reparameterization；第三部分审计 2021–2026 年核心 vol-surface VAE 文献，并给出一份实际运行过的结构复现实验。文中用 <span class="badge reported">论文报告</span>、<span class="badge executed">本次实跑</span> 和 <span class="badge audit">审计判断</span> 明确区分证据来源。
</div>

# 0. 先看结论：这个领域真正研究的是什么？

“用 VAE 做 volatility surface”并不是一个单一问题。至少有五种不同任务：

1. **压缩与表示**：把一个几十到几千维的曲面压成少量 latent factors；
2. **补全**：给定部分报价，恢复缺失的 strike/maturity 单元；
3. **无条件生成**：从随机数生成历史分布上合理的新曲面，用于 stress/scenario；
4. **条件生成或预测**：给定 VIX、spot return、过去若干日曲面等上下文，生成未来曲面分布；
5. **可控生成**：直接指定 level、skew、curvature、term structure，再生成与这些控制量一致的曲面。

它们的最优建模方式并不一样。一个在重建误差上很好的 VAE，不一定能从先验 $z\sim N(0,I)$ 生成好曲面；一个能补全随机缺失点的模型，不一定能外推整条期限；一个在离散网格上没有套利的输出，也不一定在网格之间没有套利。

<div class="metric-grid">
<div class="metric"><b>99.99579%</b><small>本次四因子 SSVI 数据中，PCA-8 的累计解释方差</small></div>
<div class="metric"><b>34.6%</b><small>本次随机隐藏 50% 时，ConvVAE 相对 MLP-VAE 的 RMSE 降幅</small></div>
<div class="metric"><b>90.8%</b><small>2026 latent-flow 论文报告的“全部静态无套利检查通过”比例</small></div>
</div>

本报告最重要的判断是：

- **先选表示，再选网络。** 直接输出 IV、输出 log-IV、输出 total variance、输出 SVI/SSVI/SDE 参数，决定了模型是否容易满足金融约束。
- **PCA 必须保留为强基线。** 曲面通常极低维；在光滑、参数化、低噪声数据上，PCA 可以轻易击败 VAE。
- **“训练数据无套利”不推出“生成样本无套利”。** 解码器在训练流形附近表现好，不代表高斯先验的尾部也落在可行域内。
- **重建约束不等于先验生成约束。** 本次实跑中，简单的 no-arbitrage penalty 把完整重建的通过率从 93.70% 提到 99.63%，却把标准先验采样的通过率从 98.61% 降到 81.44%。这是一个有用的负结果，而不是应该被隐藏的异常。
- **严格无套利最可靠的路线仍是 hard parameterization。** 例如让 VAE 生成一个本身 arbitrage-free 的 SDE/SSVI 参数，再由定价模型映射到曲面。
- **2026 年的前沿已经从“VAE 本身”转向“VAE 作为低维几何层”。** 在 latent space 上再训练 flow 或 diffusion，以修复 aggregate posterior 与简单高斯先验之间的错配。

## 0.1 建议阅读路线

- 只想理解概念：读第 2–4 节。
- 想选研究方向：读第 5–7 节，尤其比较 hard guarantee、soft penalty、repair 和 latent flow。
- 想动手：读第 8–9 节，并运行配套脚本。
- 想做论文：读第 10 节；其中“带约束的非线性动态 tensor factor model”与你之前考虑的方向直接相连。

# 1. 全文只保留这一组记号

| 记号 | 含义 |
|---|---|
| $S_t$ | 当前 underlying spot |
| $F$ | 到期对应的 forward price |
| $K$ | strike |
| $\tau$ | time to maturity，单位为年 |
| $k=\log(K/F)$ | log-forward-moneyness |
| $\sigma(k,\tau)$ | implied volatility |
| $w(k,\tau)=\sigma(k,\tau)^2\tau$ | total implied variance |
| $x$ | 展平后的整张曲面向量 |
| $z$ | VAE latent vector |
| $m$ | mask；1 表示已观测，0 表示缺失 |

除非专门讨论网络参数，编码器和解码器分别记为 $E$ 与 $D$。这样可以避免在金融记号、概率记号和神经网络记号之间反复换字母。

# 2. 什么是 volatility surface？

## 2.1 从一张期权报价反推出一个 implied volatility

考虑一只欧式 call。用 forward 形式写 Black–Scholes 价格：

$$
C(K,\tau;\sigma)
= D(\tau)\bigl[F\,N(d_1)-K\,N(d_2)\bigr],
$$

其中

$$
d_1=\frac{\log(F/K)+\tfrac12\sigma^2\tau}{\sigma\sqrt{\tau}},
\qquad
d_2=d_1-\sigma\sqrt{\tau}.
$$

这里 $D(\tau)$ 是 discount factor，$N$ 是标准正态分布函数。市场给出价格 $C_{\mathrm{mkt}}$ 后，implied volatility 定义为方程

$$
C(K,\tau;\sigma_{\mathrm{imp}})=C_{\mathrm{mkt}}
$$

的解。它不是“未来真实波动率的直接观测”，而是**把市场价格映射回 Black–Scholes 波动率坐标后得到的报价语言**。

### 为什么这个反解通常是唯一的？

关键是 vega 始终为正。利用恒等式 $F\phi(d_1)=K\phi(d_2)$，其中 $\phi$ 是标准正态密度，可逐步计算：

$$
\begin{aligned}
\frac{\partial C}{\partial \sigma}
&=D\left[F\phi(d_1)\frac{\partial d_1}{\partial\sigma}
-K\phi(d_2)\frac{\partial d_2}{\partial\sigma}\right]\\
&=DF\phi(d_1)
\left(\frac{\partial d_1}{\partial\sigma}
-\frac{\partial d_2}{\partial\sigma}\right)\\
&=DF\phi(d_1)\sqrt{\tau}>0.
\end{aligned}
$$

所以，只要市场价格位于无套利价格界内，$C(K,\tau;\sigma)$ 随 $\sigma$ 严格递增，数值求根就有唯一解。

<figure>
<img src="reproduction/black_price_and_vega.png" alt="Black price and vega as functions of volatility">
<figcaption>图 1　本报告自行生成。ATM call 价格随波动率严格上升，因此可从价格唯一反解 implied volatility。</figcaption>
</figure>

<div class="callout warning">
<strong>数值上的重要例外：</strong>深度 ITM/OTM、极短期限或接近无套利边界时，vega 很小。此时价格中很小的 bid–ask 或舍入误差会被反解放大成很大的 IV 误差。因此，构建曲面时不应把每个 raw IV 点等权处理。
</div>

## 2.2 从一个点到一条 smile，再到一张 surface

固定到期 $\tau$，让 strike 变化，得到 volatility smile 或 skew：

$$
k\longmapsto \sigma(k,\tau).
$$

再让 maturity 变化，得到二维函数：

$$
(k,\tau)\longmapsto \sigma(k,\tau).
$$

这就是 implied volatility surface。股票指数通常表现为负 skew：较低 strike 的 put wing IV 更高；不同资产类别的形状不同，FX 常用 delta quote，rates 常出现 swaption cube，crypto 的短端和 wings 可能更加陡峭。

<figure>
<img src="reproduction/sample_surface_3d.png" alt="Synthetic implied volatility surface">
<figcaption>图 2　本次复现实验的一张 SSVI 合成曲面。横轴为 log-forward-moneyness，纵轴为期限，竖轴为 IV。</figcaption>
</figure>

<figure>
<img src="reproduction/sample_smiles.png" alt="Volatility smile slices">
<figcaption>图 3　同一张曲面的三个 maturity slices。surface 可以理解为一族随期限平滑变化的 smiles。</figcaption>
</figure>

## 2.3 为什么最好用 $k=\log(K/F)$ 和 total variance？

直接使用 $K/S$ 有两个问题：spot 与 carry 变化会让同一个固定 strike 在不同日期的相对位置漂移；不同资产也难以比较。log-forward-moneyness

$$
k=\log(K/F)
$$

把 ATM-forward 固定在 $k=0$，并把左右翼变成相对稳定的坐标。

对期限方向，常用

$$
w(k,\tau)=\sigma(k,\tau)^2\tau.
$$

原因不是为了“换个写法”，而是 option price 直接由 total variance 进入：$d_1$ 与 $d_2$ 中出现的是 $\sigma\sqrt{\tau}=\sqrt{w}$。而且 calendar no-arbitrage 更自然地作用于 option price 或 total variance，而不是要求 raw IV 随期限上升。

<figure>
<img src="reproduction/iv_vs_total_variance.png" alt="Implied volatility versus total variance term structure">
<figcaption>图 4　IV 可以随期限下降，同时 total variance 继续上升。把“无 calendar arbitrage”误写成 IV 单调递增，会施加错误约束。</figcaption>
</figure>

## 2.4 曲面上的四种直观变形

为了读懂 VAE latent space，先把常见形变分开：

- **level**：整张曲面同时上移或下移；
- **skew/slope**：左翼与右翼的相对高低变化；
- **curvature/smile**：两翼相对 ATM 的弯曲程度；
- **term structure**：短端与长端的相对变化。

PCA 往往能用前三到五个线性因子描述大部分历史变化。VAE 的动机不是“PCA 完全无效”，而是希望用一个**非线性、概率化、可采样**的低维流形描述曲面。

# 3. Static no-arbitrage：曲面究竟要满足什么？

## 3.1 从 call payoff 直接推导 strike 方向约束

风险中性定价写成

$$
C(K,\tau)=D(\tau)\,\mathbb E[(S_T-K)^+].
$$

对 $K$ 求一阶导：当 $S_T>K$ 时，payoff 对 strike 的导数为 $-1$，否则为 0。因此

$$
\frac{\partial C}{\partial K}
=-D(\tau)\,\mathbb P(S_T>K),
$$

从而

$$
-D(\tau)\le \frac{\partial C}{\partial K}\le 0.
$$

这同时给出两个事实：call price 必须随 strike 下降；相邻 strike 的 vertical-spread slope 不能小于 $-D$。

若 $S_T$ 有密度 $f_T$，再求一次导数：

$$
\frac{\partial^2 C}{\partial K^2}=D(\tau)f_T(K)\ge 0.
$$

因此 call price 必须关于 strike 凸。负的二阶导对应“负概率密度”，也是 butterfly arbitrage 的连续形式。

<figure>
<img src="reproduction/call_convexity.png" alt="Decreasing convex call price curve">
<figcaption>图 5　固定期限时，call price 对 strike 必须递减且凸。三个相邻 strike 就构成一个离散 butterfly 检查。</figcaption>
</figure>

## 3.2 离散网格上怎么检查？

设同一期限的 strikes 为 $K_1<K_2<\cdots<K_n$，call prices 为 $C_i$。定义相邻斜率

$$
s_i=\frac{C_{i+1}-C_i}{K_{i+1}-K_i}.
$$

则一个简单的离散检查是

$$
-D\le s_i\le 0,
\qquad
s_{i+1}\ge s_i.
$$

前者检查 vertical spread，后者检查 convexity/butterfly。对 calendar spread，应比较同一 strike、相同现金流口径下的 undiscounted call value；后到期不应更便宜。

<div class="callout warning">
<strong>坐标陷阱：</strong>固定 $k=\log(K/F)$ 不总等于固定 strike，因为 forward 会随期限变化。许多机器学习论文在统一 forward-normalized grid 上检查 calendar 条件，这是一个实用离散代理，但不能不加说明地等同于所有实际合约上的严格 calendar no-arbitrage。
</div>

## 3.3 为什么有 bid–ask 和交易成本，仍然要求 surface admissible？

现实报价中确实可能出现无法在成本后套利的小违例。构建模型时仍追求 arbitrage-free，原因不是相信可以无摩擦连续对冲，而是：

1. surface 要在未报价的 continuum 上插值；小范围 noisy violation 可能被插值器放大；
2. negative density 会让数字期权、exotic payoff 或 risk-neutral distribution 失去意义；
3. Dupire local volatility 在简化情形下含有

$$
\sigma_{\mathrm{loc}}^2(K,T)
=\frac{\partial_T C(K,T)}{\tfrac12 K^2\partial_{KK}C(K,T)}.
$$

若分子或分母符号错误，local variance 会变成负数或爆炸；
4. Greeks、scenario P&amp;L 和 hedging 会对局部不光滑与曲率异常非常敏感；
5. 生成模型可一次产生数万张曲面，哪怕每张只有几个 cell 违例，也会污染 downstream simulation。

所以应区分：**market quotes 是否存在可执行套利**，与**一个作为定价输入的连续曲面是否经济可接受**。前者要考虑 bid–ask、深度、时延与交易成本；后者通常应该满足更干净的结构条件。

## 3.4 SVI/SSVI 为什么经常出现？

SVI 用少量参数描述一个 maturity slice 的 total variance；SSVI 进一步把各期限连接成 surface。它们的价值是可以把无限维的函数约束转成少量参数约束。Gatheral–Jacquier 给出的 SSVI 形式之一是

$$
w(k,\theta)=\frac{\theta}{2}
\left[1+\rho\varphi(\theta)k+
\sqrt{\bigl(\varphi(\theta)k+\rho\bigr)^2+1-\rho^2}
\right],
$$

其中 $\theta$ 是 ATM total variance。只要 $\theta(\tau)$ 与 $\varphi(\theta)$ 满足相应条件，就能得到静态无套利 surface。这也是本次合成实验选 SSVI 的原因：我们知道 source surfaces 的金融约束不是偶然通过的。

<details>
<summary>进阶：从 total variance 直接检查 butterfly 的连续条件</summary>

对一个固定 maturity slice，令 $w=w(k)$。Gatheral–Jacquier 使用的密度条件可写为

$$
g(k)=
\left(1-\frac{k w'(k)}{2w(k)}\right)^2
-\frac{w'(k)^2}{4}\left(\frac{1}{w(k)}+\frac14\right)
+\frac{w''(k)}{2}\ge 0.
$$

再配合适当的 wing asymptotics，可排除 butterfly arbitrage。机器学习中更常见的做法，是把解码后的 IV 转为 call prices，然后用有限差分检查 slopes 与 convexity；它更容易实现，但只保证选定网格上的离散条件。
</details>

## 3.5 一条可靠的 surface construction pipeline

1. **清洗报价**：去除 crossed market、零 bid、明显 stale、违反 intrinsic/upper bound 的点；
2. **选 price**：mid、microprice 或根据流动性加权的可实现价格；
3. **反解 IV**：保留 solver status、vega、bid–ask IV width，而不是只留一个点估计；
4. **变换坐标**：通常映射到 $(k,\tau)$，并考虑输出 log-IV 或 total variance；
5. **统一网格或保留 set representation**：规则网格适合 CNN，但会引入 interpolation；set/graph/operator 模型可直接处理不规则报价；
6. **拟合与约束**：SVI/SSVI、constrained spline、neural operator、VAE decoder 等；
7. **独立检查**：不要只复用训练 loss，另行检查 price bounds、calendar、vertical、butterfly，以及网格之间的密集插值点；
8. **记录误差单位**：price、IV decimal、volatility point、volatility basis point 不能混用。

