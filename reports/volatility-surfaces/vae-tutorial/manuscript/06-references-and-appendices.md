# 11. 参考文献与资源

以下链接优先指向 DOI、arXiv 或作者公开仓库。论文中的市场数据许可仍以各 vendor/数据库条款为准。

## 11.1 基础：volatility surface、无套利与 VAE

1. Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy 81(3), 637–654. <a href="https://doi.org/10.1086/260062">DOI</a>.
2. Merton, R. C. (1973). *Theory of Rational Option Pricing*. Bell Journal of Economics and Management Science 4(1), 141–183. <a href="https://doi.org/10.2307/3003143">DOI</a>.
3. Dupire, B. (1994). *Pricing with a Smile*. Risk 7(1), 18–20.
4. Cont, R. & da Fonseca, J. (2002). *Deformation of Implied Volatility Surfaces: An Empirical Analysis*. In *Empirical Science of Financial Fluctuations*. <a href="https://doi.org/10.1007/978-4-431-66993-7_25">DOI</a>.
5. Gatheral, J. & Jacquier, A. (2014). *Arbitrage-Free SVI Volatility Surfaces*. Quantitative Finance 14(1), 59–71. <a href="https://doi.org/10.1080/14697688.2013.819986">DOI</a>.
6. Kingma, D. P. & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR. <a href="https://arxiv.org/abs/1312.6114">arXiv</a>.
7. Rezende, D. J., Mohamed, S. & Wierstra, D. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*. ICML. <a href="https://proceedings.mlr.press/v32/rezende14.html">Proceedings</a>.
8. Higgins, I. et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR. <a href="https://openreview.net/forum?id=Sy2fzU9gl">OpenReview</a>.
9. Ackerer, D., Tagasovska, N. & Vatter, T. (2020). *Deep Smoothing of the Implied Volatility Surface*. NeurIPS 33, 11552–11563. <a href="https://proceedings.neurips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html">Proceedings</a>.
10. Cont, R. & Vuletić, M. (2023). *Simulation of Arbitrage-Free Implied Volatility Surfaces*. Applied Mathematical Finance 30(2), 94–121. <a href="https://doi.org/10.1080/1350486X.2023.2277960">DOI</a>.
11. Vuletić, M. & Cont, R. (2025). *VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces*. Applied Mathematical Finance. <a href="https://doi.org/10.1080/1350486X.2025.2471317">DOI</a>; <a href="https://github.com/milenavuletic/VolGAN">code</a>.

## 11.2 核心 vol-surface VAE / latent-generative 文献

12. Bergeron, M., Fung, N., Hull, J. & Poulos, Z. (2021/2022). *Variational Autoencoders: A Hands-Off Approach to Volatility*. <a href="https://arxiv.org/abs/2102.03945">arXiv 2102.03945</a>.
13. Ning, B., Jaimungal, S., Zhang, X. & Bergeron, M. (2023). *Arbitrage-Free Implied Volatility Surface Generation with Variational Autoencoders*. SIAM Journal on Financial Mathematics 14(4), 1004–1027. <a href="https://doi.org/10.1137/21M1443546">DOI</a>; <a href="https://arxiv.org/abs/2108.04941">arXiv</a>.
14. Zhang, W., Li, L. & Zhang, G. (2023). *A Two-Step Framework for Arbitrage-Free Prediction of the Implied Volatility Surface*. Quantitative Finance 23(1), 21–34. <a href="https://doi.org/10.1080/14697688.2022.2135454">DOI</a>.
15. Dierckx, T., Davis, J. & Schoutens, W. (2023). *Towards Data-Driven Volatility Modeling with Variational Autoencoders*. CCIS 1753, 97–111. <a href="https://doi.org/10.1007/978-3-031-23633-4_8">DOI</a>.
16. Richert, I. & Buch, R. (2024). *Interpolation of Missing Swaption Volatility Data Using Gibbs Sampling on Variational Autoencoders*. Financial Markets and Portfolio Management. <a href="https://doi.org/10.1007/s41237-023-00213-2">DOI</a>; <a href="https://arxiv.org/abs/2204.10400">arXiv</a>.
17. Gong, Z., Frys, W., Tiranti, R., Ventre, C., O’Hara, J. & Bai, Y. (2023/2024). *A New Encoding of Implied Volatility Surfaces for Their Synthetic Generation*. <a href="https://arxiv.org/abs/2211.12892">arXiv 2211.12892</a>.
18. Gopal, A. (2024). *Filling in Missing FX Implied Volatilities with Uncertainties: Improving VAE-Based Volatility Imputation*. <a href="https://arxiv.org/abs/2411.05998">arXiv 2411.05998</a>.
19. Feugang Nteumagné, B., Donfack, H. A. & Wafo Soh, C. (2025). *Variational Autoencoders for Completing the Volatility Surfaces*. Journal of Risk and Financial Management 18(5), 239. <a href="https://doi.org/10.3390/jrfm18050239">DOI</a>.
20. Wang, J., Liu, S. & Vuik, C. (2025). *Controllable Generation of Implied Volatility Surfaces with Variational Autoencoders*. <a href="https://arxiv.org/abs/2509.01743">arXiv 2509.01743</a>.
21. Chen, J., Hull, J., Poulos, Z., Rasul, H., Veneris, A. & Wu, Y. (2025). *A Variational Autoencoder Approach to Conditional Generation of Possible Future Volatility Surfaces*. Journal of Financial Data Science 7(3), 86–115. <a href="https://doi.org/10.3905/jfds.2025.1.196">DOI</a>; <a href="https://github.com/rotmanfinhub/vol-surface-vae-pub">code</a>.
22. Singh, S., Reddy, A. & Chopra, M. (2026). *Beyond the Smile: A Hybrid Convolutional VAE for Crypto Volatility Surfaces*. <a href="https://arxiv.org/abs/2606.16961">arXiv 2606.16961</a>; <a href="https://github.com/jasper-research/beyond-the-smile-paper">code/data archive</a>.
23. Brooks, O., Bajalica, D., Liu, Y. & Ben Tahar, I. (2026). *Latent Flow Matching for Arbitrage-Aware Implied Volatility Surface Generation*. <a href="https://arxiv.org/abs/2608.00616">arXiv 2608.00616</a>.
24. Buchegger, D. M. & Gonon, L. (2026). *Arbitrage-Aware Multi-Step Forecasting of Implied Volatility Surfaces: Modelling Surface Trajectories Using Latent Diffusion*. <a href="https://arxiv.org/abs/2608.22478">arXiv 2608.22478</a>.
25. Wang, J., Liu, S. & Vuik, C. (2026). *Latent-Space No-Arbitrage Geometry of Generative Models for Implied Volatility Surfaces*. <a href="https://arxiv.org/abs/2609.00332">arXiv 2609.00332</a>.

## 11.3 相邻方法，适合作为 baseline 或下一步阅读

26. Wiedemann, R., Jacquier, A. & Gonon, L. (2025). *Operator Deep Smoothing for Implied Volatility*. ICLR. <a href="https://arxiv.org/abs/2406.11520">arXiv</a>.
27. Jin, C. & Agarwal, S. (2025). *Forecasting Implied Volatility Surface with Generative Diffusion Models*. <a href="https://arxiv.org/abs/2511.07571">arXiv</a>.
28. Choudhary, V., Jaimungal, S. & Bergeron, M. (2023/2024). *FuNVol: A Multi-Asset Implied Volatility Market Simulator Using Functional Principal Components and Neural SDEs*. <a href="https://arxiv.org/abs/2303.00859">arXiv 2303.00859</a>; <a href="https://github.com/vedantch/FuNVol">code</a>.
29. Sohn, K., Lee, H. & Yan, X. (2015). *Learning Structured Output Representation Using Deep Conditional Generative Models*. NeurIPS. CVAE 的基础参考。
30. Lipman, Y. et al. (2023). *Flow Matching for Generative Modeling*. ICLR. <a href="https://arxiv.org/abs/2210.02747">arXiv</a>.

# 附录 A. Masked VAE 究竟在近似哪个条件分布？

设观测 cells 为 $x_O$，缺失 cells 为 $x_M$。理想目标是

$$
p(x_M\mid x_O)
=
\int p(x_M\mid z,x_O)p(z\mid x_O)\,dz.
$$

如果 decoder 给定 $z$ 后各 cells 条件独立，可近似写成

$$
p(x_M\mid x_O)
\approx
\int p(x_M\mid z)p(z\mid x_O)\,dz.
$$

难点是 $p(z\mid x_O)$ 不可直接计算。三种常见做法对应不同近似：

### A.1 Latent optimization

先训练完整曲面 VAE。测试时把 $z$ 当参数，解

$$
\widehat z
=
\arg\min_z
\sum_{j\in O}
\bigl(D(z)_j-x_j\bigr)^2
+\lambda\|z\|^2.
$$

其中 $\lambda\|z\|^2$ 来自标准正态 prior 的负 log-density。它给的是近似 MAP point，不自然提供完整 conditional uncertainty。

### A.2 Masked amortized encoder

直接训练

$$
q(z\mid x_O,m)=E(x\odot m,m).
$$

训练时随机化 $m$，让一个 encoder 学会许多 missing patterns。推断快，但遇到训练中从未覆盖的整块缺失，仍可能 distribution shift。

### A.3 Pseudo-Gibbs

从一个初始缺失填充值开始，反复：

$$
z^{(r)}\sim q(z\mid x_O,x_M^{(r-1)}),
$$

$$
x_M^{(r)}\sim p(x_M\mid z^{(r)}),
$$

并始终把 $x_O$ 固定为真实观测。它试图用完整数据 encoder 构造条件链，但除非近似 posterior 足够准确，这个链的 stationary distribution 不一定等于真实 $p(x_M\mid x_O)$；burn-in 与 mixing 也必须诊断。

# 附录 B. 为什么逐样本 KL 不保证 aggregate posterior 等于 prior？

定义数据经验分布 $q(x)$，encoder posterior $q(z\mid x)$，aggregate posterior

$$
q_{\mathrm{agg}}(z)
=
\int q(z\mid x)q(x)\,dx.
$$

VAE regularizer 是

$$
\mathbb E_{q(x)}
\operatorname{KL}\bigl(q(z\mid x)\|p(z)\bigr).
$$

把 joint $q(x,z)=q(x)q(z\mid x)$ 代入，可分解为

$$
\mathbb E_{q(x)}
\operatorname{KL}\bigl(q(z\mid x)\|p(z)\bigr)
=
I_q(x;z)
+
\operatorname{KL}\bigl(q_{\mathrm{agg}}(z)\|p(z)\bigr),
$$

其中 $I_q(x;z)$ 是在 encoder joint distribution 下的 mutual information。

推导只需在 log-ratio 中乘除 $q_{\mathrm{agg}}(z)$：

$$
\log\frac{q(z\mid x)}{p(z)}
=
\log\frac{q(z\mid x)}{q_{\mathrm{agg}}(z)}
+
\log\frac{q_{\mathrm{agg}}(z)}{p(z)}.
$$

对 $q(x,z)$ 取期望，第一项就是 mutual information，第二项就是 aggregate KL。

这揭示一个 trade-off：增大 $\beta$ 虽然推动 $q_{\mathrm{agg}}$ 接近 prior，也同时惩罚 latent 中包含的曲面信息，可能导致 posterior collapse。减小 $\beta$ 保住 reconstruction，却可能让直接采 $p(z)$ 进入训练编码从未访问的区域。这正是为什么 latent flow/diffusion 会有价值：它单独学习 $q_{\mathrm{agg}}$，不要求用同一个 KL 项同时完成压缩和 prior matching。

# 附录 C. 离散静态套利检查的最小推导

固定 maturity，归一化 call price 为 $c(K)$。三个 strike $K_1<K_2<K_3$。

### C.1 单调性与 vertical spread

由 payoff $(S_T-K)^+$ 随 $K$ 下降，

$$
c(K_2)\le c(K_1).
$$

又因为 call-spread payoff 每增加一单位 strike 最多下降一单位，secant slope 应满足

$$
-1
\le
\frac{c(K_2)-c(K_1)}{K_2-K_1}
\le0.
$$

### C.2 Convexity 与 butterfly

call price 对 strike 的 slope 应随 strike 上升：

$$
\frac{c(K_2)-c(K_1)}{K_2-K_1}
\le
\frac{c(K_3)-c(K_2)}{K_3-K_2}.
$$

等距 strike 时，这等价于

$$
c(K_1)-2c(K_2)+c(K_3)\ge0.
$$

不等距 strike 时不能直接使用普通二阶差分，必须比较 secant slopes。

### C.3 Calendar spread

在 forward/discount 口径一致的 normalized price 表示下，对固定 strike 或固定合适坐标，较长期限 call 不能更便宜。实际离散实现必须确认比较的是同一 forward-moneyness/strike 口径；不同 forward 下粗暴比较同一个 $k$ 可能引入坐标误差。

### C.4 网格通过的边界

离散检查只覆盖给定 cells 与 finite-difference stencils。它不自动约束：

- 两个网格点之间的插值；
- 最左/最右 strike 之外的 tails；
- $\tau\downarrow0$ 的边界；
- stochastic surface dynamics 的 dynamic arbitrage。

因此报告应写“通过指定网格的 static checks”，而不是无条件写“arbitrage-free”。

# 附录 D. 本次运行的文件清单

| 文件 | 内容 |
|---|---|
| `reproduce_vol_surface_vae.py` | 从数据生成到评估的完整可运行脚本 |
| `results.json` | 所有配置、环境、completion、generation、reconstruction 与 latent audit |
| `training_history.csv` | 每 epoch 的 train/validation loss components |
| `synthetic_ssvi_data.npz` | 1,800 张曲面、网格、split 与标准化统计 |
| `synthetic_ssvi_factors.csv` | 四个真实生成因子 |
| `MLP-VAE.pt` | masked MLP checkpoint |
| `ConvVAE.pt` | 2D ConvVAE checkpoint |
| `ConvVAE+NA.pt` | 带 reconstruction no-arbitrage penalty 的 checkpoint |
| `sample_completion.csv` | 图 10 对应的逐 cell 真值、mask 与预测 |
| `*.png` | 报告中所有自行生成图形 |

<div class="callout">
<strong>最终 takeaway：</strong>VAE 最有价值的角色不是“自动把一张 surface 变漂亮”，而是提供一个可做条件推断、概率生成与动态建模的 nonlinear latent representation。真正困难的金融部分仍然是：输入报价如何 point-in-time 构造、约束在哪个域成立、latent prior 是否覆盖安全区域、预测分布是否校准，以及输出是否在 pricing/hedging 中稳定。
</div>
