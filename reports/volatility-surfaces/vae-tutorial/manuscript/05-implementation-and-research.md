# 9. 从零实现：一份不依赖魔法的训练与评估配方

## 9.1 先运行本报告配套实验

```bash
python reproduce_vol_surface_vae.py \
  --output-dir reproduction
```

脚本不会联网，也不依赖市场数据。它会依次：生成 SSVI surfaces、做离散静态套利筛选、保存数据、训练三个模型、评估七类 masks、从先验采样、输出 JSON/CSV/PNG/checkpoints。

为了快速检查代码路径，也可以使用：

```bash
python reproduce_vol_surface_vae.py \
  --quick \
  --output-dir smoke_test
```

快速模式只用于 smoke test，不能替代报告中的正式运行。

## 9.2 数据张量和 mask 的形状

推荐保留二维 grid，直到确实需要送进全连接层：

```python
# surface: (batch, n_tenor, n_moneyness)
# mask:    同形状；1=observed, 0=hidden

value_channel = surface * mask
encoder_input = torch.stack([value_channel, mask], dim=1)
# -> (batch, 2, n_tenor, n_moneyness)
```

不能只把 missing values 填成 0 而不提供 mask。标准化后 0 通常正好等于训练均值；网络无法知道某个 0 是“真实位于均值”还是“没有报价”。

## 9.3 一个最小 ConvVAE

```python
class ConvVAE(nn.Module):
    def __init__(self, n_tenor=6, n_moneyness=7,
                 channels=16, latent_dim=16):
        super().__init__()
        self.n_tenor = n_tenor
        self.n_moneyness = n_moneyness
        self.channels = channels

        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        flat = channels * n_tenor * n_moneyness
        self.to_mu = nn.Linear(flat, latent_dim)
        self.to_logvar = nn.Linear(flat, latent_dim)

        self.from_latent = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )

    def encode(self, x, mask):
        b = x.shape[0]
        x = x.view(b, self.n_tenor, self.n_moneyness)
        m = mask.view_as(x)
        h = self.encoder(torch.stack([x * m, m], dim=1)).flatten(1)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        b = z.shape[0]
        h = self.from_latent(z).view(
            b, self.channels, self.n_tenor, self.n_moneyness
        )
        return self.decoder(h).flatten(1)

    def forward(self, x, mask):
        mu, logvar = self.encode(x, mask)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        return self.decode(z), mu, logvar
```

对于 6×7 这样极小的 grid，保留同样 spatial resolution 的卷积通常比反复 stride/downsample 更安全，否则 2–3 层后尺寸就会坍缩。

## 9.4 正确归一化 hidden 与 observed loss

不同样本的 missing rate 不同，所以必须先在**每个样本内部**除以 hidden cell 数，再对 batch 取平均：

```python
def masked_vae_loss(recon, target, mask, mu, logvar,
                    beta=1e-3, observed_weight=0.1):
    sq = (recon - target).square()
    hidden = 1.0 - mask

    hidden_mse = (
        (sq * hidden).sum(dim=1)
        / hidden.sum(dim=1).clamp_min(1.0)
    )
    observed_mse = (
        (sq * mask).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1.0)
    )
    kl = -0.5 * (
        1.0 + logvar - mu.square() - logvar.exp()
    ).sum(dim=1)

    return (
        hidden_mse
        + observed_weight * observed_mse
        + beta * kl
    ).mean()
```

若直接把整个 batch 的所有隐藏误差相加再除总数，missing cells 多的样本会自动获得更大权重。这不一定错误，但必须明确它是 cell-weighted 还是 surface-weighted objective。

## 9.5 从 IV 转成 call price再做离散套利检查

假设 forward 归一化为 1，$K=e^k$：

```python
def normalized_black_call(iv, tau, log_moneyness):
    # iv: (batch, tenor, strike)
    total_var = (iv.square() * tau).clamp_min(1e-10)
    root = total_var.sqrt()
    d1 = -log_moneyness / root + 0.5 * root
    d2 = d1 - root
    return torch.special.ndtr(d1) - torch.exp(log_moneyness) * torch.special.ndtr(d2)
```

随后用 strike 而不是 $k$ 计算 secant slope：

```python
strike = torch.exp(log_moneyness)
slope = torch.diff(call, dim=-1) / torch.diff(strike)

calendar_penalty = torch.relu(
    call[:, :-1] - call[:, 1:]
).square().mean()

vertical_penalty = (
    torch.relu(slope).square().mean()
    + torch.relu(-1.0 - slope).square().mean()
)

butterfly_penalty = torch.relu(
    -torch.diff(slope, dim=-1)
).square().mean()
```

这里的 convexity 条件来源于 call price 对 $K$ 的二阶导数非负。若直接对等距 $k$ 网格做普通二阶差分，会忘记 $K=e^k$ 的非线性坐标变换。

## 9.6 训练时和评估时的 mask 必须分离

训练 mask 可以随机，以提高 amortized inference 的覆盖面；评估必须固定并可重现：

```python
train_rate = Uniform(0.10, 0.50)

# 固定 test battery
schemes = [
    "random_10", "random_30", "random_50",
    "row", "column", "wings", "long_tenor",
]
```

只测试 random missing 会过度乐观。真实报价缺失通常具有结构：整个长期限不活跃、某一 delta 没有 quote、深翼同时缺失、特定资产整块稀疏。

## 9.7 不能只做一次 seed

正式研究至少建议：

1. 5 个模型训练 seeds；
2. 每个 seed 使用同一组固定 test masks；
3. 报告 surface-level bootstrap interval；
4. 分别统计 reconstruction、posterior completion、prior generation；
5. 报告 admissibility 的点估计和 binomial/seed uncertainty；
6. 保存每个 seed 的 latent prior diagnostics。

最新 latent-geometry 工作说明，即使五个 seeds 的 RMSE 几乎一样，admissible latent area 与 prior probability 仍可能明显不同。

## 9.8 真实市场数据的推荐 pipeline

<div class="two-col">
<div>
<h3>输入侧</h3>
<ol>
<li>构造 point-in-time forward、discount 与 dividend/borrow inputs；</li>
<li>按 bid–ask、open interest、volume、quote age 清洗；</li>
<li>保留原始 irregular coordinates 和 observation mask；</li>
<li>若必须 grid 化，保存插值来源与置信度；</li>
<li>所有 normalization 只用训练期。</li>
</ol>
</div>
<div>
<h3>输出侧</h3>
<ol>
<li>优先预测 log-IV 或 total variance；</li>
<li>在 price space 检查 calendar/vertical/butterfly；</li>
<li>按 bid–ask 衡量 violation 的经济幅度；</li>
<li>另测 Greeks、local-vol 稳定性与下游 calibration；</li>
<li>对生成模型检查 latent tails 与 extreme regimes。</li>
</ol>
</div>
</div>

## 9.9 复现论文结果时的命令级 checklist

```text
[ ] 固定论文/仓库 commit hash
[ ] 下载并校验数据 checksum
[ ] 记录 vendor release 与字段定义
[ ] 重建 exact date split
[ ] 重建 quote filters 与 interpolation
[ ] 锁定 Python / CUDA / PyTorch / NumPy versions
[ ] 逐项打印 model architecture 与 parameter count
[ ] 固定 seed 和 deterministic settings
[ ] 从空目录运行 single command
[ ] 自动生成论文表格，而不是手抄数字
[ ] 至少再跑多 seed robustness
```

对 Chen et al. 的 OptionMetrics 工作，需要合法 WRDS 数据后运行公开 preprocessing；对 Singh et al.，优先从其带 provenance 的 release/Zenodo bundle 开始，而不是只复制单个 Python 文件；对 Ning et al.，还需区分公开 demo 与完整 CTMC/SDE calibration pipeline。

# 10. 接下来真正值得做的研究问题

## 10.1 Constrained nonlinear dynamic tensor factor model

你之前提出的方向可以写成：

$$
X_t(i,j)
=
\underbrace{\sum_{r=1}^{R} f_{t,r}a_r(i)b_r(j)}_{\text{可解释的线性 tensor 部分}}
+
\underbrace{R(z_t,i,j)}_{\text{非线性 residual}}
+
\epsilon_t(i,j).
$$

然后让状态满足

$$
(f_{t+1},z_{t+1})
\sim p(\,\cdot\mid f_t,z_t,u_t),
$$

其中 $u_t$ 可包含 spot return、rates、VIX、realized volatility 或 macro state。最终输出通过约束映射

$$
X_t=G(f_t,z_t)
$$

保证或近似保证 admissibility。

这个框架有四个可发表的难点：

1. **识别性**：怎样让 tensor factors 跨 seed 稳定对应 level/skew/curvature？
2. **约束几何**：怎样保证整个 factor/latent support，而非训练点，落在可行域？
3. **不规则观测**：怎样从每天不同的 quote coordinates 推断状态，不先粗暴 grid 化？
4. **动态一致性**：怎样让生成的多日曲面和 underlying/rates 联合演化合理？

只说“用 nonlinear decoder 替代 tensor factor”还不够新；把可识别结构、irregular observation operator、dynamic latent process 与 no-arbitrage support guarantee 合在一起，才构成更清晰的 research contribution。

## 10.2 把 bid–ask 与 quote reliability 写进 likelihood

市场 IV 不是精确真值。对 cell $j$，可写

$$
x^{\mathrm{mid}}_j
=
D(z)_j+\varepsilon_j,
\qquad
\varepsilon_j\sim N(0,s_j^2),
$$

其中 $s_j$ 由 half-spread、quote age、size、venue reliability 决定。重建项变成

$$
\sum_j
\frac{m_j}{s_j^2}
\bigl(x^{\mathrm{mid}}_j-D(z)_j\bigr)^2
+
\sum_j m_j\log s_j^2.
$$

这样 liquid ATM quotes 权重高，stale deep-wing quotes 权重低；模型还能输出 uncertainty，而不是把所有误差归因于 latent state。Gopal 的 heteroscedastic VAE 是这一方向的开端，但还可以加入 correlated observation errors 与 bid/ask interval likelihood。

## 10.3 从固定网格转向 set/operator encoder

真实每天观测的是集合

$$
\{(k_n,\tau_n,\sigma_n,\text{metadata}_n)\}_{n=1}^{N_t},
$$

$N_t$ 与坐标每天变化。更自然的 encoder 是 permutation-invariant set network、graph neural operator 或 cross-attention：

$$
z_t=E\bigl(\{(k_n,\tau_n,\sigma_n,q_n)\}\bigr).
$$

pointwise decoder 再回答任意 query $(k,\tau)$：

$$
\widehat\sigma(k,\tau)=D(z_t,k,\tau).
$$

优点是避免先插值再学习，并可直接处理稀疏 RFQ/OTC quotes。困难是 continuous-domain constraints 需要对 query domain 做 certificate 或自适应 violation search。

## 10.4 学“可行概率质量”，而不只学可行 manifold

2026 latent-flow 与 latent-geometry 工作共同指出两个对象：

- decoder 的可行集合 $\mathcal A$；
- sampling distribution 对该集合的质量 $Q(\mathcal A)$。

可研究的 objective 是同时优化：

$$
\text{distribution fit}
+
\lambda\,\mathbb E_{z\sim Q}[-\min(M(z),0)]
+
\gamma\,\Pr_{z\sim Q}\{M(z)<0\}.
$$

还可构造 barrier-aware flow，使 vector field 在靠近边界时沿切向移动或向内反射。与简单 sample-and-repair 相比，这会更好地保留目标分布。

## 10.5 多 horizon 动态：不要把 diffusion 当成装饰

对未来 $H$ 日，目标不应只是把 horizon 作为一个额外 scalar 输入并分别回归。需要区分：

- 各 horizon 的 marginal forecast；
- 同一路径上不同 horizon 的 joint consistency；
- surface 与 underlying return 的 leverage relation；
- forecast interval 的 coverage；
- 短期 persistence 与长期 mean reversion。

一种较轻量的做法是先学习 constrained latent state $z_t$，再训练 irregular-time state-space transition：

$$
z_{t+\Delta}
=
A_\Delta z_t+b_\Delta(u_t)+\Sigma_\Delta(z_t,u_t)^{1/2}\xi.
$$

diffusion/flow 只有在 residual distribution 明显 non-Gaussian 或 multi-modal 时才需要。先用 Gaussian transition、mixture density 与 quantile baselines，才能证明复杂生成模型带来的增量。

## 10.6 评估下游 Greek 稳定性

两张 surface 的 IV RMSE 很接近，decoder 的导数却可能完全不同。对 pricing/hedging，至少还应检查：

$$
\frac{\partial \widehat\sigma}{\partial k},
\qquad
\frac{\partial^2 \widehat\sigma}{\partial k^2},
\qquad
\frac{\partial \widehat w}{\partial \tau},
$$

以及由其得到的 local volatility、risk-neutral density、option Greeks。网络若在 cells 之间有高频振荡，grid RMSE 看不出来，却会放大到 density/local-vol 中。

## 10.7 一个具体、可执行的论文设计

<div class="callout success">
<strong>题目雏形：</strong><em>Arbitrage-Constrained Nonlinear Dynamic Factor Models for Irregularly Observed Volatility Surfaces</em>。
</div>

建议四阶段：

1. **结构表示**：低秩 tensor basis + coordinate-conditioned residual decoder；
2. **观测模型**：set encoder，使用 bid–ask/reliability weighted likelihood；
3. **动态模型**：latent state-space baseline，再与 flow/diffusion residual 比较；
4. **约束机制**：hard SSVI base + residual margin certificate，或 safe-latent projection。

理论部分可以研究：

- 在 decoder Lipschitz 与正 margin 条件下的局部安全半径；
- grid certificate 到连续域 certificate 的误差界；
- structured component 与 nonlinear residual 的识别条件；
- irregular observation 下 latent-state estimation error。

实证部分应同时使用 synthetic ground truth 与真实 SPX/FX/crypto：synthetic 数据检验因素恢复与约束，真实数据检验 completion、forecast calibration、tail scenarios 和下游 hedging。

