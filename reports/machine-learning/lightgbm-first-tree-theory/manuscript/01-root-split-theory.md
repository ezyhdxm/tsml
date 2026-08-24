# 6. Root stump：Gaussian null 下的有限样本精确分布

现在先只允许 root 发生一次 split，即 `num_leaves=2`。假设条件于设计矩阵 $X$：

$$
y=\mu\mathbf 1+\varepsilon,
\qquad
\varepsilon\mid X\sim N(0,\sigma^2I_n).
$$

令 centered residual

$$
r=M_0y,
\qquad
M_0=I_n-\frac1n\mathbf 1\mathbf 1^\top.
$$

因为任意 root split contrast 都满足 $a_s^\top\mathbf 1=0$，所以

$$
a_s^\top r=a_s^\top\varepsilon.
$$

定义

$$
Z_s=\frac{a_s^\top r}{\sigma}.
$$

对任意预先固定的候选 split $s$，

$$
Z_s\mid X\sim N(0,1),
$$

因而

$$
\boxed{
\frac{\Gamma_s}{\sigma^2}
=Z_s^2
\sim\chi_1^2.
}
$$

这是条件于 $X$ 的**有限样本精确结论**，不是大样本近似。

## 6.1 所有 candidates 的联合分布

令候选集合为

$$
\mathcal S=\{s_1,\ldots,s_M\}.
$$

把所有 standardized contrasts 堆成向量

$$
Z=(Z_{s_1},\ldots,Z_{s_M})^\top.
$$

则

$$
\boxed{
Z\mid X\sim N(0,K),
\qquad
K_{st}=a_s^\top a_t.
}
$$

LightGBM 实际选择

$$
\widehat s
=
\arg\max_{s\in\mathcal S}\Gamma_s
=
\arg\max_{s\in\mathcal S}Z_s^2.
$$

所以最大训练 gain 的精确 null law 是

$$
\boxed{
\frac{\Gamma_{\max}}{\sigma^2}
=
\max_{s\in\mathcal S}Z_s^2,
\qquad
Z\sim N(0,K).
}
$$

这里没有必要把候选数粗略压缩成一个含糊的 $M_{\mathrm{eff}}$。同一 feature 的相邻 thresholds、不同但相关的 features、missing direction 等候选之间的全部相关性，都由 Gram matrix $K$ 精确编码。

> **定理 1（root max-gain law）.** 在固定设计、iid Gaussian global null、平方损失、候选集合与 $Y$ 无关的条件下，第一棵 stump 的最大 raw gain 条件分布等于一个相关标准 Gaussian 向量坐标平方的最大值。

## 6.2 精确条件 Monte Carlo

当 $M$ 很大时，不需要显式构造 $M\times M$ 的 $K$。可以直接：

1. 固定实际使用的 binned $X$、missing pattern、feature subsampling realization 与 admissibility rules；
2. 模拟 $\varepsilon^{(b)}\sim N(0,I_n)$；
3. 对模拟 response 运行完全相同的 root split scanner；
4. 保存 $T^{(b)}=\max_s(a_s^\top\varepsilon^{(b)})^2$。

观测统计量

$$
T_{\mathrm{obs}}=\Gamma_{\max}/\widehat\sigma^2
$$

可用 Monte Carlo p-value

$$
\widehat p
=
\frac{1+\sum_{b=1}^B\mathbf 1\{T^{(b)}\ge T_{\mathrm{obs}}\}}
{B+1}
$$

校准。若 $\sigma$ 未知，最稳妥的做法是对每个模拟样本执行与真实数据相同的 scale estimation，或者在 exchangeable null 下直接 permutation。简单地把 observed gain 除以全样本 residual variance 后套 $\chi_1^2$，并不能校准“取最大”这一步。

## 6.3 为什么固定 split 的 p-value 不能用于被选中的 split

对每个 fixed split，

$$
\Pr(\Gamma_s/\sigma^2>c)=\Pr(\chi_1^2>c).
$$

但对选中的 split，

$$
\Pr(\Gamma_{\widehat s}/\sigma^2>c)
=
\Pr\left(\max_sZ_s^2>c\right),
$$

后者必然不小于前者。若直接对被选中的 split 使用普通 $\chi_1^2$ p-value，它不是在检验一个预先指定的 contrast，而是在检验“经过巨大搜索后最极端的 contrast”，从而严重 anti-conservative。这正是 multiple testing / post-selection 问题的最精确表达。

# 7. 单个连续 feature：CUSUM、Brownian bridge 与 endpoint effect

固定一个没有 ties 的连续 feature $X_j$，按其值从小到大重新排列 observation。记排序后的 centered residual 为

$$
r_{(1)},\ldots,r_{(n)},
\qquad
\sum_{i=1}^nr_{(i)}=0.
$$

在第 $k$ 个位置切分，左侧为前 $k$ 个 observation，定义 partial sum

$$
S_k=\sum_{i=1}^kr_{(i)}.
$$

由于全样本 residual sum 为零，右侧 residual sum 为 $-S_k$。于是 root raw gain 是

$$
\begin{aligned}
\Gamma_{j,k}
&=
\frac{S_k^2}{k}
+
\frac{S_k^2}{n-k}\\
&=
\boxed{
\frac{nS_k^2}{k(n-k)}.
}
\end{aligned}
$$

这就是 standardized CUSUM statistic 的平方。

## 7.1 Brownian-bridge 极限

令 $t=k/n$，并定义

$$
B_n(t)
=
\frac{S_{\lfloor nt\rfloor}}{\sigma\sqrt n}.
$$

在 $X_j$ 与 iid error 独立的 null 下，centered partial-sum process 满足

$$
B_n(\cdot)\Rightarrow B^0(\cdot),
$$

其中 $B^0$ 是标准 Brownian bridge。因此

$$
\frac{\Gamma_{j,k}}{\sigma^2}
=
\frac{B_n(t)^2}{t(1-t)}.
$$

若 minimum leaf fraction 被固定为 $\alpha\in(0,1/2)$，只扫描

$$
t\in[\alpha,1-\alpha],
$$

则

$$
\boxed{
\max_{\alpha n\le k\le(1-\alpha)n}
\frac{\Gamma_{j,k}}{\sigma^2}
\Rightarrow
\sup_{t\in[\alpha,1-\alpha]}
\frac{B^0(t)^2}{t(1-t)}.
}
$$

这正是 maximally selected statistic、score fluctuation test 与 model-based recursive partitioning 中反复出现的 Brownian-bridge 结构 [@lausen1992maxstat; @hothorn2006ctree; @zeileis2008mob]。

## 7.2 相邻 thresholds 为什么高度相关

令 $k\le \ell$。对应 standardized split scores 的相关系数为

$$
\boxed{
\operatorname{Corr}(Z_k,Z_\ell)
=
\sqrt{
\frac{k(n-\ell)}{\ell(n-k)}
}.
}
$$

当 $\ell=k+1$ 且两者离边界不近时，该相关系数非常接近 $1$。因此，把一个 feature 上的 $n-1$ 个 thresholds 当成 $n-1$ 个独立检验会严重夸大 multiplicity。正确对象是一个高度相关的 Gaussian bridge process。

## 7.3 固定 minimum leaf count 与固定 leaf fraction 的区别

如果 `min_data_in_leaf` 随 $n$ 成比例增长，即

$$
\min(n_L,n_R)\ge\alpha n,
$$

那么单 feature 的最大 null gain 是 $O_p(1)$。

如果 minimum leaf count 始终固定为某个常数 $m$，允许 $t$ 越来越接近 $0$ 或 $1$，则标准化因子 $1/[t(1-t)]$ 将放大 endpoint fluctuations。经典 Darling–Erdős 型结果表明，最大平方统计量会以 $\log\log n$ 的慢速量级增长 [@darling1956maximum]。这解释了为什么在大样本中仍允许极小 leaves，会让纯噪声 threshold scan 变得越来越激进。

LightGBM 的 histogram bins 会把单 feature 候选数限制在 `max_bin` 附近，但只要边界 bin 含很少 observation，endpoint instability 仍然可能存在。

# 8. 多个 features：最大统计量、上界与独立基准

设总候选数为 $M$。不要求候选相互独立，Gaussian tail 与 union bound 给出

$$
\Pr\left(
\max_{1\le s\le M}Z_s^2>x
\right)
\le
2M e^{-x/2}.
$$

因此取

$$
\boxed{
x_\alpha=2\log\frac{2M}{\alpha}}
$$

可保证

$$
\Pr(\Gamma_{\max}>\sigma^2x_\alpha)\le\alpha.
$$

这个界通常偏保守，因为它忽略候选间相关性，但它揭示了最关键的量级：

$$
\Gamma_{\max}=O_p(\sigma^2\log M).
$$

## 8.1 独立候选的 extreme-value benchmark

如果理想化地假设 $Z_1,\ldots,Z_M$ 独立标准正态，则令

$$
b_M=2\log M-\log\log M-\log\pi.
$$

有

$$
\Pr\left(
\frac{\max_sZ_s^2-b_M}{2}\le x
\right)
\to
\exp(-e^{-x}).
$$

因此

$$
\mathbb E\max_sZ_s^2
\approx
2\log M-\log\log M+O(1).
$$

真实 tree candidates 高度相关，所以这个公式不是精确答案，而是一个有用的“最激进独立搜索”尺度。

## 8.2 高基数 feature 的 selection bias

若 feature $X_j$ 产生 $M_j$ 个可行 thresholds，而另一个 feature 只产生很少候选，即使两者都与 $Y$ 独立，前者也有更多机会产生极端 gain。于是

$$
\Pr(\widehat j=j)
$$

会偏向候选更多的 feature。经典 CART exhaustive search 的 variable-selection bias 正源于此；conditional inference trees 通过先做 multiplicity-adjusted variable-level test，再选择 threshold，试图把“变量选择”和“切点选择”分开 [@hothorn2006ctree]。

LightGBM 的 histogram 能减小但不能消除这种偏差：不同 feature 的 non-missing bins、category cardinality、missing directions 与 admissible split 数仍可能不同。

# 9. 有 signal 时：非中心分布与选对 split 的条件

考虑 fixed-design alternative

$$
y=\mu\mathbf 1+f+\varepsilon,
\qquad
\varepsilon\sim N(0,\sigma^2I_n),
$$

其中 $f$ 已中心化，即 $\mathbf 1^\top f=0$。对 fixed split，

$$
\frac{a_s^\top r}{\sigma}
\sim
N\left(
\theta_s,1
\right),
\qquad
\theta_s=\frac{a_s^\top f}{\sigma}.
$$

因此

$$
\boxed{
\frac{\Gamma_s}{\sigma^2}
\sim
\chi_1^2(\lambda_s),
\qquad
\lambda_s=\theta_s^2
=
\frac{(a_s^\top f)^2}{\sigma^2}.
}
$$

若一个 population split 的左右概率为 $\pi_L,\pi_R$，左右条件均值差为

$$
\delta_s
=
\mathbb E[Y\mid X\in L_s]
-
\mathbb E[Y\mid X\in R_s],
$$

则其 noncentrality 近似为

$$
\boxed{
\lambda_s
\approx
\frac{n\pi_L\pi_R\delta_s^2}{\sigma^2}.
}
$$

这说明真实 split signal 是 $O(n)$，而纯噪声最大 gain 通常只有 $O(\log M)$。因此 root split 在大样本下并非注定不稳定；真正关键的是

$$
\boxed{
\frac{n\pi_L\pi_R\delta_s^2}{\sigma^2}
\gg
\log M.
}
$$

## 9.1 选对 oracle split 的简单概率界

令

$$
m_s=|a_s^\top f|,
$$

并假设 oracle candidate $s_\star$ 唯一最大。定义 absolute-score margin

$$
\Delta
=
m_{s_\star}-\max_{s\ne s_\star}m_s>0.
$$

如果所有 noise contrasts 同时满足

$$
\max_s|a_s^\top\varepsilon|<\Delta/2,
$$

则 $s_\star$ 必然仍然拥有最大的 absolute empirical score。因此，无需候选独立，union bound 给出

$$
\boxed{
\Pr(\widehat s\ne s_\star)
\le
2M\exp\left(
-\frac{\Delta^2}{8\sigma^2}
\right).
}
$$

所以 selection consistency 需要的不是“最佳 split 有正 signal”这么弱，而是最佳 split 与次佳 split 之间的 score gap 至少压过

$$
\sigma\sqrt{\log M}.
$$

当很多相邻 thresholds 具有几乎相同的 population gain 时，精确 threshold 本身可能永远不稳定，但它们诱导的 prediction function 仍可能相近。这提示我们区分：

- **threshold selection consistency**；
- **risk consistency**；
- **局部 partition stability**。

它们不是同一个性质。关于 decision stump variable screening 与高维 CART consistency 的结果也体现了类似的 signal–complexity tradeoff [@klusowski2020stumps; @klusowski2021largescale]。
