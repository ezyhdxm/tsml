# 10. 从 stump 到完整第一棵树：tree-Haar 正交几何

现在允许第一棵树长到 $L$ 个 leaves。仍考虑平方损失、无 L1/L2/path smoothing 的基准情形。

初始 partition 为

$$
\Pi_0=\{\{1,\ldots,n\}\}.
$$

第 $m$ 次 split 后得到 partition $\Pi_m$，其中有 $m+1$ 个 leaves。设第 $m$ 次选择的 split contrast 为 $a_m=a_{\widehat s_m}$。

## 10.1 所选 contrasts 两两正交

> **定理 2（tree-Haar orthogonality）.** 对任意一条 binary recursive partition path，按每次 split 定义的 normalized child-mean contrast $a_1,\ldots,a_{L-1}$ 两两正交。

**证明思路。** 对 $m<k$：

- 如果第 $k$ 个 split 的 parent leaf 与第 $m$ 个 split 的 support 不相交，则 $a_m^\top a_k=0$。
- 如果第 $k$ 个 split 位于第 $m$ 个 split 的某个 descendant child 内，则 $a_m$ 在该 descendant leaf 上是常数，而 $a_k$ 在自己的 parent leaf 内元素和为零，所以内积仍为零。

每个 contrast 又都被归一化为单位长度，故

$$
A_L=[a_1,\ldots,a_{L-1}]
$$

满足

$$
A_L^\top A_L=I_{L-1}.
$$

这就是离散 tree-Haar basis 的几何结构。

## 10.2 Partition projection 的递推

令 $\mathcal V(\Pi_m)$ 表示在 $\Pi_m$ 每个 leaf 上为常数的向量空间，$P_m$ 为其 Euclidean orthogonal projection。初始空间只含全局常数：

$$
P_0=\frac1n\mathbf 1\mathbf 1^\top.
$$

每次 split 都增加恰好一个与旧空间正交的新方向，因此

$$
\boxed{
P_m=P_{m-1}+a_ma_m^\top.
}
$$

递推得到

$$
\boxed{
P_{L-1}
=P_0+
\sum_{m=1}^{L-1}a_ma_m^\top.
}
$$

由于 $r\perp\mathbf 1$，第一棵树对 centered response 的未缩放拟合为

$$
\boxed{
\widehat u
=P_{L-1}r
=
\sum_{m=1}^{L-1}(a_m^\top r)a_m.
}
$$

## 10.3 累计 gain 的精确 telescoping identity

第 $m$ 次 split 的 raw gain 为

$$
\Gamma_m=(a_m^\top r)^2.
$$

因 contrasts 正交，

$$
\begin{aligned}
\Gamma_{\mathrm{tree}}
&=
\sum_{m=1}^{L-1}\Gamma_m\\
&=
\sum_{m=1}^{L-1}(a_m^\top r)^2\\
&=
\left\|
\sum_{m=1}^{L-1}(a_m^\top r)a_m
\right\|_2^2\\
&=
\boxed{\|\widehat u\|_2^2}.
\end{aligned}
$$

同时

$$
r^\top\widehat u
=r^\top P_{L-1}r
=\|P_{L-1}r\|_2^2
=\Gamma_{\mathrm{tree}}.
$$

因此

$$
\boxed{
\Gamma_{\mathrm{tree}}
=r^\top\widehat u
=\|\widehat u\|_2^2
=\operatorname{SSE}_{\mathrm{root}}-
\operatorname{SSE}_{\mathrm{tree}}.
}
$$

这是一条非常强的有限样本恒等式。

## 10.4 固定 tree structure 与 adaptive tree structure

如果整棵 $L$-leaf structure 在看见 $Y$ 之前就固定，那么 $a_1,\ldots,a_{L-1}$ 是固定正交向量。在 Gaussian null 下，

$$
\frac{a_m^\top r}{\sigma}
\overset{\mathrm{iid}}\sim N(0,1),
$$

所以

$$
\boxed{
\Gamma_{\mathrm{tree}}/\sigma^2
\sim\chi_{L-1}^2.
}
$$

但 greedy tree 每一步都根据 $r$ 选择最大 gain，因此所选 basis 本身依赖 $r$。其累计 gain 不再服从 $\chi_{L-1}^2$，而通常显著更大。

## 10.5 LightGBM leaf-wise search 的数学表达

在当前 partition $\Pi_{m-1}$ 下，令每个 leaf $A$ 的候选集合为 $\mathcal S(A)$。LightGBM 的下一步近似执行

$$
(\widehat A_m,\widehat s_m)
=
\arg\max_{A\in\Pi_{m-1}}
\max_{s\in\mathcal S(A)}
(a_{A,s}^\top r)^2,
$$

只要最大值超过 stopping threshold。它比 depth-wise tree 多了一层“在所有当前 leaves 中再取最大”。因此 leaf-wise first tree 是一个带 hierarchical admissibility constraints 的 greedy orthogonal search。

注意：上述正交性不意味着 greedy path 是给定 $L$ 下全局最优 partition。早期某次局部最优 split 会限制后续可达 partition；这是 CART/GBDT 的计算性选择，而不是全局 optimization guarantee [@breiman1984cart]。

# 11. 整条 tree path 的选择事件与选择后推断

一个常见误解是：tree path 太离散、太复杂，所以不可能做精确 post-selection inference。对平方损失、无 penalty 的第一棵树，事情其实非常整洁。

## 11.1 单步选择事件是线性不等式

在第 $m$ 步，给定之前的 path，候选 contrast 集合 $\mathcal C_m$ 已由 $X$ 和此前 partition 确定。算法选择

$$
\widehat s_m
=
\arg\max_{s\in\mathcal C_m}|a_s^\top r|.
$$

再记录所选 score 的符号

$$
q_m=\operatorname{sign}(a_{\widehat s_m}^\top r)\in\{-1,+1\}.
$$

“第 $m$ 步选中 $\widehat s_m$ 且符号为 $q_m$”等价于，对所有 $s\in\mathcal C_m$，

$$
q_ma_{\widehat s_m}^\top r
\ge a_s^\top r,
$$

以及

$$
q_ma_{\widehat s_m}^\top r
\ge -a_s^\top r.
$$

每一条都是 $r$ 的线性不等式。

## 11.2 完整 path 是 polyhedron

条件于：

- 完整的 selected leaf–feature–threshold sequence；
- 每一步 selected contrast 的 sign；
- fixed design $X$、binning、missing pattern；
- feature subsampling / `extra_trees` 的随机 realization；
- deterministic tie-breaking；

整棵第一树的选择事件是所有步骤线性不等式的交：

$$
\boxed{
\mathcal E
=
\{r:Ar\le b\}.
}
$$

若树以固定 `num_leaves=L` 结束，则只需要 winner comparisons。若使用 raw gain threshold $\tau$：

- 一个被接受的 selected split 还要求
  $$
  q_ma_{\widehat s_m}^\top r>\sqrt\tau;
  $$
- 最终因为没有候选超过阈值而停止，则要求
  $$
  -\sqrt\tau\le a_s^\top r\le\sqrt\tau
  $$
  对最终所有 admissible candidates 成立。

条件于 signs 后，这些仍是线性约束。因此，基准第一棵树的完整 path event 是 polyhedral。

> **定理 3（polyhedral tree-path event）.** 在平方损失、固定候选生成规则、无 L1/L2/path smoothing、gain 为 squared linear contrast 的条件下，条件于整条 path 与 selected signs，第一棵 greedy tree 的选择事件可表示为 $Ar\le b$。

这个结论与一般 Gaussian selective inference 的 polyhedral machinery 对接 [@lee2016postselection]；针对 CART terminal nodes 和 selected splits 的专门方法可见 Tree-Values [@neufeld2022treevalues]。

## 11.3 Polyhedral lemma 的推导

设更一般地

$$
y\sim N(\mu,\Sigma),
\qquad
\mathcal E=\{Ay\le b\}.
$$

我们关心一个线性 target

$$
T=v^\top y.
$$

定义

$$
c=\frac{\Sigma v}{v^\top\Sigma v},
\qquad
z=(I-cv^\top)y.
$$

则

$$
y=cT+z,
$$

并且 Gaussian 情形下 $T$ 与 $z$ 独立。第 $j$ 条 selection constraint 变成

$$
(Ac)_jT+(Az)_j\le b_j.
$$

令 $\alpha_j=(Ac)_j$。则：

- 若 $\alpha_j>0$，得到上界
  $$
  T\le\frac{b_j-(Az)_j}{\alpha_j};
  $$
- 若 $\alpha_j<0$，得到下界
  $$
  T\ge\frac{b_j-(Az)_j}{\alpha_j};
  $$
- 若 $\alpha_j=0$，该约束只限制 $z$。

因此条件于 $z$ 与 selection event，

$$
\boxed{
T\mid z,\mathcal E
\sim
N(v^\top\mu,v^\top\Sigma v)
\ \text{truncated to}\
[V^-(z),V^+(z)].
}
$$

其中

$$
V^-(z)
=
\max_{j:\alpha_j<0}
\frac{b_j-(Az)_j}{\alpha_j},
$$

$$
V^+(z)
=
\min_{j:\alpha_j>0}
\frac{b_j-(Az)_j}{\alpha_j}.
$$

由 truncated normal CDF 可以构造：

- selected split contrast $a_{\widehat s}^\top\mu$ 的 selective p-value；
- 两个 selected child means 差异的 selective confidence interval；
- terminal leaf mean 的 selective interval；
- 条件于整条 first-tree path 的局部 effect inference。

## 11.4 为什么 ordinary leaf interval 失效

如果 leaf $A$ 是预先指定的，

$$
\bar y_A\sim N(\bar\mu_A,\sigma^2/n_A).
$$

但实际 leaf 是因为其 ancestor splits 在大量候选中产生了异常大的 mean differences 才被创建。于是条件于“这个 leaf 被创建”，$\bar y_A$ 的分布被截断、倾斜，普通 interval

$$
\bar y_A\pm1.96\frac{\widehat\sigma}{\sqrt{n_A}}
$$

不再具有 nominal coverage。问题不在于 leaf mean 公式错误，而在于它忽略了 leaf definition 自身依赖 $Y$。

## 11.5 加入 ridge/L1 后发生什么

当 $\lambda_2>0$ 时，每个 candidate gain 是 $r$ 的 quadratic form：

$$
\Gamma_s=r^\top Q_sr.
$$

比较两个 gains 变成

$$
r^\top(Q_s-Q_t)r\ge0,
$$

选择区域一般不再是 polyhedron，而是 quadratic inequalities 的交。L1 又引入 soft-threshold regimes；条件于每个 gradient sum 是否越过 $\lambda_1$ 后，gain 仍是 piecewise quadratic。

因此：

- 无 penalty 的基准模型允许最干净的 truncated-Gaussian exact inference；
- ridge/L1 仍可通过 selective Monte Carlo、hit-and-run 或沿 target line 搜索 selection region 来做推断，但不再有同样简单的闭式截断区间。

# 12. Search degrees of freedom：参数很少，复杂度仍可很大

对 centered Gaussian null，第一棵树定义一个数据依赖映射

$$
r\mapsto\widehat u(r).
$$

其 effective degrees of freedom 定义为

$$
\operatorname{df}_{\mathrm{tree}}
=
\frac1{\sigma^2}
\sum_{i=1}^n
\operatorname{Cov}(r_i,\widehat u_i).
$$

在 null 下 $\mathbb E r=0$，所以

$$
\operatorname{df}_{\mathrm{tree}}
=
\frac1{\sigma^2}
\mathbb E[r^\top\widehat u].
$$

利用第 10 节的 pathwise identity，逐 realization 都有

$$
r^\top\widehat u
=
\|\widehat u\|^2
=
\Gamma_{\mathrm{tree}}.
$$

因此：

> **定理 4（gain–df identity）.** 对平方损失、无 penalty 的第一棵 tree，在 Gaussian global null 下，
> 
> $$
> \boxed{
> \operatorname{df}_{\mathrm{tree}}
> =
> \frac1{\sigma^2}
> \mathbb E\Gamma_{\mathrm{tree}}.
> }
> $$

## 12.1 Fixed tree 的名义自由度

若 $L$-leaf partition 预先固定，则 projection rank 为 $L-1$，所以

$$
\operatorname{df}_{\mathrm{fixed}}=L-1.
$$

这对应 $L$ 个 leaf means 减去已经由 intercept 表示的一个常数方向。

## 12.2 Adaptive tree 的 search df

定义

$$
\operatorname{df}_{\mathrm{search}}
=
\operatorname{df}_{\mathrm{tree}}-(L-1).
$$

它衡量的不是 leaf coefficient 数，而是为了选择这些 directions 搜索了多少数据依赖结构。类似概念在 best subset selection 等 adaptive procedures 中被系统称为 search degrees of freedom [@tibshirani2015df]。

对 root stump，

$$
\operatorname{df}_{\mathrm{stump}}
=
\mathbb E\max_{s\in\mathcal S}Z_s^2.
$$

若 $M$ 个候选近似独立，

$$
\operatorname{df}_{\mathrm{stump}}
\approx2\log M-\log\log M+O(1).
$$

所以 stump 虽然只有一个 fitted contrast coefficient，其 effective df 可以是十几甚至几十。这个结果把“multiple hypothesis testing 造成的复杂度”量化成了一个直接可用于风险计算的数。

## 12.3 完整第一棵树的 df 分解

逐步 gain 恒等式给出

$$
\operatorname{df}_{\mathrm{tree}}
=
\sum_{m=1}^{L-1}
\frac1{\sigma^2}
\mathbb E(a_{\widehat s_m}^\top r)^2.
$$

每一项不仅包含新增一个 coefficient 的基础成本 $1$，还包含第 $m$ 步在当前 adaptive candidate set 中取最大值的 search cost。随着树变深：

- node sample size 变小；
- noise variance of local means 变大；
- 但 feature/threshold search 仍持续；
- leaf-wise growth 还在所有 leaves 之间取最大。

因此深层 split 的 search df 相对于 local information 往往更高。

# 13. Learning rate：不改变第一棵树搜索，但缩小其风险

在 ordinary GBDT 模式下，第一棵树先根据未缩放 gradients/Hessians 完成结构与 leaf outputs，然后整体乘 learning rate $\eta$。所以在基准情形中，$\eta$ 不改变第一棵树选哪个 split，只改变最终 update：

$$
\widehat F_1
=
\widehat F_0+\eta\widehat u.
$$

## 13.1 纯噪声下的训练误差改善

相对于 intercept-only baseline，训练 SSE 的下降为

$$
\begin{aligned}
\|r\|^2-\|r-\eta\widehat u\|^2
&=
2\eta r^\top\widehat u
-\eta^2\|\widehat u\|^2\\
&=
(2\eta-\eta^2)\Gamma_{\mathrm{tree}}.
\end{aligned}
$$

取期望并除以 $n$：

$$
\boxed{
\mathbb E[\text{training MSE improvement}]
=
\frac{(2\eta-\eta^2)\sigma^2}{n}
\operatorname{df}_{\mathrm{tree}}.
}
$$

只要 $0<\eta<2$，纯噪声第一棵树在训练集上也必然平均表现为“改善”。

## 13.2 纯噪声下的样本外风险

考虑同一 fixed design 上独立的新 noise vector $r^{\mathrm{new}}$，与训练数据独立。相对于 intercept-only baseline，第一棵树带来的 expected test MSE 增量是

$$
\begin{aligned}
&\frac1n\mathbb E
\left[
\|r^{\mathrm{new}}-\eta\widehat u\|^2
-
\|r^{\mathrm{new}}\|^2
\right]\\
&=
\frac{\eta^2}{n}
\mathbb E\|\widehat u\|^2\\
&=
\boxed{
\frac{\eta^2\sigma^2}{n}
\operatorname{df}_{\mathrm{tree}}.
}
\end{aligned}
$$

训练改善来自对 training noise 的 alignment，而独立 test noise 不提供相同 alignment；test 上只剩 predictor variance。

## 13.3 Incremental optimism

test increment 减去 training increment 得到

$$
\boxed{
\operatorname{optimism}_{\mathrm{increment}}
=
\frac{2\eta\sigma^2}{n}
\operatorname{df}_{\mathrm{tree}}.
}
$$

这与经典 $2\sigma^2\operatorname{df}/n$ optimism 公式一致，只是第一棵 tree increment 被 shrinkage $\eta$ 缩放。

## 13.4 小 learning rate 能做什么、不能做什么

从上式看：

- 样本外纯噪声损害是 $O(\eta^2)$；
- 训练拟合是 $O(\eta)$；
- 第一棵树的结构搜索在 ordinary GBDT 中并未因 $\eta$ 变小而改变。

因此小 learning rate 可以减轻单棵树的 variance，却不能把一个不稳定 split 变成稳定 split，也不能消除 feature-selection bias。后续许多轮 boosting 仍可能逐步积累噪声，所以必须与 early stopping、tree complexity control 联合使用。

# 14. Ridge、L1 与 `min_gain_to_split` 的精确作用

## 14.1 Ridge 对 root split 的衰减比例

平方损失 root 下 $G_A=0$。令

$$
S_L=\sum_{i\in L}r_i,
\qquad
S_R=-S_L.
$$

带 $\lambda_2=\lambda$ 时，raw gain 为

$$
\Gamma_{\lambda,s}
=
S_L^2
\left[
\frac1{n_L+\lambda}
+
\frac1{n_R+\lambda}
\right].
$$

无 ridge 时

$$
\Gamma_{0,s}
=
S_L^2
\left[
\frac1{n_L}+\frac1{n_R}
\right].
$$

两者比例为

$$
\boxed{
R_\lambda(n_L,n_R)
=
\frac{\Gamma_{\lambda,s}}{\Gamma_{0,s}}
=
\frac{n_Ln_R(n+2\lambda)}
{n(n_L+\lambda)(n_R+\lambda)}.
}
$$

若 split balanced，$n_L=n_R=n/2$，则

$$
\boxed{
R_\lambda=\frac{n}{n+2\lambda}.
}
$$

当 $n\to\infty$ 而 $\lambda$ 固定，$R_\lambda\to1$。所以固定 `lambda_l2` 对大样本 root search 的影响渐近消失；它主要在深层小 leaves 中有效。若希望 ridge 对 root 仍有非退化影响，$\lambda$ 必须与 node Hessian sum 同量级增长。

## 14.2 L1 的 hard screening effect

带 L1 时，leaf contribution 使用

$$
S_{\lambda_1}(G)^2/(H+\lambda_2).
$$

当 $|G|\le\lambda_1$，该 leaf output 被压成零。纯噪声大 node 中 $G=O_p(\sqrt H)$；因此固定 $\lambda_1$ 在 $H\to\infty$ 时也会相对变弱。L1 更适合抑制 gradient sum 很小的深层 leaves，而不是自动校准 root multiple testing。

## 14.3 `min_gain_to_split` 可以如何统计校准

在 Gaussian root null 下，若 $\sigma$ 和 candidate correlation 已知，定义

$$
q_{1-\alpha}(X)
=
(1-\alpha)\text{ quantile of }
\max_{s\in\mathcal S(X)}Z_s^2.
$$

则设置

$$
\boxed{
\tau_\alpha(X)
=
\sigma^2q_{1-\alpha}(X)
}
$$

可保证

$$
\Pr_0(\text{root 被 split}\mid X)\le\alpha.
$$

注意这是 LightGBM raw-gain 单位；若使用二阶 objective reduction convention，则阈值要除以 2。

固定常数 `min_gain_to_split` 通常不能对应统一的 $\alpha$，因为：

- target scale 与 $\sigma^2$ 会变；
- 每个 node 的 candidate set 不同；
- node sample size 与 dependence structure 不同；
- categorical ordering 或 GOSS 使 candidate generation 本身依赖 $Y$；
- 全树进行了序贯、自适应的重复搜索。

若要控制完整 tree 的 false split probability，可以使用 stepwise conditional simulation 与 alpha spending，例如第 $m$ 步分配 $\alpha_m$ 且 $\sum_m\alpha_m\le\alpha$。这比固定一个全局 raw-gain threshold 更接近正式的 sequential testing。

## 14.4 哪些参数真正减少 search multiplicity

- `max_bin`：减少每个数值 feature 的 threshold candidates。
- `feature_fraction_bynode`：减少每个 node 实际扫描的 features。
- `extra_trees=True`：对每个 feature 只检查随机 threshold，直接削弱 threshold maximization。
- `num_leaves` 与 `max_depth`：限制整条 path 的 adaptive search 次数与高阶 interaction。
- `min_data_in_leaf`、`min_sum_hessian_in_leaf`：排除高方差的小 node candidates，并控制 endpoint effect。
- `bagging_fraction`：通过 observation randomization 降低树之间相关性；对单棵树还改变其有效训练样本。

这些参数比单纯增加 leaf ridge 更直接地针对“从很多 hypotheses 中取最大”的问题。

# 15. 一般 smooth loss：fixed split 的 score-test 解释

回到一般 twice-differentiable loss。无 penalty 时，任意 fixed split 的 raw gain 已表示为

$$
\Gamma_s
=\langle z,b_s\rangle_H^2.
$$

注意

$$
\langle z,b_s\rangle_H
=
-\sum_{i=1}^ng_ib_s(i).
$$

它是 gradient/score contributions 的一个标准化 contrast。

## 15.1 正确 likelihood 下的渐近 $\chi^2$

若 loss 是正确设定模型的 negative log-likelihood，并满足独立、regularity 与 Fisher information identity：

$$
\operatorname{Var}(g_i\mid X_i)
=
\mathbb E(h_i\mid X_i),
$$

那么对预先固定的 split，

$$
U_s=-\sum_i g_ib_s(i)
$$

在 null 下渐近标准正态，从而

$$
\Gamma_s=U_s^2\Rightarrow\chi_1^2.
$$

这说明第一棵 generalized boosting stump 可被理解为在所有 feature–threshold 上扫描 score tests。

## 15.2 Misspecification 与 heteroskedasticity

若

$$
\operatorname{Var}(g_i\mid X_i)\ne h_i,
$$

则

$$
\operatorname{Var}(U_s\mid X)
=
\sum_i b_s(i)^2\operatorname{Var}(g_i\mid X_i)
$$

一般不等于 $1$。此时 LightGBM raw gain 是 Newton surrogate improvement，却不是正确 studentized 的检验统计量。候选之间可能因为 gradient noise variance 不同而被系统偏爱。

一个 robust fixed-split statistic 应使用 sandwich variance：

$$
T_s
=
\frac{U_s^2}{\widehat{\operatorname{Var}}(U_s)}.
$$

但标准 LightGBM 不会用这个 studentized statistic 选择 split。

## 15.3 Taylor remainder 与 learning rate

若 $|\partial_f^3\ell(y_i,f)|\le M$，对 update $\eta f_i$，

$$
\ell(y_i,F_0+\eta f_i)
=
\ell(y_i,F_0)
+
\eta g_if_i
+
\frac{\eta^2}{2}h_if_i^2
+R_{i,3},
$$

且

$$
|R_{i,3}|
\le
\frac{M\eta^3}{6}|f_i|^3.
$$

因此

$$
\left|\sum_iR_{i,3}\right|
\le
\frac{M\eta^3}{6}
\sum_i|f_i|^3.
$$

平方损失下三阶导数为零，理论完全精确；一般 smooth loss 下，二阶 first-tree analysis 对小 learning rate 是三阶精度。要把这个 pointwise remainder 提升为对“所有候选 trees 的统一控制”，还需要 leaf outputs 有界、candidate class complexity 可控等条件。
