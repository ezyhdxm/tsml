# 估计与检验：从 additive baseline 到 sparse interaction

# 预注册式模型阶梯

为了避免“先看 tree interaction 再解释”的研究自由度，建议在建模前固定以下 nested ladder。

## $\mathcal M_0$：additive quantity curve

$$
\mathcal M_0:
\quad
\mu_0(X,Q)=m_0(X)+h_0(\log Q).
$$

这是 separability null。$m_0$ 可以使用 LightGBM、CatBoost 或其他 tabular learner；$h_0$ 使用低维 spline / piecewise-linear basis，并由 orthogonalized estimation 得到。

## $\mathcal M_1$：liquidity-scaled quantity

$$
\mathcal M_1:
\quad
\mu_1(X,Q)
=m_1(X)
+h_1\{\log Q-\log Q^\star(X)\}.
$$

第一版 $Q^\star$ 不必完全 black-box。可比较：

$$
\begin{aligned}
\log Q^\star_1
&=\log(\text{trailing median size}),\\
\log Q^\star_2
&=\beta_0+\beta_1\log(\text{median size})
+\beta_2\log(1+\text{volume})\\
&\quad+\beta_3\log(\text{amount outstanding})
+\beta_4\log(1+\text{active dealers}),\\
\log Q^\star_3
&=g_{\mathrm{lowcap}}(X),
\end{aligned}
$$

其中 $g_{\mathrm{lowcap}}$ 应限制容量，例如 shallow tree、monotone GAM 或 regularized linear model，以免把 arbitrary price interaction 偷渡进 scale。

## $\mathcal M_2$：scale + amplitude

$$
\mathcal M_2:
\quad
\mu_2(X,Q)
=m_2(X)+A(X)h_2(u),
\qquad
u=\log Q-\log Q^\star(X).
$$

第一版 amplitude 可以是

$$
\log A(X)
=\gamma_0+
\gamma^\top Z_{\mathrm{risk}}(X),
$$

并约束 $A(X)>0$、$E[A(X)]=1$。$Z_{\mathrm{risk}}$ 只包含预先指定的 spread/rate volatility、CS01、stress、hedgeability 与 capacity variables。

## $\mathcal M_3$：sparse varying-curve interactions

$$
\mathcal M_3:
\quad
\mu_3(X,Q)
=m_3(X)+A(X)h(u)
+\sum_{j=1}^{J} Z_j f_j(u).
$$

$Z_j$ 只从高优先级 mechanism families 中选择：

- side × inventory pressure；
- information intensity；
- market stress；
- competition/protocol；
- hedgeability。

函数 $f_j$ 使用低自由度 basis，并施加 group penalty：

$$
\lambda\sum_j\|\theta_j\|_2,
$$

使整个 interaction curve 作为一组进入或退出，而不是在六个 quantity nodes 上零散选择。

## $\mathcal M_4$：unrestricted black-box challenger

$$
\mathcal M_4:
\quad
\mu_4(X,Q)=g(X,Q).
$$

可用 LightGBM 并允许 quantity 与所有 features 交互，但它只作为 performance upper-bound/challenger。若无法在 out-of-time、common-support 与 matched-pair tests 中稳定超过 $\mathcal M_2$ 或 $\mathcal M_3$，不应将其作为 normalization engine。

# quantity basis 与输出 grid

## 为什么用 $\log Q$

固定 grid

$$
100\text{k},
250\text{k},
500\text{k},
1\text{mm},
2\text{mm},
5\text{mm}
$$

跨越 50 倍。$100\text{k}\to200\text{k}$ 与 $2\text{mm}\to4\text{mm}$ 都是翻倍，在 raw dollars 中却距离不同。因此 basis 与 interpolation 应优先基于

$$
z=\log Q.
$$

## piecewise-linear basis

令 knots 为 grid 的 log values。可用 hinge basis

$$
B(z)=
\left(
1,z,(z-\kappa_2)_+,\ldots,(z-\kappa_{K-1})_+
\right),
$$

加 second-difference penalty 控制相邻 slope 变化。这样：

- grid nodes 可直接解释；
- interpolation 在 log quantity 上线性；
- 不会像高阶 cubic spline 那样在稀疏 tail 震荡；
- 可自然估计 small-size slope、large-size slope 与 curvature。

## 不预设全局 monotonicity

若业务需要 shape constraint，建议只对有机制支持的局部区间施加：

- 小 size 的 fixed-cost dilution 可允许 decreasing；
- institutional middle region 可 shrink toward flat；
- block tail 可允许 increasing/convex。

先用 out-of-time evidence 判断 turning point，而不是把全域 monotone constraint 当作先验事实。

# Cross-fitting 设计

## fold unit

所有属于同一信息事件的 rows 必须在同 fold：

- 同一 RFQ；
- 同一 dealer-run message；
- 同一 quote ladder snapshot；
- duplicate/sibling TRACE reports；
- 同一 derived target anchor。

时间外推是 deployment 的关键，因此 outer folds 应按日期/周连续分块，而非随机 row split。

## 建议的 nested design

1. **Outer time folds**：评估 $\mathcal M_0$–$\mathcal M_4$ 的 OOT loss；
2. **Inner group folds**：在 training window 内为 nuisance functions 生成 OOF predictions；
3. **Event grouping**：同 event rows 不跨 inner/outer folds；
4. **Hyperparameter selection**：只用 inner validation；
5. **Final curve tests**：基于 outer-fold predictions 与 cluster bootstrap。

这可以防止一个 common failure：quantity curve 在 random split 中看似稳定，只因为同 bond/dealer 的相邻 quotes 同时出现在 train/test。

# Additive null 的直接检验

# Conditional contrast curves

对每个预先指定的 modifier $Z$，估计

$$
\Delta_Z(q,q_0)
=
E[\widehat\mu(X,q)-\widehat\mu(X,q_0)\mid Z].
$$

在 additive null 下，各 $Z$ groups 应共享同一 curve。可以使用以下 statistics。

## Integrated squared heterogeneity

$$
T_{\mathrm{ISE}}(Z)
=
\sum_g \pi_g
\int
\left\{
\Delta_g(q,q_0)-\Delta(q,q_0)
\right\}^2
w(q)dq.
$$

## Supremum heterogeneity

$$
T_\infty(Z)
=
\max_g\sup_{q\in\mathcal Q_{\mathrm{support}}}
\left|
\Delta_g(q,q_0)-\Delta(q,q_0)
\right|.
$$

## Business-weighted node statistic

$$
T_{\mathrm{biz}}(Z)
=
\sum_{q\in\mathcal G}
\omega_q
\left|
\Delta_g(q,q_0)-\Delta(q,q_0)
\right|,
$$

其中 $\omega_q$ 按实际 RFQ/quote quantity distribution 或业务 importance 设置。不要让极少使用的 $5\text{mm}$ 节点与 $500\text{k}$ 同权，除非 block business 明确要求。

# Nested out-of-time loss tests

定义与业务一致的 loss，例如 spread-bp MAE、Huber、asymmetric side loss 或 weighted pinball loss。比较

$$
D_t
=
L_t(\mathcal M_a)-L_t(\mathcal M_b)
$$

的 time-block distribution，而不只比较总样本平均。需要报告：

- mean/median OOT improvement；
- 每周/月 improvement 的稳定性；
- stress vs normal regimes；
- quantity buckets；
- common-support subset；
- matched-pair subset；
- dealer/bond cold-start subsets。

interaction model 若只在一个 stress episode 或一小组 dealers 上改善，不等于具有稳定生产价值。

# Specification test 的角色

Härdle–Mammen 类型检验比较受限 parametric/additive fit 与 unrestricted nonparametric fit [@hardle1993comparing]。在本问题中可将受限模型设为 $\mathcal M_0$ 或 $\mathcal M_2$，unrestricted fit 为 $\mathcal M_3/\mathcal M_4$，使用 dependence-aware bootstrap 估计 null distribution。

但 formal p-value 不是唯一标准：样本极大时极小、无业务意义的 interaction 也会显著。因此必须同时报告 effect size：

$$
\max_{q,g}
|\Delta_g(q,q_0)-\Delta(q,q_0)|
$$

以及 OOT loss 改善。

# Varying-coefficient 视角

若选定 modifier $Z$，可写

$$
Y
=m(X)+h_0(Q)+Z h_1(Q)+\varepsilon.
$$

这属于 varying-coefficient / varying-effect family [@hastie1993varying]。对连续 $Z$，更一般地写

$$
Y
=m(X)+\sum_{k=1}^{K}
\beta_k(Z)B_k(Q)+\varepsilon.
$$

为避免维数爆炸，应：

- 每次只研究一个 mechanism family；
- 对 $\beta_k(Z)$ 使用低维 basis；
- 对整个 curve 使用 group shrinkage；
- 做 hierarchical selection：main effect 存在后才允许 interaction；
- 最终用 purified decomposition 报告 pure interaction。

# 为什么 SHAP interaction 不是主检验

tree SHAP interaction 可以帮助发现候选，但不能直接回答 separability：

1. correlated $Q$ 与 $X$ 下 attribution 依赖 background/conditionality convention；
2. main 与 interaction decomposition 可能不唯一；
3. feature duplication/proxy 会重分配 attribution；
4. SHAP magnitude 不等于 reference-quantity contrast bias；
5. in-sample tree interaction 容易放大 low-support tails。

因此 SHAP 的角色应是：

- challenger discovery；
- 找出可能遗漏的 $Z_j$；
- 检查局部 failure cases；

而最终保留 interaction 的证据应来自 explicit conditional contrast、purified effect、OOT improvement 与 matched variation。

# Interaction family 的分层检验

建议预先把候选 variables 分成五组：

$$
\mathcal Z=
\{Z_{\mathrm{liq}},
Z_{\mathrm{risk}},
Z_{\mathrm{inv}},
Z_{\mathrm{info}},
Z_{\mathrm{comp}}\}.
$$

检验顺序：

1. $\mathcal M_0$ vs $\mathcal M_1$：relative liquidity 是否解释主要异质性；
2. $\mathcal M_1$ vs $\mathcal M_2$：risk amplitude 是否进一步稳定 curves；
3. 对每个剩余 family 做 global curve test；
4. 只有 global family test 通过，才展开具体 variables/nodes；
5. 对 retained interaction 做 OOT confirmation。

这样比同时扫描数百个 `quantity × feature` 更能控制 multiple testing 与研究者自由度。

# Bootstrap 与 simultaneous uncertainty

## 外层 day-block bootstrap

抽取连续 day/week blocks，以保留 market-state persistence。block length 可由 residual dependence diagnostics 决定，并对不同 length 做 sensitivity。

## 内层 cluster preservation

抽中的 day block 内，保留完整：

- dealer-run message；
- RFQ/event；
- bond/dealer repeated observations；
- ladder levels；
- duplicate-report group。

## simultaneous bands

对整个 curve 构造

$$
\widehat\Delta(q,q_0)
\pm c_{1-\alpha}\widehat\sigma(q),
$$

其中 $c_{1-\alpha}$ 来自 bootstrap 中 standardized sup statistic，而不是在六个 nodes 上各给 95% pointwise interval。否则用户会把多个 node 的偶然偏差误读为曲线形状。

# Matched-pair validation

构造近同时、同 dealer–bond–side 的 quantity pair $(q_1,q_2)$，真实 difference 为

$$
D^{\mathrm{obs}}
=Y(q_2)-Y(q_1).
$$

模型预测为

$$
D^{\mathrm{pred}}
=\widehat\mu(X,q_2)-\widehat\mu(X,q_1).
$$

验证：

- MAE/RMSE of $D^{\mathrm{pred}}$；
- sign accuracy；
- calibration slope/intercept；
- by $q_2/q_1$ ratio；
- by elapsed time；
- by liquidity/stress/side；
- pair support and balance。

这比 raw quote-level MAE 更直接检验 quantity curve，因为 quantity-independent mid error 在 pair difference 中部分抵消。

# Residual calibration diagnostics

对 normalization 后 residual

$$
e^{\mathrm{norm}}
=Y-\widehat m(X)-\widehat h(Q)
$$

检查：

$$
E[e^{\mathrm{norm}}\mid Q\text{ bucket}],
$$

以及

$$
E[e^{\mathrm{norm}}\mid Q\text{ bucket},Z\text{ bucket}].
$$

若 additive model 正确：

- quantity-bucket mean 接近 0；
- conditional curves 无系统 slope；
- large-size residual variance 可以增加，但 mean 不应系统漂移；
- side-specific residual 不应呈相反 quantity trend。

同时检查 PIT residual dependence、dealer clusters 与 bond clusters；相关性不一定表示 quantity misspecification，但会影响 uncertainty 与 retraining cadence。

# 一个可复现的估计伪代码

```python
# Outer loop: strictly out-of-time folds
for train_idx, test_idx in time_block_splits(events):
    train = data.loc[train_idx]
    test = data.loc[test_idx]

    # Inner grouped cross-fitting for nuisance functions.
    y_hat_oof = cross_fit_predict(
        learner=m_model,
        X=train[X_cols],
        y=train[target],
        groups=train[event_group],
    )

    B_train = quantity_basis(np.log(train["quantity"]))
    B_test = quantity_basis(np.log(test["quantity"]))

    B_hat_oof = np.column_stack([
        cross_fit_predict(
            learner=basis_model,
            X=train[X_cols],
            y=B_train[:, k],
            groups=train[event_group],
        )
        for k in range(B_train.shape[1])
    ])

    theta = penalized_regression(
        X=B_train - B_hat_oof,
        y=train[target].to_numpy() - y_hat_oof,
        penalty="second_difference",
    )

    # Refit nuisance on the outer training window only.
    m_model.fit(train[X_cols], train[target] - B_train @ theta)
    pred_test = m_model.predict(test[X_cols]) + B_test @ theta

    store_outer_predictions(test.index, pred_test)
```

实际实现还应加入：

- side-specific target convention；
- leave-one-dealer-out reference mid；
- availability model；
- semantic filtering；
- scale/amplitude models；
- cluster weights；
- support score；
- fold-pure rolling features。

# 模型选择的 go/no-go 规则

以下阈值应由业务 tolerance 校准，不是普适常数。建议同时定义：

## 统计门槛

- global curve heterogeneity test；
- simultaneous band 是否排除 0；
- bootstrap stability；
- multiple-testing adjusted evidence。

## 预测门槛

- interaction model 的 OOT loss improvement；
- matched-pair difference improvement；
- quantity-bucket calibration bias；
- stress-period robustness。

## 复杂度门槛

- retained interaction 数量；
- low-support node 占比；
- curve crossing / oscillation；
- retraining instability；
- 解释与 audit 成本。

一个示例 decision rule：只有当某 interaction family 同时满足

1. common-support OOT MAE 改善超过预先设定的 materiality threshold；
2. 至少三个连续 time folds 同方向改善；
3. matched-pair calibration 改善；
4. pure interaction contrast 超过业务容忍值；
5. 不依赖单一 dealer/episode；

才从 $\mathcal M_2$ 升级到 $\mathcal M_3$。否则保留 scale–amplitude model。
