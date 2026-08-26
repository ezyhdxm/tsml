# 12. 文献地图：按什么顺序读？

## 第一层：建立方法全景

1. **Rehfeld et al. (2011)**：先读。它直接比较 interpolation、slotting、Gaussian kernel 与 Lomb--Scargle，并通过模拟展示 sampling irregularity 如何改变误差 [@rehfeld2011]。
2. **Edelson and Krolik (1988)**：理解 DCF 如何从真实 pair lags 出发、为什么拒绝无原则 interpolation [@edelson1988]。

## 第二层：理解现代 direct estimators

3. **Kreutzer et al. (2023)**：理解 S-ACF 的 selection/weight construction，以及它与 all-pairs kernel 的 weighting difference [@kreutzer2023]。
4. **Alexander (2013)**：了解 equal-population bins 与 Fisher transform 如何改善 sparse DCF/ZDCF [@alexander2013]。
5. **Bjørnstad and Falck (2001)**：把 ACF 看成 nonparametric covariance curve，并重视 simultaneous envelope 与 PSD 问题 [@bjorstad2001]。

## 第三层：进入 model-based continuous time

6. **Eyheramendy et al. (2018)**：IAR transition、$\phi^{\Delta}$ ACF 与 irregular likelihood [@eyheramendy2018]。
7. **Brockwell (2014)**：从 OU/CAR(1) 扩展到 CARMA 和更一般 continuous-time state-space structure [@brockwell2014]。

## 第四层：金融数据特有问题

8. **Engle and Russell (1998)**：把 intertrade durations 本身视为 stochastic process，而不是无关紧要的 missingness [@engle1998]。
9. **Chen, Ning, and Cai (2015)**：理解 informative observation times 为什么会使标准 inference 偏离目标 [@chen2015]。

# 13. 最终建议

对你当前的 corporate-bond prediction residual，我建议把主分析固定为下面的组合，而不是寻找一个“唯一正确 irregular ACF 函数”：

| component | default | purpose |
|---|---|---|
| event-time dependence | within CUSIP/session, lags 1/2/3/5/10 | 看连续几笔 trade 是否同方向错 |
| clock-time dependence | all-pairs Gaussian kernel | 看 wall-clock persistence |
| support | pair count + weight concentration | 判断哪些 lag 真有数据 |
| uncertainty | date/session block or multiway cluster bootstrap | 避免把 overlapping pairs 当 iid |
| pooling | pair-weighted + bond-equal/shrunk | 区分活跃市场与典型 bond |
| time clock | session time；另做 calendar/business sensitivity | 避免 overnight 解释混乱 |
| confound audit | target overlap、duplicates、common factor、sampling intensity | 避免把机械结构解释成 alpha |
| parametric summary | OU half-life only after shape check | 用一个数字总结，而不是取代曲线 |

最核心的一句话是：

> Irregular sampling 并没有破坏 autocorrelation 的定义；它暴露了 regular ACF 中被固定网格隐藏的选择——lag 的含义、pair 的权重、可识别的时间尺度、以及 observation process 是否外生。

只要这四件事写清楚，估计方法之间的关系就不会再显得跳跃。

# References
