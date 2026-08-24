# TSML 阅读列表

使用 GitHub 的 **Edit this file**，把读完的项目从 `- [ ]` 改成 `- [x]`。建议在每一项下面补一行自己的疑问、反例或可复现实验；之后可以围绕这些标记继续迭代报告。

## A. 第一棵 LightGBM 树：主报告

**报告入口：** [HTML](reports/machine-learning/lightgbm-first-tree-theory/index.html) · [可编辑 Markdown](reports/machine-learning/lightgbm-first-tree-theory/SOURCE.md)

- [ ] 读摘要、第 1–3 节：明确第一棵树为何等价于 weighted pseudo-response tree。
- [ ] 读第 4–6 节：逐行复核 fixed split contrast、LightGBM raw gain 与 root Gaussian max law。
- [ ] 读第 7–9 节：理解 Brownian bridge、跨 feature multiplicity 与 signal margin。
- [ ] 读第 10 节：自己证明完整 first tree 的 tree-Haar contrasts 两两正交。
- [ ] 读第 11 节：理解为什么条件于 path/sign 后 selection event 是 polyhedral。
- [ ] 读第 12–13 节：复核 `df = E[gain]/sigma^2` 以及 learning-rate train/test risk 公式。
- [ ] 读第 14–19 节：区分 ridge、L1、general loss、correlated data 与真实 LightGBM 实现。
- [ ] 读第 20–23 节和附录：形成自己的 root-null、full-tree-null 与 honest-gain 实验方案。
- [ ] 在报告 Markdown 中写下至少 3 个疑问或希望扩展的 theorem。
- [ ] 实现并验证 `num_leaves=2` 时 model-reported gain 与手算 SSE reduction 一致。
- [ ] 对自己的 feature matrix 跑一次 dependence-preserving root null experiment。
- [ ] 对完整第一棵树生成 leaf cluster-support report。

## B. 第一棵树理论：原始文献阅读顺序

### B1. Boosting 与 gain

- [ ] [Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine*](https://doi.org/10.1214/aos/1013203451) — 先掌握函数空间中的 stagewise gradient view。
- [ ] [Chen & Guestrin (2016), *XGBoost*](https://doi.org/10.1145/2939672.2939785) — 重点读二阶 surrogate、leaf output 与 split gain。
- [ ] [Ke et al. (2017), *LightGBM*](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) — 重点读 GOSS/EFB；注意它不是 post-selection inference 论文。
- [ ] [LightGBM pinned source snapshot](https://github.com/lightgbm-org/LightGBM/tree/3ec5b99b3a2e257e20d862e783dad40ad6511e0e) — 对照 leaf output、gain、categorical ordering 与 leaf-wise growth。

### B2. Threshold scan 与 unbiased recursive partitioning

- [ ] [Lausen & Schumacher (1992), *Maximally Selected Rank Statistics*](https://doi.org/10.2307/2532740) — 理解 threshold maximization 不是固定检验。
- [ ] [Hothorn, Hornik & Zeileis (2006), *Unbiased Recursive Partitioning*](https://doi.org/10.1198/106186006X133933) — 看 multiplicity-adjusted variable selection 与 stopping。
- [ ] [Zeileis, Hothorn & Hornik (2008), *Model-Based Recursive Partitioning*](https://doi.org/10.1198/106186008X319331) — 把 score fluctuation、Brownian bridge 与 recursive splitting 连起来。

### B3. Search degrees of freedom 与选择后推断

- [ ] [Tibshirani (2015), *Degrees of Freedom and Model Search*](https://www3.stat.sinica.edu.tw/statistica/J25N3/J25N323/J25N323.html) — 理解 nominal parameters 与 search df 的差别。
- [ ] [Lee et al. (2016), *Exact Post-Selection Inference*](https://doi.org/10.1214/15-AOS1371) — 重点掌握 polyhedral lemma。
- [ ] [Neufeld, Gao & Witten (2022), *Tree-Values*](https://jmlr.org/papers/v23/21-0722.html) — 看 CART terminal-node inference 如何正式条件化 tree selection。

### B4. 高维、相关数据与长期扩展

- [ ] [Klusowski & Tian, *Large Scale Prediction with Decision Trees*](https://arxiv.org/abs/2104.13881) — 对照“搜索多”与 prediction consistency 并不矛盾。
- [ ] [Klusowski & Tian, *Nonparametric Variable Screening with Optimal Decision Stumps*](https://arxiv.org/abs/2011.02683) — 研究 stump 的 signal 与高维筛选条件。
- [ ] [Rabinowicz & Rosset (2022), *Tree-Based Models for Correlated Data*](https://www.jmlr.org/papers/v23/21-0885.html) — 对 bond/time/cluster dependence 最直接。
- [ ] [Zhou & Hooker (2022), *Boulevard*](https://jmlr.org/papers/v23/21-0078.html) — 作为后续多树 limiting distribution 的入口。

## C. 现有 TSML 报告

### C1. Machine learning 与 time series

- [ ] [Autoresearch for Time-Series LightGBM](reports/machine-learning/autoresearch_time_series_lightgbm.html)
- [ ] [Time-Series ML Pipeline](reports/time-series/time_series_ml_pipeline.html)

### C2. Market microstructure / queue models

- [ ] [Poisson Process for Order Book — HTML](reports/market-microstructure/poisson_process_for_order_book.html) · [Markdown](reports/market-microstructure/poisson_process_for_order_book.md)
- [ ] [Order Book Master Equation — HTML](reports/market-microstructure/order_book_master_equation_derivation.html) · [Markdown](reports/market-microstructure/order_book_master_equation_derivation.md)
- [ ] [Single Queue Stationary Distribution and First Hitting Times — Markdown](reports/market-microstructure/single_queue_stationary_and_first_hitting_times_guide.md)

### C3. Corporate-bond modeling：dealer-run features

- [ ] [Dealer-run feature engineering framework](reports/corporate-bond-modeling/dealer-runs/dealer_run_feature_engineering_framework.html)
- [ ] [Dealer-run implementation review](reports/corporate-bond-modeling/dealer-runs/dealer_run_implementation_review.html)
- [ ] [Dealer-run / CPP interaction update plan](reports/corporate-bond-modeling/dealer-runs/dealer_run_cpp_interaction_update_plan.html)

### C4. Corporate-bond vendor model reconstructions

- [ ] [Vendor-modeling index](reports/corporate-bond-modeling/vendor-models/bond_pricing_vendor_modeling_index.html)
- [ ] [Tradeweb Ai-Price reconstruction](reports/corporate-bond-modeling/vendor-models/tradeweb_aiprice_model_reconstruction.html)
- [ ] [MarketAxess CP+ reconstruction](reports/corporate-bond-modeling/vendor-models/marketaxess_cpplus_model_reconstruction.html)
- [ ] [Trumid FVMP reconstruction](reports/corporate-bond-modeling/vendor-models/trumid_fvmp_model_reconstruction.html)

## D. 阅读记录模板

复制下面的小节到本文件底部，或者直接写进对应报告的 Markdown：

```markdown
### YYYY-MM-DD — 报告或论文标题

- 我确认理解的结论：
- 我不相信/还没想通的步骤：
- 我能构造的反例：
- 应该补的 simulation：
- 与当前 bond model 的连接：
```
