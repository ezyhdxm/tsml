# 2026-09-03：补充 surface 动机与经典方法

## 新增

- 第 3A 节：统一 Black–Scholes 模型与 IV 报价坐标；确定性时间变动只能解释 term structure；混合分布 smile；修模型与 surface 的关系；边际与路径信息；Dupire 的逐步推导；smile dynamics 与 delta。
- 第 3B 节：经典曲面构造、price-space convex fitting、SVI/SSVI、Heston/SABR/local vol、PCA/GP；VAE 的潜在增量、基线公平性、部署成本、P 与 Q 的区别；不把 reconstruction gain 当作交易收益。
- `classical_demo.py`：三个确定性教学例子，两个图表；不使用市场数据、不训练模型。

## 修正与验证

- 更正原第 3.2 节过度概括的 calendar 解释：在确定性 carry / 连续比例分红等假设下，归一化价格或 total variance 在固定 log-forward-moneyness 的单调性不是仅仅的代理条件；有限网格与其他合约条件仍需区分。
- IV 唯一反解明确要求价格严格位于上下界之间。
- 新增章节放在 VAE 概率推导之前，避免跳过“为什么要建曲面”。
- 正文 MathML 计数与 Pandoc AST 比对，拒绝 merror，检查 14 处图像内嵌；普通表格增加移动端局部滚动。
- 原 VAE 脚本、实验 JSON、训练日志、补全案例及 SSVI 数据哈希保持不变。渲染直接复用仓库内已验证的合成数据，避免跨平台重新生成导致浮点字节差异。

新数学例子是教学计算，不是原论文复现，也不构成新市场 benchmark。公开 ZIP 仍不包含原始三个模型 checkpoint。
