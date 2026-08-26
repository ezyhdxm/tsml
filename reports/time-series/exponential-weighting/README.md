# Exponential weighting: theory and practice

本目录包含报告 **《指数加权方法：从 EMA 到状态空间、EWMA 协方差与不规则采样》**。

## 阅读入口

- `index.html`：HTML 阅读入口。正文压缩在文件内，现代浏览器会在本地解压；若浏览器不支持，可直接读 `SOURCE.md`。
- `SOURCE.md`：可编辑 Markdown，是正文 source of truth。
- `references.bib`：参考文献数据库。

## 代码

- `exponential_weighting.py`：normalized EWA、recursive EMA、irregular-time weighting、EWM covariance、RiskMetrics covariance、local-level gain 与 expert exponential weights。
- `test_exponential_weighting.py`：验证权重几何、half-life / mean age / effective sample size、Kalman 映射、协方差修正与 point-in-time recursion。

```bash
python -m pytest -q
```

报告只使用公开文献、数学推导与合成示例，不包含公司数据或内部实现。完整带图 standalone HTML 和复现脚本由本报告的生成记录保留；仓库版本用文字图注替代大体积 SVG，以便迭代正文。
