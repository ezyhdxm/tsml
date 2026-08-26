# 非规则采样时间序列的自相关：可编辑源码入口

这份长报告按论证顺序拆成六个可编辑 Markdown 章节。`manuscript/` 是正文的 source of truth；不要只修改 rendered HTML。

1. [`00-foundations.md`](manuscript/00-foundations.md)：问题、clock-time/event-time estimands 与 pair-product identification。
2. [`01-direct-estimators.md`](manuscript/01-direct-estimators.md)：DCF/slotting、Gaussian kernel、ZDCF、S-ACF 与 spectral route。
3. [`02-models-and-inference.md`](manuscript/02-models-and-inference.md)：OU/IAR/CARMA、resampling bias、bootstrap 与 nonstationarity。
4. [`03-bond-residuals.md`](manuscript/03-bond-residuals.md)：corporate-bond residual 的 series、pooling、informative sampling 与 target overlap。
5. [`04-workflow-and-code.md`](manuscript/04-workflow-and-code.md)：不跳步的实务流程与 Python 用法。
6. [`05-reading-and-recommendations.md`](manuscript/05-reading-and-recommendations.md)：文献地图与最终建议。

## 重新生成单文件 HTML

在本目录运行：

```bash
python make_figures.py
pandoc manuscript/*.md \
  --standalone \
  --toc --toc-depth=3 \
  --citeproc --bibliography=references.bib \
  --mathml \
  --css=style.css \
  --embed-resources \
  -o index.full.html
```

仓库中的 [`index.html`](index.html) 是轻量阅读入口，并链接到预渲染的章节 HTML；这样在 GitHub 上更容易逐章审阅和迭代。
