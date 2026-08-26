# Irregularly Sampled Time-Series Autocorrelation

这份报告从 estimand 开始，区分：

- **clock-time ACF**：lag 是 5 分钟、30 分钟、1 小时；
- **event-time ACF**：lag 是下一笔、下两笔、下五笔 observation。

随后依次讨论 pair support、DCF/slotting、Gaussian kernel、S-ACF、spectral methods、OU/IAR/CARMA、bootstrap inference，以及 corporate-bond residual 中的 pooling、informative trade times 与 overlapping-target bias。

## 阅读与编辑

- [`index.html`](index.html)：HTML 阅读入口。
- [`SOURCE.md`](SOURCE.md)：Markdown 章节索引。
- [`manuscript/`](manuscript/)：可编辑正文，source of truth。
- [`irregular_acf.py`](irregular_acf.py)：pair construction、kernel/event-time ACF、cluster bootstrap 与 OU half-life。
- [`make_figures.py`](make_figures.py)：固定 seed 的 OU simulation 与图形复现。
- [`references.bib`](references.bib)：参考文献。

## Public-repo boundary

只包含公开文献、抽象数学和合成 OU 数据。不要提交公司数据、真实 residual 输出、内部路径或 credentials。
