# TSML Research Notes

这个仓库保存可公开分享的时间序列、树模型、利率期限结构、波动率曲面、市场微观结构和 corporate-bond modeling 技术报告。

## 阅读与编辑入口

- [阅读列表与完成标记](READING_LIST.md)
- [English report index](ENGLISH_REPORT_INDEX.md)
- [报告编辑与渲染流程](meta/REPORT_EDITING_WORKFLOW.md)
- **Credit Products Pricing & Risk Management**：[中文 HTML](reports/corporate-bond-modeling/credit-products-pricing-risk-guide/index.html) · [可编辑 Markdown](reports/corporate-bond-modeling/credit-products-pricing-risk-guide/SOURCE.md) — public-only 教程，覆盖 CDS、CDX、CDX option/tranche、TRS、Bond ETF、CRT/SRT、ABX/CMBX、callable/sinkable bonds，以及 FRTB CSR/Curvature/JTD 映射。
- **Vol Surface × VAE 教程与文献复现**：[中文 HTML](reports/volatility-surfaces/vae-tutorial/index.html) · [可编辑 Markdown](reports/volatility-surfaces/vae-tutorial/SOURCE.md) · [复现材料与说明](reports/volatility-surfaces/vae-tutorial/README.md)
- **利率期限结构模型**：[中文 HTML](reports/interest-rates/term-structure-models/index.html) · [中文 Markdown](reports/interest-rates/term-structure-models/SOURCE.md) · [English HTML](reports/interest-rates/term-structure-models/index.en.html) · [English Markdown](reports/interest-rates/term-structure-models/SOURCE.en.md)
- [第一棵 LightGBM 树的统计理论：Markdown](reports/machine-learning/lightgbm-first-tree-theory/SOURCE.md)
- [第一棵 LightGBM 树的统计理论：HTML](reports/machine-learning/lightgbm-first-tree-theory/index.html)
- [Irregular sampling autocorrelation：Markdown](reports/time-series/irregular-autocorrelation/SOURCE.md)
- [Irregular sampling autocorrelation：HTML](reports/time-series/irregular-autocorrelation/index.html)
- [Exponential weighting theory：Markdown](reports/time-series/exponential-weighting/SOURCE.md)
- [Exponential weighting theory：HTML](reports/time-series/exponential-weighting/index.html)

## 目录结构

```text
.
├── README.md
├── READING_LIST.md
├── ENGLISH_REPORT_INDEX.md
├── meta/
│   └── REPORT_EDITING_WORKFLOW.md
└── reports/
    ├── volatility-surfaces/
    │   └── vae-tutorial/
    ├── interest-rates/
    │   └── term-structure-models/
    ├── machine-learning/
    │   ├── lightgbm-first-tree-theory/
    │   └── autoresearch_time_series_lightgbm.html
    ├── time-series/
    │   ├── exponential-weighting/
    │   ├── irregular-autocorrelation/
    │   └── time_series_ml_pipeline.html
    ├── market-microstructure/
    │   ├── order_book_master_equation_derivation.{md,html}
    │   ├── poisson_process_for_order_book.{md,html}
    │   └── single_queue_stationary_and_first_hitting_times_guide.md
    └── corporate-bond-modeling/
        ├── credit-products-pricing-risk-guide/
        ├── dealer-runs/
        └── vendor-models/
```

## Source 与 rendered artifact

新的长报告应把可编辑 Markdown 和渲染 HTML 放在同一目录。Markdown 是 source of truth；HTML 供直接阅读。完整报告可以按章节拆分到 `manuscript/`，并由 `SOURCE.md` 作为可编辑入口。详细命令见 [`meta/REPORT_EDITING_WORKFLOW.md`](meta/REPORT_EDITING_WORKFLOW.md)。

## 公开仓库边界

本仓库只放公开理论、公开资料整理和不包含真实工作数据的技术报告。公司数据、内部实现、私有研究记录和 credentials 不应进入本仓库。
