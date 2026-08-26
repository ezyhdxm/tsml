# TSML Research Notes

这个仓库保存可公开分享的时间序列、树模型、市场微观结构和 corporate-bond modeling 技术报告。

## 阅读与编辑入口

- [阅读列表与完成标记](READING_LIST.md)
- [报告编辑与渲染流程](meta/REPORT_EDITING_WORKFLOW.md)
- [第一棵 LightGBM 树的统计理论：Markdown](reports/machine-learning/lightgbm-first-tree-theory/SOURCE.md)
- [第一棵 LightGBM 树的统计理论：HTML](reports/machine-learning/lightgbm-first-tree-theory/index.html)
- [Irregular sampling autocorrelation：Markdown](reports/time-series/irregular-autocorrelation/SOURCE.md)
- [Irregular sampling autocorrelation：HTML](reports/time-series/irregular-autocorrelation/index.html)

## 目录结构

```text
.
├── README.md
├── READING_LIST.md
├── meta/
│   └── REPORT_EDITING_WORKFLOW.md
└── reports/
    ├── machine-learning/
    │   ├── lightgbm-first-tree-theory/
    │   └── autoresearch_time_series_lightgbm.html
    ├── time-series/
    │   ├── irregular-autocorrelation/
    │   └── time_series_ml_pipeline.html
    ├── market-microstructure/
    │   ├── order_book_master_equation_derivation.{md,html}
    │   ├── poisson_process_for_order_book.{md,html}
    │   └── single_queue_stationary_and_first_hitting_times_guide.md
    └── corporate-bond-modeling/
        ├── dealer-runs/
        └── vendor-models/
```

## Source 与 rendered artifact

新的长报告应把可编辑 Markdown 和渲染 HTML 放在同一目录。Markdown 是 source of truth；HTML 供直接阅读。详细命令见 [`meta/REPORT_EDITING_WORKFLOW.md`](meta/REPORT_EDITING_WORKFLOW.md)。

## 公开仓库边界

本仓库只放公开理论、公开资料整理和不包含真实工作数据的技术报告。公司数据、内部实现、私有研究记录和 credentials 不应进入本仓库。
