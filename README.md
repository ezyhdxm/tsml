# tsml

A versatile research and iteration framework for machine learning on time-series, temporal panel, event-driven, and irregularly sampled data.

This repository is intentionally domain-agnostic. It provides a reusable workflow for point-in-time data construction, time-aware splitting, baseline design, temporal diagnostics, feature engineering, residual analysis, paired model comparison, robustness analysis, production monitoring, and reproducible experiment handoff.

## Main document

Open [`time_series_ml_pipeline.html`](./time_series_ml_pipeline.html) for the living framework.

## Iteration philosophy

Each model change should begin with a falsifiable hypothesis, be evaluated on future/out-of-sample observations, and end with a clear decision: keep, revert, or investigate.

For irregular timestamps, actual elapsed time is a first-class variable rather than an afterthought.

## Repository role

This repo is intended to be the source of truth across notebooks, experiments, collaborators, and future AI-assisted research sessions. Domain-specific experiments can be added separately without narrowing the core framework.
