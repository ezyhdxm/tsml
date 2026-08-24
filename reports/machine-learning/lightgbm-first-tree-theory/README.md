# 第一棵 LightGBM 树的统计理论

本目录是一份可迭代的研究报告。

- **Markdown 总入口：** [`SOURCE.md`](SOURCE.md)
- **分章源文件：** [`manuscript/`](manuscript/)
- **渲染结果：** [`index.html`](index.html)
- **参考文献：** [`references.bib`](references.bib)
- **渲染样式：** [`style.css`](style.css)
- **阅读进度：** [仓库总阅读列表](../../../READING_LIST.md)

## 内容范围

报告从第一 boosting round 的二阶 surrogate 出发，完整处理：

1. weighted pseudo-response 表示；
2. LightGBM raw gain 与二阶 objective reduction 的因子区别；
3. L2 root split 的有限样本精确 Gaussian maximum law；
4. threshold scan 的 Brownian bridge；
5. 完整 leaf-wise 第一棵树的 tree-Haar 正交几何；
6. polyhedral selection event 与选择后推断；
7. search degrees of freedom、learning rate 与样本外风险；
8. ridge、L1、binary、Poisson、相关数据与真实 LightGBM 实现偏离；
9. root/full-tree null simulation 与 honest gain audit。

## 重新渲染

在本目录运行：

```bash
pandoc \
  manuscript/00-frontmatter-and-foundations.md \
  manuscript/01-root-split-theory.md \
  manuscript/02-full-tree-selection-and-risk.md \
  manuscript/03-losses-dependence-and-implementation.md \
  manuscript/04-audits-implications-and-appendices.md \
  --standalone \
  --toc --toc-depth=3 \
  --citeproc --bibliography=references.bib \
  --mathml \
  --css=style.css \
  --embed-resources \
  --metadata title-prefix="TSML" \
  -o index.full.html
```

仓库中的 `index.html` 是一个轻量 loader；它从同目录 `payloads/` 读取 gzip-compressed rendered HTML。这样既保留完整浏览效果，也避免把一大段机器生成 HTML 混入 Markdown review。下载或移动时请保留 `index.html` 与 `payloads/` 的相对目录结构。

## 编辑约定

只在 `manuscript/*.md` 中修改正文和公式；参考文献修改 `references.bib`；不要直接手改 rendered HTML 或 payload。修改完成后重新渲染，并重新生成 payload。
