# Corporate Bond Quote Quantity 的可分离性与交互结构

本目录是一份可迭代的研究报告，研究 corporate-bond dealer quote 的 quantity effect 是否可从其他 fair-value / liquidity features 中分离，以及何时必须保留 interaction。

- **Markdown 总入口：** [`SOURCE.md`](SOURCE.md)
- **分章源文件：** [`manuscript/`](manuscript/)
- **渲染结果：** [`index.html`](index.html)
- **参考文献：** [`references.bib`](references.bib)
- **渲染样式：** [`style.css`](style.css)
- **阅读进度：** [仓库总阅读列表](../../../READING_LIST.md)

## 内容范围

报告依次处理：

1. quantity-neutral quote target 与“可分离”的精确定义；
2. fixed cost、客户议价、inventory risk、information risk、competition 与 hedgeability 如何产生不同的 quantity interaction；
3. 为什么 naive 的“先拟合其他 feature，再拟合 residual 对 quantity”一般不识别 additive quantity curve；
4. 如何用 joint additive model、orthogonalized spline、varying-coefficient model 和 purified functional ANOVA 检验 interaction；
5. quantity endogeneity、客户/dealer composition、quote-size 语义、support/overlap 与 point-in-time leakage；
6. 从 additive baseline 到 scale–amplitude–shape model 的分层建模与生产化验证。

## 重新渲染

在本目录运行：

```bash
pandoc \
  manuscript/00-frontmatter-and-research-question.md \
  manuscript/01-economic-mechanisms-and-hypotheses.md \
  manuscript/02-identification-and-separability.md \
  manuscript/03-estimation-tests-and-model-comparison.md \
  manuscript/04-implementation-validation-and-literature.md \
  --standalone \
  --toc --toc-depth=3 \
  --citeproc --bibliography=references.bib \
  --mathml \
  --css=style.css \
  --embed-resources \
  --metadata title-prefix="TSML" \
  -o index.full.html
```

仓库中的 `index.html` 是一个轻量 loader；它从同目录 `payloads/` 读取 gzip-compressed rendered HTML。下载或移动时请保留 `index.html` 与 `payloads/` 的相对目录结构。

## 编辑约定

只在 `manuscript/*.md` 中修改正文和公式；参考文献修改 `references.bib`；不要直接手改 rendered HTML 或 payload。报告只使用公开文献、抽象数学和合成示例，不包含任何公司数据或内部实现。
