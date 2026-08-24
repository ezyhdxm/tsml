# Report editing workflow

`tsml` 中可迭代报告采用以下约定：

1. Markdown 是正文与公式的 source of truth。
2. HTML 是由 Markdown 生成的 rendered artifact。
3. 编辑时先改 `.md`，再重新生成对应 `.html`；不要只修改 HTML。
4. 行内公式使用 `$...$`，display 公式使用：

   ```markdown
   $$
   E[T\mid X]=\frac{1}{\lambda}.
   $$
   ```

5. 新的完整报告优先使用独立目录。短报告可用单个 `report.md`；长报告可按章拆到 `manuscript/`：

   ```text
   reports/<topic>/<report-name>/
   ├── README.md
   ├── SOURCE.md
   ├── manuscript/
   │   ├── 00-frontmatter.md
   │   ├── 01-main-result.md
   │   └── 02-extensions.md
   ├── index.html
   ├── references.bib   # 如需要
   └── style.css        # 如需要
   ```

6. 推荐 Pandoc 渲染方式：

   ```bash
   pandoc manuscript/*.md \
     --standalone \
     --toc --toc-depth=3 \
     --citeproc --bibliography=references.bib \
     --mathml \
     --css=style.css \
     --embed-resources \
     -o index.full.html
   ```

7. `tsml` 是公开仓库。不得提交公司数据、内部代码、未公开截图、凭证、路径、主机名或其他敏感信息。公开报告只使用公开资料、抽象数学与合成示例。
8. 每次新增或移动报告后，同步更新根目录的 `README.md` 与 `READING_LIST.md`。
