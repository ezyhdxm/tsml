# Time Series Cross-Validation — 教程式综述

[阅读 HTML](index.html) · [编辑 Markdown](SOURCE.md)

以 forecasting procedure 为评估对象，覆盖信息可用性、rolling/expanding、blocked/hv-block、随机 K-fold 的适用边界、purging/embargo/CPCV、nested selection、重训与多 horizon、不规则事件与 panel、MAE/quantile、配对损失和依赖稳健推断。15 节正文、3 个附录、24 项带阅读问题的参考资料。检索截至 2026-09-04；不是穷尽式系统综述。

## 编辑和渲染

`SOURCE.md` 是正文 source of truth；`style.css` 与 `template.html` 控制展示。不要仅修改生成 HTML。

```bash
# 已测试 Pandoc 3.1.11.1；渲染仅使用 Python 标准库。
python render.py
```

生成 `index.html` 和 `build_validation.json`。HTML 内嵌样式与交互脚本，公式为原生 MathML，不依赖 CDN，可离线阅读。长公式、表格、代码在窄屏内局部横向滚动。目录、horizon/标签延迟演示及打印布局包含在文件内。Pandoc 3.1.11.1 的 zh-CN 翻译警告只涉及默认词条；自定义模板已经提供中文标题，不影响公式转换。

## 合成实验复现

```bash
python -m pip install numpy==2.3.5
python experiments.py
```

三组实验各 200 次重复；生成 `results.json`。表格数值是固定版本的运行结果，不能将任意改动后的 DGP 与旧表混用。需要改变实验时，运行后同步更新正文表格与条形图。

A：稳定 AR(1) + OLS；B：回归斜率突变；C：重叠 24 步目标的 MAE 与标准误。它们不是金融数据回测，也不是所引论文的完整复现。MCSE、重复间 SD 与单次时间序列标准误在正文分别定义。

附录 A 的 calendar-time splitter 需要 NumPy、pandas；测试环境 pandas 2.2.3。它只决定索引和 eligibility，不替代原始数据 point-in-time 审计或完整嵌套学习系统。

## 已完成检查

`build_validation.json`：公式、内部链接与离线资源检查。
`splitter_checks.json`：13 项代码检查，包括延迟标签、未知验证标签、严格截止、时区、排序、window/gap 和文内实验代码一致性。
`browser_checks.json`：Chromium 1440px 与 390px 视口的整页溢出、脚本错误、标签成熟交互检查；这些是本地实际运行的冻结结果，不冒充其他浏览器的验证。

所有内容仅使用公开文献、一般数学与独立合成数据，不含工作数据或内部实现。
