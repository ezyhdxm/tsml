# Convertible Bond Pricing · 可转债定价教程

[中文 HTML](index.html) · [可编辑正文](SOURCE.md)

从精确的欧式“债券 + call”分解出发，推导提前转股、回售、发行人赎回、Tsiveriotis–Fernandes 分拆及显式违约模型。包括条件期望到 PDE 的逐步推导、survival/default 树、有限差分框架、条款状态、联合校准、Greeks 与违约跳跃对冲。

**内容边界**：普通自愿转股可转债；理论、公开资料与合成实验。不是生产估值系统，不使用真实公司/客户/交易数据。文献核对日期为 2026-09-05；摘要级阅读与实际数值复现已在正文分别标明。

## 阅读

`index.html` 是独立 HTML，公式使用原生 MathML，图表及交互脚本全部内嵌，不需要 CDN 或联网。使用支持 MathML 的现代浏览器打开。手机上长公式、表格和代码在各自区域内横向滚动；正文不应横向溢出。关闭 JavaScript 仍能阅读正文、公式与静态数值表，只有图表/交互计算器不显示。

交互计算器支持调整股价、存续态扩散波动率和风险中性违约强度，并切换欧式、American、票息、call、put 等简化合约。它使用 300 步显式违约树，不是 TF/QuantLib 引擎。通知期、20/30 日触发、重设条款及实际结算机制只在正文分析，未伪装成已实现的功能。

## 文件

| 文件 | 用途 |
|---|---|
| `SOURCE.md` | 正文与公式的 source of truth，含最小 Python 基准 |
| `index.html` | 由 Markdown 渲染的单文件阅读版 |
| `experiments.py` | 重新计算收敛表、六种合约、Greeks 和十项模型检查 |
| `results.json` | 本次实际运行产生的合成数值结果；不是市场数据 |
| `interactive.js` | 网页端计算器与三个数值图表 |
| `style.css`, `template.html`, `render.py` | 可复用的排版与渲染材料 |
| `build_validation.json` | 数学公式、内部链接、占位符及依赖检查 |
| `validate_browser.py`, `browser_validation.json` | 浏览器校验程序和本次运行记录 |

## 修改与渲染

先编辑 `SOURCE.md`，不要只修改 `index.html`。渲染需要 Python 3.10+ 与 Pandoc；本次出版使用 **Pandoc 3.1.11.1**。渲染本身仅用 Python 标准库，不重新跑实验，也不联网。

```bash
python render.py
```

需要重算数值时，安装 NumPy、SciPy，再运行：

```bash
python experiments.py
python render.py
```

`experiments.py` 是合成教学模型，明确限制了事件日期对齐和半年度票息 schedule。更换参数后，`results.json` 会更新，图表与交互基准会使用新文件；正文中手写的表格/文字**不会自动改写**，应同步核对并更新 `SOURCE.md`。浮点数最后若干位可随软件环境改变。

浏览器检查需要 Playwright 和 Chromium：

```bash
python -m pip install playwright
python -m playwright install chromium
python validate_browser.py
# 系统已有 Chromium 时：
CHROMIUM_PATH=/usr/bin/chromium python validate_browser.py
```

本次校验：220 个 MathML 公式、无未渲染公式；1440px 桌面及 390px 手机视窗无文档级横向溢出；浏览器无 JavaScript 运行错误；六种合约在 600/1,200 步下的 Python/JavaScript 结果一致至浮点精度；交互范围角点测试通过。数值基准的十项检查全部通过。以上只是针对声明模型和测试点的验证，不是对任意合约的生产级认证。

## 发布流程

仓库 workflow `.github/workflows/render-convertible-bond-pricing.yml` 从可编辑素材重建 HTML，并维护根 README 与阅读列表入口。初次发布使用校验和保护的临时文本运输包恢复文件，生成后删除运输目录；后续直接修改可读源文件即可。
