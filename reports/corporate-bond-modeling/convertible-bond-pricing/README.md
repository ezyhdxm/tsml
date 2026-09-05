# Convertible Bond Pricing · 可转债定价与 Black–Scholes

[完整中文 HTML](index.html) · [新增 Black–Scholes 专题](index.html#bs-replication) · [原可转债正文](SOURCE.md) · [新增专题 Markdown](foundations/)

原报告的 15 节正文和 2 个附录全部保留；第 4 节之后新增 BS1–BS9 基础专题：自融资复制与完整正态积分、离散 delta hedge、股票 bid–ask 的精确现金账本、成本与风险权衡、确定时间变化 volatility、Dupire local volatility、Heston 随机方差与未被覆盖的风险，以及可复算对冲实验。

**内容边界**：公开理论、公开资料与合成实验，不使用公司或客户交易数据，不是生产估值系统。数值实验与理论扩展分开标记：未实现 Heston Fourier/PDE 引擎、local-vol/LSV 校准、最优 no-trade band、Leland 策略回测、无差异报价或 deep hedging。波动率消息揭示实验不是 Heston 模拟。

## 阅读与编辑

`index.html` 是完整单文件：原生 MathML、内嵌图表和脚本，无 CDN 或运行时联网依赖。现代浏览器可直接打开；手机上长公式、表格和代码在各自区域滚动。关闭 JavaScript 后仍可阅读正文和静态表。

原来的可转债交互计算器保留。新增图表允许调整股票 half-spread 和风险权重，并比较常数 volatility 与独立波动率消息模型。它使用冻结的 Monte Carlo 结果；改变 spread 只缩放固定策略费用，不重新优化策略，也不生成市场报价。

Markdown 是 source of truth：`SOURCE.md` 保存原报告，`foundations/01-...md` 至 `05-...md` 保存新增内容。`render.py` 按显式文件顺序将专题插入原第 5 节之前，原章节编号及锚点不变。请编辑对应 Markdown 后重新渲染，不要仅修改 HTML。

## 文件

| 文件 | 用途 |
|---|---|
| `SOURCE.md` | 原可转债正文与两个附录 |
| `foundations/` | 九节新增专题的五个 Markdown 源文件 |
| `index.html` | 完整合并后的阅读版 |
| `experiments.py`, `results.json` | 原可转债树、六种条款设定与十项检查 |
| `hedging_experiments.py`, `hedging_results.json` | 新增精确 GBM 对冲模拟、费用账本、消息风险与九项检查 |
| `interactive.js` | 原可转债计算器与图表 |
| `hedging_lab.js` | 新增费用/风险和非零误差极限图表 |
| `style.css`, `template.html`, `render.py` | 排版与离线 MathML 渲染 |
| `build_validation.json` | 公式、锚点、占位符、外部依赖及 HTML 校验和 |
| `validate_browser.py`, `browser_validation.json` | 原交互计算器的浏览器校验 |
| `validate_hedging_browser.py`, `hedging_browser_validation.json` | 新专题布局、图表和交互的浏览器校验 |

## 复现

仅渲染需要 Python 3.10+ 与 Pandoc；出版使用 Pandoc 3.1.11.1。渲染本身不联网、不重新跑实验：

```bash
python render.py
```

重算新增实验的出版环境为 Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0：

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0
python hedging_experiments.py --paths 32768 --seed 20260905
python render.py
```

需要重算原可转债实验时另运行 `python experiments.py`。两组结果与检查互相独立，不能拿原树模型的测试为尚未实现的随机波动率引擎背书。

新实验每个场景为 32,768 条独立路径；四个对冲网格共享嵌套股票路径。模拟测度是明确指定的 Q，不是历史概率 P。所有 P&L 和费用折现至今天，包含开仓、调仓和现金结算后的平仓；程序逐路径核对两套现金账本。参数、软件版本、MCSE、VaR/ES、极限方差积分及检查结果见 JSON。

修改参数后，JSON 与图表可以更新，但正文静态数字不会自动改写；必须同步核对 Markdown。浮点末位和随机实现可随环境不同而变化。Monte Carlo 均值标准误、单次对冲风险、离散化误差和模型误差是不同对象。

## 浏览器与发布校验

```bash
python -m pip install playwright
python -m playwright install chromium
python validate_browser.py
python validate_hedging_browser.py
# 使用系统浏览器：CHROMIUM_PATH=/usr/bin/chromium python validate_hedging_browser.py
```

测试记录文件保存对应版本的实际检查结果。检查涵盖桌面与手机文档宽度、MathML、JavaScript 错误、图表非空、滑块响应及 Python/JavaScript 数值一致性。它们验证声明模型和测试点，不构成任意合约的生产认证。

`render-convertible-bond-pricing.yml` 在 main 上重建 HTML。`build-convertible-foundations.yml` 用于明确触发的专题实验复算和发布前检查；生成的文件仍以可读 source、JSON 和 HTML 保存。
