# Vol Surface × VAE：教程、文献与结构复现

**资料截止：2026-09-03。** 本目录保存完整中文教程及其公开复现材料，不包含公司数据、内部实现或第三方受许可市场数据。

## 阅读与下载

- [独立 HTML 报告](index.html)：原生 MathML，14 处内嵌图像；下载后可离线用浏览器阅读。
- [完整 Markdown](report.md) · [分章编辑入口](SOURCE.md)
- [公开复现包 ZIP](public_reproduction_bundle.zip)
- [完整 VAE 实验脚本](reproduce_vol_surface_vae.py) · [新增经典方法示例](classical_demo.py)
- [原始实验结果 JSON](reproduction/results.json) · [训练日志 CSV](reproduction/training_history.csv)
- [补全案例 CSV](reproduction/sample_completion.csv)
- [合成 SSVI 数据](reproduction/synthetic_ssvi_data.npz) · [真实生成因子 CSV](reproduction/synthetic_ssvi_factors.csv)
- [文件校验清单](MANIFEST.sha256) · [渲染检查](build_validation.json) · [原始文件哈希](publication.json)

GitHub 的文件页面不直接渲染任意 HTML；使用 **Download raw file** 下载 `index.html`，然后在浏览器中打开。仓库链接不是已部署的网站地址。

## 新增：为什么需要 surface？经典方法与 VAE 的收益

建议先读 [第 3A 节](manuscript/00a-why-surface.md) 和 [第 3B 节](manuscript/00b-classical-methods.md)。它们区分 Black–Scholes 模型与 IV 换算公式、静态报价与路径动态，逐步讲解 price-space constrained fitting、SVI/SSVI、local vol、Heston、SABR、PCA 与 Gaussian conditional completion，并讨论何时不需要 VAE。

新增三个可重跑教学例子：同一个混合终值分布产生不同 IV；bid/ask 内的受约束价格修复；相同 vanilla 边际分布但不同路径数字期权价格。结果见 [classical_examples](classical_examples/)，修订范围见 [REVISION.md](REVISION.md)。

## 原有内容与证据边界

教程包含 Black 定价、IV 反解、total variance 与静态无套利；逐步推导 VAE 的 ELBO、Gaussian KL、reparameterization 和 masked inference；审计 2021–2026 年 14 篇核心工作的数据、网络、训练和复现条件。

VAE 实验是在 1,800 张合成 SSVI 曲面上比较 PCA-8、masked MLP-VAE、ConvVAE 和 ConvVAE+NA 的**结构复现**，不是专有 FX/SPX 或公开 crypto benchmark 的严格复刻。保留 PCA 胜出、以及重建可行性改善而先验采样退化的负结果。本次新增章节没有重新训练模型，也不声称已证明 VAE 带来交易 P&L。

## 公开包与原始会话 ZIP 的区别

`results.json`、训练日志、补全案例和 VAE 实验脚本保留原始字节。SSVI 数据由原 seed 确定性重建并验证 SHA-256；VAE 图表使用已保存结果重绘，不重新计算 benchmark。

`public_reproduction_bundle.zip` 包含本版报告、源码、结果、日志、合成数据、教学示例和图表，**不包含原始三个 `.pt` 模型权重，也不是原始会话 ZIP 的逐字节副本**。原始带 checkpoint 的包仍保留在创建报告的会话下载中。报告附录 D 描述原始完整实验产物，不表示所有产物都已上传仓库。

## 编辑与渲染（不训练）

以 `manuscript/` 内的 11 个章节文件为 source of truth；`report.md` 与 `index.html` 是生成文件。

```bash
# Python 3.13.5；另安装 Pandoc 3.1.11.1
python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 matplotlib==3.10.8
python render.py
# 仅重跑新增的数学例子：
python classical_demo.py --output-dir classical_examples
```

渲染流程校验冻结文件、重绘图像、生成原生 MathML，将正文公式数与 Pandoc AST 逐项计数比对，并更新公开 ZIP。修改该目录会触发 render-only GitHub Actions 工作流。

## 重新运行 VAE 实验

另需安装 PyTorch 和 scikit-learn。原始环境见 `reproduction/results.json`，其中 PyTorch 为 `2.10.0+cpu`。

```bash
python reproduce_vol_surface_vae.py --output-dir new_run
python reproduce_vol_surface_vae.py --quick --output-dir smoke_test
```

新结果应写入新目录，不覆盖冻结的 `reproduction/`。不同平台、版本或训练配置不保证逐字节复现原始 checkpoints。
