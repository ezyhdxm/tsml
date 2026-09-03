# Vol Surface × VAE：教程、文献与结构复现

**资料截止：2026-09-03。** 本目录保存完整中文教程及其公开复现材料，不包含公司数据、内部实现或第三方受许可市场数据。

## 阅读与下载

- [独立 HTML 报告](index.html)：337 个 MathML 公式、12 处内嵌图像；下载后可离线用浏览器阅读。
- [完整 Markdown](report.md) · [分章编辑入口](SOURCE.md)
- [公开复现包 ZIP](public_reproduction_bundle.zip)
- [完整实验脚本](reproduce_vol_surface_vae.py)
- [原始实验结果 JSON](reproduction/results.json) · [训练日志 CSV](reproduction/training_history.csv)
- [补全案例 CSV](reproduction/sample_completion.csv)
- [合成 SSVI 数据](reproduction/synthetic_ssvi_data.npz) · [真实生成因子 CSV](reproduction/synthetic_ssvi_factors.csv)
- [文件校验清单](MANIFEST.sha256) · [渲染检查](build_validation.json) · [发布说明与原始文件哈希](publication.json)

GitHub 的文件页面不直接渲染任意 HTML；使用 **Download raw file** 下载 `index.html`，然后在浏览器中打开。仓库链接不是已部署的网站地址。

## 内容

教程从 Black 定价、IV 反解、smile/skew/term structure、total variance 与离散静态无套利开始；随后逐步推导 VAE 的 ELBO、Gaussian KL、reparameterization 和 masked conditional inference。文献部分审计 2021–2026 年 14 篇核心工作的数据、网络、训练和复现条件，并讨论 constrained nonlinear dynamic factor models。

实验是在 1,800 张合成 SSVI 曲面上比较 PCA-8、masked MLP-VAE、ConvVAE 和 ConvVAE+NA 的**结构复现**，不是对专有 FX/SPX 数据或公开 crypto benchmark 数字的严格复刻。报告保留了 PCA 显著胜出、以及重建无套利改善而先验采样退化的负结果。

## 本次发布与原始会话 ZIP 的区别

本目录的 `results.json`、训练日志、补全案例和实验脚本均保留原始字节。合成数据由原 seed 确定性重建，并验证原始 SHA-256；图表只使用已保存结果重绘，**发布过程不训练模型，也不重新计算 benchmark 数字**。

`public_reproduction_bundle.zip` 是仓库公开包，**不是原始会话 ZIP 的逐字节副本**。它包含报告、源码、结果、日志、合成数据和图表，但不包含原始三个 `.pt` 模型权重。原始带 checkpoint 的 ZIP 仍保留在创建报告的会话下载中。报告附录 D 描述的是原始完整实验产物，不表示每种产物都已上传本仓库。

本次仅修正一处排版：第 8.4 节 PCA ridge 目标中，移除 `10^{-4}` 前的 stray tab/`iny` 残留；系数和实验结果不变。

## 编辑与渲染（不训练）

以 `manuscript/` 中的九个章节文件为 source of truth；`report.md` 与 `index.html` 是生成文件。

```bash
# Python 3.13.5；另安装 Pandoc 3.1.11.1
python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 matplotlib==3.10.8
python render.py
```

渲染流程会校验冻结文件、重绘图像、生成原生 MathML、检查公式和图片数量，并更新公开 ZIP。修改该目录会触发仓库的 render-only GitHub Actions 工作流。

## 重新运行实验

重新训练需要额外安装 PyTorch 和 scikit-learn。原始运行环境记录在 `reproduction/results.json`，其中 PyTorch 为 `2.10.0+cpu`。

```bash
python reproduce_vol_surface_vae.py --output-dir new_run
# 仅检查代码路径：
python reproduce_vol_surface_vae.py --quick --output-dir smoke_test
```

请把新结果写入新目录，不要覆盖已冻结的 `reproduction/`。不同平台、版本或训练配置不保证逐字节复现原始 checkpoints。
