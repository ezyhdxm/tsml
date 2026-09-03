# Vol Surface × VAE：可编辑源文件

[阅读 HTML](index.html) · [完整 Markdown](report.md) · [目录与复现说明](README.md)

以 `manuscript/` 内的分章 Markdown 为 source of truth。`report.md` 和 `index.html` 由 `render.py` 生成，不应直接编辑。

## 章节

1. [曲面与静态无套利基础（第 0–3 节）](manuscript/00-surface-foundations.md)
2. [VAE 概率模型与逐步推导（第 4 节）](manuscript/01-vae-foundations.md)
3. [文献地图与精读：第一部分](manuscript/02a-literature.md)
4. [文献精读：第二部分](manuscript/02b-literature.md)
5. [文献精读：条件生成、ConvVAE、flow、diffusion 与 latent geometry](manuscript/02c-literature.md)
6. [方法比较与复现性审计（第 6–7 节）](manuscript/03-comparison-and-reproducibility.md)
7. [实际执行的合成 SSVI 实验（第 8 节）](manuscript/04-experiment.md)
8. [实现教程与研究方向（第 9–10 节）](manuscript/05-implementation-and-research.md)
9. [参考文献与数学附录](manuscript/06-references-and-appendices.md)

## 渲染

在本目录运行 `python render.py`。依赖和版本见 [README](README.md)。渲染只使用已保存实验结果，不重新训练模型。推送本目录变更时，render-only 工作流会校验公式、内嵌图像和冻结数据，然后更新 HTML 与公开复现包。

原始三个模型 checkpoint 未包含在本次公开包中；完整范围与原始会话 ZIP 的区别见 README。
