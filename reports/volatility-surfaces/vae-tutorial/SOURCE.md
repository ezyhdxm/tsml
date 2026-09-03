# Vol Surface × VAE：可编辑源文件

[阅读 HTML](index.html) · [完整 Markdown](report.md) · [目录与复现说明](README.md)

以 `manuscript/` 内的分章 Markdown 为 source of truth。`report.md` 和 `index.html` 由 `render.py` 生成，不应直接编辑。

## 新增导读

对“为什么需要 surface，而不是直接修 BS 模型？”有疑问，先读第 3A 节，再读第 3B 节；它们位于 VAE 概率推导之前。

## 章节

1. [曲面与静态无套利基础（第 0–3 节）](manuscript/00-surface-foundations.md)
2. [为什么需要 surface：报价、分布与动态模型（第 3A 节）](manuscript/00a-why-surface.md)
3. [经典方法与 VAE 的增量收益（第 3B 节）](manuscript/00b-classical-methods.md)
4. [VAE 概率模型与逐步推导（第 4 节）](manuscript/01-vae-foundations.md)
5. [文献地图与精读：第一部分](manuscript/02a-literature.md)
6. [文献精读：第二部分](manuscript/02b-literature.md)
7. [文献精读：条件生成、ConvVAE、flow、diffusion 与 latent geometry](manuscript/02c-literature.md)
8. [方法比较与复现性审计（第 6–7 节）](manuscript/03-comparison-and-reproducibility.md)
9. [实际执行的合成 SSVI 实验（第 8 节）](manuscript/04-experiment.md)
10. [实现教程与研究方向（第 9–10 节）](manuscript/05-implementation-and-research.md)
11. [参考文献与数学附录](manuscript/06-references-and-appendices.md)

## 渲染与示例

在本目录运行 `python render.py`。依赖和版本见 [README](README.md)。渲染使用已保存 VAE 实验结果，并运行不涉及训练的新增数学示例；不会重新训练模型。推送本目录变更时，render-only 工作流校验公式、内嵌图像和冻结数据，然后更新 HTML 与公开复现包。

单独运行新增示例：`python classical_demo.py --output-dir classical_examples`。修订说明见 [REVISION.md](REVISION.md)。

原始三个模型 checkpoint 未包含在公开包中；与原始会话 ZIP 的区别见 README。
