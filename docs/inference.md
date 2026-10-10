# 当前一拍草稿 A 的自由推理

本文描述复制源码的 native 一拍草稿路线。计划中的 24 拍局部流、全曲 G 和共享 Conformer 见 [新音频 encoder 设计](../../0wd-v3-refactor-docs/new-audio-encoder.md)，尚未成为当前推理实现。

一拍草稿 A 的 checkpoint 内嵌草稿生成器；当前默认条件来自该生成器，不是旧 B 的四拍自回归 latent。native 路线需要匹配的外部教师音频缓存、song id、时间信息及实际启用约束所需资产。缺少 native 缓存时不会自动在线编码波形；新歌曲须先通过绑定教师的外部生产路径取得缓存。输入检查见 [inference/api.py](../src/maimai_baseline/inference/api.py)。

## 计算与复用

1. 读取 native22 的 P0/C0/C1、有效位、绝对拍坐标和几何；每窗分别为 176/528/264 个 256 维位置。
2. 将 P0 中心 16 拍拼接为全曲 bank；A 的 decoder 使用全曲音频 memory，局部路由使用所属窗 P0，C0/C1 由当前 A 投影为时间候选。
3. 根据音频范围一次生成全曲每拍一个 128 维草稿 token，并按目标窗提取前、中、后三段共 48 个条件槽，边界使用 mask。推理使用干净草稿，不加训练噪声。
4. A 以已生成的完整因果谱面历史预测后续 MOMENT 时间及 NOTE 字段，复用已经计算的音频、草稿和可用的历史状态。

流式生成及局部回退复用同一份全曲草稿；不会在每个候选或回退步骤重新生成。实现见 [generation.py](../src/maimai_baseline/inference/generation.py) 与 [style.py](../src/maimai_baseline/inference/style.py)。当前全曲 P0 memory 不应写成未来设计的“decoder 仅看 24 拍”。

生产者、预处理和输出合同必须匹配 checkpoint 绑定；缺失歌曲、variant 或缓存结构错误按实际 loader 报错处理。不能通过改写 manifest 把另一教师的缓存冒充已有绑定。有关结构和来源的边界见 [runtime-versioning](runtime-versioning.md)。

## 输出与约束

当前详细 trace 默认关闭，不传 --trace；必要错误、截断和 Physical bypass 记录仍保留。普通与流式路径都必须遵守所选 checkpoint 的条件和约束语义。

CLI 的 --physical-hard-mask 与 --miacode-slide-check 默认开启；启用硬约束需提供匹配的 Physical path catalog，Slide 检查依赖硬约束。实际生成应按本次参数、绑定资产和输出记录确认约束是否应用。实现与规则见 [Physical 文档](physical.md)。退出成功不自动说明输出未截断或没有 bypass；谱面质量仍由实际自由生成结果判断。

历史 B、四拍 latent 及其他推理辅助入口的存在不使其成为当前一拍路线的默认步骤。新 encoder 的在线编码方式、草稿改接和历史权重兼容性尚待实现与确认。当前数据与训练参考见 [audio-draft](audio-draft.md)、[data-and-tools](data-and-tools.md) 和 [luna-training](luna-training.md)。
