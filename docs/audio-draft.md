# 一拍 audio draft：复制源码的实现参考

本文说明本副本继承的一拍草稿实现，不代表新 encoder 已实现，也不指定新实验的教师权重、数据和训练轮数。入口说明见 [README](../README.md) 与 [current-runtime](current-runtime.md)；未来结构见 [新音频 encoder 设计](../../0wd-v3-refactor-docs/new-audio-encoder.md)。

## 当前音频输入

native-audio-draft128-beat-v1 使用外部教师导出的冻结 native22 缓存：前 4 拍、目标 16 拍、后 2 拍，共 22 拍；保存 FP32 eval 输出、有效位、绝对拍坐标和插值几何。每窗宽度均为 256：

| 尺度 | 每窗形状 | 当前用途 |
| --- | --- | --- |
| P0 | 176 × 256 | 局部路由；中心 16 拍拼接全曲 bank |
| C0 | 528 × 256 | A 的音频时间候选 |
| C1 | 264 × 256 | A 的另一尺度音频时间候选 |

P0 为每拍 8 个位置；C0/C1 的名义分辨率分别为每拍 24/12 个位置，实际采样位置以缓存坐标和几何为准，不能把它们当作固定毫秒间隔。全曲 P0 bank 由各窗 P0[32:160] 的中心 16 拍拼接，连同有效位和拍坐标使用。草稿读取全曲 bank；A 的 decoder 音频交叉注意力也使用全曲 bank，局部路由读取所属窗的 P0。C0/C1 的候选投影由当前 A 执行并训练，缓存本身保持冻结。

当前训练和 native 推理读取这些缓存，不在每个 step 在线重跑外部教师。缓存读取合同见 [teacher_audio.py](../src/maimai_baseline/data/teacher_audio.py)。计划中的 24 拍局部流、全曲 G 与共享 Conformer 尚未实现；本节的 22 拍不能替换成设计中的 24 拍。

## 草稿与 A 条件

当前生成器为宽度 256、8 heads、8 blocks、FFN 1024 的非自回归网络：拍查询进行双向 self-attention，并 cross-attend 全曲 P0。输出每拍一个 128 维 token，查询锚点为 j + 0.5，末端使用无可训练仿射参数的 LayerNorm。条件包括真实难度、连续 level 和音频范围；输入不含真值谱面，也不借真值谱面长度确定查询范围。

每个目标 16 拍窗口取前一段、当前段、后一段共 48 个条件槽；边界不足部分由 mask 处理。训练草稿时，固定容量的谱面行查询仅由草稿 latent 重建谱面，另有目标 16 拍的 MOMENT 数量监督，每拍类别为 0 至 96。目标是字段 CE、时间 SmoothL1、数量 CE 三项分别求均值后相加；CE 在 FP32 中计算并使用 0.05 label smoothing。MOMENT 时间使用窗内绝对拍位置，持续时长使用 log1p 表示。这是草稿预训练目标，区别于 A 的 L3 与 Physical 目标。实现见 [audio_draft.py](../src/maimai_baseline/model/audio_draft.py)。

A 训练读取指定草稿 checkpoint 生成的干净缓存，草稿生成器及原音频模块冻结。identity/mirror 共用 canonical 草稿 token，另加可训练的 view embedding。训练时每个窗口以 50% 概率加 0.05 × draft_condition_std 的高斯噪声；一次条件扰动供路由和各 decoder 条件残差复用。标准差来自 A 训练条件，推理使用干净条件。实现见 [baseline.py](../src/maimai_baseline/model/baseline.py)。

## 两种验证数据模式

通用准备入口默认选择独立的 32 首验证歌曲，使用验证来源 manifest，并排除实际草稿训练歌曲及音频别名；抽样 seed 为 20260811。

源码另支持从已验收的 full complete-tail 训练输入留出 64 首，必须显式选择：

    prepare-audio-draft-data ... --validation-from-training --validation-song-count 64

该模式只从草稿训练计划移除留出歌曲，保留剩余 occurrence、顺序及原 batch 边界，空 batch 删除；验证使用留出歌曲已接受的完整窗口。A 计划仍保留这些歌曲，因此这是草稿验证集，不是 A 的留出验证集；从完整 A 训练计划计算的条件噪声统计也会包括它们。默认独立 32 首与显式 full 64 首是两种模式，不能混称。实现见 [audio_draft 数据准备](../src/maimai_baseline/data/audio_draft.py) 与 [draft_validation.py](../src/maimai_baseline/data/draft_validation.py)。

## 入口与后续边界

- prepare-audio-draft-data：从给定输入准备 canonical 草稿计划及原视图 A 计划。
- train-audio-draft：训练草稿生成器和重建头。
- build-draft-cache：用明确指定的草稿 checkpoint，以 FP32 eval 生成干净 token 及 A 训练条件统计。
- train --stage A：读取缓存，训练 A 的 L3 与 Physical 路线。

历史四拍草稿、chart encoder、mu/GMM、B/SFT 入口仍可见于复制源码；入口存在不表示已适配当前一拍路线或未来新 encoder。新 encoder 的重新训练、草稿改接、历史权重复用和新训练配方仍待确定。复制代码不构成这些工作的实现或训练结果。

训练恢复见 [luna-training](luna-training.md)，自由生成见 [inference](inference.md)，资产边界见 [data-and-tools](data-and-tools.md)。
