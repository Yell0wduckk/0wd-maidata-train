# 冻结 Encoder/Drafter 后适配主 Decoder

入口：`python -m maidata.decoder_training.adapt`。本次按用户确认的 **主 Decoder ckpt45＋GT plan** 衔接，只增加源码、文档和 CPU 验证，不自动启动适配训练或续接正在进行的 Drafter 实验。

## 更新范围

| 部分 | 来源与行为 |
|---|---|
| 音频 Encoder | ckpt45 生产 P0 时保存的冻结快照；保持 eval、无梯度、不执行前向，不进入 optimizer |
| Drafter / Planner | 已完成的冻结 Drafter checkpoint 离线导出 Z/预测 plan；训练不构建或执行 D/P/R |
| Decoder | 从 ckpt45 严格加载已有 M 权重；全部 M 参数继续更新，包括音频投影、Z 投影、条件 attention、主干和字段头 |
| Z | 开启 `decoder_z_cross_attention`，输入宽度由新 D/P 缓存推导；默认逐拍 256 维 |
| plan | 默认沿用 GT plan 缓存，训练与验证一致；仍支持显式 `--plan-source predicted` |

ckpt45 的 Z 分支此前关闭，`condition.z.weight/bias` 是未训练参数。新入口明确重新初始化这两个参数；若从已有 Z 模型开始且宽度一致，则完整保留它们。其他 M 权重均严格加载，缺键、多键或尺寸不符直接报错，不使用隐式部分加载。共享的条件 attention、source embedding 与 plan 网络继续继承原 M。

首次适配是 **权重初始化**：使用 `--init-checkpoint`，新建 optimizer、scheduler，从适配第 1 轮开始；不继承旧 M 的 Adam 状态或第 45 轮游标。之后仅使用适配自己的完整轮 `--resume`，恢复 M、冻结 E、optimizer、scheduler、轮游标和 RNG；不得将原 joint checkpoint 作为 cached 模式的完整续训输入。

## P0 与 Z 的衔接

直接复用上一任务已由主 M ckpt45 生产的 P0：

```text
原 M: /root/autodl-tmp/decoder-cont-v2-no-z-gt-20261011/training/checkpoints/epoch-045.ckpt
P0:   /root/autodl-tmp/drafter-split-p0-m45-20261011/p0
D/R/P 完成权重（完成后才可使用）:
      /root/autodl-tmp/drafter-split-p0-m45-20261011/training/checkpoints/epoch-050.ckpt
```

这些是已确认的路径合同，不代表本文确认了远端文件当前已完成。P0 包含 `encoder.pt` 与生产者元数据，不重新计算音频。D/P exporter 现在识别 frozen-E Drafter checkpoint 的 `audio_producer`，与所选 P0 的已记录身份相配；不再要求“整个 Drafter checkpoint 等于生产 P0 的主 M checkpoint”。不重新哈希或扫描已验收的音频、P0 bank、原谱、Physical 或输入 catalog。

新增 Z 必须来自这套 P0 上训练的拆分 Drafter，不能使用旧 128 维缓存或由另一 Encoder 生产的 Z。训练仅比较已记录的生产者身份，正常按需加载。初始化时，M 的 Encoder 配置和权重须与 P0 内快照一致，避免将 ckpt45 的缓存与另一 Encoder 的权重混装在推理 checkpoint 中。

P0 是公开目标中心拼接的 **全曲 bank**，不是 ckpt45 原联训阶段各目标窗各自编码的 24 拍 bank。本入口明确记录 `decoder_train_audio_scope=full_song`，M 读取该曲全曲 P0；不将邻窗中心表示冒充原窗局部表示。后续推理按 checkpoint 使用相同全曲语义。Z 仍按原 M 前 4＋目标 16＋后 4 拍选择，完整空尾 Z/plan 有效；P0 音频 padding、不同歌曲 padding 保持无效。

## 使用方式

准备一份独立 `adaptation.json`，复用已验收的主 M 训练配置中的训练/验证计划、v5 窗口监督、GT plan、Physical/几何路径、loss 权重与其他配方。首次适配的学习率、scheduler horizon、绝对停止轮由运行配方显式设置；本次未指定新的训练轮数，不在入口内硬编码。

其中 `decoder_train_checkpoint` 用于离线导出，指向已完成的 **Drafter** checkpoint；`--init-checkpoint` 用于初始化 **M**，二者不能混用。其余新增配置均集中于 `entry.CONFIG`，原 `maidata.decoder_training` 仍默认 `joint`。

先导出新 Z/预测 plan（即便训练使用 GT plan，也从该 bank 取 Z）：

```sh
python -m maidata.decoder_training.dp_cache \
  --config adaptation.json \
  --p0-cache /root/autodl-tmp/drafter-split-p0-m45-20261011/p0 \
  --output NEW_DP_CACHE
```

缓存导出配置的 train/validation 计划应为后续 M 使用的计划；保持原 split 和全部基础窗/空尾窗，重复 occurrence 共用一份 Z/plan。导出默认 workers 12，CUDA BF16；CPU FP32 仅用于小型实现测试。

显式启动适配（示例命令，本次不执行）：

```sh
python -m maidata.decoder_training.adapt \
  --config adaptation.json \
  --init-checkpoint /root/autodl-tmp/decoder-cont-v2-no-z-gt-20261011/training/checkpoints/epoch-045.ckpt \
  --p0-cache /root/autodl-tmp/drafter-split-p0-m45-20261011/p0 \
  --dp-cache NEW_DP_CACHE \
  --plan-source gt --gt-plan-cache EXISTING_GT_PLAN_CACHE \
  --output NEW_ADAPTATION_OUTPUT
```

适配入口强制 `cached`、`full_song` 和 Z 开启，`--show-config` 不读取资产或 checkpoint。完整轮恢复时用 `--resume NEW_ADAPTATION_OUTPUT/checkpoints/last.ckpt` 替代 `--init-checkpoint`；原 ckpt45 路径可以离线，恢复不再打开它。P0/Z 缓存仍需可读。

模型保存真实冻结 E 和更新后的 M，因此完成后的 v4 checkpoint 可由现有 `maidata.inference` 加载，配上对应 Drafter 权重即可；推理不依赖训练时的 P0 路径。

## 保留的训练合同与验证边界

不更改已保存的计划、occurrence、history dropout、监督、loss 分母、row budget 或逻辑 batch 边界。缓存模式仅替换音频输入来源；缓存按 audio_variant 去重，多 M 分块仍正确累积音频投影梯度。workers 默认 12，checkpoint 只按完整轮保存，validation 每 5 轮及末轮，完整诊断只在末轮。

70 项受影响 CPU 回归通过，覆盖权重继承与 Z 投影初始化、冻结 E 无前向/无更新、M 与 Z 投影实际梯度、BF16 缓存接 FP32 M、分块梯度对照、GT 条件/occurrence 对齐、P0 到新 D/P 导出、完整轮恢复一致性及正常推理加载。CUDA、正式适配训练与谱面质量尚未验证；不设额外准确率、loss 或性能阈值。
