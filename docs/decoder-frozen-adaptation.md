# 冻结 Encoder/Drafter 后适配主 Decoder

入口：`python -m maidata.decoder_training.adapt`。按用户确认的 **主 Decoder ckpt45＋GT plan** 衔接；2026-10-11 用户要求保持原采样与音频窗口，授权微调 **20 轮停止／50 轮调度、LR 1e-4、warmup 200**。实际实验交给原训练会话执行。

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

直接复用上一任务已由主 M ckpt45 生产的全曲 P0 供 D/P 导出及 Encoder 快照加载；M 另读精确局部 P0：

```text
原 M: /root/autodl-tmp/decoder-cont-v2-no-z-gt-20261011/training/checkpoints/epoch-045.ckpt
P0:   /root/autodl-tmp/drafter-split-p0-m45-20261011/p0
D/R/P 完成权重（完成后才可使用）:
      /root/autodl-tmp/drafter-split-p0-m45-20261011/training/checkpoints/epoch-050.ckpt
```

这些是已确认的路径合同，运行会话负责核对当前文件状态。全曲 P0 包含 `encoder.pt` 与生产者元数据，D/P 直接复用它。D/P exporter 识别 frozen-E Drafter checkpoint 的 `audio_producer`，与所选 P0 的已记录身份相配；不要求“整个 Drafter checkpoint 等于生产 P0 的主 M checkpoint”。不重新哈希或扫描已验收的音频、P0 bank、原谱、Physical 或输入 catalog。

新增 Z 必须来自这套 P0 上训练的拆分 Drafter，不能使用旧 128 维缓存或由另一 Encoder 生产的 Z。训练仅比较已记录的生产者身份，正常按需加载。初始化时，M 的 Encoder 配置和权重须与 P0 内快照一致，避免将 ckpt45 的缓存与另一 Encoder 的权重混装在推理 checkpoint 中。

M 使用 `decoder_train_local_p0_cache`，表示每个原目标起点分别编码的 **前 4＋目标 16＋后 4 拍** P0，包含完整空尾窗。生产入口复用冻结 Encoder 快照和既有 Mel/aux32，经原 `collate_online_audio` 构造输入；只补产此前未保存的局部神经特征，不重建 Mel、aux32、Physical、谱面或原全曲 P0。不从全曲中心 bank 裁剪或拼接来近似原窗表示。

适配与推理均保持 `decoder_train_audio_scope=window`，音频 attention 每窗只读取对应局部 bank；重复 occurrence 和不同难度的同起点窗口共享该 bank。Z 同样沿用前 4＋目标 16＋后 4 拍，完整空尾 Z/plan 有效，范围外 mask 不变。此前 `c179d5a` 的全曲 M 输入方案已被本次修正覆盖，不用于正式实验。

## 使用方式

正式配置为 [`configs/decoder-adapt-m45-z-linux.json`](../configs/decoder-adapt-m45-z-linux.json)，复用原主 M 的训练/验证计划、v5 窗口监督、GT plan、Physical/几何路径与 loss 权重。微调从自身第 1 轮开始，完整轮编号为 1…20，调度 horizon 50，LR 1e-4、warmup 200；不把旧第 45 轮游标当成本轮新调度的进度。

其中 `decoder_train_checkpoint` 用于离线导出，指向已完成的 **Drafter** checkpoint；`--init-checkpoint` 用于初始化 **M**，二者不能混用。其余新增配置均集中于 `entry.CONFIG`，原 `maidata.decoder_training` 仍默认 `joint`。

设置独立实验根 `/root/autodl-tmp/decoder-adapt-m45-z-20261011`，其中 `source` 为本次提交源码，`local-p0` 为局部音频缓存，`dp` 为冻结 Drafter 输出，`training` 为适配输出。先补产局部 P0，再导出新 Z/预测 plan（即便训练使用 GT plan，也从该 bank 取 Z）：

```sh
python -m maidata.decoder_training.local_p0_cache \
  --config configs/decoder-adapt-m45-z-linux.json \
  --output /root/autodl-tmp/decoder-adapt-m45-z-20261011/local-p0

python -m maidata.decoder_training.dp_cache \
  --config configs/decoder-adapt-m45-z-linux.json \
  --p0-cache /root/autodl-tmp/drafter-split-p0-m45-20261011/p0 \
  --output /root/autodl-tmp/decoder-adapt-m45-z-20261011/dp
```

缓存导出配置的 train/validation 计划应为后续 M 使用的计划；保持原 split 和全部基础窗/空尾窗，重复 occurrence 共用一份 Z/plan。导出默认 workers 12，CUDA BF16；CPU FP32 仅用于小型实现测试。

受影响的原 batch CUDA 检查通过后，由训练会话启动适配；检查只验证此改动的冻结/梯度/局部音频与 batch 成本，不追加旧资产全量扫描。命令为：

```sh
python -m maidata.decoder_training.adapt \
  --config configs/decoder-adapt-m45-z-linux.json \
  --init-checkpoint /root/autodl-tmp/decoder-cont-v2-no-z-gt-20261011/training/checkpoints/epoch-045.ckpt \
  --p0-cache /root/autodl-tmp/drafter-split-p0-m45-20261011/p0 \
  --local-p0-cache /root/autodl-tmp/decoder-adapt-m45-z-20261011/local-p0 \
  --dp-cache /root/autodl-tmp/decoder-adapt-m45-z-20261011/dp
```

适配入口强制 `cached`、`window` 和 Z 开启，GT plan/output 由配置指定。`--show-config` 不读取资产或 checkpoint。完整轮恢复时用 `--resume TRAINING_OUTPUT/checkpoints/last.ckpt` 替代 `--init-checkpoint`；原 ckpt45 路径可以离线，恢复不再打开它。缓存和快照仍需可读，已成功生产的缓存不因训练重启而重复生产。

模型保存真实冻结 E 和更新后的 M，因此完成后的 v4 checkpoint 可由现有 `maidata.inference` 加载，配上对应 Drafter 权重即可；推理不依赖训练时的 P0 路径。

## 保留的训练合同与验证边界

不更改已保存的计划、occurrence、history dropout、监督、loss 分母、row budget 或逻辑 batch 边界。缓存模式仅替换音频输入来源；缓存按 audio_variant 去重，多 M 分块仍正确累积音频投影梯度。workers 默认 12，checkpoint 只按完整轮保存，validation 每 5 轮及末轮，完整诊断只在末轮。

本次 66 项受影响 CPU 回归通过，覆盖原在线 Encoder 与局部缓存 values/beats/mask 一致、空尾/负拍/不同歌曲/重复窗口归属、原采样顺序及 batch 边界、权重继承与 Z 初始化、冻结 E 无前向/无更新、M/Z 梯度、BF16 缓存接 FP32 M、完整轮恢复及正常推理加载。CUDA、正式训练启动及完成证据由训练会话报告；不设额外准确率、loss 或性能阈值。
