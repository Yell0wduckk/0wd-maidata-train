# 主 Decoder 阶段二训练

入口为 `python -m maidata.decoder_training`。本阶段加载明确指定的阶段一 E/D/P 配套 checkpoint，固定生产链，只更新 M 及 M 自己的音频 K/V、Z/plan 接入层、小头。阶段一 `maidata.training` 保持独立；R 不构建、不运行，也不计算 R/P 辅助监督。可选联合微调、整曲自动生成与离线隐藏表示缓存不属于此入口。

## 模块与数据流

| 模块 | 职责 |
|---|---|
| `decoder_training.plan` | 读取既有原谱 Decoder 采样计划、split、逐轮 occurrence 顺序及完整音频组 |
| `decoder_training.data` | 读取 Mel/aux32、字段与 Physical；构造 GT 历史、边界、目标及整曲输入 |
| `decoder_training.support_cache` | 恢复真实窗前动作与首次归责账本；复用等价前缀支持，重算边界与终端 |
| `decoder_training.network` | 组合固定 E/D/P 与独立 M；全曲 P0、Z、预测 plan12 的公开张量传递 |
| `decoder_training.batching` | 按实际 M 输入长度连续分块，不改变 occurrence 顺序或裁剪历史 |
| `decoder_training.supervision` | CPU 整理支持和分母、批量小头计算、三组独立 loss 归约 |
| `decoder_training.recipe/checkpoint/runner` | 配方、配套生产者身份、严格完整轮恢复、训练及验证 |

每个完整音频组只运行一次 E，唯一中心 P0 拼为全曲 bank；同一音频的不同难度/等级条件分别生成 Z。P 只读取目标窗 Z；重复 occurrence 共享对应预测 plan12。M 读取全曲 P0、默认前4＋目标16＋后4拍的 Z，以及无位置编码的预测 plan12；不读取 GT Planner 或直接难度值。

E/D/P 始终 `eval`、`requires_grad=False`，生产表示使用 `no_grad`。M 的投影在其外正常反传。全曲音频 K/V 在一个输入 batch 内共享，直到该 batch 的最后分块完成反传；optimizer 更新之后重新建立，禁止跨更新复用。每个窗口并行 teacher forcing，重新计算包括历史在内的全部 self-attention，不跨窗携带 self K/V。

## 原谱、历史和 Physical

消费已保存的主 Decoder 计划，训练侧保留基础窗口和复杂节奏的附加 occurrence，验证侧仅基础窗口；目标默认16拍、stride16拍，保留完整空尾窗。不会重新划分数据、增建镜像或按 Physical 违规删掉 GT。

历史为完整起音组的4拍 GT 前缀，不设 token 裁剪预算；首组保留原始 delta 和绝对位置。边界间隔使用全曲最后真实起音，即使它在4拍之外。训练历史 dropout 默认0.15，由 `(seed, epoch, window_id, occurrence)` 决定，与 DataLoader worker 分配无关；dropout 只移除历史输入并注入 M(0)，验证不启用。终端使用名义音频上下文窗尾 U，保留精确窗外和尾部删失标签，不把 U 改为全曲 audio_end。

字段和训练 Physical 均为原谱 v2，Physical 通过原有 source SHA 绑定字段生产者。正常读取缓存恢复动作、pending 语法和具体冲突账本；窗前仍有效的长 hold/slide 不受4拍历史裁剪或 dropout 影响。当前窗控制记录与重复 occurrence 不重复提交历史动作。只有语法前缀、动作及归责账本等价时才复用该行缓存，直到重新等价之前按在线规则查询；终端另行查询。完整源谱支持不会每轮重新生成。

窗尾是 Physical 时间支持的显式区间分界，避免用关闭窗口后的代表点判定窗内候选。该修复仅影响消费窗口：离线完整原谱没有窗口边界，其字段、动作、候选缓存格式与语义不变，无需重建已验收缓存。负拍音频网格早于源谱 M(0) 时，窗口可生成源序列中没有的首个正 delta MOMENT。

## Batch 与损失

`decoder_train_audio_groups_per_batch` 指每个输入 batch 的完整音频组数；`decoder_train_gradient_accumulation_steps` 个输入 batch 组成一次 optimizer update。组及 occurrence 按已保存的 seed/epoch 规则排列，不回填、不丢弃、不重复扩增。

`decoder_train_token_budget` 控制 M 的连续窗口分块，计费为 `窗口数 × 本块最长实际输入 token 数`，包含历史、边界及重复实例的 padding；不使用 R 的固定207行。单窗口超过预算时作为完整单块保留并在成本报告记录，预算不是历史截断上限。冻结 E/D/P 仍处理完整音频组，因此该数值也不是全模型显存的硬上限。

先在 CPU 为整个逻辑 batch 汇总分母，再逐块反传：

- 主监督：每个有效预测决策 NLL 的总和／总数；唯一合法分类、NA、历史目标、padding 不计，终端时间计数。
- Physical：Router 类型与内容各计单位；NOTE 内部 kind/pad 固定0.5/0.5，有效零贡献单位仍在共同分母中。四类最后按 `(1,1,4,1)/4` 归约，flags 不连接 Physical 梯度。
- 两项几何：分别按有效 ROUTE 总数归约，独立于主监督和 Physical 分母。

批量小头实现与 `decoder.supervision.decoder_loss` 逐行参考比较总和、计数和梯度。分块及梯度累积不采用窗口均值的平均。

## 配方与执行

唯一配置位于 `entry.CONFIG`，所有 `decoder_train_*` 均有实际消费者。尚未确定的训练数值为 `None`，运行前必须显式提供，不能回退到阶段一数值：

- 阶段一 checkpoint；学习率、weight decay、gradient clip、warmup update 数；scheduler 总轮数与绝对停止轮。
- 精度 `fp32` 或 CUDA `bf16`；每批完整音频组数、梯度累积数、M token 分块预算。
- Physical、judge、trajectory 三个辅助系数；已有 Route-71 距离矩阵路径及独立输出目录。

默认 AdamW，workers12，seed 与现有计划一致。所有路径可通过 JSON 显式改为目标主机路径；本地默认路径只定位既有资产，并不表示所有远端载荷已下载。本次 pilot320 Physical 本地目录只有生产摘要等元数据；实际训练必须指向含 `catalog.json` 和 `charts/...` 的既有完整产物目录，不能用摘要冒充载荷。

```powershell
python -m maidata.decoder_training --show-config
python -m maidata.decoder_training --config decoder-stage2.json
python -m maidata.decoder_training --config decoder-stage2.json --resume OUTPUT/checkpoints/last.ckpt
```

原 batch CUDA 检查是显式命令，在独立临时模型上执行第1轮指定的完整逻辑 update，保持该配方的分组、occurrence、累积与精度，不保存训练 checkpoint：

```powershell
python -m maidata.decoder_training --config decoder-stage2.json --cuda-check --check-step 0
```

报告含冻结/更新范围、E/D/P 调用次数、各块实际 padding token 成本、耗时和显存。`check-step` 为零基索引。该入口不启动后续训练，也不设置性能或质量通过阈值。

## Checkpoint 与验证

阶段二使用独立 schema `maidata.decoder-training.v1`。初始化参数是阶段一配套 checkpoint，`--resume` 只接阶段二完整轮 checkpoint，两者不能混用。保存 M、optimizer、scheduler、完整轮/step、所有 RNG、DataLoader generator、生产者身份、配置和输入绑定；E/D/P 权重由原始绑定 checkpoint 严格加载，不复制成新的生产者。

初始化核对 E/D/P 架构、张量 shape/dtype/有限值、字段及音频归一化绑定，原有生产者文件级 source guard 存在时保留。阶段二恢复核对自身源码、配方、计划、字段/Physical 生产者、几何和指定上游身份。只允许延长绝对停止轮且不超过原 scheduler horizon；不重新 warmup 或重新抽取已完成轮。身份检查读取必要元数据和正常消费资产，不遍历原音频、旧特征或全量缓存载荷重新验收。

每5轮及末轮 validation，重合只执行一次；完整诊断仅末轮。`last` 每个完整轮保存，`best` 只在轮末验证改善时更新，编号 checkpoint 每5轮及末轮保存；异常和轮内不保存 checkpoint。验证 loss 用于诊断及 checkpoint 记录，谱面质量仍由实际生成结果评估。

CPU 验证覆盖原谱缓存读写、在线规则对照、拼接和并行轨道、负拍起点、旧长动作、历史 dropout、空尾窗、批量/逐行 loss 与梯度、完整组分块等价、固定 E/D/P、真实 M 更新及中断后逐位一致恢复。正式训练和 CUDA 实测需在确定配方与配套 checkpoint 后执行；本次实现未启动它们。
