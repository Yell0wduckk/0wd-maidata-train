# 主 Decoder 阶段二训练

入口为 `python -m maidata.decoder_training`。本阶段读取明确指定的阶段一 checkpoint 所产出的离线全曲 P0、全曲 Z 和预测 plan12 缓存，**训练与验证只构建 M，不构建或运行 E/D/P/R**。仅更新 M 自己的音频 K/V、Z/plan 接入层、主干及小头。阶段一 `maidata.training`、离线导出和可选联合微调分别独立；缺失或错配缓存直接报错，不回退在线生成。

## 模块与数据流

| 模块 | 职责 |
|---|---|
| `decoder_training.plan/ordering` | 保留既有实例集合与 split，逐轮全局打乱窗口，从窗口缓存取得实际输入长度并按 row budget 组批 |
| `decoder_training.frozen_cache/conditioning` | 读取已完成的 P0/Z/plan 缓存及身份，按窗口选取公开张量，不执行网络 |
| `decoder_training.data` | 按 batch 去重读取缓存张量，选择 GT 历史视图并恢复随机窗口顺序 |
| `decoder_training.preparation/transport` | worker 内整理去重映射、padding 和分块；按 dtype 打包 IPC、pin memory 与分块 H2D |
| `decoder_training.export_windows/support_cache` | 离线恢复真实动作与首次归责账本，准备窗首、窗内和终端支持 |
| `decoder_training.window_cache` | 读取已打包的输入与监督张量，执行历史 dropout、padding 和行索引平移 |
| `decoder_training.network` | 只持有 M；接收已选取的公开缓存张量，执行可训练投影 |
| `decoder_training.batching` | 按实际 M 输入长度连续分块，不改变 occurrence 顺序或裁剪历史 |
| `decoder_training.supervision` | 离线整理支持与计数；训练执行批量小头及三组独立 loss 归约 |
| `decoder_training.recipe/checkpoint/runner` | 配方、配套生产者身份、完整轮状态恢复、训练及验证 |

每个输入 batch 内，相同 audio_variant 共享一份离线全曲 P0 bank；D/P 缓存保留按原难度/等级生产的全曲 Z，以及每个基础窗和空尾窗的预测 plan12。通过 chart_to_group、window_id、起点和音频标识对应选择，同 batch 的重复 occurrence 共享同一份条件。M 读取全曲 P0、默认前4＋目标16＋后4拍的 Z，以及无位置编码的预测 plan12；不读取 GT Planner 或直接难度值。窗口随机采样改变监督的组批方式，全曲音频和 Draft 条件范围保持原定义。

离线导出时 E/D/P 使用 `eval/no_grad`，缓存张量不带梯度。训练时 M 的投影正常反传，不能缓存其投影结果跨参数更新使用。全曲音频 K/V 在一个输入 batch 内按歌曲共享，attention 将当前块的有效 query 按歌曲分组，读取同一份 K/V，并裁去歌曲 bank 两端的无效 padding；结果恢复原窗口顺序。K/V 保留计算图直到该 batch 的最后分块完成反传；optimizer 更新之后重新建立。每个窗口并行 teacher forcing，重新计算包括历史在内的全部 self-attention，不跨窗携带 self K/V。

缓存路径由 `decoder_train_p0_cache`、`decoder_train_dp_cache` 显式提供，沿用 [P0 导出](frozen-p0-cache.md)和 [D/P 导出](frozen-dp-cache.md)格式。启动读取 manifest 索引和原生产者元数据，不重新校验身份、架构、归一化或计划，不读取或哈希生产者 checkpoint。配套输入由用户和训练会话选定。训练读取 bank 时不重复检查 shape、dtype、有限值、拍坐标、mask 或归属，也不扫描和重哈希载荷。P0/Z 保持原始绝对位置，完整空尾 Z/plan 有效；不同歌曲的 batch padding 与 Z 上下文范围外的槽无效。

## 原谱、历史和 Physical

消费已保存的主 Decoder 计划，训练侧保留基础窗口和复杂节奏的附加 occurrence，验证侧仅基础窗口；目标默认16拍、stride16拍，保留完整空尾窗。不会重新划分数据、增建镜像或按 Physical 违规删掉 GT。

历史为完整起音组的4拍 GT 前缀，不设 token 裁剪预算；首组保留原始 delta 和绝对位置。边界间隔使用全曲最后真实起音，即使它在4拍之外。训练历史 dropout 默认0.15，由 `(seed, epoch, window_id, occurrence)` 决定，与 DataLoader worker 分配无关；dropout 只移除历史输入并注入 M(0)，验证不启用。终端使用名义音频上下文窗尾 U，保留精确窗外和尾部删失标签，不把 U 改为全曲 audio_end。

字段和训练 Physical 均为原谱 v2。独立离线导出器复用已有完整谱面支持，恢复真实动作、pending 语法及冲突账本，完成窗口边界适配和监督张量化。窗前仍有效的长 hold/slide 不受4拍历史裁剪或 dropout 影响。训练 DataLoader 只读所选窗口的张量，不读取整谱 JSON、不解压字段或 Physical、不重放语法、不查询 Physical、不生成时间候选。历史保留和丢弃共享同一份监督，只改变输入和监督行位置。

窗尾是 Physical 时间支持的显式区间分界，避免用关闭窗口后的代表点判定窗内候选。该修复仅影响消费窗口：离线完整原谱没有窗口边界，其字段、动作、候选缓存格式与语义不变，无需重建已验收缓存。负拍音频网格早于源谱 M(0) 时，窗口可生成源序列中没有的首个正 delta MOMENT。

## Batch 与损失

训练每轮保留既有全部 occurrence（包括复杂节奏追加实例），按 seed/epoch 对窗口实例全局打乱，然后沿此顺序连续装入 batch。同曲窗口可以分散到不同 batch，不再以完整歌曲组约束主 Decoder 的监督采样；验证保持既有固定顺序和基础窗口集合。既有计划中的原始组顺序保留实例来源，消费端不改写计划文件，也不重新划分数据或增加倍率。组批不回填、不丢弃，读取时临时聚合同曲，collate 后按 sequence_index 恢复随机顺序。

`decoder_train_row_budget` 默认 **8192**，只约束 `窗口实例数 × 最长实际 M 输入长度`，不把全曲 P0 计入阶段二预算。长度包括保留的4拍历史、注入边界、当前窗内容和 batch padding，按本 occurrence 的确定性历史 dropout 计算；终端只作目标，不另占输入位置。目标和历史长度直接读取窗口缓存 manifest，不在训练启动或每轮读取字段载荷计算。音频张量规模仍单独记录，联合音频＋谱面预算留到可能的第三阶段。

`decoder_train_gradient_accumulation_steps` 默认 **1**，每个 batch 反传后立即更新一次。`decoder_train_chunk_token_budget` 默认 **8192**，只控制同一更新内部的连续计算分块；所有块共享逻辑 batch 分母。单窗口超预算时保留完整历史和目标，并记录实际规模；不裁剪、不丢弃、不重排。

先在 CPU 为整个逻辑 batch 汇总分母，再逐块反传：

- 主监督：每个有效预测决策 NLL 的总和／总数；唯一合法分类、NA、历史目标、padding 不计，终端时间计数。
- Physical：Router 类型与内容各计单位；NOTE 内部 kind/pad 固定0.5/0.5，有效零贡献单位仍在共同分母中。四类最后按 `(1,1,4,1)/4` 归约，flags 不连接 Physical 梯度。
- 两项几何：分别按有效 ROUTE 总数归约，独立于主监督和 Physical 分母。

批量小头实现与 `decoder.supervision.decoder_loss` 逐行参考比较总和、计数和梯度。分块及梯度累积不采用窗口均值的平均。

## 计算复用与传输

通过独立命令准备当前选择范围的窗口张量：

```sh
python -m maidata.decoder_training.export_windows --config decoder-stage2.json --output WINDOW_CACHE --workers 12
```

将 `decoder_train_window_cache` 指向输出目录。该步骤复用已有字段、Physical，不重建源缓存或 P0/Z/plan；只处理计划中的基础窗和空尾窗，重复 occurrence 共用一份产物。每个窗口保存不压缩的 dtype 连续张量缓冲及轻量布局，新增产物在离线生成时检查一次落盘往返。历史长度和目标长度保存在 manifest。

DataLoader worker 只读取所选窗口输入与监督张量，以及所需 P0/Z/plan；按 seed/epoch/occurrence 选择历史视图、平移标签索引、padding，再按 dtype 打包。标签、合法 mask、Physical 罚项及其计数已离线确定。训练进程不做整谱解压、校验、规则重放或候选生成；不扫描输入张量的形状、有限值及归属。模型投影和概率运算的重复输入检查在训练/validation 调用中关闭；合法性 mask、概率归一化与梯度裁剪是实际计算，继续保留。

IPC 使用现有 `training.transport` 的 dtype 连续缓冲实现，整曲条件与每个 M 分块分别打包；只传轻量 occurrence 元数据，不传规则状态、TimingMap 或整棵 GT/支持对象。整曲条件传入设备一次，token/支持随当前分块传输和释放，避免把全曲所有候选同时放入显存。模型只消费公开张量，打包形式不进入网络接口。

NOTE 输入、MMoE gate、kind→pad/flags 使用固定形状张量运算，避免 CUDA 布尔索引产生动态 `nonzero`；flags 的条件适配每个 NOTE 仅计算一次。常规 loss 统计在设备累积，输出 update/epoch 日志时统一回读；不逐步执行额外有限值断言。

| 对照项 | 原仓库实现 | 当前阶段二 |
|---|---|---|
| IPC/H2D | `maimai_baseline.training.inputs._collate_full_input`、`data.batch_transport.PackedBatch` | 条件及 M 分块分别按 dtype 打包，并保持分块显存边界 |
| 全曲 P0 K/V | `model.rope.mapped_attention`、`data.attention_layout.cross_layout` 按歌曲聚合 query、裁 key padding | 独立 `decoder.batching.audio_query_layout` 与 `Attention.mapped`，共享 K/V 张量并恢复原顺序 |
| 条件准备 | 原路径读取冻结音频特征并按歌曲映射 | worker 选择缓存 Z/plan 和共享 P0；训练前向不构造 E/D/P 或执行 CPU layout fallback |
| 采样与归约 | 原训练维护完整音频组、既有计划及逻辑 batch | 保留 occurrence 集合，全局随机窗口组批；内部计算块仍按共同分母反传 |

原仓库对照路径为 `repos/main-new-audio-draft-beat/src/maimai_baseline`，仅作实现参考。当前网络和训练均不导入该包。按歌曲聚合 attention 会改变 dropout 随机数的消费顺序；数值/梯度等价测试使用 dropout=0，分布语义不变；同版本完整轮状态恢复另用非零 dropout 验证。

## 配方与执行

唯一配置位于 `entry.CONFIG`，所有 `decoder_train_*` 均有实际消费者。尚未确定的训练数值为 `None`，运行前必须显式提供，不能回退到阶段一数值：

- 原始阶段一 checkpoint 身份及配套 P0、D/P 缓存路径；学习率、weight decay、gradient clip；scheduler 总轮数与绝对停止轮。
- 精度 `fp32` 或 CUDA `bf16`。
- Physical、judge、trajectory 三个辅助系数；已有 Route-71 距离矩阵路径及独立输出目录。

默认 AdamW、warmup **200次 optimizer update**、row budget **8192**、梯度累积 **1**、workers **12**，均保持可配置。seed 与现有计划一致；scheduler 按各轮随机组批产生的实际更新数计算 horizon，不沿用整曲组批的步数。所有路径可通过 JSON 显式改为目标主机路径；本地默认路径只定位既有资产，并不表示所有远端载荷已下载。本次 pilot320 Physical 本地目录只有生产摘要等元数据；实际训练必须指向含 `catalog.json` 和 `charts/...` 的既有完整产物目录，不能用摘要冒充载荷。

```powershell
python -m maidata.decoder_training --show-config
python -m maidata.decoder_training --config decoder-stage2.json
python -m maidata.decoder_training --config decoder-stage2.json --resume OUTPUT/checkpoints/last.ckpt
```

原 batch CUDA 检查是显式命令，在独立临时模型上执行第1轮指定的完整逻辑 update，保持该配方的分组、occurrence、累积与精度，不保存训练 checkpoint：

```powershell
python -m maidata.decoder_training --config decoder-stage2.json --cuda-check --check-step 0
```

报告标明 `conditioning=offline_cache`、仅构建 Decoder、E/D/P 调用数为0，并包含 M 更新范围、各输入 batch 的音频/谱面 padding 成本与超限标志、各 M 块成本、数据等待耗时、更新耗时和显存。`check-step` 为零基索引。该入口不启动后续训练，也不设置性能或质量通过阈值。

## Checkpoint 与验证

阶段二使用独立 schema `maidata.decoder-training.v1`。`decoder_train_checkpoint` 仅记录原始阶段一生产者路径；训练读取 manifest 中保存的生产者信息，不重新哈希 checkpoint 或加载 E/D/P 权重。`--resume` 只接阶段二完整轮 checkpoint，两者不能混用。保存 M、optimizer、scheduler、完整轮/step、所有 RNG、DataLoader generator、原始生产者身份、所选缓存 manifest 路径、配置和输入来源，不把消费者源码写成缓存生产者。

启动时读取缓存 manifest 和配置；不遍历旧载荷重新验收。源码版本仅作记录，不绑定 checkpoint，不生成或比较输入文件哈希。恢复不以源码、recipe、budget、计划、缓存身份或停止轮变更为由拒绝加载；是否继续某个 checkpoint 由用户和训练会话确定。接口加载所选模型、optimizer、scheduler、完整轮游标和 RNG；实际不匹配的参数张量由 PyTorch 的状态加载返回具体错误。恢复使用当前配置安排后续窗口和批次，并保留 checkpoint 中的 optimizer/scheduler 状态，不自行重置 warmup。

每5轮及末轮 validation，重合只执行一次；完整诊断仅末轮。每轮按保存的 occurrence 集合与随机顺序消费，去重歌曲数与音频读取实例数分别记录。`last` 每个完整轮保存，`best` 只在轮末验证改善时更新，编号 checkpoint 每5轮及末轮保存；异常和轮内不保存 checkpoint。验证 loss 用于诊断及 checkpoint 记录，谱面质量仍由实际生成结果评估。

CPU 验证覆盖原谱缓存、在线规则对照、拼接/并行轨道、负拍起点、旧长动作、历史 dropout、空尾窗、批量/逐行 loss 与梯度及分块等价；独立导出检查覆盖生产者/归一化错配；训练检查覆盖不重新校验已验收输入、多曲多难度与重复实例映射、Z上下文padding、预测plan及BF16缓存供FP32训练。禁止E/D/P构造、原始音频读取和生产者权重加载的测试仍可完成M更新，M的音频/Z/plan投影保留梯度。共享/复制K/V、packed worker、无前向CPU layout fallback和小头复用检查保留。完整轮状态恢复使用模型及历史dropout，覆盖零worker与测试用2个persistent workers，逐位核对M、optimizer、scheduler、全部RNG与每轮指标，并验证源码和配方变更不触发恢复门禁；正式默认仍为12 workers。吞吐需另外记录真实 DataLoader 与 CUDA 更新耗时。
