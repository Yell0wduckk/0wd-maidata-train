# 主 Decoder 阶段二训练

入口为 `python -m maidata.decoder_training`。本阶段**重新随机初始化音频 Encoder E_M 与主 Decoder M，联合更新两者**；始终读取阶段一 E_D/D/P 的固定离线全曲 Z。plan12默认读取P的预测，可显式切换成独立离线GT plan作对照。训练与验证构建E_M/M，不构建或运行教师E_D/D/P/R，不加载阶段一权重初始化E_M。E_M在线消费已验收Mel/aux32，M的谱面损失经音频K/V正常反传至E_M；固定Z/plan不会随E_M更新漂移，缺失条件不在线补算。

## 模块与数据流

| 模块 | 职责 |
|---|---|
| `decoder_training.plan/ordering` | 保留既有实例集合与 split，逐轮全局打乱窗口，从窗口缓存取得实际输入长度并按 row budget 组批 |
| `decoder_training.frozen_cache/conditioning` | 读取固定 Z/plan 及来源，按窗口选择上下文，整理在线 P0 的坐标与有效位 |
| `decoder_training.gt_plan_cache` | 一次离线复用原谱和MusicPlanTargets生产GT plan12；训练仅读取小型bank |
| `decoder_training.data` | 按 batch 去重读取 Mel/aux32 和固定条件，准备完整音频覆盖、GT 历史与随机窗口顺序 |
| `decoder_training.preparation/transport` | worker 内整理去重映射、padding 和分块；按 dtype 打包 IPC、pin memory 与分块 H2D |
| `decoder_training.export_windows/support_cache` | 离线恢复真实动作与首次归责账本，准备窗首、窗内和终端支持 |
| `decoder_training.window_cache` | 读取已打包的输入与监督张量，执行历史 dropout、padding 和行索引平移 |
| `decoder_training.intervals/compact_windows` | 从Physical区间直接生成loss积分区间；另支持已有窗口张量转换，不重查规则 |
| `decoder_training.network` | 持有独立 E_M/M；只编码所选局部 P0，再接入固定 Z/plan |
| `decoder_training.shared_audio` | 累积各 M 分块的音频 K/V 梯度，最后统一反传到真实 K/V 投影和 E_M |
| `decoder_training.batching` | 按实际 M 输入长度连续分块，不改变 occurrence 顺序或裁剪历史 |
| `decoder_training.supervision` | 离线整理支持与计数；训练执行批量小头及三组独立 loss 归约 |
| `decoder_training.recipe/checkpoint/runner` | 配方、配套生产者身份、完整轮状态恢复、训练及验证 |

每个输入batch内，相同audio_variant只读取一份音频输入，E_M仅编码当前监督窗口需要的局部音频：前4＋目标16＋后4拍。同曲同起点的重复occurrence及不同难度窗口共享一份局部P0；不同目标窗各自读取自己的24拍P0，不读取同batch其他窗口或整曲P0。完整空尾窗也按其起点编码真实上下文，范围外mask无效。轻量G0与其投影仍按歌曲准备一次，保持既有全曲上下文结构；局部L/S分支不再遍历整曲覆盖窗。固定D/P缓存与GT plan来源不变。

音频K/V按唯一局部窗口共享，worker准备各监督窗到局部bank的映射，audio attention只读取对应bank。Z保持原来的前4＋目标16＋后4拍范围，plan12无位置编码，不直接读取难度值。按需音频范围记录为`decoder_train_audio_scope=window`；推理加载新checkpoint后使用相同局部范围，历史缺少此字段的checkpoint继续使用原整曲音频语义。

`decoder_train_encoder_window_chunk_size`默认8，只控制Encoder局部窗口计算块，不改变逻辑batch、row budget、采样、顺序或loss分母。每块采用non-reentrant梯度检查点，反向重算该块局部激活，保留dropout RNG；G0及其K/V图按曲共享。正常前向每个需要的窗口只计算一次，反向允许重算，不detach可学习音频。

M计算块通过原有音频K/V梯度桥汇总梯度，再统一反传真实投影与E_M图。所有块及累积batch完成后，E_M/M共同裁剪并执行一次optimizer/scheduler更新。离线Z/plan无梯度，更新后重新计算在线音频。保留完整GT历史、监督字段、Physical支持和分母。


固定条件由 `decoder_train_dp_cache` 显式提供，沿用 [D/P 导出](frozen-dp-cache.md)格式；在线音频使用 `decoder_train_audio_cache` 与 `decoder_train_mel_root` 的既有输入。`decoder_train_p0_cache` 仅保留为可选离线导出路径，阶段二训练不读取 P0 缓存。启动只读取所需索引与原生产者元数据，不重新验收旧资产，不读取或哈希生产者 checkpoint。配套输入由用户和训练会话选定，读取时不重复扫描 shape、dtype、有限值、拍坐标、mask 或归属。P0/Z 保持原始绝对位置，完整空尾 Z/plan 有效；不同歌曲的 batch padding 与 Z 上下文范围外的槽无效。

## Plan 条件来源与对照

`decoder_train_plan_source`默认`predicted`，可选`gt`；CLI的`--plan-source`覆盖JSON。所选来源同时用于训练和validation，不做逐batch混合，不改变Z、谱面监督、历史dropout、采样实例或E_M/M初始化。

| 来源 | M接收的plan12 | 输入路径 |
|---|---|---|
| `predicted` | P输出的8维sigmoid连续值＋4类softmax概率 | 既有`decoder_train_dp_cache` |
| `gt` | MusicPlanTargets的8维归一化统计＋4类one-hot | 独立`decoder_train_gt_plan_cache`；Z仍来自原D/P缓存 |

GT缓存复用原谱，一次离线计算，与原字段/Physical、窗口监督和D/P缓存分开；训练不解压整谱或计算统计。缓存按audio_variant保存小型bank，使用window_id对应plan，重复occurrence共享同一行；空窗及完整尾窗也有效，为8个0和`single` one-hot。GT模式不重新生成Z，也不训练Planner。

显式生产GT小缓存后再运行GT训练；此生产命令不由训练或串行launcher自动调用：

```sh
python -m maidata.decoder_training.gt_plan_cache --config decoder-stage2.json --output GT_PLAN_CACHE --workers 12
python -m maidata.decoder_training --config decoder-stage2.json --plan-source predicted --output PREDICTED_OUTPUT
python -m maidata.decoder_training --config decoder-stage2.json --plan-source gt --gt-plan-cache GT_PLAN_CACHE --output GT_OUTPUT
```

两种来源分别从同seed的新E_M/M、optimizer和scheduler开始。串行执行、12维字段顺序和比较口径见[Plan对照](decoder-plan-comparison.md)。GT validation同样使用真实统计条件，仅作oracle诊断；新曲自动推理仍由固定E_D/D/P提供预测plan，包括GT训练的checkpoint。GT训练用于比较准确统计条件的作用，不表示新曲自动取得真实谱面统计。

## 原谱、历史和 Physical

消费已保存的主 Decoder 计划，训练侧保留基础窗口和复杂节奏的附加 occurrence，验证侧仅基础窗口；目标默认16拍、stride16拍，保留完整空尾窗。不会重新划分数据、增建镜像或按 Physical 违规删掉 GT。

历史为完整起音组的4拍 GT 前缀，不设 token 裁剪预算；首组保留原始 delta 和绝对位置。边界间隔使用全曲最后真实起音，即使它在4拍之外。训练历史 dropout 默认0.15，由 `(seed, epoch, window_id, occurrence)` 决定，与 DataLoader worker 分配无关；dropout 只移除历史输入并注入 M(0)，验证不启用。终端使用名义音频上下文窗尾 U，保留精确窗外和尾部删失标签，不把 U 改为全曲 audio_end。

字段和训练 Physical 均为原谱 v2。独立离线导出器复用已有完整谱面支持，恢复真实动作、pending 语法及冲突账本，完成窗口边界适配和监督张量化。窗前仍有效的长 hold/slide 不受4拍历史裁剪或 dropout 影响。训练 DataLoader 只读所选窗口的张量，不读取整谱 JSON、不解压字段或 Physical、不重放语法、不查询 Physical、不生成时间候选。历史保留和丢弃共享同一份监督，只改变输入和监督行位置。

窗尾是 Physical 时间支持的显式区间分界，避免用关闭窗口后的代表点判定窗内候选。该修复仅影响消费窗口：离线完整原谱没有窗口边界，其字段、动作、候选缓存格式与语义不变，无需重建已验收缓存。负拍音频网格早于源谱 M(0) 时，窗口可生成源序列中没有的首个正 delta MOMENT。

## Batch 与损失

训练每轮保留既有全部 occurrence（包括复杂节奏追加实例），按 seed/epoch 对窗口实例全局打乱，然后沿此顺序连续装入 batch。同曲窗口可以分散到不同 batch，不再以完整歌曲组约束主 Decoder 的监督采样；验证保持既有固定顺序和基础窗口集合。既有计划中的原始组顺序保留实例来源，消费端不改写计划文件，也不重新划分数据或增加倍率。组批不回填、不丢弃，读取时临时聚合同曲，collate 后按 sequence_index 恢复随机顺序。

`decoder_train_row_budget` 默认 **8192**，只约束 `窗口实例数 × 最长实际 M 输入长度`，本次不改变组批边界或把全曲音频计入预算。长度包括保留的4拍历史、注入边界、当前窗内容和 batch padding，按本 occurrence 的确定性历史 dropout 计算；终端只作目标，不另占输入位置。目标和历史长度直接读取窗口缓存 manifest，不在训练启动或每轮读取字段载荷计算。音频张量规模和在线 E_M 成本单独记录。

`decoder_train_gradient_accumulation_steps` 默认 **1**，每个 batch 反传后立即更新一次。`decoder_train_chunk_token_budget` 默认 **8192**，只控制同一更新内部的连续计算分块；所有块共享逻辑 batch 分母。单窗口超预算时保留完整历史和目标，并记录实际规模；不裁剪、不丢弃、不重排。

先在 CPU 为整个逻辑 batch 汇总分母，再逐块反传：

- 主监督：每个有效预测决策 NLL 的总和／总数；唯一合法分类、NA、历史目标、padding 不计，终端时间计数。
- Physical：Router 类型与内容各计单位；NOTE 内部 kind/pad 固定0.5/0.5，有效零贡献单位仍在共同分母中。四类最后按 `(1,1,4,1)/4` 归约，flags 不连接 Physical 梯度。
- 两项几何：分别按有效 ROUTE 总数归约，独立于主监督和 Physical 分母。

批量小头实现与 `decoder.supervision.decoder_loss` 逐行参考比较总和、计数和梯度。分块及梯度累积不采用窗口均值的平均。

MOMENT（delta）、HOLD_DURATION、SLIDE_DURATION 的训练概率直接积分紧凑区间，不枚举全部 tick/cell。仅合并端点严格相接、GT归属相同、四类Physical新增归责位相同的基础合法cell；保留不连续支持的缺口、GT边界和无穷尾部。每个合并区间的Logistic质量等于原cell质量之和，因此基础归一化、GT NLL及四类Physical补集概率保持原数学定义。有效决策计数、NOTE五五分配和逻辑batch分母沿用合并前计数，不按合并后的区间数量计数。

区间端点、CDF质量及并集归约使用FP64，完成每个决策的概率归约后再输出FP32 loss；不在大量窄格的FP32 log概率上累加。模型主体精度与参数不变。紧凑区间只供loss积分，推理仍按原cell/代表tick生成，不改变可生成时间或Physical规则。末轮分类诊断保留accuracy；时间诊断使用GT cell的概率总和、NLL总和及计数，不对合并区域的质量做argmax冒充时间格准确率。

## 计算复用与传输

通过独立命令准备当前选择范围的窗口张量：

```sh
python -m maidata.decoder_training.export_windows --config decoder-stage2.json --output WINDOW_CACHE --workers 12
```

将 `decoder_train_window_cache` 指向输出目录。该步骤复用已有字段、Physical，不重建源缓存或 P0/Z/plan；只处理计划中的基础窗和空尾窗，重复 occurrence 共用一份产物。每个窗口保存不压缩的 dtype 连续张量缓冲及轻量布局，新增产物在离线生成时检查一次落盘往返。历史长度和目标长度保存在 manifest。

窗口导出直接从Physical整数区间生成v2时间积分张量，不先枚举逐tick候选或逐桶duration再压缩。只在基础合法性、四类新增归责或GT归属改变处切分，MOMENT窗尾和无穷尾部独立处理。duration按原四tick桶代表值映射支持，同时检查GT实际tick的基础合法性；完整BPM映射与原始有效选择数保持不变。时间张量生成成本随区间数量增长，不随候选tick总数增长。整谱Physical仍保留原区间与具体冲突证据，不能仅按四类mask相同就合并丢失首次归责账本。

已生成的紧凑v2窗口语义不变，无需重建。已有v1窗口可直接转换到新目录：

```sh
python -m maidata.decoder_training.compact_windows --source OLD_WINDOW_CACHE --output COMPACT_WINDOW_CACHE --workers 12
```

转换仅读取已完成的窗口张量，保留token、顺序、长度、计数及非时间监督；不读取字段、Physical、音频或P0/Z/plan。将 `decoder_train_window_cache` 改为新目录，从原完整轮checkpoint恢复，保留optimizer、scheduler和RNG；不改运行中的源码或原缓存。

DataLoader worker 读取所选窗口输入/监督、固定 Z/plan 与去重歌曲的 Mel/aux32，按 seed/epoch/occurrence 选择历史视图、平移标签索引、padding，再按 dtype 打包。所选Encoder窗口布局、M有效行与attention布局均在worker根据输入坐标和mask准备；P0数值由训练进程中的E_M提供，不传输占位的完整隐藏bank。标签、合法 mask、Physical 罚项及其计数已离线确定。训练进程不做整谱解压、校验、规则重放或候选生成；不扫描输入张量的形状、有限值及归属。模型投影和概率运算的重复输入检查在训练/validation 调用中关闭；合法性 mask、概率归一化与梯度裁剪是实际计算，继续保留。

IPC 使用现有 `training.transport` 的 dtype 连续缓冲实现，按需局部音频输入/布局、固定条件与每个 M 分块分别打包；只传轻量 occurrence 元数据，不传规则状态、TimingMap 或整棵 GT/支持对象。音频输入与固定条件传入设备一次，token/支持随当前分块传输和释放，避免把全曲所有候选同时放入显存。模型只消费公开张量，打包形式不进入网络接口。

NOTE 输入、MMoE gate、kind→pad/flags 使用固定形状张量运算，避免 CUDA 布尔索引产生动态 `nonzero`；flags 的条件适配每个 NOTE 仅计算一次。常规 loss 统计在设备累积，输出 update/epoch 日志时统一回读；不逐步执行额外有限值断言。

有效音频与谱面行索引、小批SDPA布局均在worker整理现有张量时生成，并随dtype缓冲传输，不读取字段或Physical、不改离线缓存。每层四个LayerNorm、self-attn QKV、两路cross-attn query及三路输出投影、FFN专家与gate均只计算有效谱面行；条件K/V仅投影有效Z/plan槽。self-attn通过拼接现有Q/K/V权重执行一次线性投影，不改变参数名称或optimizer槽位。

独立`decoder.row_layout`按有效token数的二次幂档组织窗内self-attn；内部无效位置按序移除，原始绝对拍坐标不变。每个小批只补齐到组内最长窗口，padding key位于全部真实query之后，使用`is_causal=True`保持原因果关系。`decoder.packed_attention`只消费张量和布局，输出按原窗口/行索引恢复，不改随机窗口顺序、batch成员、row budget或loss分母。self K/V仍恢复原窗口形状及有效位，增量续接和不可变快照沿用现有接口。

Z/plan attention同时按query长度和有效key数分组；独立key mask保留空条件及plan有效性，原有效槽的拍坐标和归属不变。Z/plan通过互不重叠的两半Q/K特征实现单次SDPA：Z使用旋转后分量，plan使用未旋转分量，显式保留原head维度的缩放；仍为同一个联合softmax，plan没有位置编码。分组减少attention内部padding，不把它计为有效token或改变8192的预算口径。

| 对照项 | 原仓库实现 | 当前阶段二 |
|---|---|---|
| IPC/H2D | `maimai_baseline.training.inputs._collate_full_input`、`data.batch_transport.PackedBatch` | 条件及 M 分块分别按 dtype 打包，并保持分块显存边界 |
| 全曲 P0 K/V | `model.rope.mapped_attention` 逐曲SDPA，切片反向回传整批bank | 独立 `decoder.batching.audio_query_layout` 与 `Attention.mapped`，有效位置投影、小批SDPA及一次梯度收集 |
| 条件准备 | 原路径读取冻结音频特征并按歌曲映射 | worker准备完整音频输入/布局和固定Z/plan；在线E_M只计算所选局部P0，不构造D/P或执行CPU layout fallback |
| 采样与归约 | 原训练维护完整音频组、既有计划及逻辑 batch | 保留 occurrence 集合，全局随机窗口组批；内部计算块仍按共同分母反传 |

原仓库对照路径为 `repos/main-new-audio-draft-beat/src/maimai_baseline`，仅作实现参考。当前网络和训练均不导入该包。小批attention和跳过padding会改变dropout随机数的消费顺序，保持相同分布语义；数值/梯度对照使用dropout=0，当前联合路径另以CPU FP32核对直接计算图。新v2完整轮checkpoint恢复E_M/M成套状态；只有M的旧v1不能用于联合训练续训。同版本完整轮恢复另用非零dropout验证。

## 配方与执行

唯一配置位于 `entry.CONFIG`，所有 `decoder_train_*` 均有实际消费者。尚未确定的训练数值为 `None`，运行前必须显式提供，不能回退到阶段一数值：

- 配套D/P缓存与音频输入路径；学习率、weight decay、gradient clip；scheduler总轮数与绝对停止轮。教师来源直接读取D/P manifest，训练无需提供或读取阶段一checkpoint。
- 精度 `fp32` 或 CUDA `bf16`。
- Physical、judge、trajectory 三个辅助系数；已有 Route-71 距离矩阵路径及独立输出目录。

默认 AdamW、warmup **200次 optimizer update**、row budget **8192**、梯度累积 **1**、workers **12**，均保持可配置。`decoder_train_fused_optimizer=True`在CUDA选用fused AdamW，CPU使用普通实现；恢复时按当前配置选择执行内核，并完整加载原lr、moments和step，不重置优化器。seed 与现有计划一致；scheduler 按各轮随机组批产生的实际更新数计算 horizon，不沿用整曲组批的步数。所有路径可通过 JSON 显式改为目标主机路径；本地默认路径只定位既有资产，并不表示所有远端载荷已下载。本次 pilot320 Physical 本地目录只有生产摘要等元数据；实际训练必须指向含 `catalog.json` 和 `charts/...` 的既有完整产物目录，不能用摘要冒充载荷。

`decoder_train_checkpoint`与`decoder_train_p0_cache`均为离线导出工具的可选路径，运行阶段二联合训练时允许缺省；不把旧checkpoint或P0缓存设为启动依赖。

```powershell
python -m maidata.decoder_training --show-config
python -m maidata.decoder_training --config decoder-stage2.json
python -m maidata.decoder_training --config decoder-stage2.json --resume OUTPUT/checkpoints/last.ckpt
```

原 batch CUDA 检查是显式命令，在独立临时模型上执行第1轮指定的完整逻辑 update，保持该配方的分组、occurrence、累积与精度，不保存训练 checkpoint：

```powershell
python -m maidata.decoder_training --config decoder-stage2.json --cuda-check --check-step 0
```

报告标明`conditioning=online_audio_fixed_draft_plan`、`encoder_initialization=scratch`及实际`plan_source`，构建E_M/M，D/P调用数为0，并包含两模块更新范围、各输入batch的音频/谱面padding成本与超限标志、各M块成本、数据等待耗时、更新耗时和显存。`check-step`为零基索引。该入口不启动后续训练，也不设置性能或质量通过阈值。

## Checkpoint 与验证

阶段二联合训练使用 schema `maidata.decoder-training.v2`，`model_state`同时保存`encoder.*`（E_M）和`decoder.*`（M）。训练读取D/P manifest中的原始生产者信息，`decoder_train_checkpoint`可缺省，不哈希checkpoint或加载教师权重。新训练从随机初始化的E_M/M开始；`--resume`恢复v2完整轮的两套权重、全部optimizer参数槽、scheduler、完整轮/step、所有RNG与DataLoader generator，同时记录教师来源、所选manifest、配置和输入来源。

checkpoint的recipe记录`decoder_train_plan_source`，GT缓存路径作为输入来源记录；恢复不因plan来源变化新增硬门禁。两种fresh对照应各自使用独立输出目录，不将第一项checkpoint用作第二项初始化或续训。

旧`maidata.decoder-training.v1`只有M，缺少训练后的E_M及其optimizer状态，明确拒绝作为本方案完整轮续训；阶段一权重也不能作为v2续训输入。该区别属于真实模型/训练状态差异，不新增源码哈希、旧资产扫描或审批门禁。

启动时读取缓存 manifest 和配置；不遍历旧载荷重新验收。源码版本仅作记录，不绑定 checkpoint，不生成或比较输入文件哈希。恢复不以源码、recipe、budget、计划、缓存身份或停止轮变更为由拒绝加载；是否继续某个 checkpoint 由用户和训练会话确定。接口加载所选模型、optimizer、scheduler、完整轮游标和 RNG；实际不匹配的参数张量由 PyTorch 的状态加载返回具体错误。恢复使用当前配置安排后续窗口和批次，并保留 checkpoint 中的 optimizer/scheduler 状态，不自行重置 warmup。

每5轮及末轮 validation，重合只执行一次；完整诊断仅末轮。每轮按保存的 occurrence 集合与随机顺序消费，去重歌曲数与音频读取实例数分别记录。`last` 每个完整轮保存，`best` 只在轮末验证改善时更新，编号 checkpoint 每5轮及末轮保存；异常和轮内不保存 checkpoint。验证 loss 用于诊断及 checkpoint 记录，谱面质量仍由实际生成结果评估。

CPU验证覆盖既有字段/Physical监督、历史dropout、空尾窗和独立损失归约；联合训练另覆盖真实小E_M/M更新、禁止D/P/R构造、固定Z/plan不带梯度、音频与条件映射、packed传输及worker布局。梯度桥与直接真实计算图比较分块/未分块和累积batch的loss、E_M/M梯度及更新，检查每batch E_M只前向和反向一次。v2恢复使用非零dropout逐位核对两模块、optimizer、scheduler和全部RNG，并拒绝旧v1。正式默认仍为12 workers；CPU测试不代表CUDA原batch、吞吐或正式训练已验证。
