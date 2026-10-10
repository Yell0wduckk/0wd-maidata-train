# Decoder 模块与接口

`maidata.decoder` 实现[设计正本](../../0wd-v3-refactor-docs/new-decoder.md)的网络、字段小头、时间概率、窗口输入与三类监督。默认六层、前两层 Router 共享段、后四层 dense MMoE；直接读取全曲 P0 与局部 Z/预测 plan12。网络不持有 Encoder、Draft、Planner、Physical 或数据文件。阶段二训练循环与checkpoint由独立的[`maidata.decoder_training`](decoder-training.md)接入；整曲生成调度由独立的[`maidata.inference`](maidata-inference.md)接入。

## 模块边界

| 模块 | 职责 |
|---|---|
| `types` | 张量、投影 memory、不可变 self K/V 和分阶段结果 |
| `embedding` | 五类输入、NOTE/ROUTE 配额、固定零 NA、连续时间编码 |
| `attention / blocks / network` | 独立投影、绝对拍 RoPE、联合 Z/plan attention、Router、共享 FFN/MMoE |
| `heads` | kind→pad→独立 flags、route→首段 track_break、三个时间参数头 |
| `probability` | Logistic mixture、有限区间与尾部质量、基础支持归一化、显式采样 |
| `windows / batching` | 完整起音组历史、窗口 teacher forcing、15%训练历史 dropout、张量整理与公共 Z 窗口选择 |
| `supports` | 外部规则记录→基础/新增归责/原始拒绝支持；可不连续时间区间；窗口 oracle 适配 |
| `losses / supervision` | 主监督、Physical、两项几何损失的独立 sum/count；不包含训练配方 |
| `resources` | 从显式 NPZ 路径只读加载已有 Route-71 距离，不加载旧 Python 包或重建几何 |
| `session` | 绑定调用者的 pending 字段、Router/后段结果、K/V 与规则快照，整体回退 |

只有 `maidata.nn.position` 的无状态数学和 `maidata.chart/time` 的公共定义被复用。`batching.select_condition_windows` 调用 Draft 的公开纯张量窗口选择器，网络本身不导入 Draft。规则适配接收外部 AttributionState，Physical 不读取模型隐藏状态、音频、logits 或未来 GT 字段。

## 配置与构建

所有默认值在 [entry.CONFIG](../src/maidata/entry.py)，通过 [model.build_decoder](../src/maidata/model.py) 显式传给构造器。新增 `decoder_*` 参数独立于 Encoder/Draft 宽度；`audio_dim` 和 `latent_dim` 由各生产者配置传入，分别经过消费方投影。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m maidata --component decoder
```

默认 `dim=256 / layers=6 / router_layer=2 / heads=8`，前段 FFN1024、后段4个FFN512和五类gate、小头MLP256。NOTE配额 `(64,128,16,16,16,8,8)`，ROUTE `(240,16)`；两组各自求和为 Decoder 宽度。头维数须为偶数；层数、Router位置、heads、FFN、experts、配额、dropout、RoPE和时间分量数均可配置。常规数值默认 dropout0.1、LayerNorm eps1e-5、条件残差标量门初值1e-3；门是可学习直接乘数，沿用原版。

时间输入正弦默认base10000/scale1。三个独立头默认K5，初始位置/尺度、reference和minimum_scale在同一CONFIG中，改变K须同步提供K个初始值。MOMENT/HOLD的reference1000ms，SLIDE为96tick；scale为`softplus(raw)*reference+minimum_scale`。不设置学习率、训练集、验证集或损失总系数默认值。

## 前向与条件复用

`TokenBatch` 的主要形状为 `[B,L]`；flags为`[B,L,5]`。NA ID为-1，合法0不作NA。时间输入已转换为M的秒值或duration拍值；`beats`是当前已知输入的绝对起音坐标，不能填下一目标的位置。`Memory`接受任何具有`values/beats/valid`的公开张量对象。True表示有效。

```python
from maidata.entry import CONFIG
from maidata.model import build_decoder, decoder_time_distributions
from maidata.decoder.batching import collate_windows

decoder = build_decoder(CONFIG)
tokens, next_types, supervised = collate_windows(examples, device=device)
audio_kv = decoder.prepare_audio(full_p0)  # 每层、每个不同音频投影一次
memory = decoder.prepare_memory(audio_kv, window_to_audio, local_z,
                                predicted_plan12, plan_valid)
output = decoder(tokens, next_types, memory)
# output.type_logits [B,L,5]；output.hidden [B,L,d_model]
```

`prepare_memory`只计算一次共享Z/plan适配，再计算每层自己的K/V。Z逐拍投影、plan变成一个独立无位置条目；各自有效位独立。Z评分使用旋转后的Q/K，plan评分使用未旋转的Q/K，两组分数进入同一个softmax。输入不含difficulty/level，也不接受真值plan。

所有投影保留计算图，不在模块内detach或冻结。相同参数版本和同次前向可共享`audio_kv`；更新权重、输入、坐标或mask后由调用者重新准备。`prepare_memory`的结果标识当前窗口条件；每个新窗重新prefill全部self-attention层。

整块训练可在 CPU 用 `batching.audio_key_spans/audio_query_layout` 准备歌曲 query 映射，作为 `prepare_memory(..., audio_layout=...)` 传入。此时各层直接保留唯一歌曲 K/V，按歌曲读取当前块的有效 query 后恢复窗口位置，不按窗口复制全曲 bank。layout 只属于该 token batch 的形状、有效位和 ownership；增量调用使用其自身对应的布局或普通窗口 memory。网络不读取 Dataset、采样计划或打包格式。

## 增量、内容阶段与回退

```python
routed = decoder.begin(current_token_batch, memory, cache=previous_cache)
# 对 routed.logits 应用语法、基础支持和所需的独立 Physical 拒绝，再选类型。
completed = decoder.complete(routed, selected_next_types)
# 按 selected_next_types 调用 decoder.heads；同类型改内容不再运行主干。
next_cache = completed.cache
```

`begin`不接受下一类型或GT内容，只运行共享前段；`complete`加一次独立目标类型embedding并运行后段。训练右移GT类型仅传给后段。历史最后位置的后继是外部边界M，使用M的目标类型但不计监督。

`ContentHeads`公开`kind`、`pad_logits`、`flag_logits`、`route`、`break_logits`和`time_parameters`。pad与flags阶段分别从同一个最终h出发，独立条件表不绑定输入embedding。flags按适用性调用/计损失；pad统一返回33槽，按钮行的后25槽为负无穷。几何距离不参与输入或路线分类权重。

对同一个`routed`改选类型会从原前缀重算后段K/V；此前返回的缓存不原位更改。`DecodeSession`可绑定PhysicalState或AttributionState，快照同时记录pending字段、模型阶段与规则自身的完整快照；回退不以token数推算动作数。进入新窗的`reset_attention`同时确认调用者pending为空、规则语法允许结束，再清模型缓存；未完成音符或链时失败且保留原缓存，持久规则动作和最后真实起音始终保留。

## 窗口、支持与损失

`window_from_config`从真实整曲字段序列构造窗口。默认target16拍、stride16由外部采样计划负责；历史4拍按完整起音组保留，不设token预算。首组保留原始delta与绝对位置；历史为空仍从全曲最后真实起音算边界delta。训练整窗dropout概率0.15，验证不drop；drop仅改变输入，不改变目标和真实起音摘要。`history_prefix`也可消费推理已生成的真实起音组，旧窗口边界/终端不进入这些组。

终端标签保持独立`TimeTarget`，不进入模型输入或真实事件；U取名义音频窗口右边界，不裁到audio_end。时间候选包含所有U前的正tick格点与一个U尾部；最后有限cell延伸至U，与尾部无缝分界。窗内无音频候选由基础支持排除，达到窗尾的关闭候选不查询音符补全。

训练支持沿用`maidata.training-physical-original.v2`阶段记录格式，使用`penalty`新归责位；推理拒绝使用独立`physical`原始位。边界改变了查询游标和可选类型，不能把整曲原M/NOTE的缓存行直接贴到窗首。`supports.window_stages(example, attribution)`提供实际窗口oracle：调用者传入窗首之前的完整真实Physical状态，适配器临时注入边界、生成各阶段支持并最终恢复原快照。它不从裁剪历史重建动作、不修改缓存。既有缓存内满足相同前缀语义的支持行可直接传给loss；数据调度负责该映射，不在网络中读文件或批量重建。

duration使用字段层的4tick half-up标签、首尾裁剪cell与唯一代表tick；每个候选的基础支持、训练归责和推理拒绝均查询该代表tick。Physical对每类非法候选cell的质量求和，支持不连续区间，不假设单个safe_upper，也不在bucket内部增加其他可生成tick。GT保留精确整数tick，先检查其基础支持再映射bucket，不能借舍入接受超出音频范围的标签。Physical非法GT仍保留；实际生成使用原始拒绝支持，不以训练新增归责位替代硬拒绝。

```python
from maidata.decoder.supervision import decoder_loss
from maidata.decoder.resources import load_route_distances

geometry = load_route_distances(explicit_route71_npz).to(device)
sums = decoder_loss(decoder.heads, output, examples, stages, timings, audio_ends,
                    distributions=decoder_time_distributions(CONFIG), geometry=geometry)
loss = sums.objective(physical_weight=physical_weight,
                      judge_weight=judge_weight, trajectory_weight=trajectory_weight)
```

`DecoderLoss`可跨窗口或分块相加，最后才求均值。主监督按实际分类/时间决策计数；Physical按有效类型/内容单位计数，NOTE内部kind/pad各0.5，四类固定`(1,1,4,1)/4`；几何两项按ROUTE位置独立平均。有效零罚项保留Physical分母，flags完全不参与Physical梯度。使用逻辑batch分块反传时须先确定整个逻辑batch的分母，再按同一分母缩放各分子；不能先对每块求均值。

阶段二训练对delta和duration采用离线合并的时间积分区间：相邻cell仅在基础合法、GT归属及四类新增归责一致时合并，保留非连续缺口和GT边界。直接对并集求Logistic质量，避免对全部可生成格逐格计算loss；生成候选、分布、归责和决策计数不变。见[训练概率与缓存](decoder-training.md)。

## 验证范围

定向CPU测试位于`tests/test_decoder_{network,time_windows,supervision,session}.py`，覆盖默认/非默认结构、K变化、NA与合法零、teacher/incremental一致、因果性与目标类型隔离、plan位置不变性、空mask、FP32及CPU BF16反向、真实字段/规则接口、窗尾与BPM、三套loss、跨模块梯度与冻结、链/分支快照。几何只读加载、旧数学对照和无旧包运行时依赖分别验证。

```powershell
$env:TEMP = (Join-Path (Get-Location) '.test-temp')
$env:TMP = $env:TEMP
python -m pytest tests/test_decoder_network.py tests/test_decoder_time_windows.py tests/test_decoder_supervision.py tests/test_decoder_session.py --basetemp .test-temp/decoder
```

未执行CUDA、正式训练、整曲生成、上传或缓存重建；CPU数值与接口检查不代表这些流程已验证。现有字段、Physical、音频和几何缓存合同不因本次网络/归约实现改变。
