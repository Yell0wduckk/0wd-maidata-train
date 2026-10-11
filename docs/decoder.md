# Decoder 模块与接口

`maidata.decoder` 实现主 M 的类型 Router、内容小头、独立字段查询和离散时间循环。当前语义以[局部查询与时间实施说明](draft-decoder-local-query-time384-plan.md)为准。默认六层，前两层共享 Router，后四层 dense MMoE；M 保留类型、NOTE kind→pad→flags、ROUTE→track_break 和时间的全部监督。R 不监督 pad 不改变 M 的 pad 预测。

网络不持有 Encoder、Draft、Planner、Physical 或文件。阶段二由[`decoder_training`](decoder-training.md)联合训练独立 E_M/M；固定 E_D/D/P 提供 Z，训练默认读取准确对应 GT 谱面的 GT plan12。自由推理由 P 预测 plan12，见[整曲推理](maidata-inference.md)。

## 模块和配置

| 模块 | 职责 |
|---|---|
| `network / blocks / attention` | 因果历史、类型 Router、MMoE、消费方投影与绝对拍 RoPE |
| `field_query` | 独立 Q/output、字段角色与进度条件、临时位置的音频/局部 Z 读取 |
| `heads` | kind、pad、flags、route、track_break、三个独立 385 类时间头 |
| `time.discrete / time.distribution` | 共用编码、written-BPM 位置、目标路径、完整合法域概率与删失集合 NLL |
| `windows / batching / supports` | 真实起音历史、窗口边界、外部规则支持与张量整理 |
| `losses / supervision` | 主监督、首次归责 Physical、两项几何的独立 sum/count |
| `session` | 不可变 KV、pending 字段和完整规则状态的整体回退 |

唯一架构配置位于 `entry.CONFIG`。默认 M 宽 256、6 层、Router 第 2 层、8 heads，前段 FFN1024，后段 4 个 FFN512 专家和 5 类 gate。NOTE 配额 `(64,128,16,16,16,8,8)`、ROUTE `(240,16)`；输入时间正弦编码仍用于已提交真实完整 token。三个输出头固定 385 类，旧 Logistic mixture 分量、中心、尺度配置已退出实际消费者。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m maidata --component decoder
```

## 主历史与独立字段查询

`TokenBatch.beats` 是已知真实输入 token 的绝对拍坐标，用于主 self-attention/KV。它不能填待预测目标位置。`Decoder.begin(tokens, memory, cache=...)` 只接受已知前缀，不接受 next-type、字段 query、位置列表或 GT mask；`complete(routed, next_types)` 才选择后段内容分支。GT 类型不能通过临时查询选择泄漏给 Router。

```python
audio_kv = decoder.prepare_audio(local_p0)
memory = decoder.prepare_memory(audio_kv, window_to_audio, local_z, plan12, plan_valid)
output = decoder(tokens, next_types, memory)
# indices [Q,2] 指向真实主序列 [window,row]，其余张量 [Q]
field_h = output.field_reader(output, indices, field_query_beats,
                             field_roles, origin_ticks, elapsed_ticks)
logits = decoder.heads.time_logits(field_h, 'MOMENT')  # [Q,385]
```

字段查询各行独立，没有 query self-attention；增加后续 GT 查询不会改变早期查询。内容基态来自当前已知前缀，完整 route 链保留在其因果历史中。查询添加字段角色与已知累计进度，以独立 Q/output 读取已投影的最后一层音频/Z KV；K/V 仅按同参数版本准备一次并保留计算图，不重新运行主干。plan 已经条件化主前缀，不进入局部字段 Z 读取。

普通内容及 duration 查询选 `floor(q)-1..floor(q)+1` 三个 Z；MOMENT 选 `floor(q)..floor(q+4)`，仅依赖已知临时游标。缺失槽 mask，无循环补齐。字段音频只读取已提供的本地 bank；主干已有的因果历史与 Z/plan 条件范围保持。query Q 和 key 使用同一绝对拍坐标系。plan 的主干评分保持无时间 RoPE。

NOTE 的 kind/pad/flags 复用同一字段基态，pad 与 flags 各有条件适配。pad 统一返回 33 槽，按钮行后 25 槽为负无穷；NA=-1 不与合法 0 混淆。ROUTE 与首段 track_break 复用路线查询；同类型内容重试不重跑主干。改选 Router 类型会从原前缀重算受影响的后段 KV。

## 384 tick 与 CONTINUE

96 tick/拍不变，类别 `0..383` 表示余数 `1..384`，类别 `384` 是 CONT。正值 d 唯一编码为 `CONT * ((d-1)//384)` 加末余数 `(d-1)%384+1`。MOMENT/Slide 至少 1 tick，Hold 完整值至少 2 tick；Hold 的后续余数 1 合法。

CONT 锁定本字段类型，累计 384 tick，只更新临时 field query；不追加主 self-KV、不产生 MOMENT/动作、不推进 Grammar/Physical/charged。短值合成完整时间后才向原规则层提交一次。三个独立时间头显式关闭 autocast，以 FP32 线性输出 logits；主干可使用 BF16。

MOMENT 从当前真实游标或显式窗首边界开始。短值到达或越过窗尾 b 时关闭本窗；CONT 游标首次跨 b 时立即关闭，没有最终余数，也没有假起音。结果区分 `closed_by_continue`、控制下界和 query tick。下一窗保持最后真实起音及活动动作，重新注入边界并 prefill。

Hold query 从 NOTE 起音开始，Slide 从全局拍轴的 `onset+96` 启动位置开始。累计时长始终以 NOTE written BPM 转换为实际秒，再由完整 BPM 映射反算 query 拍位；跨 BPM 不直接加 `elapsed/96`。duration 完整合法域同时受本地 U 和整曲有效音频末尾约束。

## 支持、概率与归约

新监督从精确原谱 v2 和完整 Physical 整数区间编译，不能从旧 4 tick duration 标签反推精确 tick。训练仅以 basic support 归一化，首次 penalty 位只用于 Physical loss；生成以同一完整完成的 raw 四类联合合法性作 hard mask。CONT 合法要求存在一个完整合法长值，不能分别按四类存在性拼接。

精确时间 loss 是监督路径各步 NLL 之和；右删失末步对兼容短值加 CONT 的集合概率取 NLL。一个原始时间字段只计一次主 loss 分母和一次 Physical 内容分母。Physical 展开全部可达且 query 仍在窗口内的 CONT 深度，按 CONT 前缀概率相乘。MOMENT 跨谱面窗尾的短值和 CONT 是零罚终止叶；duration 跨音频窗尾的 CONT 是未完成字段的零罚叶，不再展开，但窗内发出、窗外结束的短值仍保留完整源区间的 Physical 处罚。稳定合法补集 logsumexp 与 FP32 epsilon floor 保留，避免 `1-q` 消减和全违规无穷梯度。

NOTE 的 kind/pad 各 0.5，四类权重 `(1,1,4,1)/4`；flags 不连接 Physical 梯度。几何两项各按 ROUTE 数归约。所有窗口/分块先相加分子和计数，再使用逻辑 batch 的共同分母；内部查询成本与真实主序列长度分开记录。

GT duration 保留完整源值、组织、支持和真实动作。监督包含跨出本地音频 U 的最后一个 CONT 后停止；最后非 CONT 从窗内发出时，其落点超出 U 也正常监督。后续 teacher history/Physical 继续消费完整 GT，CONT 不变成主序列事件。`compile_support_tree` 有 GT 时采用此监督规则；无 GT 的生成路径保持现有完整 duration 的局部 cap，不把训练截断标记当成生成 MOMENT。原谱/Physical v2 源资产和迁移不受监督规则影响。

新完整轮 checkpoint 为 `maidata.decoder-training.v3`，架构身份 `maidata.decoder-field-query-tick384.v1`。旧 v1/v2 的时间头和 optimizer 不能冒充新架构完整轮 resume。新监督窗口为 `maidata.decoder-window-tensors.v4`，区分旧 v3 的局部 cap 标签；完整原谱和 Physical 源仍为 v2。迁移和从零生产共用编译器，命令见[阶段二训练](decoder-training.md)。

CPU 定向测试覆盖 Router 隔离、并行/增量等价、独立字段读取、梯度、完整字段概率/归约、CONT 不写 KV/Physical、链/分支回退与完整轮恢复。CPU 证据不代表 CUDA 原 batch、吞吐、正式训练或真实权重评谱已经执行。
