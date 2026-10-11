# Draft 局部条件重建接口

`maidata.draft` 实现逐拍 D、独立局部字段重建 R、目标窗 Planner P、原谱监督与辅助 loss。D 保持一拍一个 128 维 Z。完整训练组合位于 `maidata.training`；主 Decoder 仍独立持有字段读取和完整谱面生成。

## 网络和条件

所有构造默认值来自唯一 `entry.CONFIG`。D 默认 8 层、256 维、8 头、4 个 FFN512 专家，输出 128 维；R 默认 2 层、128 维、4 头、FFN512；P 默认 128 维、4 头、MLP256。构建不扫描数据：

```powershell
$env:PYTHONPATH = 'src'
python -m maidata --component draft
python -m maidata --component reconstruction
python -m maidata --component planner
```

`reconstruction_target` 从精确原谱 v2 创建独立查询。GT 只安排查询和 loss 索引；网络不接收 GT 总行数、完整未来类型表、count 答案或目标时间。查询之间没有 self-attention，padding 有效位仅控制本查询的计算和 loss。

| 角色 | 已知条件 | 查询位置 | 监督 |
|---|---|---|---|
| MOMENT | 已知游标、最后真实起音间隔、累计 tick | 原点加已累计的全局 tick | 下一 delta 或终止集合 |
| NOTE | 当前 MOMENT、同刻 NOTE 序号 | 当前 MOMENT | kind 与适用 flags |
| ROUTE | 当前 MOMENT、已知 kind、轨道序号和当前链前缀 | 当前 MOMENT | route_id、适用 track_break |
| HOLD_DURATION | 当前 MOMENT、已知 NOTE kind | 起点及累计 written-BPM 时长对应的拍位 | 完整 hold tick |
| SLIDE_DURATION | 当前轨道完整 route 链、NOTE 起点、轨道序号 | 等待一拍后的启动位置及累计实际时长对应的拍位 | 完整 slide tick |

R 不预测 type、pad 或空槽。NOTE 的待预测 kind 不输入其查询；ROUTE 的当前答案不输入链前缀。同刻与轨道序号可无界，通过连续特征编码；route embedding 与链内连续位置组合后汇总，保留整条已知链。

普通字段只读取 `floor(q)-1..floor(q)+1` 三个 Z。MOMENT 只读取 `floor(q)..floor(q+4)` 的至多五个 Z，选择仅由已知游标决定。CONT 后重新选择。R 的窗口上限默认前4＋目标16＋后4；查询不扩大该消费者的窗口。每层先对窗口 Z 投影一次 K/V，再为各查询 gather 局部槽，保留梯度。

P 使用独立窗口和 mask，读取目标 `[s,s+16)` 全部有效 Z。GT plan12 的八项统计、四类定义及空窗标签不变，可复用既有缓存。R 查询、长度和局部 mask 不进入 P 或 D。主 Decoder 训练默认使用 GT plan，自动推理使用 P 预测，详见[阶段二训练](decoder-training.md)。

## 时间、边界与计数

三时间头均输出 385 类 logits，索引 0..383 对应正 tick 1..384，384 为 CONT。共用 `maidata.time.discrete.compile_time_target`：`n=(d-1)//384, r=(d-1)%384+1`。96 tick 仍是一拍；Hold 的完整最小值为2，其 CONT 后的余数1可以合法。

窗首起音直接创建 NOTE 查询，无零 delta MOMENT。每个有效拍保留 97 类 MOMENT count，零类监督空拍。末尾创建角色仍为普通 MOMENT 的终止查询；exact/censored/closed-by-continue 元数据只用于标签和控制，不作为模型输入。首次短值或 CONT 跨谱面窗尾时停止该字段；CONT 不代表假起音。右删失使用短值尾部加 CONT 的集合 NLL。

duration 保持 NOTE written-BPM 单位。Slide 从 `T(onset+96)` 启动；CONT query 用实际秒数反算拍位，跨 BPM 后不更换 written 单位。生成域为本地音频窗尾与真实音频尾的交集。超 cap GT 保存在 `WindowTarget.deferred_durations`，包含源索引、精确 tick、cap 和 route 链；导出明确标记 `trainable=false`。collator 报出受影响窗口，不截短、不删除实例、不生成部分训练样本。该边界处理机制仍延期。

## 张量和组合

`BeatMemory(values, beats, valid)`、`DraftInputs` 与 `WindowLatents` 仍保持原公开接口。E 按曲准备一次 G0，同音频不同条件共享完整 P0 图；D 按 variant/level/difficulty 生成全曲 Z。更新后不复用旧隐藏值。

```python
targets = collate_reconstruction(window_targets, target_beats=16, device=device)
r_output, p_output = predict_auxiliary(
    z, groups, starts, valid, reconstruction, planner, config,
    layouts=layouts, queries=targets.queries)
terms = auxiliary_losses(r_output, p_output, targets, plan_targets, config)
```

`ReconstructionQueries` 只含 role、绝对 query/origin 拍位、elapsed、局部序号、已知 kind、route 前缀及其 mask、历史间隔和本查询有效位。`ReconstructionTargets` 另持有分类标签、精确源 tick、385 类目标集合、basic 域与每原始时间字段的首次计数位。两者分开传入网络和 loss。

`ReconstructionOutput.field_logits[name]` 为 `[N,Q,C]`，`time_predictions[name]` 保存 `[N,Q,385]` logits，`count_logits` 为 `[N,T,97]`。Q 为 batch 内查询 padding 宽度，不是模型容量或可见总行数。构建不再要求207行。

五个 `LossTerm(total,count)` 为 classification、time、count、planner_continuous、planner_category。分类按适用字段 CE，时间按 basic 域归一的整条路径 NLL／右删失集合 NLL；一条完整时间字段只计一次 time 分母，CONT 只增加分子和计算成本。count 保留逐拍监督；P 保持 MSE 与类别 CE。逻辑 batch 汇总分子/分母后加权，不按窗口均值平均。全 padding 为可反传有限零。

阶段一现有配方、完整音频组顺序和 batch 边界保留，详见[训练](encoder-draft-training.md)。`pretrain_batch_reference_rows=207` 仅保留原分批 proxy，不限制或填充 R 查询；真实 query/substep 成本单独记录。新阶段一 checkpoint 为 `maidata.encoder-draft-training.v2`，新 R 不冒充旧完整轮 resume；未改 E/D/P 的历史生产者可明确用于固定教师对照，权重更新后的神经特征须重新前向导出。

## 离线导出与验证

```powershell
python -m maidata.draft.export_targets --chart /path/to/charts/10/5.json.gz --start-beat 0 --start-beat 16 --output /path/to/new-targets.json
python -m maidata.draft.capacity --plan /path/to/plan.json --split /path/to/split.json --output /path/to/query-cost
```

两工具与训练共用 `reconstruction_target`，输出精确标签、查询条件、原字段计数与 query/substep 成本；capacity 不再设定网络行上限。只显式运行选定范围，不成为训练启动门禁；不重算音频、统计或 Physical 几何。

定向 CPU 测试见 `tests/test_draft_local_queries.py`、`test_draft_data.py`、`test_draft_networks.py`、`test_draft_integration.py` 与阶段一 runtime/layout/recipe 测试。覆盖局部梯度范围、查询隔离、链条件、变 BPM、空尾窗、完整字段分母、E/D/R/P 共享计算、梯度与恢复。CPU 证据不表示 CUDA 原 batch 成本、正式训练或谱面质量已验收。
