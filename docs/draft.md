# Draft 模块与接入接口

`maidata.draft` 已实现可训练的 D、独立重建头 R、Planner P、窗口组合、原谱标签适配和辅助 loss。结构以[设计正本](../../0wd-v3-refactor-docs/new-draft.md)为准。这里没有主 Decoder、训练循环、数据抽样计划、checkpoint 或整模型 AR 推理；旧模型不参与运行时导入。

## 模块职责

| 模块 | 输入与输出 | 依赖边界 |
|---|---|---|
| `windows` | 真实音频范围→覆盖布局；公开局部 P0→全曲 bank；全曲 Z→指定上下文 | 仅张量/坐标，不持有 Encoder、D、R、P |
| `conditions` | 源 `lv_<difficulty>` /显式数值等级→独立条件；按 variant/level/difficulty 去重 | 读取元数据，不读取谱面事件或隐藏状态 |
| `generator.DraftGenerator` | 全曲 P0＋独立难度/等级/span＋query 坐标→每拍 Z | 不导入 Encoder、字段头、Planner、Decoder |
| `reconstruction.ReconstructionHead` | R 上下文 Z→分类、绝对时间、逐拍 count | 独立参数，公共字段词表；无行 self-attention、无 GT 输入 |
| `planner.PlannerHead` | 目标窗 Z→8维连续 logits＋4类 logits＋可微 plan12 | 独立单次池化，不读取邻窗 Z、GT plan 或 R 输出 |
| `chart_data / targets / plan_targets` | 已有原谱 v2 记录→真实起音组、R 行与 simai 统计 | 复用公共字段与 TimingMap，不重算 Physical、不接入网络 |
| `losses` | 预测＋标签→分别归一化的损失项 | 不修改网络、冻结策略或优化器，不指定总权重 |
| `supervision` | 复用同一全曲 Z，组合 R/P 窗口和损失项 | 仅调用独立接口，不持有 Encoder 或 Decoder |
| `maidata.nn.position` | RoPE /连续正弦数学 | Encoder 与 D/P 共用无状态 RoPE；不共享投影参数 |

网络不读磁盘、标签模块不读隐藏状态。所有构造参数显式传入，默认值只在 [entry.CONFIG](../src/maidata/entry.py) 维护；[model](../src/maidata/model.py) 分别提供 `build_model`（Encoder）、`build_draft`、`build_reconstruction`、`build_planner`。单独构建 Draft 不导入 Encoder、Physical、输入生产或旧包。

## 配置与构建

```powershell
$env:PYTHONPATH = 'src'
python -m maidata --show-config
python -m maidata --component draft
python -m maidata --component planner
```

D 默认8层/256维/8头/4专家（每专家 FFN512），输出128维；难度 embedding8，level归一化(10,5)，span除以16；dropout0。P 默认128维/4头/MLP256/dropout0.1。R 默认2层/128维/4头/FFN512/dropout0.1，前16＋目标16＋后16拍；各宽度独立，不要求等于 Encoder。

`reconstruction_query_rows=None` 表示尚未绑定数据容量。调用方用 `required_query_rows` 汇总所选训练/验证的新字段行数元数据（每窗包含 `SEGMENT_END`），写入配置后构建 R；不按每个样本实际长度改变查询数量。超容量明确报错。它不要求现在选择训练集，也不会自动扫描数据。

`target_beats` 与音频余量、R余量分别配置；D/P 的 RoPE 共用 Encoder 的 base/scale。R 保留独立连续正弦表示，`reconstruction_position_base/scale` 默认10000/1。CE smoothing和SmoothL1 beta分别由配置传入，默认0.05/1；学习率、阶段长度、数据集和损失组合权重未设置。

## 全曲计算与复用

以下是消费接口顺序，`grids` 来自已有音频输入缓存，`rows` 是所选谱面的源元数据；这里只说明组合，不指定数据计划。

1. 对每个唯一 audio variant，用 `TimingMap.from_audio_timing(row['timing'])`、真实 `audio_end_seconds`、缓存 `first_beat` 和 `target_beats` 构造 `song_coverage`。
2. `audio_windows(coverages, device=...)` 返回覆盖全曲的音频中心窗起点与 `window_to_audio`。经既有 `preparation.cache.song_inputs/window_inputs` 准备输入；转换到与网络相同设备。首窗前文按既有有效位补零。
3. Encoder `prepare_song` 一次、`encode_windows` 覆盖全部音频中心，再调用 `assemble_full_p0(output.P0, starts, mapping, coverages, p0_rate=config['fpn_rates'][0])`。每个中心唯一归属，不平均重复窗口；margin不进入bank。额外完整尾窗没有实际P0，不必额外编码一次全无效音频。
4. `read_condition(row, numeric_level=...)` 读取精确数值等级；缺失或仅有 `14+` 时必须提供对应数值，不回填0。`make_draft_inputs(conditions, audio_variants, coverages, device=...)` 返回按条件去重的 `DraftInputs` 与每谱到组的映射。
5. `z = draft(full_p0, inputs)` 生成完整 Z；查询包括音频末窗的剩余拍及追加完整空尾窗。`span_beats=B(audio_end)-B(0)` 独立于取整、尾窗和谱面结束。`project_audio`/`from_projected` 可显式复用同一图内的各层音频 K/V；常规 `forward` 已在条件映射前按唯一音频投影一次。
6. 用 `select_windows(z, window_groups, starts, window_valid, target_beats=..., before_beats=..., after_beats=...)` 分别选择 R、P、Decoder 上下文。P 前后余量均为0；R默认16/16；Decoder默认4/4由其消费者传入。占位窗通过 `window_valid=False` 表达，不把有效空窗变成padding。

全曲 P0 和 Z 只计算一次，再供本 step 的多个目标窗共用。整个图完成反传后更新参数；更新后不能继续复用旧图。条件键为 `(audio_variant, level, difficulty_id)`，只共享音频的不同等级不能共享 Z。

## 张量协议

| 对象 | 字段 |
|---|---|
| `BeatMemory` | `values[N,L,D]`、绝对 `beats[N,L]`、bool `valid[N,L]`；可接收 Encoder 同形公开对象 |
| `DraftInputs` | `query_beats/draft_valid[N_draft,Lz]`；`audio_mapping/difficulty_id/level/span_beats[N_draft]`；前两项索引为long |
| `WindowLatents` | `memory`、`target_start[N_window]`、整数 `target_beats`、bool `window_valid[N_window]` |
| `PlannerOutput` | `continuous_logits[Nw,8]`、`config_logits[Nw,4]`、可微 `plan_values[Nw,12]`、bool `valid[Nw]` |
| `ReconstructionOutput` | `field_logits[name][Nw,Q,classes]`、`time_predictions[name][Nw,Q]`、`count_logits[Nw,T,97]`、bool `valid[Nw]` |

Decoder 消费公开全曲 P0、自己选取的 `WindowLatents.memory` 和预测 `plan_values/valid`，自行实现投影、来源向量及一次条件 attention；Draft 不创建这些模块。plan12 已按 sigmoid8＋softmax4 构成，保留梯度，不附拍坐标。训练与自动推理均用预测值，GT 只交给 loss。普通推理只运行 D/P，不需要 R。

## 标签与损失

`chart_data.from_record(record)` 消费当前原谱 v2 的 tokens、绝对 tick、owner映射及 TimingMap；保留连续 ROUTE 和并行轨道的原序列。真实 NOTE 起音在窗首时仍生成 R-MOMENT，外部边界和主 Decoder terminal不进入R。类型为五个公共类型加专用 `SEGMENT_END`；NOTE的button/touch pad头分别为8/33类，其他适用性复用公共规则。NA用loss忽略值表达，合法0保留。

`reconstruction_target` 生成 R 标签；`MusicPlanTargets` 按真实 NOTE（含短hold语义还原）生成8连续值和整数类别。全谱 workload 分母使用 `metadata.render_end_beat`；缺失时接受显式 `source_end_beat`，不能替换为最后NOTE或音频终点。缺少该统计元数据只阻止Planner标签计算，不阻止独立R标签适配。空谱面窗为R结束标记＋逐拍count0，Planner连续值0＋single类；Z本身不回归零。

`collate_reconstruction` / `collate_planner` 用 `None` 表示batch占位窗。R的时间标签保持tick，loss中将MOMENT转为窗内绝对拍位、duration转为log1p(ticks/96)。`auxiliary_losses` 返回 `classification/time/count/planner_continuous/planner_category` 五个 `LossTerm(total,count)`，`.mean` 是各自归一化值；训练配方显式加权。全padding时是可反传的有限零。

可以仅对指定缓存和指定窗口导出可复核标签，不做全量资产遍历：

```powershell
python -m maidata.draft.export_targets --chart /path/to/charts/10/5.json.gz --start-beat 0 --start-beat 16 --output /path/to/targets.json
```

脚本复用普通缓存读取校验，不计算音频、神经表示或Physical。导出的 `query_rows_required` 仅覆盖本次显式选取的窗口，不自动成为完整训练容量。

## 验证与执行边界

```powershell
$env:TEMP = "$PWD/.test-temp"
$env:TMP = $env:TEMP
python -m pytest tests/test_draft_networks.py tests/test_draft_data.py tests/test_draft_integration.py tests/test_maidata_entry.py tests/test_maidata_attention.py tests/test_maidata_encoder.py --basetemp .test-temp/draft
```

覆盖默认/非默认维度、全曲覆盖和负起点、唯一中心拼接、空尾窗/占位窗、NA/有效0、simai统计算法对照、变BPM、K/V复用的数值及梯度等价、D/R/P到E反传、外部消费者经Z和plan12分别回传、实际冻结、CPU bf16 autocast及无旧模块导入。simai源码对照仅在该只读仓库存在时执行，生产代码不依赖它。

已对既有原谱v2的 `10:5` 三个显式窗口导出真实标签。当前环境是CPU Torch；未执行CUDA、真实训练batch成本测量、训练、上传或主Decoder AR生成。
