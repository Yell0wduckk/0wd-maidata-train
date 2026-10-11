# 基础窗口与复杂节奏采样

`maidata.sampling` 将固定歌曲／谱面清单转换为唯一基础窗口和逐轮监督实例。它只读取所选原谱缓存，不持有模型，不提取音频，不重建 Physical。可读取既有训练／验证划分；训练循环和 batch 大小独立设置。复杂窗口过采样只用于主Decoder监督，Encoder＋Draft阶段一的R/P全部基础窗各监督一次。

## 入口与配置

从独立源码仓库根目录运行：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
./.venv-audio/Scripts/python.exe -m maidata.sampling --selection ../artifacts/pilot320-master-20261010/selection.json --split ../artifacts/pilot320-master-20261010/encoder-draft-split.json --partition all --output ../artifacts/pilot320-sampling-20261010 --seed 20261010 --epoch 1
```

架构及采样默认值统一读取 `src/maidata/entry.py` 的普通 `CONFIG`；CLI 可显式覆盖本次计划的参数，实际值写入 `plan.json`。

| 配置／CLI参数 | 默认值 | 作用 |
|---|---:|---|
| `target_beats` / `--target-beats` | 16 | 目标窗口长度，整数拍 |
| `target_stride_beats` / `--target-stride-beats` | 16 | 监督窗口步长，满足 `0<stride<=target` |
| `audio_before_beats` / `--audio-before-beats` | 4 | 前侧音频上下文 |
| `audio_after_beats` / `--audio-after-beats` | 4 | 后侧音频上下文及终端观察上限 |
| `chart_history_beats` / `--chart-history-beats` | 4 | 谱面历史区间长度 |
| `complex_rhythm_min_kinds` / `--complex-rhythm-min-kinds` | 3 | 复杂节奏所需的最少短 delta 种类 |
| `complex_rhythm_multiplier` / `--complex-rhythm-multiplier` | 2.0 | 主Decoder复杂窗口总曝光倍率，至少1，可为非整数；E＋Draft计划显式用1 |
| `--seed` | 20261010 | 抽取与排列的随机种子 |
| `--epoch` | 1 | 从1开始的轮号，参与确定性排列和非整数倍率的子集选择 |

音频余量、历史长度均为非负整数拍。默认目标窗连续且不重叠；改为小于 target 的 stride 时监督窗可以重叠，但独立全曲音频中心仍按 target 步进。没有 Slide 额外加权参数。

`--selection` 读取 `selected_chart_ids` 和同序的 `charts`，因此 pilot320 的23份 Re:Master 与320份 Master 都会进入计划。默认缓存根由清单的 `sources.chart_catalog` 定位；迁移到其他机器时可传 `--chart-cache` 指向对应缓存，不需要改写已选歌曲。

### 使用既有训练／验证划分

`--partition` 默认为 `all`，表示完整清单概览。选择 `train` 或 `validation` 必须同时传入 `--split`，读取既有 `maidata.encoder-draft-split.v1`，不重新选歌。只读取当前侧的谱面载荷；核对两侧谱面覆盖与互斥，以及歌曲、audio_asset、audio_variant 不跨侧，使用的都是已有标识元数据。

主Decoder训练侧使用配置中的复杂窗口倍率；验证侧固定为基础窗口1倍，即使显式传入其他倍率也不会追加实例，实际生效值写入配置。以下命令为主Decoder计划，沿用固定的288首训练／32首验证划分：

```powershell
./.venv-audio/Scripts/python.exe -m maidata.sampling --selection ../artifacts/pilot320-master-20261010/selection.json --split ../artifacts/pilot320-master-20261010/encoder-draft-split.json --partition train --output ../artifacts/pilot320-sampling-20261010/train --seed 20261010 --epoch 1
./.venv-audio/Scripts/python.exe -m maidata.sampling --selection ../artifacts/pilot320-master-20261010/selection.json --split ../artifacts/pilot320-master-20261010/encoder-draft-split.json --partition validation --output ../artifacts/pilot320-sampling-20261010/validation --seed 20261010 --epoch 1
```

`all` 计划仍对全部复杂窗应用配置倍率，供主Decoder全清单统计；主Decoder消费相应 `train`／`validation` 计划。Encoder＋Draft使用[独立阶段一计划](../../artifacts/pilot320-encoder-draft-sampling-20261010/README.md)，训练7924实例、验证904实例，无追加过采样；可以从现成all窗口元数据将倍率显式设为1后用 `partition_plan` 派生，不必重新读取原谱载荷。直接构建新的E＋Draft计划时须传 `--complex-rhythm-multiplier 1`。阶段一不以重复R/P窗口或额外loss倍率代替过采样。

## 基础窗口与时间坐标

使用每份谱面绑定的完整 TimingMap，令 `b0=floor(B(0秒))`、`e=B(audio_end_seconds)`，目标长度为 T、监督步长为 S：

```text
N = max(1, ceil((e-b0-T)/S)+1)
音频覆盖监督窗起点：b0+i*S，0<=i<N
额外完整空尾窗起点：最后一个音频覆盖监督窗起点+T
```

所有窗口为 `[start,end)`，从歌曲自己的首个拍网格开始；保留负拍起点，不对齐到全局16拍倍数。必须覆盖整首音频并额外保留一个完整目标空尾窗，不能以最后 NOTE 的起音、结束或最后一个非空窗口截短。开头空白、中间休止、持续动作期间没有新起音的窗口，以及最后 NOTE 之后的全部窗口都保留。

窗口内只存引用与坐标，音频上下文为 `[start-before,end+after)`，终端观察上限 `U=end+after`，不会裁到真实音频末端。历史区间为 `[start-history,start)`，不会在曲首强行裁到0。这里不物化谱面历史、边界 token、终端时间标签或历史 dropout；消费侧按字段／Decoder接口处理，不能将这些区间当作所有音频位置都有效。

独立音频组记录 `encoder_starts_beats`：以 T 步进覆盖真实音频，供消费侧建立全曲 P0，不为额外空尾窗增加一个音频中心。`actual_audio_start_beat`、`actual_audio_end_beat`、`span_beats`来自真实音频时轴；`draft_end_beat`描述当前默认 Draft 覆盖终点，`supervision_end_beat`描述全部监督窗口终点。

非默认重叠布局可能需要更长的 Draft 查询域。例如 T=16、S=10、音频在31拍结束时，监督起点为0、10、20，额外尾窗为 `[36,52)`，而默认不重叠 Draft 域到48拍。采样清单完整保留52拍终点；接入此配置时消费侧须扩展有效 Draft 域，不能裁掉监督尾窗或改变全曲 P0 的中心覆盖。默认 T=S=16 时两种终点一致。

## 复杂节奏与重复实例

只取 `tokens` 中 NOTE 对应的整数 `token_map.tick`，按窗口去重、排序后计算相邻真实起音的差。保留 `0<delta_tick<96` 的不同整数值，种类数大于等于3为复杂。MOMENT、窗首偏移、跨窗间隔、ROUTE、duration 和终端查询均不贡献起音或 delta；同刻多押只算一个起音，变 BPM 不改变 tick 种类。

每个基础窗口都有 `occurrence=0`。主Decoder训练默认复杂窗口再增加 `occurrence=1`，总计2次；其余窗口总计1次。含 Slide 的复杂窗口仍为2次，只有 Slide 的窗口没有额外曝光。重复实例引用同一 `window_id`，不复制谱面载荷或改变窗口边界。可选联合微调也只增加M的监督，R/P仍按唯一基础窗归约。

非整数倍率 m 的处理为：每个复杂窗口先获得 `floor(m)` 次总曝光，再从全部复杂窗口中不放回抽取 `round_half_up(frac(m)*复杂窗口数)` 个，各增加一次。全部基础窗口始终保留，不使用有放回加权抽样替代基础遍历。

同一 `audio_variant` 的所有谱面、窗口和重复实例集中成一个完整音频组。seed 和 epoch 决定组间、组内排列；相同输入、配置、seed、epoch 可复现相同结果。组不是 batch，不自行设置 batch 边界。共享音频不代表共享不同难度／等级的 Draft Z，梯度及参数更新后的复用边界仍遵守[训练计划](../../0wd-v3-refactor-docs/training-plan.md#6-计算复用与阶段成本)。

## 输出与接口

| 文件 | 内容 |
|---|---|
| `plan.json` | 配置、seed、epoch、来源引用、总计／分难度／逐谱统计、完整音频组及其实例偏移 |
| `windows.jsonl` | 每个唯一基础窗口一行，含区间、上下文、历史、空尾窗、起音数、Slide及复杂节奏统计 |
| `occurrences.jsonl` | 按音频组排列的监督实例；`window_id` 引用基础窗，`occurrence`区分基础和追加，`sequence_index`为全局零基顺序 |

音频组的 `[occurrence_start,occurrence_stop)` 是零基、左闭右开区间。同一组的 `chart_ids` 和完整 `encoder_starts_beats` 供后续训练数据适配使用。`policy.split` 记录 `all/train/validation`；存在划分文件时保留 `sources.split` 引用。逐谱 `selection_index` 和 `chart_index` 保留原清单的一基歌曲／谱面编号，分侧后不重编号。没有预填 DataLoader、optimizer 或 checkpoint 状态；不同轮号的结果应保存在各自目录，实际完整轮恢复保留所绑定计划。

Python 接口为 `build_plan(selection_path, config=..., seed=..., epoch=..., chart_cache=..., split=..., partition='all')`、`chart_windows(record, config=...)`、`make_occurrences(windows, multiplier=..., seed=..., epoch=...)` 和 `write_plan(plan, output)`。`partition_plan(plan, split_path, partition=...)` 可从包含 `windows`／`occurrences` 的完整 all 内存计划派生分侧计划，只读 split JSON，重新计算实例、统计及组偏移，不重读谱面载荷或修改输入计划。

`build_plan`只读取所选侧谱面的 gzip 载荷，保留普通 schema/view、映射形状、整数 tick、来源／TimingMap绑定检查；不逐文件重哈希或重新跑完整语法／Physical验收。真实 NOTE 落在基础覆盖范围外或额外空尾窗时报告具体谱面错误，不静默丢掉。

## pilot320 结果

当前[采样计划](../../artifacts/pilot320-sampling-20261010/plan.json)使用固定320首、343份原谱，默认配置、seed=20261010、epoch=1：

| 难度 | 基础窗口 | 复杂窗口／追加实例 | 总实例 |
|---|---:|---:|---:|
| Master | 8,251 | 1,442 | 9,693 |
| Re:Master | 577 | 145 | 722 |
| 合计 | 8,828 | 1,587 | 10,415 |

基础窗口含656个无新起音窗口，其中343个为额外完整空尾窗；all概览的采样实例增加17.98%，分为320个完整音频组。这是采样实例数量，不是训练耗时估算。

已按固定划分生成主Decoder计划；Encoder＋Draft的独立计划使用同一批基础窗：

| 计划 | 歌曲／谱面 | 基础窗口 | 复杂窗口 | 追加实例 | 总实例 |
|---|---:|---:|---:|---:|---:|
| [训练](../../artifacts/pilot320-sampling-20261010/train/plan.json) | 288／307 | 7,924 | 1,418 | 1,418 | 9,342 |
| [验证](../../artifacts/pilot320-sampling-20261010/validation/plan.json) | 32／36 | 904 | 169 | 0 | 904 |
| [E＋Draft训练](../../artifacts/pilot320-encoder-draft-sampling-20261010/train/plan.json) | 288／307 | 7,924 | 1,418 | 0 | 7,924 |
| [E＋Draft验证](../../artifacts/pilot320-encoder-draft-sampling-20261010/validation/plan.json) | 32／36 | 904 | 169 | 0 | 904 |

训练侧含589个无新起音窗口，其中307个为完整空尾窗；验证侧分别为67和36。合成检查覆盖完整音频、空尾窗、负拍、变 BPM、窗口边界、复杂节奏阈值、非整数倍率、确定性、音频组、分侧载荷读取及仅从计划派生，采样与CONFIG入口共16项定向CPU检查通过。真实计划已生成并独立复核，主Decoder训练已通过 maidata.decoder_training.plan 消费既有计划并复用 make_occurrences；本次不重新生成计划。
