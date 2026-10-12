# 当前模型整曲推理与 maidata 物化

`python -m maidata.inference` 加载当前 `maidata` 的两套独立音频 Encoder、Draft D、Planner P 与 Decoder M，生成五类新字段和 Chart IR。阶段一 E_D 服务固定 Draft/Planner，阶段二 E_M 服务 M 的直接音频输入。`python -m maidata.inference.materialize` 独立把 IR 写成可播放的 `maidata.txt` 与配套音频。两者不导入旧 `maimai_*` Python 包，不读取旧模型版本表；旧仓库仅作为整曲调度、MiaCode 调用和 simai 排版的实现参考。

窗口与历史遵循 [推理设计](../../0wd-v3-refactor-docs/new-inference.md)、[Decoder 设计第5～6节](../../0wd-v3-refactor-docs/new-decoder.md)以及[当前字段接口](chart-fields-physical.md)。

## 推理输入

- 联合入口的 `maidata.joint-training.v1` 可单独通过 `--joint-checkpoint` 加载；E只构建一次，真实局部P0供D和M复用，包含完整空尾窗。训练/验证GT plan、默认推理P预测的差异仍明确记录。联合推理时不再传两个阶段checkpoint，其他参数沿用下方示例。见[联合训练](joint-training.md)。
- 阶段二 `maidata.decoder-training.v4` checkpoint 提供联合训练后的 E_M/M。Z开启或使用预测plan时，还需阶段一 `maidata.encoder-draft-training.v3` checkpoint提供固定E_D/D/P；旧v1/v2及128维Z教师不通过隐式部分加载或补零扩维复用。Z关闭且plan为gt/none时只需阶段二权重。R不构建、不运行。
- 教师与M分别按各自checkpoint的`recipe`构造，Encoder参数和网络容量独立，不用M配置覆盖D/P。当前推理共用一份音频grid，因此两者的`samples_per_beat`和`mel_bins`必须一致；加载器在构建网络前报告不匹配字段及两侧值，Python直接调用`prepare_conditions`也遵守同一输入约束。目标窗长度须一致，启用Z时M的Z投影宽度须匹配教师；关闭Z且使用GT/无plan时不加载或检查教师。不把E_M替换成E_D，也不让D读取E_M的P0。加载器严格恢复各模块，不比较权重或源码哈希，不加载optimizer、训练计划或训练缓存清单。
- M要求架构身份`maidata.decoder-causal-cont-tick384.v2`，并严格加载v4的E_M/M成套参数。旧v1/v2连续时间头不兼容本次独立CONT序列与385类头，不能仅修改schema后加载或冒充完整轮续训。
- 阶段二默认GT plan，见[plan对照](decoder-plan-comparison.md)。推理默认读取P预测plan12；用户可显式指定gt或none作对照。`generation.json`分别记录`training_plan_source`与实际`inference_plan_source`，不会把GT结果标为predicted。
- 音频、完整 BPM/first 时间映射、目标 difficulty_id 与精确数值 level。无需源谱、真实 planner 或谱面结束时间。
- 二选一：`--grid` 直接读取既有训练格式音频 NPZ，或者 `--mel-stats` 使用训练时固定的 Mel 统计从原音频提取 Mel128/全曲 aux32。引用旧 Mel 的 grid 可用 `--mel-root` 搬迁路径。已验收 grid 按指定输入直接读取，不全量扫描或重哈希。
- `--geometry` 指向既有 Physical path-catalog；`--slide-data` 指向 MiaCode `slide_data.json`。只读资源，不重建几何。

所有文件路径显式配置，无远端或本机绝对路径内置默认值。使用 WAV 最便于和原输入对齐；音频解码由已有 soundfile 完成。新音频提取沿用 `maidata.preparation.features`，需要现有 `audio-inputs` 依赖。不能对新曲单独重估 Mel 的均值/方差。

## 命令

以下为 Linux 示例，路径替换为实际资产。在仓库根运行；Windows PowerShell 使用 `$env:PYTHONPATH='src'`，后续参数相同。

```bash
export PYTHONPATH=src
python -m maidata.inference \
  --encoder-draft-checkpoint /path/to/encoder-draft/last.ckpt \
  --decoder-checkpoint /path/to/decoder/last.ckpt \
  --audio /path/to/song.wav \
  --timing /path/to/timing.json \
  --grid /path/to/audio-cache/grids/song.npz \
  --mel-root /path/to/referenced-mel \
  --geometry /path/to/physical/path-catalog \
  --slide-data /path/to/slide_data.json \
  --difficulty 5 --level 14.5 \
  --device cuda --precision bf16 --temperature 1 --seed 20261010 \
  --output /path/to/generated/song
```

`--timing` JSON 示例：

```json
{"first_seconds": 0.125, "bpm": [[0, 1, 120], [64, 1, 180]]}
```

每个 BPM 项依次是拍坐标的分子、分母、BPM；首项拍坐标必须为0。恒定 BPM 可改用 `--bpm 120 --first-seconds 0.125`。输入 grid 必须是这首音频及该 timing 对应的 grid。直接从音频提取时，用 `--mel-stats /path/to/training-mel-stats.json` 替换 `--grid/--mel-root`。修改 first/BPM 后不能把旧 timing 的缓存重新标记为新 timing；调整偏移的推理可直接走 `--mel-stats`，按新 timing 提取输入。

`first_seconds` 是谱面第0拍对应的音频秒数。推理目标窗固定从第0拍开始，随后为16、32……拍（实际步长沿用checkpoint的目标窗长度），不再从音频缓存的最早拍开始。全曲Encoder输入和局部音频输入都排除负拍；第一个窗的前4拍位置仍保留固定布局，但Mel/aux32全零且mask无效，不输入first之前的音频帧。后续窗口仅使用第0拍及之后的历史音频。first为负时，第0拍仍是推理起点，音频第0秒之前的部分补零并标为无效，原音频不移动、不裁切。既有grid只在内存裁剪或补齐，不修改缓存、训练窗口或特征算法；Physical仍按原音频范围限制真实音符。

```bash
python -m maidata.inference.materialize \
  --ir /path/to/generated/song/model.ir.json \
  --audio /path/to/song.wav \
  --miacode /path/to/miacode_chart_ir \
  --difficulty 5 --level 14.5 --title "曲名" --artist "作者" \
  --output /path/to/playable/song
```

`--miacode` 可为 Linux 可执行文件、Windows `miacode_chart_ir.exe`，或其所在目录。物化入口不需要 checkpoint、Torch 或 GPU。difficulty/level 与推理请求保持一致。

## 配置与运行行为

| 参数 | 默认值与含义 |
|---|---|
| `--device / --precision` | cuda / bf16；CPU 可显式选择 cpu / fp32 |
| `--temperature` | 默认0，为确定性 argmax；正值为温度采样 |
| `--z-cross-attention / --no-z-cross-attention` | 默认沿用checkpoint；可显式关闭Z。不能给未训练Z投影的checkpoint启用Z |
| `--plan-source` | 默认predicted；gt读取`--gt-plan`逐窗JSON，none完全屏蔽plan key |
| `--gt-plan` | `{"windows":[{"start_beat":0,"plan12":[12个数值],"valid":true}]}`，覆盖含完整尾窗的全部目标窗 |
| `--seed` | 20261010；独立 CPU sampling generator，重试不重复同一候选 |
| `--encoder-window-batch` | 8；控制两套Encoder的局部窗口分块大小，不改变全曲P0/Z或AR目标窗 |
| `--max-tokens-per-window` | 512；当前新字段 token 数上限，不是旧行数或历史截断预算 |
| `--max-tokens-per-moment` | 20；不含MOMENT本身；达到阈值后停止扩展，只补齐当前动作必需的ROUTE/时值，再推进MOMENT或合法窗末终端；必要收束字段允许超出阈值 |
| `--max-backtracks-per-window` | 128；合法补全搜索的回退次数上限 |
| `--rollback-tokens` | 16；保留的搜索回退深度，不限制音符链表示或模型输入历史长度 |
| `--physical-hard-mask` | 开启；使用原始 Physical 拒绝位，完整动作提交时再次判定 |
| `--trace` | 默认关闭；显式开启才记录逐token选择/回退 |
| `--progress / --no-progress` | 默认开启；向stderr打印阶段、窗口与节流后的token进度；不启用详细trace |
| `--progress-interval` | 2秒；Encoder分块与窗内token进度的最短输出间隔，阶段和窗口起止立即输出 |
| `--force` | 默认关闭；显式覆盖同名产物，不删除输出目录中的其他文件 |

网络容量、RoPE、音频余量、Z范围、谱面历史和目标窗口使用 checkpoint 保存的模型配方；时间词表固定为1..384 tick加CONT，共385类。目标 stride 必须等于目标长度，保证目标中心连续、无重复。换容量须使用相应训练权重，不能用 CLI 强行宽松加载。

推理读取并准备一次音频输入，E_D与E_M分别计算自己的G0；可学习K/V和隐藏表示不共享。E_D按覆盖窗拼接全曲P0_D，D一次生成含完整空尾窗的Z，P预测各窗plan12。E_M按需要生成的目标窗编码局部24拍P0_M，含完整空尾窗，M只读取当前窗的对应bank。每窗选择固定Z上下文和plan12，内容重试及Physical回退不重跑Encoder或D/P。

显式使用`--no-z-cross-attention --plan-source gt --gt-plan ...`，或`--no-z-cross-attention --plan-source none`时，仅加载阶段二E_M/M，不需要`--encoder-draft-checkpoint`，不构建或执行E_D/D/P。none将独立plan有效位设为false，不把全零向量当有效条件；Z和plan均关闭时条件attention没有有效key。GT与none报告分别记录来源、实际Z开关及每窗`plan_valid`。关闭Z的推理覆盖在严格加载原始checkpoint参数结构之后应用，不更改训练配置或checkpoint。

M使用独立CONT输入；续接锁定时间类型，每次CONT执行主干并追加KV，不调用Physical接受、不产生新起音。最终短值合成完整时间后提交一次真实动作。温度0逐步取合法argmax，正温度逐步采样；窗口报告记录time_sampling。Hold/Slide按NOTE written BPM换算秒并反算绝对拍，Slide首次仅调整后段音频/Z查询坐标。

默认先使用Physical hard mask；窗口回退预算耗尽后，从当前因果前缀仅放宽Physical拒绝，继续强制补齐和生成。每窗`physical_fallback`记录触发点与原因，实际接受的违规逐条记录在`physical_bypass`；下一窗重新开启请求的hard mask。语法、音频范围和基本时值支持始终生效。显式`--no-physical-hard-mask`则从开始放宽Physical。放宽后仍无合法语法补全、非有限预测或超过生成上限时报错并写`generation-error.json`，不将截断谱面标为成功。

## Physical 推理查询与复用

推理在选择每个 ROUTE 候选时提前检查第一段的 `path_coverage`：第一段路径、NOTE 出现时刻和一拍后的启动时刻不随 duration 或后续拼接变化，必然冲突的候选立即拒绝。即使恢复了旧的延迟前缀，也不能在该第一段上继续追加。保留原来的路径豁免、覆盖方向和时间边界；拼接后段的启动时间依赖总 duration 分配，仍由完整轨道检查。此提前拦截仅用于推理，训练的 AttributionState、归责和已生产支持缓存不变。

整曲生成使用独立的 `physical.inference.InferencePhysicalState`，迁移旧仓库的局部字段查询、存在性短路和前缀内缓存。训练的 `AttributionState` 继续维护首次归责与惩罚支持；正常整曲推理只读取原始规则决定，不计算训练归责、继承冲突或冲突覆盖。

- 类型阶段只需确认存在一个可用 NOTE，不枚举全部 kind×pad；kind阶段找到可用位置即停止；pad阶段只查询已选kind。候选结果在同一前缀内复用，外观flags和track break共享查询，short标记仍独立检查音频边界。
- 查询、最终候选确认与接受复用同一前缀的规则结果。语法、动作、窗口边界或快照改变后刷新缓存，跨窗活动动作与回退语义保持；不按裁剪后的谱面历史重建Physical状态。
- 同一活动动作集合共用占手区间panel、handoff、几何与容量查询；尚未完成的NOTE→ROUTE→duration仍可复用这些区间。容量直接做资源匹配和过载判定，不枚举训练归因需要的最小冲突见证集合；branch逐条计占手，启动抵消按共同owner共享一次。
- Touch重复在位置与完整动作处检查，不在type/kind阶段提前禁用整个Touch类别。普通ROUTE保持未完成状态，允许继续拼接；分支、WiFi和完整动作仍受规则约束。
- 时值保留精确且可能不连续的合法区间，类型与内容共享已计算支持；最短候选合法时可立即确认存在，最短候选非法时仍求完整区间，不据此剪掉后面的合法值，不改回旧版单一 `safe_upper`。MOMENT终端及空窗仍可合法关闭，不要求窗内存在可放置NOTE。
- `time.discrete`直接索引完整整数支持，在每个CONT深度构造385类basic/raw mask；duration域交本地U和有效音频末尾。旧`inference.time_support`仅保留为历史连续分布工具，实际新推理不调用其4tick/CDF候选。

## 进度输出

CLI默认在stderr显示权重加载、音频输入、几何资源、Draft encoder与Decoder encoder各自的全曲准备/局部窗口、Draft、Planner、自回归生成与产物写入的开始和完成。自回归逐窗显示窗口序号、拍范围，窗内每隔约2秒显示已接受token数、回退次数和当前拍；即使未达到输出间隔，窗口完成也立即显示。stdout仍只保留原有最终JSON，重定向结果或脚本解析时无需滤除进度。使用`--no-progress`静默，或用`--progress-interval 5`降低刷新频率；这些设置不改变采样、约束、模型条件或详细trace。

Python接口`prepare_conditions(..., progress=None)`与`generate(..., progress=None)`默认静默；可传入接收事件字典的callback。事件包括`stage_start/stage_done/stage_error`、`encoder_progress`、`window_start/window_progress/window_done`。CLI reporter只读取已有的Python计数和单调时钟，不为进度读取GPU标量或调用`torch.cuda.synchronize()`。输出的时间用于观察壁钟进度，不是逐阶段精确CUDA基准。

## 新字段跨窗历史

1. 以真实起音选择 `[a-chart_history_beats,a)` 的完整组，包含所属 NOTE、连续/并行 ROUTE 与完整 duration。不设历史token预算，不把长动作拆成窗口内的短动作。
2. 真实起音组单独持有绝对tick和原始相邻起音delta；跨窗口首组的delta根据全曲上一真实起音重算。历史裁剪不改写首组delta，不把历史左边界当成上一事件。
3. 前窗外部边界和终端 `TimeTarget` 不进入真实谱面或历史。窗首直接生成NOTE会成为真实起音组，下次prefill恢复普通MOMENT。最终IR也从整曲真实起音重建delta，不能只删除控制token后拼接。
4. 新窗边界delta始终从全曲最后真实起音计算；例如最后真实起音20拍，32拍窗的前4拍历史为空，边界仍保留20→32的时间。连续空窗不更新最后真实起音。推理不启用历史dropout。
5. 切窗丢弃旧谱面K/V，使用当前Z/plan重新prefill全部历史。历史最后token的后继类型固定为边界MOMENT；窗口内才增量复用K/V。
6. Physical状态独立持续存在；历史prefill不重复提交动作。跨窗hold/slide、释放和交接关系按真实时间保留，即使其起音已不在4拍历史内。切窗前必须完成当前音符及路线字段，duration不因越过窗尾终止或截短。
7. 时间短值到达窗尾b或CONT临时游标首次跨b只关当前窗，不预约下窗事件；CONT终止分别记录`closed_by_continue`、`terminal_lower_bound_tick`和`terminal_query_tick`。U为b加音频后余量，duration候选同时受U及有效音频限制。始终覆盖真实音频及一个额外完整空尾窗，不因休止提前结束歌曲。

## 模块与产物

| 模块 | 职责 |
|---|---|
| `checkpoint` | 显式权重与模型配方加载 |
| `conditioning` | 共享输入/布局，分别计算E_D与E_M全曲P0，再从E_D计算D/P；不调用M或谱面标签 |
| `sampling` | Router/内容分层采样、385类时间字段内部循环、完整值规则支持 |
| `time.discrete` | 与训练/迁移共用的时间编码、支持和临时query坐标 |
| `physical.inference` | 推理专用raw规则查询、局部字段支撑及前缀缓存 |
| `generation` | 全曲窗口、真实历史、prefill与增量K/V、同步规则快照回退 |
| `materialize / simai` | 原生MiaCode调用、simai排版、语义回读、音频配套 |

推理写出 `model.tokens.json`（字段/NA/起始锚点）、`model.ir.json`、`generation.json`（权重路径与来源、实际配方、窗口/预测plan12、终端、bypass与是否完成）。不将源码来源说明变为加载门禁。

物化先调用MiaCode render，再dump回读，比较新字段的绝对起音、NOTE/flags、路线顺序/分支、duration及BPM时间轴；允许each组内原生排序和派生180Hz投影取整。超过4拍的空白间隔沿原仓库方式压缩，按4拍换行，不移动真实事件。出现负拍起音时平移导出拍原点并同步调整 `&first` 和BPM控制，音符的音频绝对时间不变。

成功写出 `maidata.txt`、`track.<原扩展名>`、`render.ir.json`、`reparsed.ir.json` 和 `materialization.json`。音频原样复制，不裁切、不变速；连续链保持共同head，分支保持并行，WiFi/短hold恢复为对应simai写法。

## 验证范围

小型真实模型的CPU执行覆盖v3严格权重加载、全曲条件复用、Encoder分块等价、预测plan12、增量/prefill一致、字段条件采样、CONT不写KV/Physical、窗尾控制、回退、生成上限、跨窗完整动作及连续空窗。原生MiaCode保持完整语义token消费；CONT从不进入IR/maidata。CPU测试不代表真实权重CUDA整曲生成或谱面质量验收。

这些测试使用小模型/合成输入，不代表已对真实训练checkpoint完成CUDA整曲生成、速度或谱面质量验收。实际谱面质量以用户评谱为准。
