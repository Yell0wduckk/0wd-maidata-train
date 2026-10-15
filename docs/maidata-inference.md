# 当前模型整曲推理与 maidata 物化

`python -m maidata.inference` 推荐使用单个联合 checkpoint：共享音频 Encoder E、Draft D、Planner P 与 Decoder M，生成五类新字段，实时转译完整音符并更新可播放的 `maidata.txt`。联合模型可用两层硬盘缓存复用 E 的 P0 和 D/P 的 Z、plan12。显式 `--decoder-checkpoint` 保留两阶段路线，E_D 服务固定 Draft/Planner，E_M 服务 M，本次缓存不支持此路线。`python -m maidata.inference.materialize` 独立把 IR 写成可播放的 `maidata.txt` 与配套音频。运行时不导入旧 `maimai_*` 包、不调用 MiaCode、不做回读。

窗口与历史遵循 [推理设计](../../0wd-v3-refactor-docs/new-inference.md)、[Decoder 设计第5～6节](../../0wd-v3-refactor-docs/new-decoder.md)以及[当前字段接口](chart-fields-physical.md)。

## 推理输入

- `--joint-checkpoint` 按保存的 `recipe` 构造 E/D/P/M，分别严格加载 `model_state` 中对应组件；E只构建一次，真实局部P0供D和M复用，包含完整空尾窗。`schema`、`architecture` 的训练版本标签仅作来源记录，可不同或缺省，不作为推理门槛；训练方式、优化器、训练计划和仅训练使用的辅助头也不限制推理。联合推理时不再传两个阶段checkpoint，其他参数沿用下方示例。见[联合训练](joint-training.md)。
- 两阶段路线的 Decoder checkpoint 提供 E_M/M。Z开启或使用预测plan时，还需另一 checkpoint 提供固定E_D/D/P。两个文件同样不限制训练版本标签或 joint/cached 等训练方式；所有必要组件必须有完整且兼容的权重。Z关闭且plan为gt/none时只需 E_M/M 权重。R不构建、不运行。
- 教师与M分别按各自checkpoint的`recipe`构造，Encoder参数和网络容量独立，不用M配置覆盖D/P。当前推理共用一份音频grid，因此两者的`samples_per_beat`和`mel_bins`必须一致；加载器在构建网络前报告不匹配字段及两侧值，Python直接调用`prepare_conditions`也遵守同一输入约束。目标窗长度须一致，启用Z时M的Z投影宽度须匹配教师；关闭Z且使用GT/无plan时不加载或检查教师。不把E_M替换成E_D，也不让D读取E_M的P0。加载器严格恢复各模块，不比较权重或源码哈希，不加载optimizer、训练计划或训练缓存清单。
- 兼容性依据实际网络配置、组件权重名称与形状，以及跨模型音频 / Z / 窗口维度。必要组件缺失、组件内部多余参数或形状不符仍报错，保持 `load_state_dict(strict=True)`，不随机补参数、忽略必要权重或补零扩维。旧连续时间头等真实架构差异仍无法加载。预编码只加载 E，预解码只加载 D/P，常驻会话与生成按所需组件使用相同规则；缓存自身的内容版本和身份匹配不变。本次仅调整推理，训练续训规则不变。
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
  --joint-checkpoint /path/to/joint/last.ckpt \
  --audio /path/to/song.wav \
  --timing /path/to/timing.json \
  --grid /path/to/audio-cache/grids/song.npz \
  --mel-root /path/to/referenced-mel \
  --geometry /path/to/physical/path-catalog \
  --slide-data /path/to/slide_data.json \
  --difficulty 5 --level 14.5 \
  --device cuda --precision bf16 --temperature 1 --seed 20261010 \
  --cache-dir /path/to/inference-cache \
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
  --ir /path/to/existing/model.ir.json \
  --audio /path/to/song.wav \
  --difficulty 5 --level 14.5 --title "曲名" --artist "作者" \
  --output /path/to/playable/song
```

上面的独立物化命令用于已有 IR；默认推理无需再执行此命令。物化使用纯 Python，不需要 MiaCode、checkpoint、Torch 或 GPU；`--miacode` 和 `--timeout` 已删除。两入口均支持 `--title/--artist/--designer/--cabinet`，difficulty/level 与推理请求保持一致。

## 联合模型预编码与预解码

`--cache-dir` 为可选参数。不传时照常在线计算；指定后按需补算缺失层，命中时不再读取 grid/Mel 载荷或提取音频特征。全部命中只构建、运行 M，不构建 E/D/P；修改 difficulty/level 复用音频层，仅重算 D/P。关闭 Z 且 plan 为 gt/none 时完全跳过 D/P。M 的音频投影及因果 KV 仍按本次生成计算，不保存在这两层缓存中。

也可以在生成谱面之前独立完成两步预计算，无需 Physical 几何、源谱或输出目录：

```bash
python -m maidata.inference.preencode \
  --joint-checkpoint /path/to/joint/last.ckpt \
  --audio /path/to/song.wav --timing /path/to/timing.json \
  --grid /path/to/audio-cache/grids/song.npz \
  --cache-dir /path/to/inference-cache --device cuda --precision bf16

# audio-cache 使用上一条 stdout JSON 中 audio_cache.path 的实际路径。
python -m maidata.inference.predraft \
  --joint-checkpoint /path/to/joint/last.ckpt \
  --audio-cache /path/to/inference-cache/audio/KEY.pt \
  --difficulty 5 --level 14.5 \
  --cache-dir /path/to/inference-cache --device cuda --precision bf16
```

`preencode` 仅构建 E，按原有 batch 顺序编码含完整空尾窗的局部 P0，并保存 D 所需全曲中心 P0；局部 P0 不从全曲中心裁剪。`predraft` 仅构建 D/P，只读取已编码缓存与联合权重，不读取原音频或 grid。两个入口重复调用也会命中；stdout 仅输出一次最终 JSON（路径、hit、窗口数），stderr 沿用进度输出，支持 `--no-progress`。不复制音频、不写谱面或 `generation.json`。

随后运行上面的普通推理命令并使用同一 `--cache-dir`，即可直接复用。更改 `--seed`、`--temperature`、生成上限或标题后再次推理，使用新的 `--output` 目录；这些参数不影响缓存身份。普通推理仍保留原音频、timing 和 grid/Mel 统计参数，用于选择正确缓存及写出配套音频。`generation.json.cache` 记录 audio/draft 的路径和命中情况；不需要 D/P 时 draft 为 null。`--force` 只覆盖生成目录的既有产物，不清理缓存。

音频项存放在 `audio/KEY.pt`，schema 为 `maidata.inference-audio.v1`；Draft 项位于 `draft/KEY.pt`，schema 为 `maidata.inference-draft-plan.v1`。文件保存完整身份、coverage、窗口起点、原 dtype 的 CPU 张量及完成状态，读取时迁移至目标设备。音频层身份包含完整 BPM/first、音频与特征来源、checkpoint、Encoder 配方（含 P0 归一化）、窗口 batch、设备和精度；Draft 层另含音频缓存 key、D/P 配方、Z 间隔和 difficulty/level。预测 plan 缓存不受 gt/none 覆盖影响。

来源只比较绝对路径、文件大小、纳秒修改时间及配方元数据；引用 Mel 的 grid 还读取小型侧车 JSON 并记录引用文件元数据。不扫描目录、不计算音频/权重/源码内容哈希；KEY 仅是小型身份文档的摘要。同路径替换文件须更新文件元数据。历史联合配方未记录 P0 归一化时沿用 False，不继承新训练默认值。

不同身份使用不同缓存项，每项通过同目录唯一临时文件原子替换，Windows 占用最多重试约一秒；未发布的临时文件不会命中。当前请求对应的音频或 Draft/Plan 项若损坏、截断，或身份/张量形状、数值、坐标不一致，会记录原因，只清理该文件，并按原请求重建一次。坏 Draft/Plan 复用有效音频，只运行 D/P；坏音频在预编码、普通推理或交互会话中使用原音频参数重跑 E。报告记录 `hit=false`、`invalidated=true` 和 `error`；修复后再次调用正常命中。权限、内存/显存、模型加载/计算和保存失败照常报错，不触发清理或循环重试。显式 `predraft --audio-cache` 只读指定音频文件，不能删除任意传入路径；该音频文件无效时需用原参数执行预编码，checkpoint/device/precision 不匹配也不删除。某层失败时保留已成功发布的其他层，下次调用可继续复用；不把预计算成功当成整曲生成成功。

Python 在线与缓存入口共用 `encode_joint_audio(...) -> (local_conditions, center_p0)` 和 `decode_joint_draft(..., local_conditions, center_p0, difficulty=..., level=...) -> Conditions`。`JointCheckpoint.models(device=..., components=...)` 严格恢复所选模块；原 `load_joint_models(...)` 默认仍加载 E/D/P/M。缓存不包含 R、IR 或可变生成状态。

## 配置与运行行为

| 参数 | 默认值与含义 |
|---|---|
| `--device / --precision` | cuda / bf16；CPU 可显式选择 cpu / fp32 |
| `--temperature` | 默认0，为确定性 argmax；正值为温度采样 |
| `--repetition-window` | 默认32；最近 n 个真实结构 token，仅对其中出现过的 pad / route 降权 |
| `--repetition-penalty` | 默认1；重复候选 logit 减去此值，0 关闭，同一值重复多次也只扣一次 |
| `--z-cross-attention / --no-z-cross-attention` | 默认沿用checkpoint；可显式关闭Z。不能给未训练Z投影的checkpoint启用Z |
| `--plan-source` | 默认predicted；gt读取`--gt-plan`逐窗JSON，none完全屏蔽plan key |
| `--gt-plan` | `{"windows":[{"start_beat":0,"plan12":[12个数值],"valid":true}]}`，覆盖含完整尾窗的全部目标窗 |
| `--seed` | 默认20261010；`-1` 在每次任务开始时随机抽取一次实际种子，写入 `generation.json.options.seed`；独立 CPU sampling generator，重试不重复同一候选 |
| `--encoder-window-batch` | 8；控制两套Encoder的局部窗口分块大小，不改变全曲P0/Z或AR目标窗 |
| `--cache-dir` | 默认不启用；仅联合模型，复用或自动补算音频、Draft/Plan 两层硬盘缓存 |
| `--max-tokens-per-window` | 512；当前新字段 token 数上限，不是旧行数或历史截断预算 |
| `--max-tokens-per-moment` | 20；不含MOMENT本身；达到阈值后停止扩展，只补齐当前动作必需的ROUTE/时值，再推进MOMENT或合法窗末终端；必要收束字段允许超出阈值 |
| `--max-backtracks-per-window` | 128；合法补全搜索的回退次数上限 |
| `--rollback-tokens` | 16；保留的搜索回退深度，不限制音符链表示或模型输入历史长度 |
| `--physical-hard-mask` | 开启；使用原始 Physical 拒绝位，完整动作提交时再次判定 |
| `--trace` | 默认关闭；显式开启才记录逐token选择/回退 |
| `--progress / --no-progress` | 默认开启；向stderr打印阶段、窗口与节流后的token进度；不启用详细trace |
| `--progress-interval` | 2秒；Encoder分块与窗内token进度的最短输出间隔，阶段和窗口起止立即输出 |
| `--stream-output / --no-stream-output` | 默认开启；完整音符/轨道分支与回滚立即更新同一路径的maidata.txt。关闭后仅在整曲成功时写出 |
| `--force` | 默认关闭；显式覆盖同名产物，不删除输出目录中的其他文件 |

网络容量、RoPE、音频余量、Z范围、谱面历史和目标窗口使用 checkpoint 保存的模型配方；时间词表固定为1..384 tick加CONT，共385类。目标 stride 必须等于目标长度，保证目标中心连续、无重复。换容量须使用相应训练权重，不能用 CLI 强行宽松加载。

联合路线共享 E 的真实局部 P0，中心拼接后供 D 读取；两阶段路线则由 E_D 与 E_M 分别计算自己的 G0 和 P0。D 一次生成含完整空尾窗的 Z，P 预测各窗 plan12，M 读取当前窗对应的局部 P0。内容重试及 Physical 回退不重跑 Encoder 或 D/P；联合路线启用硬盘缓存后可跨生成调用复用这些固定输入。

显式使用`--no-z-cross-attention --plan-source gt --gt-plan ...`，或`--no-z-cross-attention --plan-source none`时，仅加载阶段二E_M/M，不需要`--encoder-draft-checkpoint`，不构建或执行E_D/D/P。none将独立plan有效位设为false，不把全零向量当有效条件；Z和plan均关闭时条件attention没有有效key。GT与none报告分别记录来源、实际Z开关及每窗`plan_valid`。关闭Z的推理覆盖在严格加载原始checkpoint参数结构之后应用，不更改训练配置或checkpoint。

M使用独立CONT输入；续接锁定时间类型，每次CONT执行主干并追加KV，不调用Physical接受、不产生新起音。最终短值合成完整时间后提交一次真实动作。温度0逐步取合法argmax，正温度逐步采样；窗口报告记录time_sampling。Hold/Slide按NOTE written BPM换算秒并反算绝对拍，Slide首次仅调整后段音频/Z查询坐标。

默认先使用Physical hard mask；窗口回退预算耗尽后，从当前因果前缀仅放宽Physical拒绝，继续强制补齐和生成。每窗`physical_fallback`记录触发点与原因，实际接受的违规逐条记录在`physical_bypass`；下一窗重新开启请求的hard mask。语法、音频范围和基本时值支持始终生效。显式`--no-physical-hard-mask`则从开始放宽Physical。放宽后仍无合法语法补全、非有限预测或超过生成上限时报错并写`generation-error.json`，不将截断谱面标为成功。

经过判定区后的撞尾（`path_guard`）例外采用 NOTE 采样后的修复，不在 type / kind / pad / flags 选择时硬屏蔽。间隔按 Slide 实际进入该判定区至 NOTE 起音的秒数计算：`150 <= gap <= 200 ms` 自动设置 `ex=1`，保留 kind、pad、其他 flags 及后续 Hold / Slide 内容；`0 < gap < 150 ms` 放弃这个 NOTE，强制回到 MOMENT 并重新选择时间，不在原起音改选 Touch。已有 EX 也不能跳过150 ms以内的回退。多个路径同时命中时任一严重撞尾即回退，初次重试时间至少到所有命中路径的150 ms边界；不是将其他音符或未来窗口整体向后移动。Slide 起点 `launch_guard`、占手、重叠和其他 Physical 规则不变。

Hold / Touch Hold 与 Slide 最终 route 的最后判定区另有推理时概率修复：duration 完成后比较位置字段和时长的合法原概率；换 route 排除同终点并重推 duration，换 pad 保留 duration。判定、CONT 概率与状态合同见 [最终判定区修复](inference-hold-slide-final-area.md)。

若撞尾发生于 MOMENT 后的第一个 NOTE，撤回尚未落音的 MOMENT 再重采样，避免连续 MOMENT 或遗留空起音；若同一起音已有完成音符，则保留它们并选择下一 MOMENT。前缀、规则、K/V、流式文本和 IR 同步回退。每窗生成报告及交互结果的 `tail_repairs` 记录加 EX / MOMENT 修复，搜索撤回也撤销对应修复记录；这些有意修复不计为 Physical bypass。整曲、续写和范围生成共用该流程。

加 EX 不增加模型前向，只修改尚未送回模型的 NOTE。MOMENT 回退复用原前缀和 Router 前半层输出，但重新执行时间类型对应的后半层；延后起音的 NOTE 也需重新预测，长时值可能增加 CONT 步骤。不重跑 Encoder、Drafter / Planner 或整段前缀，不能将其计算量描述为完全不变。CPU 小模型测试核对了实际 begin / complete 调用次数及重算后 K/V 与接受序列的一致性。

## Physical 推理查询与复用

推理在选择每个 ROUTE 候选时提前检查第一段的 `path_coverage`：第一段路径、NOTE 出现时刻和一拍后的启动时刻不随 duration 或后续拼接变化，必然冲突的候选立即拒绝。即使恢复了旧的延迟前缀，也不能在该第一段上继续追加。保留原来的路径豁免、覆盖方向和时间边界；拼接后段的启动时间依赖总 duration 分配，仍由完整轨道检查。此提前拦截仅用于推理，训练的 AttributionState、归责和已生产支持缓存不变。

整曲与交互生成使用推理层的 `GenerationPhysicalState`，继承 `physical.inference.InferencePhysicalState` 的局部字段查询、存在性短路和前缀内缓存，只将上述 `path_guard` 留到采样后修复。原始 Physical oracle、flags无关的训练判定、训练 `AttributionState` 及既有支持缓存不变；正常推理不计算训练归责、继承冲突或冲突覆盖。

- 类型阶段只需确认存在一个可用 NOTE，不枚举全部 kind×pad；kind阶段找到可用位置即停止；pad阶段只查询已选kind。候选结果在同一前缀内复用，外观flags和track break共享查询，short标记仍独立检查音频边界。
- 查询、最终候选确认与接受复用同一前缀的规则结果。语法、动作、窗口边界或快照改变后刷新缓存，跨窗活动动作与回退语义保持；不按裁剪后的谱面历史重建Physical状态。
- 同一活动动作集合共用占手区间panel、handoff、几何与容量查询；尚未完成的NOTE→ROUTE→duration仍可复用这些区间。容量直接做资源匹配和过载判定，不枚举训练归因需要的最小冲突见证集合；branch逐条计占手，启动抵消按共同owner共享一次。
- Touch重复在位置与完整动作处检查，不在type/kind阶段提前禁用整个Touch类别。普通ROUTE保持未完成状态，允许继续拼接；分支、WiFi和完整动作仍受规则约束。
- 时值保留精确且可能不连续的合法区间，类型与内容共享已计算支持；最短候选合法时可立即确认存在，最短候选非法时仍求完整区间，不据此剪掉后面的合法值，不改回旧版单一 `safe_upper`。MOMENT终端及空窗仍可合法关闭，不要求窗内存在可放置NOTE。
- `time.discrete`直接索引完整整数支持，在每个CONT深度构造385类basic/raw mask；duration域交本地U和有效音频末尾。旧`inference.time_support`仅保留为历史连续分布工具，实际新推理不调用其4tick/CDF候选。

## 进度输出

CLI默认在stderr显示权重加载、音频输入、几何资源、Draft encoder与Decoder encoder各自的全曲准备/局部窗口、Draft、Planner、自回归生成与产物写入的开始和完成。自回归逐窗显示窗口序号、拍范围，窗内每隔约2秒显示已接受token数、回退次数和当前拍；即使未达到输出间隔，窗口完成也立即显示。stdout仍只保留原有最终JSON，重定向结果或脚本解析时无需滤除进度。使用`--no-progress`静默，或用`--progress-interval 5`降低刷新频率；这些设置不改变采样、约束、模型条件或详细trace。

Python接口`prepare_conditions(..., progress=None)`与`generate(..., progress=None)`默认静默；可传入接收事件字典的callback。事件包括`stage_start/stage_done/stage_error`、`encoder_progress`、`window_start/window_progress/window_done`。CLI reporter只读取已有的Python计数和单调时钟，不为进度读取GPU标量或调用`torch.cuda.synchronize()`。输出的时间用于观察壁钟进度，不是逐阶段精确CUDA基准。

## 实时谱面输出

CLI默认在自回归生成期间直接更新`output/maidata.txt`。Tap/Touch完成后立即显示，Hold/TouchHold等待完整时值，Slide等待完整轨道与时值；后续分支完成后更新同一个Slide。裸MOMENT、未完成音符或分支不进入预览。搜索恢复前缀时立即撤销被回滚的可见内容，已提交窗口保持。每次通过同目录临时文件原子替换，Windows读者短暂占用文件时有限重试；音频在生成前只复制一次。

预览的E位于最后可显示起音后一tick，BPM控制只包含该范围；first与最终输出一致。成功时按完整音频终点收尾并写出`generation.json`的`complete=true`。推理期间文件可能包含当前窗口的暂定音符，不能仅凭文件存在判断整曲完成。失败保留最后一份语法完整的预览，写出`generation-error.json`的`complete=false`并报错，不写成功报告。`--no-stream-output`恢复整曲成功后才写谱面的方式；与`--no-progress`、`--trace`独立。

Python接口`generate(..., stream=None)`与`decode_window(..., stream=None)`可接收事件字典callback，默认不发布。事件依次为`window_start`（window/start_tick）、`token`（window/token/token_count/complete）、`rollback`（window/token_count）；`generate`提交窗口后另发布`window_end`（window/groups）。token是当前新字段对象，complete表示语法处于done或track_done；rollback的token_count是保留前缀长度。终端和内部CONT不作为真实字段发布。`on_groups`仍只在成功提交窗口后调用，不受流式开关影响。`StreamingPreview`消费这些事件并持有SimaiWriter的提交；不能再将同一writer的add_groups同时绑定on_groups，以免重复提交。事件不写stdout，不使用旧行协议或外部物化进程。

## 新字段跨窗历史

1. 以真实起音选择 `[a-chart_history_beats,a)` 的完整组，包含所属 NOTE、连续/并行 ROUTE 与完整 duration。不设历史token预算，不把长动作拆成窗口内的短动作。
2. 真实起音组单独持有绝对tick和原始相邻起音delta；跨窗口首组的delta根据全曲上一真实起音重算。历史裁剪不改写首组delta，不把历史左边界当成上一事件。
3. 前窗外部边界和终端 `TimeTarget` 不进入真实谱面或历史。窗首直接生成NOTE会成为真实起音组，下次prefill恢复普通MOMENT。导出从真实起音的绝对位置确定间隔，不能只删除控制token后拼接。
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
| `streaming` | 当前窗口完整前缀预览、回滚撤销与窗口提交 |
| `render / simai / materialize` | 字段及已有IR的纯Python转译、持续拍号与四拍排版、原子文本写入及音频配套 |

默认推理将已接受的完整音符直接转为simai预览，跳过IR构造和最终token包重建；搜索回滚同步撤销预览。窗口成功结束后将完整真实起音组提交，回退、CONT、空窗和跨窗动作不会产生重复音符。最后补齐尾部和排版，产物为 `maidata.txt`、`track.<原扩展名>` 与 `generation.json`（权重来源、实际配方、窗口/预测plan12、终端、bypass、stream_output与是否完成）；不生成 `model.ir.json` 或 `model.tokens.json`。失败保留最后预览与 `generation-error.json`，不报告完整生成。

独立物化入口仍消费已有Chart IR，保留NOTE/flags、连续链、并行分支、WiFi、短hold、written-BPM时值和BPM/meter/HS控制的写法。超过4拍的空白间隔保持原有压缩和四拍换行，不移动真实事件。每行起始显式写出细分 `{n}`，同一行内才合并重复声明：连续24tick输出 `{16}1,1,`，72tick接24tick输出 `{16}1,,,1,`。跨越四拍边界的间隔按边界拆开，例如第3.5拍起的一拍间隔拆为行尾和下一行行首的两个八分间隔；拆分后按下一段的实际细分输出，不提前恢复旧细分，避免 `{4}{8}` 连写。整曲、流式预览、独立物化及交互片段共用该排版，片段依据自身绝对拍位换行。出现负拍起音时仍平移导出拍原点并同步调整 `&first` 和BPM控制，音符的音频绝对时间不变。

独立物化写出 `maidata.txt`、`track.<原扩展名>`、`render.ir.json` 和 `materialization.json`（v2，renderer为python-simai），不生成 `reparsed.ir.json`，不报告未经执行的roundtrip。音频原样复制，不裁切、不变速。运行时完全没有MiaCode render/dump步骤；独立解析器的语义对照仅在开发测试中执行。

## MiaCode 配对 IR 与交互会话

重复惩罚在合法性 mask 后、温度和 top-k / top-p 前应用；只匹配 NOTE 的实际 pad 标识或 ROUTE 的 route_id，不影响类型、flags、track break 或时间。历史包含已有真实前缀和本次接受的 MOMENT / NOTE / ROUTE / duration，不包含注入的窗口边界、内部 CONT 或失败候选。跨窗延续历史，回滚恢复对应前缀；关闭惩罚保持原采样和 RNG 行为。`GenerationOptions` 与报告中的 `options` 使用 `repetition_window/repetition_penalty`。

`--emit-ir` 为可选参数，独立 CLI 默认关闭；MiaCode 启用后每至多 250 ms 原子发布 `chart.snapshot.json`，字段为 `revision/chart/ir/complete`，成功结束强制同步并写出 `model.ir.json`。原 `maidata.txt` 输出保持兼容。IR 直接从完成的生成字段构造，已提交窗口复用已解码 IR，当前窗口仅包含完整前缀；回滚同时撤销文本和 IR，复用已加载 Geometry，不额外执行物化或原谱解析。

`python -m maidata.inference.interactive` 使用相同联合 checkpoint 和预计算目录，NDJSON `chart` 请求携带完整 `maimai.chart_ir.v1` 与 revision，后端确认 `chart_ready` 后可发送对应 revision 的 `suggest`。空谱也是有效初始请求。`suggest` 支持 continue/range、温度、top-k、top-p、seed；seed=-1 时每次请求独立抽取一次实际种子，result.seed 返回实际值。result 包含文本、区间及 `ir_fragment`，由编辑器在接受结果时合入本地缓存。取消或过期建议不修改正式谱面。

`suggest` 共用 `repetition_window=32/repetition_penalty=1`。continue 另支持 `max_tokens=32`、`max_beats=4`、`max_four_beat_blocks=1`，后两个为非负整数，0 关闭。起点为 b 拍时，拍数截止为 `floor(b)+max_beats`，四拍截止为 `4*floor(b/4)+4*max_four_beat_blocks`；第9拍开始、四拍段数1截止到12，段数2截止到16。最早的时间限制、下一真实音符、请求终点及音频末尾限制新起音；token 数到达后补完当前音符 / 分支，允许少量超出，已开始的 Hold / Slide 不因输出截止截短。range 不使用续写长度配置。

计算仍按全曲的完整16拍目标窗口（以 checkpoint 配方为准），输出截止单独传入，不改变音频、Z、Plan、Physical窗口和时长 cap；跨窗不以光标位置建立新窗口。移除隐含的两窗口上限。result 的 `limit_tick` 为有效时间上界，`stop_reason` 为 `token_limit/beat_limit/four_beat_limit/next_note/audio_end/requested_end`，同时记录惩罚参数。即使本段没有音符，仍可返回可接受的空拍文本，使空谱续写推进到下一段。接受时分割空白逗号区间，文本和 IR 保持一致，区间外音符不移动。

交互片段沿用全曲四拍换行；终点恰好落在四拍边界时保留结尾换行，不添加没有音符 / 间隔的下一行细分声明。后续片段声明自己的实际细分，连续接受不会将四拍边界接回上一行。

交互请求的 `plan_source` 支持 `predicted`（默认）和 `none`，分别使用预测 Plan 或完全屏蔽其条件。每次请求固定该值，续写与范围生成共用；切换时重建受影响窗口的条件记忆并清空其自注意力 K/V，保持音频投影、Z 和已预测的 Draft / Plan 缓存不变，不重跑 E/D/P。整曲入口继续使用 `--plan-source none`，预编码 / Draft / Plan 预解码仍照常输出。

窗口内使用真实前缀的最长公共部分复用 K/V；前缀修改使受影响后缀失效，跨窗口分别 prefill，Physical 由对应真实前缀恢复。`reused_rows/computed_rows` 统计前缀复用/补算行数，不含新生成字段的全部计算量。详细 trace 仍默认关闭。

## 验证范围与限制

小型真实模型的CPU执行覆盖v3严格权重加载、全曲条件复用、Encoder分块等价、预测plan12、增量/prefill一致、字段条件采样、CONT不写KV/Physical、窗尾控制、回退、生成上限、跨窗完整动作及连续空窗。测试使用原生MiaCode独立解析Python输出，对照全部71条路线及8个起点、flags、链/分支、时值和变BPM时间轴；CONT从不进入maidata。CPU测试不代表真实权重CUDA整曲生成或谱面质量验收。

这些测试使用小模型/合成输入，不代表已对真实训练checkpoint完成CUDA整曲生成、速度或谱面质量验收。实际谱面质量以用户评谱为准。


## 独立位置与形状重复惩罚

`--pad-repetition-window` / `--pad-repetition-penalty` 配置 note 位置惩罚；
`--route-repetition-window` / `--route-repetition-penalty` 配置 slide 形状惩罚。
两组分别默认32个真实结构 token、logit减1；力度0仅关闭对应组。
先应用合法性 mask，再按各自窗口扣减一次，随后执行温度和top-k/top-p，
不改变候选合法性或无放回机制。窗口均按全部真实结构 token 计数，
包含已有前缀和已接受内容，跨计算窗口保留历史；回滚、拒绝和人工修改沿用真实前缀规则。

原有 `--repetition-window` / `--repetition-penalty` 及 Python API 同名参数保留为共享默认值；
显式指定的pad/route参数优先（包括0）。历史保留两组窗口的最大值，各自只查询自己的尾部。
交互请求使用 `pad_repetition_window`、`pad_repetition_penalty`、
`route_repetition_window`、`route_repetition_penalty`，旧请求字段按相同优先级兼容。
生成报告和交互结果记录实际解析的四个值。两组力度均0时不改变原采样或RNG行为。


## Slide 判定区长度上限

`--max-slide-judge-areas`（交互请求 `max_slide_judge_areas`）默认为40，必须为正整数。
读取已加载几何中的判定区序列，逐条链式轨道累计；相接且相同的首尾判定区只计一次，
一个判定区包含多个 pad 仍算一个区。不按空间距离、route token数或时间长度计算。
累计恰好等于上限时仍可继续选完整一段；超过上限后下一字段只能为 `SLIDE_DURATION`，
不裁剪已选段，也不改变现有duration的合法性/时间/Physical规则。
新分支重新计数，搜索回滚按恢复的grammar前缀恢复长度；Physical fallback仍保留此上限。
只限制推理采样，不修改原谱解析、训练规则或已有权重。报告记录本次固定上限。

推理CLI、Python生成选项和常驻交互默认温度统一为0.4；位置和形状惩罚分别默认
32个结构token、力度1，均开启，显式力度0仍可分别关闭。
