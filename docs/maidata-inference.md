# 当前模型整曲推理与 maidata 物化

`python -m maidata.inference` 加载当前 `maidata` 的 Encoder、Draft D、Planner P 与 Decoder M，生成五类新字段和 Chart IR。`python -m maidata.inference.materialize` 独立把 IR 写成可播放的 `maidata.txt` 与配套音频。两者不导入旧 `maimai_*` Python 包，不读取旧模型版本表；旧仓库仅作为整曲调度、MiaCode 调用和 simai 排版的实现参考。

窗口与历史遵循 [推理设计](../../0wd-v3-refactor-docs/new-inference.md)、[Decoder 设计第5～6节](../../0wd-v3-refactor-docs/new-decoder.md)以及[当前字段接口](chart-fields-physical.md)。

## 推理输入

- 显式指定两份权重：阶段一 `maidata.encoder-draft-training.v1` checkpoint 提供 E/D/P；阶段二 `maidata.decoder-training.v1` checkpoint 提供 M。R不构建、不运行。
- 网络按 checkpoint 的 `recipe` 构建；E/D/P与M的表示参数须一致，权重严格加载。不比较源码哈希，不加载 optimizer、训练计划或训练缓存清单。
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

每个 BPM 项依次是拍坐标的分子、分母、BPM；首项拍坐标必须为0。恒定 BPM 可改用 `--bpm 120 --first-seconds 0.125`。输入 grid 必须是这首音频及该 timing 对应的 grid。直接从音频提取时，用 `--mel-stats /path/to/training-mel-stats.json` 替换 `--grid/--mel-root`。

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
| `--temperature` | 1；0 为确定性 argmax，正值为温度采样 |
| `--seed` | 20261010；独立 CPU sampling generator，重试不重复同一候选 |
| `--encoder-window-batch` | 8；控制 E 分块大小，不改变全曲 P0/Z 或 AR 目标窗 |
| `--max-tokens-per-window` | 512；当前新字段 token 数上限，不是旧行数或历史截断预算 |
| `--max-backtracks-per-window` | 128；合法补全搜索的回退次数上限 |
| `--rollback-tokens` | 16；保留的搜索回退深度，不限制音符链表示或模型输入历史长度 |
| `--physical-hard-mask` | 开启；使用原始 Physical 拒绝位，完整动作提交时再次判定 |
| `--trace` | 默认关闭；显式开启才记录逐token选择/回退 |
| `--force` | 默认关闭；显式覆盖同名产物，不删除输出目录中的其他文件 |

网络容量、RoPE、time mixture、音频余量、Z范围、谱面历史和目标窗口使用 checkpoint 保存的模型配方；没有第二套推理架构。目标 stride 必须等于目标长度，保证目标中心连续、无重复。旧配方没有 stride 字段时使用目标长度。换容量须使用相应训练权重，不能用 CLI 强行宽松加载。

推理仅计算一次全曲 G0；每个音频覆盖窗编码一次，提取中心 P0拼接成全曲bank；D一次生成含额外完整尾窗的Z，P对各目标窗预测plan12。M的音频投影按曲复用，每窗选择Z上下文和plan12。窗口切换、内容重试、Physical回退均不重跑E/D/P。

若显式使用 `--no-physical-hard-mask`，语法和音频范围仍生效，实际接受的Physical违规记录在 `physical_bypass`，报告明确记录关闭状态。默认运行不存在自动关约束或静默接受违规的回退。搜索失败、非有限预测或超过生成上限报错并写 `generation-error.json`，不将截断谱面标为成功。

## 新字段跨窗历史

1. 以真实起音选择 `[a-chart_history_beats,a)` 的完整组，包含所属 NOTE、连续/并行 ROUTE 与完整 duration。不设历史token预算，不把长动作拆成窗口内的短动作。
2. 真实起音组单独持有绝对tick和原始相邻起音delta；跨窗口首组的delta根据全曲上一真实起音重算。历史裁剪不改写首组delta，不把历史左边界当成上一事件。
3. 前窗外部边界和终端 `TimeTarget` 不进入真实谱面或历史。窗首直接生成NOTE会成为真实起音组，下次prefill恢复普通MOMENT。最终IR也从整曲真实起音重建delta，不能只删除控制token后拼接。
4. 新窗边界delta始终从全曲最后真实起音计算；例如最后真实起音20拍，32拍窗的前4拍历史为空，边界仍保留20→32的时间。连续空窗不更新最后真实起音。推理不启用历史dropout。
5. 切窗丢弃旧谱面K/V，使用当前Z/plan重新prefill全部历史。历史最后token的后继类型固定为边界MOMENT；窗口内才增量复用K/V。
6. Physical状态独立持续存在；历史prefill不重复提交动作。跨窗hold/slide、释放和交接关系按真实时间保留，即使其起音已不在4拍历史内。切窗前必须完成当前音符及路线字段，duration不因越过窗尾终止或截短。
7. 终端候选到达窗尾b或U尾部只关当前窗，不预约下窗事件；U为b加音频后余量，不裁到音频末端。始终覆盖真实音频及一个额外完整空尾窗，不因休止提前结束歌曲。

## 模块与产物

| 模块 | 职责 |
|---|---|
| `checkpoint` | 显式权重与模型配方加载 |
| `conditioning` | 特征适配、一次性全曲E/D/P；不调用M或谱面标签 |
| `sampling` | Router/内容分层采样、时间cell积分、候选规则支持 |
| `generation` | 全曲窗口、真实历史、prefill与增量K/V、同步规则快照回退 |
| `materialize / simai` | 原生MiaCode调用、simai排版、语义回读、音频配套 |

推理写出 `model.tokens.json`（字段/NA/起始锚点）、`model.ir.json`、`generation.json`（权重路径与来源、实际配方、窗口/预测plan12、终端、bypass与是否完成）。不将源码来源说明变为加载门禁。

物化先调用MiaCode render，再dump回读，比较新字段的绝对起音、NOTE/flags、路线顺序/分支、duration及BPM时间轴；允许each组内原生排序和派生180Hz投影取整。超过4拍的空白间隔沿原仓库方式压缩，按4拍换行，不移动真实事件。出现负拍起音时平移导出拍原点并同步调整 `&first` 和BPM控制，音符的音频绝对时间不变。

成功写出 `maidata.txt`、`track.<原扩展名>`、`render.ir.json`、`reparsed.ir.json` 和 `materialization.json`。音频原样复制，不裁切、不变速；连续链保持共同head，分支保持并行，WiFi/短hold恢复为对应simai写法。

## 验证范围

小型真实E/D/P/M的CPU执行覆盖权重加载、全曲条件复用、E分块等价、预测plan12、增量/prefill一致、字段条件采样、回退、生成上限、跨窗完整动作及连续空窗。原生MiaCode覆盖复杂链/分支/WiFi、短hold、变BPM、负拍与空谱的完整导出回读。入口测试实际串联 checkpoint→grid→模型→IR→MiaCode→maidata与音频；原始音频提取另验证固定训练特征算法。

这些测试使用小模型/合成输入，不代表已对真实训练checkpoint完成CUDA整曲生成、速度或谱面质量验收。实际谱面质量以用户评谱为准。
