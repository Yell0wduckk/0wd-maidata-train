# 当前源码状态与实现边界

本文件区分已实现的新 `maidata` 音频 Encoder 和独立副本保留的旧模型。Encoder 规格位于[设计正本](../../0wd-v3-refactor-docs/new-audio-encoder.md)，模块、张量协议和 aux32 生产约定位于[实施文档](../../0wd-v3-refactor-docs/encoder-execution-plan.md)。来源与复制排除项见 [import-and-layout](import-and-layout.md)。

## 新 maidata：独立音频 Encoder

入口为 `python -m maidata`，调用方式与 CPU 验证命令见 [maidata-entry](maidata-entry.md)。`entry.py` 顶部普通 `CONFIG` 集中18项架构参数，构造器无第二套默认值；`--help` 与 `--show-config` 仅依赖标准库。默认命令延迟导入 Torch，成功构建 `AudioEncoder` 并提示 Decoder 与训练未实现。

当前代码位于 `src/maidata/audio/{types,frontend,geometry,attention,blocks,pyramid,encoder}.py`，不导入旧 `maimai_baseline`、`maimai_dataset` 或朋友仓库。已实现：

- 独立频谱 tokenizer、aux32 融合与逐拍 G0 tokenizer；频率 query 和容量由配置传入。
- `prepare_song` 按曲计算 G0 及非末块的 K/V，保持计算图；`encode_windows` 按绝对拍坐标初始化每窗 S，执行共享参数、同步读取的 L/S Conformer。
- 局部联合 attention、完整 head RoPE、L 专属时间卷积、两次 stride transition，以及按坐标插值的局部 FPN。
- `forward(song_inputs, window_inputs)` 组合前两步，只返回命名的 P0/P1/P2，各自携带 values、beats、valid。

默认24拍输入的内部 C 长度为576/288/144，公开 P 长度为192/96/48；其他合法容量、深度和窗口由同一配置推导。没有 C0/C1/C2 高分辨率直出、最终 S 或 Decoder 可绕过本结构读取的全曲输出。

Encoder 合成验证见[入口文档](maidata-entry.md#cpu-合成验证)。独立输入模块 `src/maidata/preparation/{timing,features,produce,cache}.py` 已实现全曲输入生产、既有 full Mel 引用和 SongInputs/WindowInputs 构造，见[缓存说明](encoder-audio-cache.md)。真实样本验证包含变 BPM、历史 WAV、默认 Encoder 前向及旧直接 FFT 数值对照；未执行 CUDA 或训练。协作约定保留在 Markdown 中，不新增政策门禁；旧模型及其 `runtime-versions.toml` 保持旧路径专属。

## 已复制的旧模型实现

当前来源路线为：外部冻结 native 音频 → 一拍音频草稿 → 用户选定兼容草稿 checkpoint → 干净 token 缓存 → A。源实验采用 draft60/A90；该旧配方不是新 encoder 的训练安排。

- `BaselineTransformer`、旧 CLI 和三个原有 Python 包保留。旧模型版本及兼容范围只由 `src/maimai_baseline/runtime-versions.toml` 维护，不约束新 Encoder。
- native 路径读取外部 P0/C0/C1，冻结保留的 `audio_encoder`/`song_audio_encoder`；训练步不执行这两个音频网络。
- 一拍草稿为整曲双向音频条件网络，每拍输出 128 维 token；重建头与 A 读取前/中/后三个 16 拍窗口共 48 槽。草稿输入、内部交互和谱面因果历史仍是全曲范围。
- 主 A 保留结构化事件/字段、自回归生成历史、L3 与 Physical；当前一拍路线没有额外 B 训练步骤。

## 音频与窗口的实际形状

| 项目 | 旧模型当前代码 |
|---|---|
| Mel | 128 频带；24 kHz PCM，每个拍网格中心独立 FFT1024、periodic Hann、log1p 与固定统计 |
| 局部输入 | 绝对拍网格 `k/24`，完整 BPM 映射后以 half-up 量化 PCM 中心；反射波形边界 |
| 目标窗与音频上下文 | 目标 16 拍；前 4＋目标 16＋后 2＝22 拍 |
| 保留的局部 encoder 类 | 四块 Conformer，深度分配 (1,1,2)，24→12→6→6 token/拍 |
| 保留的全曲 encoder 类 | 一拍一个 [24,128] patch＋24 个有效位，经 3096→256 投影，再经两块全曲注意力 |
| native 缓存 | P0 [176,256]、C0 [528,256]、C1 [264,256]，另含 mask、拍坐标和插值几何 |
| native Decoder 音频 | 各原生窗口 P0[32:160] 中心 16 拍拼接的全曲 bank |

当前 full 与本次新增 Mel 均直接在拍网格中心提取，不从 hop240 的旧均匀 Mel 插值。间隔随 BPM 变化，但每个位置固定使用1024样本分析窗。旧元数据里的 192 是旧全曲表示宽度，不能解释为 192 个 Mel 频带。

旧直接提取源码定位：`src/maimai_baseline/data/audio.py:extract_at_centers` 与 `src/maimai_baseline/time/grid.py`。新输入模块独立移植所需算法，不运行时导入旧包。其余旧窗口/模型定位：`src/maimai_dataset/window.py`、`src/maimai_baseline/model/{audio,song_audio,baseline,song_dense,audio_draft}.py`、`src/maimai_baseline/data/teacher_audio.py`。

## 草稿验证的两种现有模式

默认准备模式从训练歌曲之外固定选择 32 首验证。来源最新 full 模式显式传 `--validation-from-training --validation-song-count 64`，从 full 中选出 64 首，只从草稿训练引用中移除对应歌曲；A 仍保留全部 full 歌曲。两者不能混写成同一个默认验证集。

教师权重选择由实际命令参数决定。最新来源说明使用微调后的教师（epoch4/step1550）；导出脚本还保留较早 full A90 的默认路径。副本没有这些权重，路径与来源不能仅凭旧说明推断。详见 [草稿参考](audio-draft.md)。

## 新 Encoder 尚未衔接的部分

新 Encoder 的全曲 G₀、每窗动态 S、局部 L、块内共享处理和三路局部输出已进入代码。Mel/aux32 离线缓存生产及 Encoder 输入适配已实现；Decoder、Router、MMoE、草稿、训练循环、loss 和 checkpoint 尚未接入，不能把可构建 Encoder 称为可训练完整模型的入口。

aux32 采用全曲统一的18维声学量、4维节奏统计和10维时间上下文，L 的 `k/R` 网格与 G0 的 `j+0.5` 拍中心按位置取值。网络模块只消费准备好的张量，输入模块独立计算全曲声学源及派生网格。新隐藏表示不自动兼容现有 native cache；内容未变的原始音频/Mel 不因新源码失效。

两个训练衔接问题仍待决定：

1. 当前 dense 训练为全曲因果历史补齐窗口；历史行需要的局部音频不能因新 Decoder 只读单窗就自动省掉。
2. 当前草稿读取全曲 fine P0；新方案只计算所需 fine 窗口时，草稿的输入或训练安排需要另行适配。

## 独立副本的执行边界

复制了旧源码、配置、schemas、必要 JSON/C++/CUDA 资源及文档；没有复制数据、权重、缓存、环境、旧测试或原 Git 历史。本轮未运行旧模型 CLI、CUDA、训练或整模型推理；新 Encoder 的使用与验证范围见 [maidata-entry](maidata-entry.md)。

旧 launcher 仍有源实验的远端路径；数据准备工具还保留 `--main-worktree` 默认定位，MiaCode/MajRadar 等是外部依赖。实际运行前使用明确输入和输出，不能把旧 launcher 当成新实验命令。详情见 [复制与布局](import-and-layout.md)。
