# 当前源码状态与实现边界

本文件区分已实现的新 `maidata` 音频 Encoder 和独立副本保留的旧模型。Encoder 规格位于[设计正本](../../0wd-v3-refactor-docs/new-audio-encoder.md)，模块、张量协议和 aux32 生产约定位于[实施文档](../../0wd-v3-refactor-docs/encoder-execution-plan.md)。来源与复制排除项见 [import-and-layout](import-and-layout.md)。

## 新 maidata：独立音频 Encoder

入口为 `python -m maidata`，调用方式与 CPU 验证命令见 [maidata-entry](maidata-entry.md)。`entry.py` 顶部普通 `CONFIG` 集中Encoder、Draft与Decoder架构参数，构造器无第二套默认值；`--help` 与 `--show-config` 仅依赖标准库。默认命令延迟导入 Torch，成功构建 `AudioEncoder`，独立阶段一训练入口为 `python -m maidata.training`，整曲生成调度未实现。

当前代码位于 `src/maidata/audio/{types,frontend,geometry,attention,blocks,pyramid,encoder}.py`，不导入旧 `maimai_baseline`、`maimai_dataset` 或朋友仓库。已实现：

- 独立频谱 tokenizer、aux32 融合与逐拍 G0 tokenizer；频率 query 和容量由配置传入。
- `prepare_song` 按曲计算 G0 及非末块的 K/V，保持计算图；`encode_windows` 按绝对拍坐标初始化每窗 S，执行共享参数、同步读取的 L/S Conformer。
- 局部联合 attention、完整 head RoPE、L 专属时间卷积、两次 stride transition，以及按坐标插值的局部 FPN。
- `forward(song_inputs, window_inputs)` 组合前两步，只返回命名的 P0/P1/P2，各自携带 values、beats、valid。

默认24拍输入的内部 C 长度为576/288/144，公开 P 长度为192/96/48；其他合法容量、深度和窗口由同一配置推导。没有 C0/C1/C2 高分辨率直出、最终 S 或 Decoder 可绕过本结构读取的全曲输出。

Encoder 合成验证见[入口文档](maidata-entry.md#cpu-合成验证)。独立输入模块 `src/maidata/preparation/{timing,features,produce,cache}.py` 已实现全曲输入生产、既有 full Mel 引用和 SongInputs/WindowInputs 构造，见[缓存说明](encoder-audio-cache.md)。真实样本验证包含变 BPM、历史 WAV、默认 Encoder 前向及旧直接 FFT 数值对照；未执行 CUDA 或训练。协作约定保留在 Markdown 中，不新增政策门禁；旧模型及其 `runtime-versions.toml` 保持旧路径专属。

## 新字段与 Physical 数据/规则层

`maidata.chart`实现结构化token、语法、Route-71、IR往返、镜像与窗口终端标签；`maidata.time`实现96 tick拍轴、written-BPM时长、全局等待/释放和4tick标签cell；`maidata.physical`消费既有几何，支持单段/拼接链、候选支持、动作提交及快照恢复。详见[接口说明](chart-fields-physical.md)。

不导入旧模型包、Torch或音频模块。flags不参与四类Physical判定，short音符与Tap/Touch占用等价，1tick音频越界单独处理。拼接链逐段检查几何、整体占手和释放；duration支持可以不连续。字段/规则模块本身没有embedding、预测网络或调度；这些可学习组件位于独立`maidata.decoder`，旧模型及旧缓存合同保持原样。

## 新 Draft：D、R、P与独立适配

`maidata.draft` 已实现全曲P0中心组合、逐拍非AR Dense MoE主干D、固定行重建R、单query Planner P、条件元数据、原谱标签、simai统计和辅助loss。默认值及使用方法见[Draft接口](draft.md)，设计正本为[new-draft](../../0wd-v3-refactor-docs/new-draft.md)。

D只消费公开音频张量和独立条件，R/P各自只消费Z；网络不读文件，窗口组合不持有模型。Decoder的Z/plan12投影及attention未放入Draft。Encoder与D/P仅共享无状态RoPE数学，所有可学习参数独立。完整空尾窗保持有效Z与R/P监督，音频mask不替代Draft mask。

CPU验证覆盖默认/非默认容量、源字段/simai标签、复用与mask、辅助loss及跨模块梯度。pilot320的R容量为每16拍窗207行，D仍一拍一个Z。`maidata.training`已接入独立Dataset/Collator、workers12 DataLoader、全曲E/D/R/P、五项total/count归约、训练/验证和严格完整轮恢复。batch同时受最多2个完整组及加权token预算45056约束，累积1；100轮17272步，绝对50轮8647步。Linux路径可配置，CPU完整更新和中断恢复一致性已验证；CUDA检查与正式启动交给训练会话，见[阶段一训练](encoder-draft-training.md)。

## 新 Decoder：网络、窗口与监督

`maidata.decoder`独立实现五类输入、前k层Router、后段dense MMoE、内容小头、三个时间分布及主/Physical/几何loss，默认值只在`entry.CONFIG`维护。全曲P0和局部Z/预测plan12通过公开张量接口传入，生产网络与Decoder参数不共享；plan使用无位置评分，音频/Z使用绝对拍RoPE。接口和可运行构建命令见[Decoder模块](decoder.md)。

`begin/complete`分开类型决策与后段计算，teacher forcing和窗内增量路径共用参数；跨窗全部self K/V重新prefill。独立窗口适配保留原始历史delta、4拍裁剪之外的最后真实起音与名义U尾部标签。独立支持适配读取首次归责记录，或在调用者提供的完整Physical状态上生成当前窗支持；网络不执行规则查询、不读缓存。

CPU验证覆盖默认与非默认架构、并行/增量一致、目标类型隔离、plan位置与空mask、FP32/CPU BF16反向、真实字段/规则支持、非单调时间概率、独立归约、跨Encoder/D/P梯度与实际冻结、链/分支快照。完整训练循环、checkpoint和整曲生成调度不在本模块；未运行CUDA或重建缓存。

## 已复制的旧模型实现

全曲原谱离线编排和读取入口为 `maidata.chart_cache`，见[缓存说明](chart-cache.md)。
它复用当前官谱音频 variant 和既有几何，保存完整 token/时间/归属/Physical 动作，
通过显式 `accept_source` 保留违规GT；不改变正常候选拒绝，不生成镜像或训练窗口。
原谱v2采用外部边界MOMENT、严格正生成delta及slide/Route-70统一WiFi；`chart_cache.migrate`另生产teacher条件的训练Physical候选mask与非单调时间区间，模型概率和loss通过独立Decoder适配，缓存生产不依赖模型。
全量实际完成情况以产物的 `production-summary.json` 为准，清单覆盖不等于全部成功。

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

新 Encoder 的全曲 G₀、每窗动态 S、局部 L、块内共享处理和三路局部输出已进入代码。Mel/aux32离线输入、公开P0全曲组合、新Draft D/R/P及辅助loss已实现；Decoder、Router/MMoE与主loss已独立实现，阶段一E/D/R/P训练循环与checkpoint已接入；主Decoder训练及整曲生成调度仍另行实现。

aux32 采用全曲统一的18维声学量、4维节奏统计和10维时间上下文，L 的 `k/R` 网格与 G0 的 `j+0.5` 拍中心按位置取值。网络模块只消费准备好的张量，输入模块独立计算全曲声学源及派生网格。新隐藏表示不自动兼容现有 native cache；内容未变的原始音频/Mel 不因新源码失效。

新Draft始终要求覆盖全曲的P0；只抽取部分监督窗口也不能省去其他音频中心。主Decoder按其[设计](../../0wd-v3-refactor-docs/new-decoder.md)消费全曲P0、局部Z及预测plan12，并在当前窗口条件下重新处理因果历史；其网络与小头/loss已独立实现，完整训练与整曲生成调度另行接入。

## 独立副本的执行边界

复制了旧源码、配置、schemas、必要 JSON/C++/CUDA 资源及文档；没有复制数据、权重、缓存、环境、旧测试或原 Git 历史。本轮未运行旧模型 CLI、CUDA、训练或整模型推理；新 Encoder 的使用与验证范围见 [maidata-entry](maidata-entry.md)。

旧 launcher 仍有源实验的远端路径；数据准备工具还保留 `--main-worktree` 默认定位，MiaCode/MajRadar 等是外部依赖。实际运行前使用明确输入和输出，不能把旧 launcher 当成新实验命令。详情见 [复制与布局](import-and-layout.md)。
