# 当前源码状态与重构边界

本文件描述独立副本实际保留的代码。新 encoder 设计位于 [设计正本](../../0wd-v3-refactor-docs/new-audio-encoder.md)，尚未实现。来源与复制排除项见 [import-and-layout](import-and-layout.md)。

## 已复制的实现

当前来源路线为：外部冻结 native 音频 → 一拍音频草稿 → 用户选定兼容草稿 checkpoint → 干净 token 缓存 → A。源实验采用 draft60/A90；该旧配方不是新 encoder 的训练安排。

- `BaselineTransformer`、CLI 和三个 Python 包保留。版本及兼容范围只由 `src/maimai_baseline/runtime-versions.toml` 维护。
- native 路径读取外部 P0/C0/C1，冻结保留的 `audio_encoder`/`song_audio_encoder`；训练步不执行这两个音频网络。
- 一拍草稿为整曲双向音频条件网络，每拍输出 128 维 token；重建头与 A 读取前/中/后三个 16 拍窗口共 48 槽。草稿输入、内部交互和谱面因果历史仍是全曲范围。
- 主 A 保留结构化事件/字段、自回归生成历史、L3 与 Physical；当前一拍路线没有额外 B 训练步骤。

## 音频与窗口的实际形状

| 项目 | 当前代码 |
|---|---|
| Mel | 128 频带；基础音频参数 24 kHz、FFT 1024、hop 240（10 ms） |
| 局部输入 | BPM 对齐，每拍 24 个采样位置 |
| 目标窗与音频上下文 | 目标 16 拍；前 4＋目标 16＋后 2＝22 拍 |
| 保留的局部 encoder 类 | 四块 Conformer，深度分配 (1,1,2)，24→12→6→6 token/拍 |
| 保留的全曲 encoder 类 | 一拍一个 [24,128] patch＋24 个有效位，经 3096→256 投影，再经两块全曲注意力 |
| native 缓存 | P0 [176,256]、C0 [528,256]、C1 [264,256]，另含 mask、拍坐标和插值几何 |
| native Decoder 音频 | 各原生窗口 P0[32:160] 中心 16 拍拼接的全曲 bank |

上表的 10 ms 是基础 Mel hop；实际局部 encoder 输入沿拍网格重采样，间隔随 BPM 变化。旧元数据里的 192 是旧全曲表示宽度，不能解释为 192 个 Mel 频带。

当前源码定位：`src/maimai_dataset/audio.py`、`src/maimai_dataset/window.py`、`src/maimai_baseline/model/{audio,song_audio,baseline,song_dense,audio_draft}.py`、`src/maimai_baseline/data/teacher_audio.py`。

## 草稿验证的两种现有模式

默认准备模式从训练歌曲之外固定选择 32 首验证。来源最新 full 模式显式传 `--validation-from-training --validation-song-count 64`，从 full 中选出 64 首，只从草稿训练引用中移除对应歌曲；A 仍保留全部 full 歌曲。两者不能混写成同一个默认验证集。

教师权重选择由实际命令参数决定。最新来源说明使用微调后的教师（epoch4/step1550）；导出脚本还保留较早 full A90 的默认路径。副本没有这些权重，路径与来源不能仅凭旧说明推断。详见 [草稿参考](audio-draft.md)。

## 新设计尚需实现的部分

设计中的全曲 G₀、24 个动态 S、24 拍 L、四块内共享处理和 Decoder 仅读当前局部音频均未进入代码。不得只改文档或窗口长度就宣称完成架构迁移。

若目标窗继续为 16 拍，4＋16＋4 是一种 24 拍分配建议；本次没有把它定为实际配置。预计 24 拍的 C0/C1/C2 长度为 576/288/144，FPN 的 P0/P1/P2 为 192/96/48，这些是未来形状而非现有 native cache。

两个训练衔接问题仍待决定：

1. 当前 dense 训练为全曲因果历史补齐窗口；历史行需要的局部音频不能因新 Decoder 只读单窗就自动省掉。
2. 当前草稿读取全曲 fine P0；新方案只计算所需 fine 窗口时，草稿的输入或训练安排需要另行适配。

## 独立副本的执行边界

复制了源码、配置、schemas、必要 JSON/C++/CUDA 资源及文档；没有复制数据、权重、缓存、环境、测试或原 Git 历史。本次未运行 CLI、CUDA、训练或推理。

旧 launcher 仍有源实验的远端路径；数据准备工具还保留 `--main-worktree` 默认定位，MiaCode/MajRadar 等是外部依赖。实际运行前使用明确输入和输出，不能把旧 launcher 当成新实验命令。详情见 [复制与布局](import-and-layout.md)。
