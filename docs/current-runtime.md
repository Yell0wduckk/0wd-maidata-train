# 当前源码状态与实现边界

当前主 Decoder 变更以 [独立 CONT 序列合同](decoder-cont-token-plan.md) 为准：删除 FieldReader，CONT 写入主 KV，窗口升级 v5。
本文件区分已实现的新 `maidata` 音频 Encoder 和独立副本保留的旧模型。Encoder 规格位于[设计正本](../../0wd-v3-refactor-docs/new-audio-encoder.md)，模块、张量协议和 aux32 生产约定位于[实施文档](../../0wd-v3-refactor-docs/encoder-execution-plan.md)。来源与复制排除项见 [import-and-layout](import-and-layout.md)。

## 新 maidata：独立音频 Encoder

入口为 `python -m maidata`，调用方式与 CPU 验证命令见 [maidata-entry](maidata-entry.md)。`entry.py` 顶部普通 `CONFIG` 集中Encoder、Draft与Decoder架构参数，构造器无第二套默认值；`--help` 与 `--show-config` 仅依赖标准库。默认命令延迟导入 Torch，成功构建 `AudioEncoder`，独立阶段一训练入口为 `python -m maidata.training`，整曲推理入口为`python -m maidata.inference`。

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

`maidata.draft` 已实现全曲P0中心组合、逐拍非AR Dense MoE主干D、独立局部字段重建R、单query Planner P、条件元数据、精确原谱标签、simai统计和辅助loss。D保留一拍一个128维Z；R不预测type/pad/空槽，普通字段读取近邻三个Z，MOMENT读取已知游标前向四拍；P独立读取目标全部16拍。默认值及使用方法见[Draft接口](draft.md)。

D只消费公开音频张量和独立条件，R/P各自只消费Z；网络不读文件，窗口组合不持有模型。Decoder的Z/plan12投影及attention未放入Draft。Encoder与D/P仅共享无状态RoPE数学，所有可学习参数独立。完整空尾窗保持有效Z与R/P监督，音频mask不替代Draft mask。

CPU验证覆盖独立查询/未来条件隔离、局部梯度范围、空拍count0、精确时间路径/删失、原字段分母、源字段/simai统计、复用与mask及跨模块梯度。R已无207行容量；`pretrain_batch_reference_rows=207`仅保留原分批proxy，实际query/substep成本单独记录。`maidata.training`保留workers12、完整音频组和五项total/count归约，checkpoint为`maidata.encoder-draft-training.v2`。原batch边界、预算45056与累积1保持，未改训练轮数/学习率。CPU完整更新和恢复一致性已验证，CUDA原batch及正式训练未执行，见[阶段一训练](encoder-draft-training.md)。

## 新 Decoder：网络、窗口与监督

`maidata.decoder`独立实现五类目标与八类输入、前k层Router、后段dense MMoE、独立CONT输入、内容小头、三个385类时间头及主/Physical/几何loss，默认值只在`entry.CONFIG`维护。主M继续生成并监督type/kind/pad/flags/route；begin只读真实前缀，目标类型与Slide查询坐标不提前进入Router。CONT作为独立输入进入主self-KV，只有完整字段提交Physical；首次Slide仅调整后段音频/Z查询坐标。plan使用无位置评分，音频/Z使用绝对拍RoPE，双Encoder保持独立。接口见[Decoder模块](decoder.md)。

三种时间共用`relative-tick384-continue`；时间CE/集合NLL和Physical按原字段计数，Physical只监督GT实际前缀步骤并按原字段平均，保留零罚窗尾叶；duration窗外短值仍按完整源区间处罚。written-BPM单位保留，Slide从启动位置读取，变BPM以实际秒数反算query。`begin/complete`分开类型决策与后段计算，teacher forcing和窗内增量路径共用参数；跨窗全部self K/V重新prefill。独立窗口适配保留原始历史delta、4拍裁剪之外的最后真实起音与名义U尾部标签。独立支持适配读取首次归责记录，或在调用者提供的完整Physical状态上生成当前窗支持；网络不执行规则查询、不读缓存。

CPU验证覆盖默认与非默认架构、并行/增量一致、目标类型隔离、plan位置与空mask、FP32/CPU BF16反向、真实字段/规则支持、非单调时间概率、独立归约、跨Encoder/D/P梯度与实际冻结、链/分支快照。训练循环和checkpoint位于独立的`maidata.decoder_training`；整曲生成由独立`maidata.inference`调度，本次只对显式选定少量真实窗口重编译v3缓存，未运行CUDA或整库迁移。

阶段二已接入原谱v2字段/Physical、全局打乱的既有Decoder occurrence、GT历史与确定性15% history dropout；保持仍有效的完整物理动作，训练读取离线窗口支持。当前重新随机初始化音频Encoder E_M与M并联合更新；每batch按audio_variant去重读取已验收Mel/aux32，E_M只编码当前监督窗所需的前4＋目标16＋后4拍局部窗口，同曲同起点共享P0。G0按曲准备一次；选中的局部窗口统一批量编码，不使用activation checkpoint反向重算，Encoder布局仅准备和传输一份。M音频attention只读取对应局部P0；新checkpoint记录window范围，推理使用同一范围。Z始终读取阶段一E_D/D/P的固定离线缓存；plan12默认读取GT原谱统计小缓存，显式`--plan-source predicted`可选择固定P预测，训练与validation使用相同来源。训练不构建或运行教师E_D/D/P/R，也不从阶段一权重初始化E_M。固定条件缺失不在线补算；导出与消费模块独立。GT缓存生产、两组独立fresh训练及远端串行launcher见[plan对照](decoder-plan-comparison.md)。

默认row budget8192、累积1、warmup200，全部可配置，音频成本另行记录，本次不改变组批预算。M计算块使用整个逻辑update的独立分母；单M块直接反传真实音频图，多M块才通过音频K/V梯度桥收集各块梯度，最后只反传真实E_M及音频投影一次，E_M/M共同裁剪和更新。worker准备Encoder/P0/M布局，训练不解压整谱或查询Physical，不重新验收旧输入、不哈希生产者checkpoint。联合训练checkpoint为`maidata.decoder-training.v4`，保存成套E_M/M、optimizer、scheduler、完整轮和全部RNG；旧v1/v2连续时间头及v3字段读取分支不能作为新架构完整轮续训。源码/配方/budget/计划身份不作为额外恢复门禁。保留workers12、完整轮checkpoint、每5轮及末轮validation与原batch CUDA测量入口；本次性能修复的CPU梯度与恢复验证、CUDA原batch成本记录由独立修复报告提供。其余未确定配方保持None，见[阶段二训练](decoder-training.md)。

统一新窗口编译器从精确原谱v2和完整Physical整数区间生产`maidata.decoder-window-tensors.v5`监督，增量与从零入口共用；完整源资产和迁移不变。GT duration、真实动作与历史保持完整，监督到跨音频窗CONT或窗内发出的最后短值；旧v3局部cap监督需重编译标签，不能混用。迁移命令见[缓存迁移](time384-cache-migration.md)。E/D/P权重变化后的神经特征需重新前向导出。

## 监督窗口采样

`maidata.sampling` 已实现固定清单到基础窗口与逐轮实例的离线计划，入口为 `python -m maidata.sampling`，见[采样说明](sampling.md)。默认目标16拍、stride 16拍，保留整曲音频覆盖及额外完整空尾窗。主Decoder训练对短于1拍的不同真实起音delta种类至少3种的窗口总采样2倍，Slide不额外加权；Encoder＋Draft预训练全部基础窗各1次，不过采样。默认值读取唯一 `entry.CONFIG`。

模块仅使用标准库及现有TimingMap，按所选侧读取原谱v2载荷，不扫描其他谱面、不计算音频或Physical、不重新哈希输入。输出唯一基础窗、按完整audio_variant组排列的occurrence，以及可复现的seed/epoch/config和组偏移；只引用既有训练/验证划分，不重新创建split，不物化谱面历史、终端标签或batch。

pilot320的320份Master及23份Re:Master已有8,828个基础窗和1,587个复杂窗，包含343个完整空尾窗，分为320个音频组。按固定split派生的[Encoder＋Draft计划](../../artifacts/pilot320-encoder-draft-sampling-20261010/README.md)为训练288首／307谱、7,924实例，验证32首／36谱、904实例，全部基础窗各1次。[主Decoder计划](../../artifacts/pilot320-sampling-20261010/README.md)为训练9,342实例、验证904实例，all概览10,415实例。派生阶段一计划只读取现成窗口元数据，不重读源谱。采样与唯一CONFIG入口的16项定向CPU检查通过，真实输出与既有逐窗口统计一致，分侧组与实例独立复核通过；主Decoder训练已通过 maidata.decoder_training.plan 消费既有计划并复用 make_occurrences；本次不重新生成计划。

## 当前整曲推理与物化

`python -m maidata.inference`显式加载阶段一E_D/D/P与阶段二v4的E_M/M权重。固定E_D产生Draft的P0并计算全曲Z/预测plan；独立E_M产生M的全曲P0，参数与隐藏表示分开。GT plan训练的M自由推理仍读取P预测，报告分别记录训练和推理plan来源。读取配套grid或使用固定Mel统计提取音频，按新字段规则维护真实起音组、边界MOMENT、跨窗活动动作与完整规则快照。两套Encoder各自按曲复用准备结果，D/P及M音频投影整次生成复用；各窗重新prefill历史，始终覆盖音频及一个完整空尾窗。旧连续时间头不兼容新385类架构。Physical硬约束默认开启，trace默认关闭；路径、温度、种子与生成上限可配置。

`python -m maidata.inference.materialize`独立调用MiaCode render/dump，核对新字段与时间轴后输出`maidata.txt`和`track.<音频扩展名>`。两入口不导入旧模型包、不增加源码哈希门禁。小型真实E/D/P/M已完成CPU FP32/BF16到MiaCode的端到端测试；真实训练checkpoint的CUDA生成与谱面质量未验收。命令、跨窗合同和模块边界见[推理与物化](maidata-inference.md)。

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

新 Encoder 的全曲 G₀、每窗动态 S、局部 L、块内共享处理和三路局部输出已进入代码。Mel/aux32离线输入、公开P0全曲组合、新Draft D/R/P及辅助loss已实现；Decoder、Router/MMoE与主loss已独立实现，阶段一E/D/R/P训练循环与checkpoint已接入；主Decoder阶段二训练及checkpoint由独立`maidata.decoder_training`接入，整曲生成和可播放maidata物化由独立`maidata.inference`接入。

aux32 采用全曲统一的18维声学量、4维节奏统计和10维时间上下文，L 的 `k/R` 网格与 G0 的 `j+0.5` 拍中心按位置取值。网络模块只消费准备好的张量，输入模块独立计算全曲声学源及派生网格。新隐藏表示不自动兼容现有 native cache；内容未变的原始音频/Mel 不因新源码失效。

新Draft始终要求覆盖全曲的P0；只抽取部分监督窗口也不能省去其他音频中心。主Decoder按其[设计](../../0wd-v3-refactor-docs/new-decoder.md)消费全曲P0、局部Z及预测plan12，并在当前窗口条件下重新处理因果历史；其网络、小头/loss及阶段二训练已独立实现，整曲生成见[当前推理入口](maidata-inference.md)。

## 独立副本的执行边界

复制了旧源码、配置、schemas、必要 JSON/C++/CUDA 资源及文档；没有复制数据、权重、缓存、环境、旧测试或原 Git 历史。旧模型 CLI、CUDA及正式训练的验证按各任务实际记录；新 Encoder 的使用与验证范围见 [maidata-entry](maidata-entry.md)。

旧 launcher 仍有源实验的远端路径；数据准备工具还保留 `--main-worktree` 默认定位，MiaCode/MajRadar 等是外部依赖。实际运行前使用明确输入和输出，不能把旧 launcher 当成新实验命令。详情见 [复制与布局](import-and-layout.md)。
