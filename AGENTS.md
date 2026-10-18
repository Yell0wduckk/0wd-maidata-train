# 0wd-v3 独立源码仓库

## 范围和路径

本仓库属于 maidata，Git 根为 `repos-0wd-v3/src`，包根为其内层 `src/`。它有独立 `.git` 目录，不是 worktree。源码导入、编辑和提交均在这里进行；不要重定向回原工作树。原仓库只作为复制来源及只读历史参考。

进入任务先读本文件和 [当前源码状态](docs/current-runtime.md)，再按下方导航读取对应模块合同。README 仅保留模型公开状态，不承担运行导航。Encoder 重构再读同级设计目录中的 [Encoder 设计](../0wd-v3-refactor-docs/new-audio-encoder.md)。在本仓库的 linked worktree 中工作时，使用该 worktree 自己的源码、文档和配置，不将运行时导入重定向到 main。

当前 `src/maidata` 已实现独立可训练的音频 Encoder、Draft D/R/P、主 Decoder M、音频输入生产、字段/Physical 数据规则层和窗口采样计划。阶段一训练、阶段二训练、共享 E/D/R/P/M 的联合训练及各自 checkpoint/恢复已有独立入口，整曲推理与纯 Python maidata 物化已接入。保留的旧包为 `maimai_baseline`、`maimai_dataset` 和 `maimai_chart_ir`，新模型不以旧模型包作为生产依赖。所有架构默认值集中在 `src/maidata/entry.py` 顶部唯一的普通 `CONFIG` 字典；构造器无第二套默认值，尺寸随配置推导，实验覆盖配置位于 `configs/`。开发依赖由 `pyproject.toml` 和 `uv.lock` 声明。

## 当前 main 模型与兼容

- main 已合入 [Z2、全层 Q/K 归一化与逐拍起音监督](docs/z2-qknorm-onset.md)。对应显式配方为 `configs/joint-main320-z2-single-plan16-qknorm-onset-gt.json`：E/D/M 320 维、8 头、每头 40 维，E/M 前段 FFN1280，D/M 专家640，M 八层；3 完整音频组/80000 总预算，M chunk32768，峰值学习率6e-4，停止150轮/调度150轮，BF16、workers12。这是该配方的设置，不将其当作所有旧 checkpoint 或 `CONFIG` 基础默认值。
- Z 为整数拍锚点0、2、4…，包含完整空尾窗端点；每枚重建 `[t-2,t+2)`。对目标窗 `[s,s+16)`，P 读取9枚 Z，M 读取11枚 Z 加单枚无位置 Plan16。Plan GT 统计范围及缓存保持原单窗语义，P 内部保留 RoPE。当前 Z 直接取 D 的最终 LayerNorm 隐藏，宽度等于 `draft_dim`，不固定为256维。
- E/D/P/M 的 Q/K 归一化由保存配方中的各组件开关控制，在投影拆头后、RoPE 前逐头执行无仿射 RMSNorm；FP32 计算 RMS、epsilon1e-6，再恢复 dtype，保留 `1/sqrt(head_dim)`。V 与音频 tokenizer 的频率池化保持原数学。旧 recipe 缺开关时关闭，不能因没有新增参数而静默切换数学。
- R 的字段和 MOMENT96 保持按实际有效查询归约，不做拍内平均；新配方另监督有效拍是否存在真实起音，空拍及空尾标签为0，覆盖外不监督。数值数量监督保持关闭，起音存在头与数量头独立。几何、去重、损失和 mask 规则按 Z2 合同执行。
- 主 M 以 [独立 CONT 序列合同](docs/decoder-cont-token-plan.md) 为准：删除额外 FieldReader，普通字段读取主干，CONT 写入主 self-KV，三类时间头保持385类，窗口监督为v5；原谱及 Physical 为v2。训练默认 GT Plan，预测 Plan 推理由 P 产生。`decoder_train_plan_input` 控制训练/验证的 M 侧 Plan 输入，独立于 Z 输入及 P 辅助监督，保留该开关。
- 普通、预缓存、交互和增量推理按 checkpoint 保存配方恢复宽度、Z/Plan 几何、Q/K 数学及起音头；旧 Z4 和历史整数 Z1 checkpoint 保持各自语义。旧权重或神经隐藏缓存不自动兼容新架构，不以宽松加载、补维或改缓存身份掩盖真实不匹配。E/D/P 权重变化后，相关神经特征须重新前向导出；内容未变的原始音频/Mel 不因此失效。
- 推理默认使用 `--joint-checkpoint` 共享 E/D/P/M；仍保留显式两阶段路线。联合模型支持仅 E 的 `preencode`、仅 D/P 的 `predraft` 及 `--cache-dir` 复用；invalid 缓存自动清理并重建。输入、输出、缓存身份及必要错误检查按 [推理合同](docs/maidata-inference.md) 执行。

## 新 Encoder 与保留的旧模型

新字段的数据与规则层位于 `maidata.chart`、`maidata.time`、`maidata.physical`，接口见 `docs/chart-fields-physical.md`。只实现编解码、语法、时间标签、Physical查询/动作/回退，不将embedding、小头、Decoder或训练放入数据/规则模块；可学习组件由`maidata.decoder`独立持有。生产模块不导入旧包；测试可以使用保留旧实现作单段数值/规则对照。几何通过显式资源路径读取，不重建旧资产。

- `maidata` 不导入旧模型或参考仓库。帮助和配置展示仅需标准库，实际构建时延迟导入Torch；默认命令构建`AudioEncoder`，阶段一/阶段二训练与整曲推理各有独立入口。
- 新 Encoder 通过 `prepare_song`、`encode_windows`、`forward` 消费准备好的 Mel、aux32、绝对拍坐标和有效位，仅公开 P0/P1/P2 三路局部输出。默认前4＋目标16＋后4＝24拍；全曲 G₀、每窗动态 S 与局部 L 使用共享块。`maidata.preparation` 独立生产直接拍网格 Mel、全曲 aux32、坐标和 mask，构造 SongInputs/WindowInputs；不从旧均匀 Mel 插值，见 [输入缓存](docs/encoder-audio-cache.md) 与 [Encoder 接口](docs/maidata-entry.md)。
- `maidata.draft` 分离全曲 P0 中心组合、窗口选择、D、R、P、条件元数据、标签适配和辅助 loss。Draft 不持有 Encoder 或 Decoder，D/R/P 参数独立；公开张量组合保留计算图。R 使用直接索引 owner Z 的共享残差 MLP，GT 只安排有效查询和监督索引，不输入总行数、当前目标、pad 或完整未来类型表，不预测 type/pad；MOMENT 为拍内96类，duration 不进入 Draft 监督。保留的数量两头与新起音存在头按配方分别控制，不能将关闭的数量监督当作空拍监督。旧 R 及128维 Z 不兼容拆分后的阶段一 checkpoint v3。接口见 [Draft](docs/draft.md)，历史拆分合同见 [实施计划](../0wd-v3-refactor-docs/drafter-split-execution-plan.md)。
- `maidata.decoder` 独立持有网络、Router/MMoE、字段小头、时间分布及主/Physical/几何 loss；主 M 仍预测全部结构及 pad，Z/Plan 适配和条件 attention 归 Decoder 所有。当前阶段二及联合训练的 M 读取对应局部 P0；D 仍需覆盖全曲的 P0 中心，局部监督不能省略其他音频中心。两者音频范围按 [当前运行说明](docs/current-runtime.md) 区分。
- `maidata.sampling` 从既有划分生成唯一基础窗和逐轮 occurrence 引用计划，保留完整音频覆盖及额外完整空尾窗；主 Decoder 对复杂节奏窗口总采样2倍，Encoder＋Draft 全部基础窗各1次，按完整音频组组织。实际视图、occurrence、顺序及 batch 策略服从所选计划，见 [采样](docs/sampling.md)。
- 复制来源是 `main-new-audio-draft-beat` 的一拍草稿/native 音频路线。保留的旧 native 路径冻结音频模块，训练步读取外部22拍 P0/C0/C1 缓存；旧 Decoder 读取全曲 P0 bank，旧草稿读取整曲 P0。这些是旧模型合同，不作为新 Encoder/Draft/M 的默认路线。
- 新 Encoder 的 G₀ 可复用不等于冻结：tokenizer/encoder 可训练，同次前向按曲共享计算图，不能 detach 或跨参数更新缓存旧表示。
- Decoder 音频局部化不自动局部化谱面因果历史；dense 历史和草稿输入的适配需显式设计。
- 历史 B、谱面 encoder、雷达及 SFT 入口仅按其实际合同使用，入口存在不代表已适配新 Encoder。旧配置及远端 launcher 是来源参考；数据工具保留的 `--main-worktree` 默认定位和 MiaCode/MajRadar 等外部依赖需按实际任务明确，不把旧 launcher 当作新实验命令。

## 开发与运行入口

从所选源码仓库根设置 Python 包路径。以下示例指向 main；在 worktree 中将工作目录改为该 worktree 根。

```powershell
Set-Location D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/src
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m maidata --help
python -m maidata --show-config
python -m maidata
```

默认命令构建 `AudioEncoder`；`--component draft/planner/reconstruction/decoder` 分别构建独立组件。R 通过直接 Z 查询解析，无固定行容量。帮助和配置展示仅依赖标准库，实际构建延迟导入 Torch。

- `python -m maidata.training`：阶段一 Encoder＋Draft 训练。
- `python -m maidata.decoder_training`：阶段二主 Decoder 训练及恢复。
- `python -m maidata.joint_training`：共享 E/D/R/P/M 的从头联合训练及完整轮恢复。
- `python -m maidata.inference`：整曲推理与直接 maidata 导出。
- `python -m maidata.inference.preencode`、`python -m maidata.inference.predraft`：联合模型音频和 Draft/Plan 预计算。
- `python -m maidata.inference.materialize`：已有 IR 的独立物化入口。

实际命令须显式选定任务所需配置、资源、checkpoint 和输出；模块构建成功或 CPU 检查通过不代表 CUDA、正式训练或谱面质量已经验收。

## 版本和配置

旧 `maimai_baseline` 的版本及兼容范围仅维护在 `src/maimai_baseline/runtime-versions.toml`，只适用于旧路径；新 `maidata` Encoder 不读取该表，也不新设版本表。其他文档不复制当前版本白名单。`0wd-v3` 是工作区名称，不是既有 baseline/checkpoint 版本。

不新增没有实际消费者的配置字段。来源提交、复制范围和讨论结论写在文档中，不把它们变成训练启动门禁。Encoder＋Draft训练直接消费已验收输入，不在启动、Dataset、Collator、前向或loss适配中校验输入；不生成或比较输入哈希。共享缓存生产/离线工具仍可显式验收。正常mask、索引、batch预算计算保留，loss/梯度非有限值、模型构造配置和checkpoint恢复状态检查保留；不通过宽松加载掩盖新旧架构差异。

训练默认值和资产使用规则是协作约定，不将其强塞为 Encoder 入口的政策校验。模型构造检查数学配置；Encoder＋Draft训练不重复检查输入张量协议，完整轮恢复仅检查模型、配方、调度和恢复状态。

## 训练和资产规则

- 未经用户改变，DataLoader workers 固定 12；不将 CPU 测试设置带入真实训练。
- 正式训练复用已批准计划的 curriculum、视图、occurrence、顺序、完整音频组及 batch 边界，不擅自重复扩增。
- checkpoint 只按完整轮保存；完整诊断仅末轮；新训练默认每 5 轮及末轮 validation，重合一次。复制来源的草稿验证/A 无验证行为作为现有实现说明，不自动视为新 encoder 配方。
- scheduler horizon 与绝对停止轮分别设置；完整轮恢复包含 optimizer、scheduler、游标及全部 RNG。异常或轮内不另存 checkpoint。
- 默认信任内容未变的已验收资产。不因新仓库、源码身份或新架构追加旧数据全量扫描、重哈希、特征重算、统计重估或几何重建。
- 只验证本次实际改变的内容及合同要求的受影响部分；纯文档修改只检查文档。不能添加未经要求的性能、validation loss 或质量阈值，谱面质量由用户实际评谱。
- 推理默认不传详细 `--trace`，保留必要错误和 Physical bypass 记录。
- 不改父工作区 handcheck，不替换正在运行实验的绑定源码。

## 文档、测试与交接

AGENTS 维护开发/运行约定与导航，`docs/current-runtime.md` 描述实际代码，设计正本分别维护 Encoder、Draft 和 Decoder 架构，历史规格放 `docs/history/`。设计目录 `../0wd-v3-refactor-docs/` 与源码仓库并列，不属于本仓库 Git tree；仅涉及 Encoder 的实施与验证范围见 [Encoder 实施文档](../0wd-v3-refactor-docs/encoder-execution-plan.md)。

本仓库从 `repos/main-new-audio-draft-beat` 的提交 `3de8e75706aa4b8a1f486b4a42125faca3214592` 复制建立，未导入原 Git 历史；来源提交仅作记录。复制保留旧源码、配置、schemas 及必要包资源，排除原测试、fixture、权重、数据、缓存、环境和运行证据。历史文档中的 Git 提交和命令属于源仓库，详细范围见 [复制与布局](docs/import-and-layout.md)。

旧测试和 fixture 按用户要求没有复制，旧测试结果不能当作新仓库验证。新测试位于 `tests/`，按各模块范围维护；Encoder 使用小型 CPU 合成输入，命令见 [CPU 合成验证](docs/maidata-entry.md#cpu-合成验证) 和 [Draft 接口](docs/draft.md)。独立输入生产测试另覆盖直接 FFT 数值参考和显式选择的少量真实音频。临时文件、TEMP/TMP 和 pytest basetemp 放本仓库 D 盘 `.test-temp`，不放 C 盘；只清理当前会话产生的临时文件。

实现任务汇报提交、受影响验证和未完成项。实验进度只在用户直接查看的实验会话报告；主任务交接后不主动轮询或转述。SSH 上传授权按用户既有目标主机与当前任务范围执行，不据此推断新训练授权。

## 文档导航

- [当前源码状态](docs/current-runtime.md)
- [Z2、Q/K归一化与逐拍起音监督](docs/z2-qknorm-onset.md)
- [Main扩容、Q/K归一化与Route37原配方](docs/main320-route37-qknorm.md)
- [整数拍 Draft 与重叠 Plan16 的几何和兼容合同](docs/integer-draft-plan16.md)
- [共享 E/D/R/P/M 联合训练](docs/joint-training.md)
- [Maidata Encoder 入口与张量接口](docs/maidata-entry.md)
- [Draft D/R/P、窗口、标签与损失接口](docs/draft.md)
- [Draft 拆分实施记录](docs/drafter-split-implementation.md)
- [冻结主 Decoder ckpt45 的 P0 与 Drafter 训练](docs/drafter-frozen-p0-training.md)
- [Decoder、Router/MMoE、小头、窗口与损失接口](docs/decoder.md)
- [主 Decoder 阶段二训练与恢复](docs/decoder-training.md)
- [M侧训练 Plan 输入开关](docs/decoder-training-plan-input.md)
- [冻结 Encoder/Drafter、读取 P0/Z 的 Decoder 适配入口](docs/decoder-frozen-adaptation.md)
- [384 tick 时间监督缓存迁移与从零生产](docs/time384-cache-migration.md)
- [当前模型整曲推理与可播放 maidata 导出](docs/maidata-inference.md)
- [联合模型音频预编码、Draft/Plan 预解码与推理缓存](docs/maidata-inference.md#联合模型预编码与预解码)
- [全官谱音频输入缓存](docs/encoder-audio-cache.md)
- [新字段、时间与 Physical](docs/chart-fields-physical.md)
- [全官谱原谱全曲缓存](docs/chart-cache.md)
- [基础窗口与复杂节奏采样](docs/sampling.md)
- [Encoder 设计](../0wd-v3-refactor-docs/new-audio-encoder.md)
- [Draft 设计](../0wd-v3-refactor-docs/new-draft.md)
- [Decoder 设计](../0wd-v3-refactor-docs/new-decoder.md)
- [复制范围、独立 Git 与外部依赖](docs/import-and-layout.md)
- [文档索引](docs/README.md)
- [旧一拍草稿实现参考](docs/audio-draft.md)
- [旧模型训练与恢复参考](docs/luna-training.md)
- [旧模型推理参考](docs/inference.md)
- [数据与工具](docs/data-and-tools.md)
- [行内字段与 Physical](docs/physical.md)
- [旧版本维护](docs/runtime-versioning.md)
- [历史规格](docs/history/index.md)
