# 0wd-v3 独立源码仓库

## 范围和路径

本仓库属于 maidata，Git 根为 `repos-0wd-v3/src`，包根为其内层 `src/`。它有独立 `.git` 目录，不是 worktree。源码导入、编辑和提交均在这里进行；不要重定向回原工作树。原仓库只作为复制来源及只读历史参考。

进入任务先读 README、docs/current-runtime.md；encoder 重构再读同级设计目录中的 `../0wd-v3-refactor-docs/new-audio-encoder.md`。

当前 `src/maidata` 已实现独立音频 Encoder，入口为 `python -m maidata`。包含 frontend、geometry、attention、blocks、pyramid、encoder 和 types；18 项架构配置集中在 `entry.py` 顶部唯一的普通 `CONFIG`。独立的 `maidata.preparation` 负责直接拍网格 FFT、全曲 aux32 与输入缓存，说明见 `docs/encoder-audio-cache.md`。Encoder 与输入生产均不迁移 Decoder、训练循环或 checkpoint，不启动训练、上传或替换运行任务。入口说明见 `docs/maidata-entry.md`。

## 新 Encoder 与保留的旧模型

- `maidata` 不导入旧模型或参考仓库。帮助和配置展示仅需标准库，实际构建时延迟导入 Torch；默认命令成功构建 `AudioEncoder` 并提示 Decoder 与训练未实现。
- 新 Encoder 通过 `prepare_song`、`encode_windows`、`forward` 消费准备好的 Mel、aux32、绝对拍坐标和有效位，仅公开 P0/P1/P2 三路局部输出。独立输入生产已实现；Decoder、草稿、训练与恢复系统仍待单独实现。
- 复制来源是 `main-new-audio-draft-beat` 的一拍草稿/native 音频路线。当前 native 路径冻结保留的音频模块，训练步读取外部原生 22 拍 P0/C0/C1 缓存。
- 旧 Decoder 仍读取全曲 P0 bank，旧草稿仍读取整曲 P0。新 Encoder 的可配置 24 拍默认窗口、S 工作槽和局部三尺度输出已实现，但尚未接入旧 Decoder 或草稿。
- 新 Encoder 的 G₀ 可复用不等于冻结：tokenizer/encoder 可训练，同次前向按曲共享计算图，不能 detach 或跨参数更新缓存旧表示。
- Decoder 音频局部化不自动局部化谱面因果历史；dense 历史和草稿输入的适配需显式设计。
- 本次不改旧模型、资产或兼容范围。历史 B、谱面 encoder、雷达及 SFT 入口仅按其实际合同使用，入口存在不代表已适配新 Encoder。

## 版本和配置

旧 `maimai_baseline` 的版本及兼容范围仅维护在 `src/maimai_baseline/runtime-versions.toml`，只适用于旧路径；新 `maidata` Encoder 不读取该表，也不新设版本表。其他文档不复制当前版本白名单。`0wd-v3` 是工作区名称，不是既有 baseline/checkpoint 版本。

不新增没有实际消费者的配置字段。来源提交、复制范围和讨论结论写在文档中，不把它们变成训练启动门禁。保留真实架构、参数 shape、缓存布局、mask、索引、有限值和必要恢复状态检查；不通过宽松加载掩盖新旧架构差异。

训练默认值和资产使用规则是协作约定，不将其强塞为 Encoder 入口的政策校验。当前只检查真实数学与张量协议，不提前实现数据或恢复框架；后续有实际消费者时，再实现必要的表示和严格恢复检查。

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

README 导航，current-runtime 描述实际代码，设计正本维护 Encoder 架构，历史规格放 docs/history。历史文档中的 Git 提交和命令属于源仓库；本仓库没有导入其 Git 历史。

旧测试和 fixture 按用户要求没有复制，旧测试结果不能当作新仓库验证。新 Encoder 测试位于 `tests/`，使用小型 CPU 合成输入；独立输入生产测试另覆盖直接 FFT 数值参考和显式选择的少量真实音频。临时文件、TEMP/TMP 和 pytest basetemp 放本仓库 D 盘 `.test-temp`，不放 C 盘；只清理当前会话产生的临时文件。

实现任务汇报提交、受影响验证和未完成项。实验进度只在用户直接查看的实验会话报告；主任务交接后不主动轮询或转述。SSH 上传授权按用户既有目标主机与当前任务范围执行，不据此推断新训练授权。
