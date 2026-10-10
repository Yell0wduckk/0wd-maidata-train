# 0wd-v3 独立源码仓库

## 范围和路径

本仓库属于 maidata，Git 根为 `repos-0wd-v3/src`，包根为其内层 `src/`。它有独立 `.git` 目录，不是 worktree。源码导入、编辑和提交均在这里进行；不要重定向回原工作树。原仓库只作为复制来源及只读历史参考。

进入任务先读 README、docs/current-runtime.md；encoder 重构再读同级设计目录中的 `../0wd-v3-refactor-docs/new-audio-encoder.md`。

本次授权为设计落文档、源码/文档复制和旧文档清理。新 encoder 尚未实现；没有据此启动训练、上传或替换运行任务。后续按用户明确任务继续。

## 当前源码与未来设计

- 复制来源是 `main-new-audio-draft-beat` 的一拍草稿/native 音频路线。当前 native 路径冻结保留的音频模块，训练步读取外部原生 22 拍 P0/C0/C1 缓存。
- 当前 Decoder 仍读取全曲 P0 bank，草稿仍读取整曲 P0。不能把设计中的 24 拍、24 槽 S 或局部 Decoder 音频描述为已实现。
- 新设计的 G₀ 可复用不等于冻结：后续要端到端训练 tokenizer/encoder，同一步按曲共享计算图，不能 detach 或跨参数更新缓存旧表示。
- Decoder 音频局部化不自动局部化谱面因果历史；dense 历史和草稿输入的适配需显式设计。
- 本次不改模型、配置数值、数据布局或兼容范围。历史 B、谱面 encoder、雷达及 SFT 入口仅按实际合同使用，入口存在不代表适配当前路线。

## 版本和配置

版本及兼容范围仅维护在 `src/maimai_baseline/runtime-versions.toml`，其他文档不复制当前版本白名单。`0wd-v3` 是工作区名称，不是既有 baseline/checkpoint 版本。

不新增没有实际消费者的配置字段。来源提交、复制范围和讨论结论写在文档中，不把它们变成训练启动门禁。保留真实架构、参数 shape、缓存布局、mask、索引、有限值和必要恢复状态检查；不通过宽松加载掩盖新旧架构差异。

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

README 导航，current-runtime 描述实际代码，设计正本记录未来 encoder，历史规格放 docs/history。历史文档中的 Git 提交和命令属于源仓库；本仓库没有导入其 Git 历史。

测试和 fixture 按用户要求没有复制，旧测试结果不能当作新仓库验证。后续如确需测试，临时文件、TEMP/TMP 和 pytest basetemp 放本仓库 D 盘 `.test-temp`，不放 C 盘；只清理当前会话产生的临时文件。

实现任务汇报提交、受影响验证和未完成项。实验进度只在用户直接查看的实验会话报告；主任务交接后不主动轮询或转述。SSH 上传授权按用户既有目标主机与当前任务范围执行，不据此推断新训练授权。
