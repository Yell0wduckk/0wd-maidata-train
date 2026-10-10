# 版本兼容、结构与来源

兼容版本策略集中维护在 [runtime-versions.toml](../src/maimai_baseline/runtime-versions.toml)，读取逻辑见 [runtime_versions.py](../src/maimai_baseline/runtime_versions.py)，查询入口为 python -m maimai_baseline runtime-versions。文档不另抄一份版本号表，也不为尚未实现的新 encoder 指定版本数值。

独立复制目录与重命名不等于模型结构改变，也不表示已有权重已完成新架构适配。未来 encoder 的实际结构、训练和兼容范围确定后，再按中央策略维护对应合同。当前实现见 [current-runtime](current-runtime.md)，未来设计见 [新音频 encoder 设计](../../0wd-v3-refactor-docs/new-audio-encoder.md)。

## 兼容边界

版本号不能替代真实结构和数据语义检查。模型参数形状、表示类型、音频尺度、mask、位置坐标、归一化与训练状态仍由各 loader 的实际合同检查；旧四拍 latent 不因字段相似就能作为一拍草稿缓存或同语义续训输入。

源码 hash 和 commit 用于来源记录，不是通用运行兼容键；不能要求整份 provenance JSON 逐字段相等来替代中央策略。已有旧分支可能仍有自己的绑定实现，应以所选 loader 的实际行为为准。

一拍草稿使用 draft_producer_identity 提取 native 输出、预处理和实际生产者语义；该函数不把源码与 checkpoint 文件 hash 作为此新草稿合同的通用兼容字段。它仍保留原特征统计、生产者语义与输出合同，并不允许任意教师缓存互换。旧 native 分支的 producer_identity 与其字段集合不同。实现见 [teacher_audio.py](../src/maimai_baseline/data/teacher_audio.py)。

不得伪造 manifest、跳过真实不匹配、放宽参数加载或丢弃 optimizer/scheduler/RNG 来宣称完整续训。完整轮恢复见 [luna-training](luna-training.md)。已验收且内容未变的资产不因复制源码而失效，日常读取检查也不因此扩展为全量重验。

## 普通与流式推理

当前一拍路线一次生成全曲草稿，提取 48 槽条件，使用全曲 P0 bank 及完整因果谱面历史；流式和回退复用相同条件与可用历史状态。不能将其写成“latent disabled”或旧 B 的四拍自回归默认路线。

详细 trace 默认关闭，Physical 约束及必要 bypass 记录仍按实际实现执行。具体输入与输出边界见 [inference](inference.md)，资产说明见 [data-and-tools](data-and-tools.md)。历史运行报告既不是本副本的新训练结果，也不是新启动或新架构验收的凭据。
