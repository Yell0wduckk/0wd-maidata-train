# 文档索引

本仓库保留从旧工作树复制的模型，并实现独立的 `maidata` 音频 Encoder。维护文档区分两条路径；Encoder 设计正本保留在仓库同级的设计目录。完整输入生产、Decoder 和训练系统仍未接入。

| 文档 | 用途 |
|---|---|
| [maidata-entry](maidata-entry.md) | `python -m maidata`、唯一 CONFIG、张量接口和 CPU 合成验证命令 |
| [新 encoder 设计](../../0wd-v3-refactor-docs/new-audio-encoder.md) | G₀ / S / L、共享块、局部三尺度及边界 |
| [Encoder 实施文档](../../0wd-v3-refactor-docs/encoder-execution-plan.md) | 已实现模块、完整张量协议、验证范围与 aux32 生产约定 |
| [current-runtime](current-runtime.md) | 新 Encoder 和保留的旧模型的实际状态 |
| [import-and-layout](import-and-layout.md) | 复制来源、独立 Git、排除项、外部依赖 |
| [audio-draft](audio-draft.md) | 现有一拍草稿、native 缓存、32/64 首模式 |
| [luna-training](luna-training.md) | 现有训练入口、恢复合同与旧配方范围 |
| [inference](inference.md) | 一拍草稿/native 推理和物化 |
| [data-and-tools](data-and-tools.md) | 仍存在的数据与资产工具 |
| [physical](physical.md) | 行内字段、L3、Physical 与时间语义参考 |
| [runtime-versioning](runtime-versioning.md) | 旧模型专属中央版本文件、恢复与流式协议 |
| [history](history/index.md) | 旧实验规格和计划，均非当前执行说明 |

旧测试报告、原始验证证据和生成的历史网站没有导入。新 `tests/` 仅使用合成输入验证 Encoder；历史记录不代表本副本已完成验证，也不要求恢复已退役入口。
