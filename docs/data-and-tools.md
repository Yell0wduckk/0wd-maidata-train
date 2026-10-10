# 数据、缓存与工具

本副本保留源码、配置和维护文档；数据、权重、测试与运行产物不属于这次源码复制的内容。普通运行应从本副本的 src 导入，不依赖原 worktree 的 Python 源码。当前路线见 [current-runtime](current-runtime.md)。

## 当前一拍路线的资产

| 资产 | 实际作用 |
| --- | --- |
| dataset、chart/spatial/song-chart 缓存 | 已准备的谱面表示、视图与完整歌曲索引 |
| epoch-plan cache | 固定 curriculum、视图分配、occurrence、顺序、完整音频组与 batch 边界 |
| native 教师音频缓存 | 冻结的 22 拍 P0/C0/C1、mask、绝对拍坐标、插值几何及生产者绑定 |
| 草稿验证输入 | 默认独立 32 首，或显式 full 训练输入留出 64 首；后者仅对草稿留出 |
| 干净草稿缓存 | 指定草稿 checkpoint 生成的 canonical 每拍 128 维 token，供 identity/mirror 共用 |
| 条件噪声统计 | 从 A 训练计划引用的条件计算；full 留出的 64 首仍属于 A 训练 |
| Physical 资产 | 所选 L3/Physical 训练或推理所需的路径、源行映射及区间缓存 |

native 当前每窗 P0/C0/C1 形状分别为 176×256、528×256、264×256，全曲 P0 bank 由各窗中心 16 拍拼接。新设计的 24 拍输入不能替换现有 manifest 中的 22 拍合同。草稿与验证模式详见 [audio-draft](audio-draft.md)。

## 代码中保留的准备入口

下列工具是按指定任务准备资产的入口，不是日常启动时必须重跑的阶段：

| 入口 | 用途 |
| --- | --- |
| scripts/prepare-training-dataset.py、scripts/prepare-full-dataset.py | 按给定来源准备数据 |
| scripts/prepare-bpm96.py、scripts/prepare-note-view.py、scripts/build-slide-geometry.py | 对应的音频、短 note 视图和 Slide 几何资产 |
| scripts/prepare-interval-physical.py、scripts/repack-interval-physical.py | Physical 区间资产及布局转换 |
| scripts/prepare-song-charts.py、scripts/prepare-song-audio.py | 完整歌曲索引与独立全曲音频输入 |
| python -m maimai_baseline prepare-audio-draft-data | 当前草稿与 A 的输入和计划 |
| scripts/export-draft-native-audio.py | 在独立生产过程中调用明确绑定的外部教师，导出 native 音频缓存 |
| python -m maimai_baseline build-draft-cache | 指定草稿 checkpoint 的干净条件缓存与统计 |

工具名中的 full 或历史架构名不表示新分支已适配。chart encoder、mu/prior、B/SFT 的历史工具不能仅凭文件存在就加入当前一拍训练流程。具体选项以 [CLI](../src/maimai_baseline/cli.py) 和所选脚本为准。

native 导出脚本的外部教师来源是显式生产依赖，与普通训练从本副本导入源码是两件事。保留真实生产者和原特征归一化绑定，不能将资产生产者改写为当前消费者，也不能重估统计来规避不匹配。新 encoder 重新训练后如何导出或在线提供特征、如何改接草稿及是否重建缓存仍待确定，见 [设计文档](../../0wd-v3-refactor-docs/new-audio-encoder.md)。

## 复用与检查范围

默认信任已验收、内容未变的只读资产。保留正常读取过程中的形状、dtype、有限值、mask 和身份检查；不因新目录、新源码身份或新架构计划，提前遍历所有载荷、递归重哈希、解压、重算音频特征、重估后验统计或重建几何缓存。

新增或内容改变的资产按适用合同验收一次；上传完整性检查限于本次实际上传的新增或改变文件。具体错误只触发受影响部分的检查，不自动升级为全量重验。历史报告不能成为新启动凭据，可选诊断也不构成隐含前置条件。

完整轮恢复与固定计划见 [luna-training](luna-training.md)，版本和来源语义见 [runtime-versioning](runtime-versioning.md)，生成输入见 [inference](inference.md)。
