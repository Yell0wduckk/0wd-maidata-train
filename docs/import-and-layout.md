# 复制来源、布局与文档清理

## 来源和独立性

2026-10-10 按用户要求从以下工作树的已提交内容复制：

- 原目录：`D:/01YD/01code/codex/maimai-ai-charting/repos/main-new-audio-draft-beat`。
- 来源提交：`3de8e75706aa4b8a1f486b4a42125faca3214592`。
- 复制时原工作树没有未提交修改。
- 新 Git 根：`D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/src`。
- Python 包根：新 Git 根下的 `src/`。

新仓库用 `git init` 建立独立 `.git/` 目录，不拷贝原 worktree 的 `.git` 指针，不使用共享 object store，不配置原仓库为 remote，不导入原 Git 历史。来源提交只作迁移记录，不能在新仓库里直接 `git show` 旧提交。

设计目录 `../0wd-v3-refactor-docs/` 按要求与源码仓库并列，设计文档不属于内层 Git tree。

## 复制与排除

最初从 Git 跟踪文件复制 417 个文件，约 11.15 MB；352 个非 Markdown 源码/脚本/配置/资源在复制时逐项核对与来源一致。随后只整理文档并去掉下列测试依赖脚本，模型、生产脚本和配置没有修改。

保留：

- `src/` 下三个包的源码。
- `src/maimai_baseline/assets/` 的三个 JSON、`runtime-versions.toml`。
- `physical/csrc/` 的 C++/CUDA 源码。
- `configs/`、生产与数据准备脚本、`schemas/`、可选 MajRadarHost 源码。
- `pyproject.toml`、`uv.lock`、Git 文本属性和适用文档。

未复制或从新副本去除：

- `tests/` 及全部 fixture。
- `docs/evidence/`、独立验证/结果报告，以及 `docs/project-history/` 的生成网站/JSON/构建产物。
- 依赖测试夹具的 `scripts/benchmark-inference-cache.py`、`scripts/compare-decoder8-initialization.py`、`scripts/diagnose-batches.py`。
- 原 `.git`、虚拟环境、临时文件、数据、权重、缓存、训练日志和运行产物。

没有按 `diagnose*` 或 `benchmark*` 批量删除脚本；现有草稿末轮诊断等生产用途仍保留。依赖中的 pytest 及历史 pytest 配置未改动，它们不表示本副本含有原测试集。

## 文档清理

README、AGENTS、当前状态、训练、推理、数据、版本和目录 README 改为独立副本的实际情况。旧的 worktree 唯一路径、固定远端输出、已退役脚本、默认 B/四拍 prior、MajRadar/GMM 现役等陈述不再作为当前指南。

历史架构和实验计划移入 `docs/history/`，保留来源提示。验证/运行报告不作为源码文档保留；被排除证据的引用改为来源路径说明。旧 Git 命令只适用于原仓库，不是本仓库的历史恢复命令。

现有 22 拍/frozen-native 实现与未来 24 拍/可训练 G₀-S-L 设计分开说明。新设计没有通过本次复制进入模型，也没有改变现有 checkpoint/cache 合同。

## 已知外部依赖与路径限制

- native 音频需要外部教师源码快照、明确指定的 checkpoint 和已准备缓存。这些没有复制。
- `configs/run-audio-draft-16079.sh` 和后续 A launcher 仍保留源实验远端默认路径。本次不修改脚本，也不把它们推荐为新工作区默认启动方式。
- `src/maimai_baseline/data/training_dataset.py` 的 `--main-worktree` 默认仍定位原布局下的 `maidata-baseline`；不能声称默认动作会写到本仓库。显式选择目标时还需满足该工具自己的布局要求。已有已验收数据直接引用，不为副本重建。
- `tools/MajRadarHost/MajRadarHost.csproj` 的默认项目引用指向外部 MajRadar；可以显式传入项目路径。MiaCode、FFmpeg 与音频解码环境也没有打包。
- 新仓库没有给外部输入设置新的验收凭据或启动流程；复制时的源码一致性核对不是训练门禁。

## 本次验证范围

检查新 Git 独立性、源码复制一致性、测试/产物排除和维护文档本地链接。没有运行模型测试、CLI、环境安装、CUDA、正式训练、特征重算或旧资产全量扫描。
