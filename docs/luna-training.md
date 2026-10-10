# 训练入口与完整轮恢复

本文记录复制源码已有的训练行为，供后续明确配方时参考。本副本未因此启动训练，也未选定新 encoder 的教师、历史 checkpoint 或停止轮数。当前实现见 [current-runtime](current-runtime.md)，未来结构见 [新音频 encoder 设计](../../0wd-v3-refactor-docs/new-audio-encoder.md)。

## 当前一拍草稿入口

从本仓库根目录设置 PYTHONPATH=src，通过 python -m maimai_baseline 调用入口；数据、计划、checkpoint 与输出路径均须显式绑定实际任务。历史 worktree、远端目录和旧实验的 60/90 轮安排不构成本副本的执行配方。

| 入口 | 所需核心输入与作用 |
| --- | --- |
| prepare-audio-draft-data | 给定数据、缓存和完整轮计划，准备草稿与 A 输入 |
| train-audio-draft | 草稿计划、外部 native 音频缓存及输出目录 |
| build-draft-cache | 明确指定的草稿 checkpoint、A 计划及 native 音频缓存 |
| train --stage A | A 计划、干净草稿缓存、native 音频缓存及 L3/Physical 所需资产 |

--scheduler-epochs 是 scheduler 的完整轮 horizon，--stop-after-epochs 是绝对停止轮；续训时停止轮不是“再训练多少轮”。二者分别指定，并满足各入口实际计划覆盖检查，不能仅凭一个轮数推断另一个。CLI 和 JSON 配置中的已有数值属于继承配方；未来 encoder 的新配方尚未确定。

训练 DataLoader workers 固定为 12。保留原计划的 curriculum、identity/mirror 分配、occurrence、顺序、完整音频组和 batch 边界；2:1 是抽取分配比例，不能改成每个窗口重复训练三遍。不得用 CPU 测试夹具参数替换真实训练设置。入口见 [cli.py](../src/maimai_baseline/cli.py)。

B/SFT 等历史训练入口存在，不表示一拍草稿路线或未来 encoder 已完成相应适配。需要先明确所选模型的实际输入、冻结范围与 checkpoint 兼容性。

## 验证与保存

草稿准备有默认独立 32 首和显式 --validation-from-training --validation-song-count 64 两种模式，详见 [audio-draft](audio-draft.md)。full 64 首仅从草稿训练排除，A 仍保留这些歌曲。不得把它写成 A 的独立验证集。

草稿训练默认要求声明的验证输入，每 5 轮及绝对末轮执行验证，重合时只执行一次。字段 CE、时间项和 count 项分别按其有效支持数汇总。best 仅在轮末验证改善时更新，不能把训练 loss 当作验证 best。训练状态与 DataLoader RNG 按 checkpoint 恢复；验证有独立的 loader RNG。实现见 [training/audio_draft.py](../src/maimai_baseline/training/audio_draft.py)。

当前完整轮合同为：last 每个完整轮保存，编号 checkpoint 每 5 轮及末轮保存，完整诊断仅在末轮收集；轮内和异常退出不另存 checkpoint。中断从上一个完整轮恢复，最多回退一轮。末轮指本次明确的绝对停止轮，不固定等于历史实验轮数。后续 A 使用哪个草稿 checkpoint 必须明确指定，不能自动把目录中的 last 或 best 视为已选权重。

本副本继承的一拍路线中，A 的 validation 关闭，不创建验证 loader 或验证 best；草稿的 32/64 首验证不属于 A 验证。未来新 encoder 的验证安排需随新配方明确，不由本次文档迁移改变。

## 恢复与资产复用

--resume 恢复完整训练状态，包含模型、optimizer、scheduler、完整轮游标和 RNG；A 的条件噪声与 loader 随机状态也必须保持恢复语义。严格检查结构、数据绑定和计划身份，不能把旧四拍 latent 或旧计划 checkpoint 伪装成同语义的一拍续训。--init-checkpoint 与完整状态恢复是不同动作，应按实际入口说明选择。

当前副本的独立目录不使已验收、内容未变的资产失效。日常启动复用现有元数据和合同绑定，不因换目录重新全量扫描、解压、计算特征或重估统计。新增或改变的资产仅按适用合同验收一次；发生具体不匹配时，只停止和检查受影响步骤。版本与来源的区别见 [runtime-versioning](runtime-versioning.md)，资产说明见 [data-and-tools](data-and-tools.md)。

模型质量以用户查看自由生成谱面后的评价为主，validation loss 是诊断信号。CPU 检查、CUDA 原 batch 检查、实际训练、成本测量和自由生成质量是不同证据，不能互相替代；新 encoder 的这些证据尚未产生。
