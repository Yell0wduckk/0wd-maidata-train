> 历史参考，迁移于 2026-10-10。原路径：`docs/history/experiments/query3x64-experiments.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 动态 query 3×64 对照实验

当前固定源码实现方案②：`dynamic-query3x64-lift64-v1`。方案①保留在提交 `07c83d1` / 分支 `experiment/query3x64-conv64`；分别上传固定源码，不在运行目录切换模型。

## 模型

① stem 和 depthwise residual 全部改为 64 通道。三个基础 query 为 `[3,64]`，动态分支为 LN64 → Linear64→64 → GELU → Linear64→192，末层零初始化，构造隔离 RNG。中心 `(i+0.5)/3`，带宽 `1/6`，分别用 sigmoid / softplus+1e-4 参数化。全 64 个 Mel 位置可见，缩放 sqrt64，三个输出拼接后投影192→192。卷积输出逐帧频率均值生成动态增量。没有额外辅助损失。

② 保留原 32 通道卷积，频率池化前逐位置增加带偏置 Linear32→64 + GELU；动态摘要、注意力 K/V 均使用升维后的特征。其余与①相同。

| 频谱 tokenizer | 参数 | 主要 MAC/帧 |
|---|---:|---:|
| 原动态6×32 | 56,812 | 374,784 |
| ① conv64 | 72,006 | 1,200,128 |
| ② lift64 | 61,030 | 507,904 |

池化自身54,022参数，其中动态分支16,768。以上计算不包括归一化、GELU、softmax、均值及反传；不是完整 A 的耗时比例。两组都保持每帧192维和P0尺寸，B计算图不变。实际A/B更新沿用 batch-costs，记录耗时、batch规模和峰值显存；不新增独立benchmark或性能门槛。

## 运行

原 `scripts/run-query6-ar.py` 名称保留为入口。当前源码默认选用 `configs/pilot261-dynamic-query3x64-lift64-a90.json` 及对应AR配置。先用 `scripts/prepare-query6-plan.py` 从已验收模板派生仅元数据计划，再执行此入口。保留原计划边界、视图、occurrence和顺序。

每组独立运行 **新A90 → 新后验/归一化 → 新P0 → 新song-context → GMM4 B50**。后续五个阶段统一使用本组 `A/last.ckpt` 完整训练payload，不能用推理用编号checkpoint替代；A的完成检查仍验证编号末轮及完整轮游标。

沿用full来源261首/280张谱，无验证、workers12、seed20260811。A4096行，90轮停止/155轮horizon，warmup79、Physical warmup988；B整曲16384软行预算、绝对轮91–140、自身50轮horizon、warmup50、FP32 AdamW1e-4/WD.01/clip1。保留当前0.65Note+0.15Delta+0.4Router+0.2Conditional、break翻倍预算和Physical规则。last每完整轮，编号每5轮和末轮，完整诊断仅末轮。不保存轮内checkpoint。

只重建依赖新A的学习缓存。原音频/GT/几何/Physical保持原生产者，不扫描旧载荷。源码和权重摘要记录来源；兼容范围仅维护runtime-versions.toml，实际结构/参数键shape/必要恢复状态仍检查。旧6×32及两种3×64的卷积参数布局不同，不能互作续训输入。

## 外层追加与续接

`scripts/experiment-queue.py --queue /absolute/jobs.json append --job /absolute/job.json` 追加独立实验。job JSON包含唯一id、绝对cwd/log、argv字符串数组，以及可选env和provenance。argv执行本组完整A→B launcher，退出码0必须表示全部阶段成功，不能忽略内部失败。

`... --queue /absolute/jobs.json run` 按追加顺序逐个启动。一个实验非零退出或可执行文件不存在时，记录失败、exit和历史并继续下个独立实验；失败A不会启动它依赖的缓存或B。外层全部尝试完后，只要尚有失败/中断，退出1供实验任务诊断，已成功任务仍保留。

`... run --retry-failed` 对每个失败/中断实验再尝试一次，已成功实验跳过。内部stage队列恢复last的optimizer/scheduler/完整轮游标/RNG；未完成学习缓存仍须先定位具体失败并按原合同修复，不伪造ready。OOM或合同不匹配不自动改预算/workers/配方，也不无限重试。

队列单消费者锁与状态锁分离，运行中可追加；消费完退出后再追加，需要重新启动消费者。若原消费者被杀且子进程仍存活，拒绝重复启动，Sol应确认原进程结束后续接。重用已有实验队列时必须保留原job、成功/失败状态和未完成任务；不得覆盖或中止其他运行。

## 验证范围

本地CPU定向检查覆盖独立公式、初始频带、参数量、mask/无效梯度、动态模块渐进梯度、完整A更新、诊断等价、计划引用、组合checkpoint、冻结A及B下一步恢复、推理加载，以及外层失败后继续和重试。CUDA成本与真实步进由Sol在正式训练中记录，谱面质量由用户评价。
