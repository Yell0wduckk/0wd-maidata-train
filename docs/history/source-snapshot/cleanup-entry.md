> 历史参考，迁移于 2026-10-10。原路径：`docs/cleanup-entry.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 执行复用版上的通用训练入口与最小依赖

起点 b01af9f7b0135236bb9a0599f46dff7d1082cbae；分支 maintenance/cleanup-entry；worktree repos/maidata-cleanup-entry；参考 main@ec697e0。版本35.07，模型和缓存布局不变，35.00–35.07兼容范围保留。

## 修改合同

- 单阶段 `python -m maimai_baseline train --stage A|B|sft --experiment-config <配方> --epoch-plan-cache <固定计划> --output <目录>`。接受 dataset-root/manifest 或单独路径，显式路径优先。保留 fit、B脚本与SFT适配入口。不会隐式运行缓存、诊断、下一阶段或生成。
- 使用计划给定的记录、视图、occurrence、顺序与batch边界，不按pilot/full再次筛选。实际seed、LR、warmup、stop、horizon、offset、precision来自绑定配方；workers默认12。
- 删除正常读取/恢复中重复来源哈希、生产者字典和固定实验配方准入。保留实际布局、索引、mask、坐标、表示区别、严格权重加载、数值失败、完整轮optimizer/scheduler/游标/RNG及实际计划边界。
- 新B只加载指定A一次，B续训直接读自包含Bcheckpoint和最终消费计划。完整song-context不打开P0 sidecar/profile；主进程准备必要元数据一次，worker只打开消费载荷，不打开未用feature数组。旧缓存只在内存适配。

## 明确保留

保留b01af9f的音频/timing/IPC复用、dense整理、骨架query几何、Physical启动准备与缓存/载荷读取。Physical文件仅迁移来源校验删除，不改计算或按需加载。骨架/model/loss不改，谱面encoder/prior/SFT保留且不重新启用latent。先建loader再初始化模型；不增加逐步同步或诊断。

本任务不是性能实验：此前70秒尖峰后回落，无法据此认定35.06整体更慢。当前远端训练及父worktree不变；本分支不自动训练。

## 验证

只运行受影响CPU入口、缓存、恢复和父版复用/骨架行为测试；临时产物全部在本worktree的.test-temp。重点覆盖任意配方/最终计划、A更新保存恢复、B不依赖旧A与sidecar、来源说明变化可读、真实布局错误拒绝及原默认初始化/监督不变。不运行CUDA、profiler、重放、旧资产扫描或全套历史测试。

### 本次结果

62个不同CPU用例通过，索引为 `.test-temp/passed-evidence.json`。原始记录：`contracts2.xml`（24通过，另1旧夹具失败随后修复）、`data.xml`（10通过）、`regression.xml`（26通过，含该修复复测）、`launcher2.xml`（2通过）。初次旧夹具缺少新接口参数/真实计划引用的失败记录保留，没有修改生产逻辑去适配无效夹具。

- 单阶段A/B/SFT派发、路径覆盖、非固定配方/计划起点、初始化顺序、恢复前绑定seed/配方通过。SFT只验证派发；当前GT骨架损失仍未适配独立SFT。
- 真实CPU A更新、完整轮保存/恢复、下一次更新、旧35.02同布局加载、disabled推理通过。谱面encoder/prior仍未进入A执行。
- 真实CPU prior更新、自包含B保存/重载、optimizer/scheduler/RNG下一次更新一致、非prior冻结通过。原A文件删除后B仍可加载；这不表示当前latent-disabled A有已训练的posterior产物。
- B完整song-context消费不打开旧P0/profile/未使用feature数组，spawn后worker复用已准备元数据。错误归一化、latent宽度、记录映射、参数缺失与计划边界仍拒绝。
- audio/timing/spawn、LRU、骨架复用、BF16、GT多分支、Physical梯度、训练/增量输出和回滚通过。
- `git diff --check`、改动Python语法与文档链接检查通过。相对b01af9f，model/、loss/、time/、batch.py、batch_transport.py、data/dataset.py、data/song_dense.py、rhythm_scaffold.py及physical/data.py无差异。本项只是本次静态证据，不加入启动门禁。

未运行CUDA诊断、单batch重放、profiler、正式训练、完整历史测试或旧资产扫描。没有速度/质量结论；父分支与正在训练的远端源码未改。

## 入口迁移说明

`run-gt-scaffold-a90.py` 和shell launcher现在需要显式传入数据根及固定计划，不再硬编码远端数据目录；shell不再自动接五首生成。默认线程环境和文件描述符上限保留。五首提取/生成仍可独立显式调用。旧队列模块作为单阶段命令适配器，不再隐式A→缓存→B。

数据与缓存来自main的读取合同，但PhysicalDataset仍使用父版启动时准备方式，没有迁入main的按需PhysicalDataset改造。音频变体/record的timing在准备期核对一次，窗口内复用。

模型checkpoint仍按中央版本范围与实际结构加载；缓存生产软件版本只保留出处。B的随机后验归一化与确定性mu仍不能混用。A恢复使用保存的数值设置，并校验实际完整轮进度及已记录的引用/batch边界；不依赖旧run.json或生产路径一致。

保留显式计划准备工具，仅修复其对新配方返回值的调用；它们不进入训练启动。兼容a_parent模块去掉原A90/生产者门禁，实际参数仍由checkpoint loader严格加载。


## 固定提交

实现与测试提交：`39889ca5b84c39ca25668bf787ca27f247970cde`。后续文档提交仅补迁移记录，不修改源码。
