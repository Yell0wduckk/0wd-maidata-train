> 历史参考，迁移于 2026-10-10。原路径：`docs/cleanup-reuse.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 纯清理版＋去重与必要入口修复

2026-10-07。起点 `9877bd0fe892fa3d44d9a9fbf939e3b24302af82`，分支 `maintenance/cleanup-reuse`，唯一实现目录 `repos/maidata-cleanup-reuse`。中央版本35.06；模型参数与缓存布局不变，保留35.00起的兼容范围。未合并 main、上传或启动训练；正在运行的纯清理版不受影响。

## 手工移植范围

以 `36aa002` 的局部差异为参考，保留旧文件组织和未触及的压缩写法，没有整笔 cherry-pick 或以 main 文件覆盖旧文件。

| 分组 | 改动 | 保持的行为 |
|---|---|---|
| 数据与时间 | dense A 外层只保留目标引用计数；整曲音频由内层准备；A 不传输 SFT timing；复用 PreparedTiming | 原监督、上下文映射、Physical 读取／支持计算、SFT timing 消费不变 |
| 时间生命周期 | 每条记录只解析一次完整 timing；按 root、PID 分开绑定，spawn 不携带 mmap；IPC 元数据4096项 LRU | BPM／offset／meter／HS／end 均参与原完整 key；不缓存增强 item 或模型激活 |
| 骨架 | 同一步完全相同 item 复用控制器和公开特征；八层共享 query 几何 | 不跨增强视图；每层参数化投影、norm、dropout 和梯度独立；router 仍保留自己的布局 |
| 运行元数据 | 启动来源快照供执行记录和 A/B 保存复用；horizon 更新数传入 fit | 来源仍可追溯；缺少来源不触发保存时补扫；未改为 SHA 门禁 |
| 入口 | A/B 从绑定数值配方读取 seed、LR、warmup、停止轮、horizon、计划起点 | 旧 fit／B／SFT 入口、路径结构、固定计划和非数值合同保留 |

旧版已在启动期进行音频时间表核对，直接保留，包括原资产读取和检查，不移植清理版逐窗口核对再修复。保留谱面 encoder、posterior、prior、SFT；latent 仍关闭。

## 数值入口与恢复

A 沿用 `fit --experiment-config`。新增可选 `--seed`、`--warmup-steps`、`--physical-warmup-steps`、`--plan-offset`；已有 LR／stop／horizon 参数继续使用。显式参数覆盖配方默认；未传参数在续训时取 checkpoint 已保存值，冲突的 seed／LR／warmup／horizon／offset 明确拒绝。停止轮可以延长但不得超过原 horizon。

计划的原 recipe、采样 seed、视图、occurrence 和边界仍保留。运行 seed 控制初始化和 NOTE 增强，不重新抽样计划。配置与计划仅允许 A 的数值执行项不同，其余数据合同仍核对；实际消费 `[offset, offset+epochs)`，scheduler 使用 `[offset, offset+horizon)`，不足时指出范围。不会补计划或重打包。

新初始化仍先创建 loader，再构造模型并完成原 RNG 初始化。续训 checkpoint 只加载一次，在 dataset 创建前恢复数值设置；optimizer／scheduler 和 RNG 在训练迭代前恢复。模型严格加载、完整轮游标及原恢复检查保留。

B 沿用原 JSON 格式，数值字段不再与固定 B50 常量比整份相等。可选 `offset` 缺省采用绑定计划的 B 起点；LR、seed、warmup、stop、horizon、weight decay 和 gradient clip 由该 recipe 消费，游标及 scheduler 校验使用相同范围。原 A 生产者、缓存、布局和数据计划检查未移除。

**保留的限制**：当前 GT 骨架／latent-disabled 主模型仍拒绝含已训练 prior 的 checkpoint，独立 SFT 尚未适配当前损失。这次修复外围数值执行，不重新启用这些算法路径。B 的 prior 数学更新和状态恢复可以独立测试；不能据此声称当前模型支持完整正式 B 队列。

## 验证和证据

仅使用本机 CPU，临时文件和依赖放在本 worktree `.test-temp`。没有 CUDA 诊断、单 batch 重放、profiler、正式训练或旧资产扫描。

验证覆盖：音频读取及 spawn 等价、timing 的不同 root／完整 key／LRU、骨架部分与完全复用、padding／空窗／增强身份、八层几何输出／梯度／一次更新、BF16、多分支、Physical、增量／回滚、默认初始化 RNG、实际 A 更新／保存／恢复、非默认入口数值、prior 单独更新与 optimizer／scheduler／RNG 恢复、SFT 派发。

共享骨架几何在 dropout=0及0.2、固定 RNG 的对照中，输出最大绝对差均为 **0.0**；梯度和一次更新对照通过。音频和 collation 张量对照使用逐元素相等。上述证据不代表真实训练速度改善。

原始 JUnit XML 保留在 `.test-temp`，最终结果及已知限制见本节后续记录。初始测试中的本机 soundfile 缺失已通过仅安装到 `.test-temp/deps` 解决；测试适配问题的失败记录保留，不冒充首次全部通过。

### 最终结果

累计 **41个不同 CPU 用例通过**（重复复测不重复计数）。证据对应 `data-tests.xml`、`input-tests.xml` 的通过项、`reuse-tests.xml`／`reuse-retest.xml`、`audio-tests.xml` 的音频通过项、`final-entry.xml`、`real-replay.xml` 的通过项，以及最终 `final-changes.xml`。去重后的用例索引为 `.test-temp/passed-evidence.json`。

非默认 A 的 seed53、LR warmup3、Physical warmup9、计划 offset7，已贯通真实更新、完整轮保存、恢复及推理导出；同布局35.02加载亦通过。该测试发现 checkpoint 仍把 Physical warmup 强制对照原计划默认值，已局部改为保存并核对实际训练值，未删去 Physical 结构检查。

非默认 B 的 seed／LR／warmup／offset／stop 已在原入口的 CPU 调度夹具验证；真实 prior 更新后，严格恢复权重、optimizer、scheduler、Python／NumPy／Torch／后验 RNG，下一次更新一致。保存期间禁止调用来源扫描的哨兵通过。

没有把历史测试全部判为通过：

- 原版 `9877bd0` 只读对照同样复现了 prior checkpoint 被当前 latent-disabled 合同拒绝、独立 SFT 损失未适配，以及旧 SFT CLI 夹具缺少现役必需参数三项失败，记录在 `base-limitations.xml`。没有通过放宽模型合同让它们通过。
- 初次较宽选择的旧输入测试还包含已过时的 Physical bitplane／旧 attention preset／CLI 夹具，以及 Windows 沙箱不允许创建硬链接的用例；这些未作为本次通过证据，也没有扩展修整。真实音频读取测试在补齐临时依赖后通过。
- 未运行完整历史测试、GPU验证、正式 A/B/SFT 训练；没有性能结论。

## 提交和不纳入项

1. `af1eeb3`：数据、时间和音频去重。
2. `74cec6b`：骨架与运行元数据复用。
3. 当前文档所在提交：数值入口修复、35.06、测试及指南。

不引入通用 `train --stage`、任意数据路径解析、校验大删减、Physical 按需读取改造、无关文件拆分或骨架格式化。原 launcher 与配置文件不修改；不新增正式训练的诊断阶段。本次不交接训练，后续性能仍需用户授权的真实训练验证。
