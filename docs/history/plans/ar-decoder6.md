> 历史参考，迁移于 2026-10-10。原路径：`docs/plans/ar-decoder6.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 原生 AR prior／六层谱面 decoder 实验

## 1. 目标与隔离

从当前 `main@abeb2c6` 创建独立 worktree `repos/ar-decoder6`，分支 `experiment/ar-decoder6`，计划保存为 `docs/plans/ar-decoder6.md`。优先使用应用 worktree 工具，嵌套仓库无法识别时使用 Git fallback。

同一 worktree 完成两项改动：

- 移除 diffusion prior 及其退役流程，将 AR prior 纳入模型的默认构建与统一 checkpoint。
- 谱面 decoder 从4层增至6层，latent AR decoder保持8层。

保持 main 的128维谱面 encoder／latent，不引入 latent256 或 joint-ar 的改动。保留其他 worktree、未跟踪文件和远端任务。

沿用“GPT-6.1-sol实现→本聊天独立复核→既有训练会话执行”。维护 pilot/full 入口，本次只训练 **pilot 261首／280谱，A90→自身学习缓存→B50**。

## 2. 模型、checkpoint 与退役范围

### 原生 AR prior

- 模型初始化直接注册唯一的 `style_prior = SongAR`。不构建 diffusion 占位模块，B 阶段不再通过替换模块类型安装 AR。
- AR 保持128维 latent、内部256维、8头、8层单 gate MoE、GMM4、原归一化和因果移位规则。
- 将 AR 使用的 block、整曲 latent 上下文、分块注意力和音频 KV 投影移至不依赖 diffusion 的模块，保持计算语义。
- A 中 AR 冻结、不执行前向、不进入 optimizer；B 冻结全部非 prior 模块，在已注册的 AR 上加载本次归一化并训练。A 保存的 AR 为确定性初始化且未更新的状态；B 首次启动使用该状态，续训恢复已训练状态。
- AR 初始化使用独立 RNG；新增第5、6层及其条件模块也隔离初始化，避免新增随机抽样扰动未改模块。不读取旧权重或构建 diffusion 来模拟历史初始化。

### 六层谱面 decoder

- 六层均采用现有共享注意力、四个 `256→512→256` 专家及 Time／Note 独立 gate。
- 保持256维、8头、激活、dropout、残差顺序、RoPE、因果 mask 和 gate 初始化规则。
- 同步扩展音频与 latent 条件投影、每层 latent 缩放参数、KV 列表、回滚快照、统计和参数量计算；消除相关四层硬编码。
- 独立 router、输出头、字段顺序、L3 和 Physical 预算不变。四个专家、router 内部四个 norm 等无关数字不作替换。
- 层数在架构配置中集中定义，构建、合同、参数统计与推理据此派生；本实验固定为6层。

### checkpoint 与兼容

- 新 A/B checkpoint 均保存一份完整 `model_state`，包含 AR。B 不再保存 `parent_payload` 加独立 `ar_state`，也不重复保存 A 中未训练的 prior。
- B 保留必要的 A90 来源、模型配置、缓存生产者、归一化、计划和冻结范围元数据；保存本阶段 optimizer、scheduler、完整轮游标与全部 RNG。
- B checkpoint 可独立加载生成，无需另找 A 权重文件。训练冻结验证继续检查非 prior 参数未变化。
- 中央版本设为 **24.00**，训练／推理只接受24.00；物化保留15.00–24.00行语义兼容。版本范围仅维护在中央文件。
- 明确拒绝旧 checkpoint、旧复合格式及错误层数／维度，不迁移权重，不新增源码 SHA 门禁。
- 原始 GT、Mel、几何、Physical 和整曲原始音频资产继续复用，不改写生产者身份；学习缓存由本次 A90 重新生产。

### 旧流程退役

删除仅服务 diffusion 的模型、损失、采样器、ControlNet／读出器／S1 流程、CLI、脚本和测试。仍被现役 AR、缓存或原始资产工具使用的组件先独立保留。

推理默认使用 AR，保留 posterior 参考与 disabled 诊断模式；移除 diffusion／历史 global 模式及专用采样参数。A 的未训练 prior 不允许用于 AR 生成。旧实验使用 Git 历史和原绑定源码，不提供新脚本兼容分支。

## 3. 数据与固定训练配方

- 复用已验收 song-first 计划的歌曲、排除记录、引用顺序、视图、occurrence=0和 batch 边界，seed `20260811`。
- A：4096目标行预算，从头训练90轮，scheduler horizon155轮；LR `0.000565685424949238`、79步 LR warmup、988步 Physical warmup。
- A 标准正态 KL、逐维 free-bits、β和有效位置计数沿用 main，不使用联合训练目标。
- A90 后生成自己的 posterior／归一化、P0和 song-context 缓存。
- B：冻结本组 A，训练 AR 50轮；16384整曲行软预算、绝对计划轮91–140、独立50轮 horizon、50步 warmup，其余配方不变。
- workers12、无验证集；last每完整轮，编号每5轮及末轮，完整诊断仅末轮。恢复完整训练状态，异常仅从上一完整轮恢复。
- full 使用相同架构和轮次，步数由自己的计划计算，本次不启动。
- 更新现役文档、模型参数量与训练参数量，分别列明 A 可训练、A 冻结 AR、B 可训练及完整模型参数；不将参数整理描述为质量提升证据。

## 4. 验证与验收

本地实现与独立复核覆盖：

- 默认模型直接包含 AR，注册参数和 checkpoint 中无 diffusion／ControlNet 张量；完整参数量与分项计算一致。
- AR 重构前后的独立参考验证：GMM 密度、mask、因果性、梯度、完整前缀／增量 KV 与条件 KV 复用。
- 六层 MMoE 的加权参考、专家／gate 梯度、padding、空条件、未来不可见、router 不读取目标信息；新增两层实际参与前向与更新。
- 六层普通／增量生成一致，Time／Note 切换、Physical 重试／回滚、流式／普通生成及字段预算回归；所有层的条件 KV 正确复用。
- 真实 A 更新时 AR 不变且不执行；真实 B 更新时仅 AR 改变。两阶段缓存生产、装载及父模型元数据绑定正确。
- 新 A/B checkpoint 保存与独立加载、完整轮恢复后的权重／optimizer／scheduler／RNG 重放；旧格式、跨计划和错误缓存拒绝。
- pilot/full 命令、默认 AR 推理及退役入口清理完整，无现役代码反向导入 diffusion。

训练会话执行受影响的 CUDA 检查：

- A：首 batch，以及首轮按现有元数据计算谱面 decoder padding 后目标行 token 总量最大的 batch，索引重复只测一次。
- B：首 batch，检查新模型构建、A 装载、冻结范围和恢复接口；不追加未改 AR 架构的独立压力测试。
- 使用原计划真实 batch，记录前反向、更新范围、计算复用、耗时与峰值显存。诊断用独立模型和输出，正式训练重新初始化。
- 传播诊断退出码，确认诊断进程及 GPU 占用退出后再训练。OOM、资源不足或合同不匹配停止并报告，不降预算、workers或替换数据。
- 不增加全量载荷扫描、独立 benchmark、性能或质量阈值。

## 5. 实现、复核与训练交接

1. 创建 worktree、保存计划，新建 GPT-6.1-sol 实现会话，指定绝对目录；禁止自行上传、训练或修改 main。
2. 实现会话提交并回报。本聊天独立审查 prior 生命周期、checkpoint 去重、六层 KV／回滚、梯度和恢复合同；问题交回同一会话修正。
3. 通过后，将最终提交、路径、启动命令、验证结果和未覆盖项交给“6.1sol训练与实验”（`01a0dede-fc12-7850-8583-6127d5aa6a7d`）。
4. 训练会话上传到 `ssh -p 47604 root@connect.cqa1.seetacloud.com` 的独立实验目录，仅核对本次传输，保留已有任务并接入队列，执行 CUDA 检查与 pilot 完整流程。
5. B50 后生成 `11231、11749、11739、11176、438`，沿用原难度、音频、BPM与时间参数；字段 seed `20260811`／温度0，latent seed `20260812`，latent与mixture温度1。下载、物化并记录退出码、截断、Physical绕过及 MiaCode 语义重解析结果。
6. 谱面质量由用户评价。交接后本聊天不主动轮询训练，不自动合并或删除 worktree。
