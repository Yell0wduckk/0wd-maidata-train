> 历史参考，迁移于 2026-10-10。原路径：`docs/history/experiments/song-dense-no-latent-plan.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 全曲 dense decoder：禁用 latent，重新训练 A90

本文件为用户批准的实施合同。起点为干净的 `decoder8-song-dense@f71b77ee2690bf496158f5095921594d9411e1c4`。本次用户选择新建独立 worktree、固定种子从头训练，以及保留结构但冻结并跳过 latent 模块。

## 1. 起点与分工

- 分支：`experiment/decoder8-song-dense-no-latent`。
- 唯一源码、导入、测试和提交目录：`D:/01YD/01code/codex/maimai-ai-charting/repos/decoder8-song-dense-no-latent`。
- 本文先保存并提交，再交既有 gpt-6.1-sol/high 实现会话 `01a10b61-af35-7c10-afd8-f537574c99e2`。
- 实现会话完成修改、必要 CPU 验证、简洁结果文档及提交后，向主会话 `01a10311-739d-7642-8c1d-6cfe48ae64e1` 回报准确提交、验证结果、未执行项和阻断项。用户明确授权此实现完成回报。
- 主会话独立复核，阻断问题交回修复，通过后交有效训练会话“6.1sol训练与实验” `01a10bc8-e50b-7353-8717-445f41935bd5` 上传与训练。
- 不修改父 worktree、旧产物或正在运行的固定源码。本次只训练 A90，不执行 B、学习缓存生产或自动五首生成。

## 2. 模型改动与保持范围

### 关闭完整 latent 路径

- 谱面 encoder、posterior 均值/方差头、prior，以及 decoder/router 的 latent 专用投影和 gate 均冻结，不进入 optimizer。
- A 前向不调用谱面 encoder、posterior 或 prior，不采样 latent，也不计算 latent K/V。
- decoder/router 完全跳过 latent 注意力及残差注入，不能用有效的全零 latent 替代关闭，以免投影 bias 仍产生贡献。
- 移除本实验实际损失中的 KL、其升温和 posterior 统计；保留原 L3、Physical 及所有原权重。
- 保留原模块构造顺序、参数注册和固定种子初始化。禁用参数仍随完整模型保存；分别报告总参数量和实际可训练参数量。
- 一个有实际消费者的 latent 启用状态贯通模型、训练和推理合同，不添加实验名称标签。不能直接用旧 `chart_style=none` 改变模块构造与其他模块初始化。

### 原 dense 语义保持不变

- 八层 decoder 继续读取全部因果谱面历史和全曲 P0 音频。
- router 保留父版局部谱面/局部音频范围，不顺带扩展。
- 音频保持父版22拍片段编码、中央16拍拼接全曲 P0，不引入后来的整曲连续音频编码。
- 保留父版16拍目标窗口、数据监督、NOTE增强、branch ownership、Physical映射和推理回滚。
- 不带入后续四拍 K/V 历史、追加末尾空窗、四拍目标窗口等实验改动。
- 保留音频 encoder、融合和 decoder/router 正常梯度，不因禁用谱面 encoder 误关音频条件。
- 数据整理只移除已无消费者的谱面后验张量准备与传输；谱面历史、监督和音频所需数据仍保留。不重建原始资产或修改固定计划。

## 3. 版本、推理与恢复

- 中央版本34.00，训练/推理只接受本实验版本，历史物化范围扩展。
- 从头训练，不导入旧 dense A90 权重；旧 latent 模型不能作为同语义续训输入。
- 推理默认并固定 latent disabled，可直接加载本组 A checkpoint；显式请求 AR、posterior 或 latent 采样参数时清楚报错。
- 谱面字段 seed、temperature、音频条件和 Physical约束继续有效；默认不传 `--trace`。
- 完整轮恢复保留模型、实际 optimizer 参数集合、scheduler、游标、固定计划绑定和仍使用的 RNG。
- 删除本实验不再使用的 latent 专用随机状态依赖，不增加源码 SHA白名单、验收凭据或旧资产扫描。
- 更新当前运行说明、参数统计和独立 A-only 启动入口，避免沿用 A→缓存→B 队列后再跳过阶段。

## 4. 必要验证与主会话复核

仅运行受影响 CPU 正确性测试。TEMP/TMP、pytest basetemp、临时 checkpoint、日志与全部测试产物都在本 D盘 worktree。不运行独立 CUDA诊断、单 batch重放或 profiler，不重跑全部历史测试。

- 真实禁用：调用计数或失败哨兵证明 encoder、posterior、prior、latent K/V 路径均未执行；禁用参数无梯度、不进 optimizer、更新前后不变。
- 初始化与更新：相同种子下保留模块初始化与父版一致；音频、router和八层 decoder 正常获得梯度并更新。
- 条件与因果：固定音频和历史，改变未使用的 latent/后验输入不改变输出；未来谱面不泄漏，全曲历史和音频范围保持父版。
- 监督与损失：原目标、字段mask、权重、Physical映射、batch边界及更新数不变；总损失原L3+Physical，没有KL。
- 推理与恢复：固定条件并关闭dropout，训练/增量推理对齐；跨窗缓存、回滚和replay可用；完整轮恢复后下一次更新一致。
- 正式入口：小规模夹具贯通 A更新→保存→恢复→disabled推理；启动命令仅包含A，无隐含诊断、缓存或B阶段。

实现会话交准确源码和简洁结果文档。主会话读实际diff，重点复核真实关闭、初始化、冻结、损失与A-only入口，运行针对性测试后交接。只复测受影响部分；浮点误差记录实测量级，不增加性能或loss门槛。

## 5. 正式 A90

- SSH：`ssh -p 47604 root@connect.cqa1.seetacloud.com`。
- 独立根：`/root/autodl-tmp/maidata-decoder8-song-dense-no-latent-20261006`。
- 源码：`source-<复核后的完整提交>`。
- 入口：`scripts/launch-song-dense-no-latent-a90.sh <实验根>`。
- pilot261首/280谱，复用 `/root/autodl-tmp/maidata-decoder8-20261003/a-plan`。
- 原引用、视图、occurrence、顺序、完整音频组和batch边界不变。
- A4096、BF16、seed20260811、workers12，保留父版persistent workers行为。
- LR `0.000565685424949238`、LR warmup79步、Physical warmup988步、stop90/scheduler horizon155。
- 无validation/best；checkpoint仅完整轮保存，last每轮、编号每五轮及末轮；完整诊断仅第90轮。
- 记录真实轮级训练、保存、整轮墙钟、更新数与峰值显存，不用独立性能实验代替真实训练。
- 直接正式A训练，不加入CUDA资格、缓存生产或B阶段，也不伪造已通过诊断标记。
- 训练会话沿用现有排队安排，不抢占旧运行，只核验本次上传内容。遇真实OOM/错误记录具体轮次和batch，不静默降低配方。
- 用户已授权上传与A90，无需再次确认。训练进度、异常和结果只在训练会话报告；主会话交接后不主动跟踪。
