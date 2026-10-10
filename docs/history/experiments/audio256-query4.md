> 历史参考，迁移于 2026-10-10。原路径：`docs/history/experiments/audio256-query4.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# lift64 音频256维、动态4×64 query

本 worktree `audio256-query4` 从 lift64 `c2797c1dc2f26ea174fb481cb123db21ead1706e` 派生，独立运行新 A90 → GMM4 AR B50。旧3×64实验继续使用自己的源码与资产。

## 音频结构

保留32通道卷积及逐频率带偏置 Linear32→64 + GELU。四个64维基础query零初始化，中心 `(i+0.5)/4`、带宽 `1/8`，分别通过sigmoid及softplus+1e-4训练；全64个频率位置可见。逐帧频率均值进入 LN64 → Linear64→64 → GELU → Linear64→256，末层零初始化、构造隔离RNG；增量reshape为4×64后加到基础query。打分缩放sqrt64，四路拼接后256→256带偏置投影。动态分支20,928参数，池化86,984参数，含卷积和lift的tokenizer93,992参数、主要MAC/帧548,864（不计归一化、激活及反传）。

局部encoder为256维/8头/FFN1024、层数(1,1,2)；独立整曲encoder为256维/8头/FFN1024、2层。每头32维RoPE、时间分辨率、范围、dropout及mask语义不变。融合和P0/P1/P2均256维，窗口P0为[176,256]。局部encoder6,565,672参数，整曲encoder2,439,168参数。

双decoder仍256维，router仍128维；音频K/V输入均改为256维。相同宽度cross-attention使用PyTorch合并Q/K/V参数布局，仍为可学习投影。谱面后验encoder与latent保持128维，GMM4主干8层/256维/8头/FFN1024及4分量联合高斯不变，仅音频输入投影扩宽；AR共9,846,916参数。

## 入口与计划

沿用 `scripts/prepare-query6-plan.py` 和 `scripts/run-query6-ar.py`。默认配置为 `configs/pilot261-dynamic-query4x64-lift64-audio256-a90.json`，B消费配置为对应 `-ar-b50.json` 和 `-whole-song-50.json`。入口先打印命令，正式运行显式传 `--execute`。

从既有计划元数据派生新A计划，保留batch边界、歌曲顺序、视图和occurrence。B使用绝对轮91–140，按原整曲16384行软预算打包，同歌不拆分，horizon为这50轮实际batch数之和。原始full来源261首/280张谱，无验证，workers12，每谱每轮一次，seed20260811。

队列仅执行 A90 → 后验和归一化 → P0 → song-context → B计划 → GMM4 B50；缓存与B读取本次 `A/last.ckpt` 完整训练父状态。A从头初始化，不加载旧A。旧原始Mel、GT、几何与Physical输入继续复用；原始整曲Mel包的历史合同独立保留为INPUT_CONTRACT，不改写旧producer或遍历重验旧载荷。新学习缓存绑定实际新A，P0与song-context使用新的256维布局。

## 固定配方

A：4096行预算、90轮停止/155轮scheduler horizon、LR 0.000565685424949238、79步LR warmup、988步Physical warmup。B：冻结全部A，重新初始化GMM4，50轮/自身horizon、50实际步warmup，FP32 AdamW LR1e-4/WD0.01/clip1。每张谱每轮一次完整后验抽样，每音频组复用一次投影，每batch一次更新。

监督保持 `0.65 Note + 0.15 Delta + 0.4 Router + 0.2 Conditional event`，break预算和Physical合同不变。last每完整轮保存，编号每5轮及末轮保存，完整诊断仅末轮，无轮内保存。完整轮续训恢复optimizer/scheduler/游标及全部RNG。

## 兼容与成本

版本与三个操作支持区间仅维护 `src/maimai_baseline/runtime-versions.toml`。本分支训练/推理接受新架构，旧192维模型及学习缓存明确拒绝；物化保持旧行语义兼容。来源SHA仅记录，不恢复哈希准入锁。完整Q/K/V投影、AR增量KV和普通/流式推理均覆盖新布局。

特征载荷约增加33%，音频中随宽度平方增长的矩阵运算约1.78倍，不能视为整个A耗时比例。B的缓存读取及音频K/V投影增加，latent主干不变。Sol在真实A/B更新记录batch规模、耗时和峰值显存，不另设benchmark或质量门槛。

## 队列与交接

固定提交上传至SSH22857的独立目录，核对本次上传新增内容。外层使用现有 `scripts/experiment-queue.py append/run`，保留原jobs和正在运行的实验；某任务失败后继续其他独立任务，`--retry-failed` 从完整轮续接，失败A不进入其依赖B。OOM、资源不足或真实合同不匹配报告，不修改配方或无限重试。Sol在原实验任务内确认真实步进；主任务交接后不主动轮询。

## 本地验证记录

2026-09-29，Windows CPU、D:/Anaconda/python.exe。受影响测试共163项：155通过，8项因无CUDA跳过。两组执行分别103通过/4跳过、53通过/4跳过，其中2项重复；另1项旧版本兼容边界测试通过。覆盖query公式与渐进梯度、mask、真实A更新、A下一步恢复、B冻结/更新、B完整轮optimizer/scheduler/RNG恢复、计划引用、缓存生产/拒绝、QKV独立参考、AR因果与KV、流式生成及原监督预算。

执行集合：test_query3x64、test_dynamic_query6、test_frequency_pooling、test_song_ar、test_query6_workflow、test_ar_checkpoint、test_song_audio、test_song_context、test_audio_fusion、test_dual_loss_budget、test_audio256、test_p2_fusion、test_runtime_versions_streaming、test_cache_production，以及test_encoder_capacity中的形状/更新和稠密参考/mask两项。均使用本worktree的src；测试临时checkpoint迁到工作区磁盘，避免系统临时盘空间不足。运行入口help、中央版本输出和git diff --check通过。

未执行远端CUDA更新、正式训练、全量载荷扫描或旧缓存重建。CUDA成本及实际训练步进由Sol记录，谱面质量尚未评估。
