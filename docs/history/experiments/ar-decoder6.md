> 历史参考，迁移于 2026-10-10。原路径：`docs/history/experiments/ar-decoder6.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 原生 AR prior 与六层谱面 decoder

> 以下为六层实现的历史记录。当前 worktree 的八层合同见 [decoder8](decoder8.md)，启动和恢复见 [训练执行](../../luna-training.md)。六层 checkpoint 不作为八层续训输入。

本实验基于 `main@abeb2c6` 实现，提交 `1382009` 已合并到 main；当前维护目录为 `repos/maidata-baseline`，原 `repos/ar-decoder6` worktree 已删除。只维护本组独立 A90 → 本组学习缓存 → B50。正式执行仅 pilot 261 首／280 张 MASTER/Re:MASTER；full 入口维护但本次不启动。批准范围见 [计划](../plans/ar-decoder6.md)。

模型默认构建 `style_prior=SongAR`，没有 diffusion 占位。谱面 encoder／latent 保持128维；音频与谱面 decoder 为256维、8头。谱面六层 MMoE 每层共享注意力、四个 `256→512→256` FFN 专家、Time／Note 两个 gate。AR 八层单 gate MoE、GMM4，按同一次后验样本因果移位。全部 encoder 保持 dense FFN。

A 中 AR 冻结、不前向、不进入 optimizer。B 使用 A 保存的同一个原生 AR 模块，设置本组后验归一化，冻结全部非 prior 模块。AR 初始化及新增第5、6层和条件模块均隔离 CPU RNG；未改模块不因新增抽样而改变初始化。

| 参数范围 | 数量 |
|---|---:|
| 完整模型 | 36,119,626 |
| A 可训练／B 冻结的非 prior | 22,055,846 |
| A 冻结／B 可训练的 AR | 14,063,780 |
| 局部音频 encoder | 6,565,672 |
| 整曲音频 encoder | 2,439,168 |
| 谱面 encoder | 564,609 |

参数数值由固定词表的实际注册参数核对；参数增减不是谱面质量证据。

## 数据与配方

A 复用既有 song-first 计划的逐轮引用、视图、occurrence=0、歌曲／音频组和 batch 边界，seed20260811。A4096，90轮停止、155轮 scheduler horizon，LR0.000565685424949238、79步 LR warmup、988步 Physical warmup。KL、逐维 free-bits、β、L3和Physical不变。

A90 后生成自己的全曲 posterior／精确后验归一化、P0及 song-context。B16384整曲软预算，消费原计划绝对轮91–140，独立50轮 horizon、50步 warmup、LR0.0001、FP32。新消费配置仅有 A/B 调度元数据，覆盖155轮；原 A 模板中的旧阶段字段只为复用已验收计划而保留，不生成旧阶段命令。

workers固定12，无验证集，不构建验证 loader或best。last每完整轮、编号每5轮及末轮、完整训练诊断仅末轮。恢复完整轮的权重、optimizer、scheduler、游标及全部RNG。异常不保存轮内checkpoint，最多重做一轮。步数从各自实际计划计算。

GT、原始整曲Mel、Route几何和Physical继续使用原生产者合同。学习缓存由本次A90生成。正常启动只检查适用元数据和必要状态，不遍历旧载荷、重算特征或重新估计旧归一化。

## CUDA检查与队列

入口 `scripts/check-ar-decoder6-cuda.py` 使用真实训练参数。队列先在独立进程执行 A 诊断，选择首batch及首轮 `batch_size×max(target_rows)` 最大的batch；重合只测一次，报告也记录含history的实际decoder tensor padding量。A90及新缓存／B计划准备后执行B首batch诊断。诊断使用12 workers、原batch、独立模型及输出，报告前反向、更新范围、KV／编码复用、耗时、峰值allocated/reserved显存。

诊断不保存训练checkpoint，正式训练重新初始化。队列等待诊断进程结束并查询其PID已不占用GPU，再进入正式阶段；失败传播原退出码。OOM、资源不足或合同错误停止，不降低workers或预算。成功诊断仅在队列状态记一次，报告是证据，恢复时不要求它作为资产验收凭据。不追加全量扫描或独立压力测试。

## checkpoint与推理

A和B各自保存一份完整`model_state`。B仅保留A90来源／模型／缓存／计划元数据及自身训练状态，不包含`parent_payload`、`ar_state`或嵌套A权重。B可独立推理，不需要A权重文件；训练冻结验证仍逐张量比较非prior状态。

训练／推理只接受中央TOML允许的本实验版本；物化保留15.00–24.00行语义。版本数值和范围只维护在 `src/maimai_baseline/runtime-versions.toml`。错误层数、维度、旧复合checkpoint及错误缓存／计划明确拒绝，不迁移权重，不设源码或checkpoint SHA门禁。

推理默认AR，保留posterior参考与disabled诊断。A的`ready=false` prior不能AR生成。删除diffusion／历史global模式及其采样参数；旧实验恢复原Git历史及绑定源码。`--seed 20260811 --temperature 0 --local-seed 20260812 --latent-temperature 1 --mixture-temperature 1` 保持指定自由生成随机配置。

## 启动与验证状态

交付源码固定后，训练会话使用独立绝对输出目录：

```bash
bash scripts/launch-ar-decoder6-a90-b50.sh /root/autodl-tmp/maidata-ar-decoder6-20261001
```

pilot launcher 默认复用已验收 song-first A 计划，支持 `ORIGINAL_PLAN` 显式覆盖；来源路径和覆盖命令见 [训练执行](../../luna-training.md)。只派生新输出，不修改原计划，错误不退回旧模板。

维护的full入口为 `scripts/launch-full-ar-decoder6-a90-b50.sh`，仅明确要求full时执行。两个Python队列入口 `run-ar-decoder6.py`／`run-full-ar-decoder6.py` 默认只打印命令，显式`--execute`才运行。原`run-query6-ar.py`／`run-full-mmoe-decoders.py`也调用同一现役队列。

本实现会话只进行了本地CPU检查，没有SSH、上传、CUDA检查、正式训练或自由生成。验证证据和未覆盖项见 实现验证（源文档引用：`docs/evidence/ar-decoder6-implementation.md`，未导入）。谱面质量由用户评价。
