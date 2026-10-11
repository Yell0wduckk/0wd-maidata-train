# 每4拍一个Z：Drafter训练与Decoder适配

2026-10-11 用户授权将 `draft_z_stride_beats` 设为4，交给既有训练会话“主 Decoder 训练：关闭 Z、GT plan”完成Z训练和Decoder微调，其余沿用上一轮配方。本次授权包含两阶段自动衔接，覆盖此前“Draft结束先停止”的旧指令。主会话只准备配置和交接；实际上传、必要CUDA检查、训练和进度报告由训练会话执行。

## 配方与输入

| 阶段 | 初始化与输入 | 停止/调度 | LR / warmup |
|---|---|---|---|
| Drafter D/R/P | 冻结M45的Encoder，复用全曲P0；D/R/P从头初始化；每4拍一个256维Z | 50 / 100轮 | 2e-4 / 200更新 |
| Decoder适配 | 从原M45初始化M；冻结E/D，读取局部P0和本次D50导出的Z；开启Z、GT plan | 20 / 50轮 | 1e-4 / 200更新 |

配置：[Drafter](../configs/drafter-z4-frozen-p0-m45-linux.json)、[Decoder](../configs/decoder-adapt-m45-z4-linux.json)。D/R/P配置从上一轮实际run-config提取并保留；M配置沿用已完成适配的配方。新运行各自使用fresh optimizer/scheduler，从第1轮编号。禁止把逐拍D50或上次适配M20当作本轮初始化/完整续训。

- 保持D8层/256维、直接末层Z、逐拍MOMENT96与数量监督、不监督duration、P覆盖目标全部16拍（4枚Z）。
- D：AdamW、WD0.01、clip1、完整音频组上限2、budget45056、207行proxy、累积1。
- M：fused AdamW、WD0.01、clip1、row/chunk budget8192、累积1、history dropout0.15、Physical/judge/trajectory权重0.8/0.5/0.5。
- 两阶段都保持BF16、workers12、seed20261010、原pilot320计划/顺序/occurrence/批次边界。音频前4＋目标16＋后4拍；M为原 `window` 范围。N=4时M的24拍Z上下文有6枚Z。
- 只按完整轮保存；每5轮及末轮validation，完整诊断仅各阶段末轮50/20。中断恢复本次对应阶段的完整轮权重、optimizer、scheduler与全部RNG。

复用路径：

```text
M45: /root/autodl-tmp/decoder-cont-v2-no-z-gt-20261011/training/checkpoints/epoch-045.ckpt
全曲P0: /root/autodl-tmp/drafter-split-p0-m45-20261011/p0
局部P0: /root/autodl-tmp/decoder-adapt-m45-z-20261011/local-p0
GT plan: /root/autodl-tmp/pilot320-gt-plan-7adb36c-20261010
```

只核对既有manifest/生产者与正常读取合同，不重产已验收P0、Mel/aux32、原谱、Physical、窗口标签或计划，不新增全量载荷扫描/解压/重哈希。原始输入路径均在配置内。

## 执行顺序

新实验根 `/root/autodl-tmp/drafter-z4-m45-adapt-20261011`，子目录 `source`、`drafter`、`dp`、`decoder`。固定上传本次提交的完整独立源码归档，记录SOURCE_COMMIT；不覆盖旧实验绑定源码。SSH为 `root@connect.cqa1.seetacloud.com:16079`，Python为 `/usr/local/bin/python`。

先核对目标目录、所需资源和现有进程，避免重复或冲突；旧授权任务若仍占用GPU，串行排队，不中断它。只校验本次实际上传的新/改变文件。下列命令在新source目录、`PYTHONPATH=src`下运行：

```bash
python -u -m maidata.training --config configs/drafter-z4-frozen-p0-m45-linux.json --cuda-check
python -u -m maidata.training --config configs/drafter-z4-frozen-p0-m45-linux.json

# 第50轮正常完成后，用精确epoch-050产出新Z/预测plan；M训练仍用GT plan。
python -u -m maidata.decoder_training.dp_cache \
  --config configs/decoder-adapt-m45-z4-linux.json \
  --p0-cache /root/autodl-tmp/drafter-split-p0-m45-20261011/p0 \
  --output /root/autodl-tmp/drafter-z4-m45-adapt-20261011/dp

python -u -m maidata.decoder_training.adapt --config configs/decoder-adapt-m45-z4-linux.json --cuda-check
python -u -m maidata.decoder_training.adapt --config configs/decoder-adapt-m45-z4-linux.json
```

CUDA检查限定本次间隔变化的原真实batch路径：D阶段E零前向/梯度/更新，D/R/P有实际梯度和更新；M阶段冻结E、无D/P/R执行，M及Z条件分支有梯度和更新，原局部音频范围与batch边界保持。记录时间和显存，不设新性能或loss阈值，不擅自缩小batch/worker/精度。检查模型丢弃，正式运行按固定seed重新初始化。

使用可靠后台串行launcher，记录各阶段日志、进程和exit；启动需有首个实际optimizer更新、配置、进程及GPU证据。D50完成并保存完整状态后方可导出；新DP必须记录 `z_stride_beats=4`、width256、正确生产者和全部基础窗/空尾窗，并按新增资产合同验收一次。已有成功DP不因训练恢复而重复导出。最终确认M20完整checkpoint及对应D50/缓存身份，停止本次流水线；后续推理另按用户指令执行。

实验进度、完成、异常及待决定事项只在训练会话内报告；不向主会话发送进度或确认。主会话交接后不轮询。已有绑定实验或额外推理任务不因本次配置被覆盖。

## 验证边界

间隔实现来自 `a05787e`，上一轮125项受影响CPU用例已通过。本次仅增加两份配置和交接文档，已逐项核对与上次实际配方的差异仅为Z间隔与新产物路径，执行配置解析与M配方校验；不重复模型测试。CUDA、实际启动及两阶段训练结果由训练会话提供真实证据。
