> 历史参考，迁移于 2026-10-10。原路径：`docs/prior-decoder.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 冻结 A90 的 Decoder 监督 prior

本分支从 `main-new-latent-ab@fa0f002` 创建。A90 的全部参数冻结且 eval；B 从该 A 内未训练 prior 开始，不读取旧 B 权重或 optimizer。原右移 GT 谱面前缀和字段监督保持，条件来自同次 prior 的硬分量选择及 Gaussian 重参数化。目标为 NLL+L3，系数各1。具体梯度、mask、微批和恢复合同见 [current-runtime](../../current-runtime.md)。

## 本轮生产

在独立输出运行 `prepare-prior-decoder-data`，输入原 A4096计划、路径清单、encoder60 checkpoint和旧 A90 P0。显式传入 `--chart-encoder-checkpoint` 与 `--device cuda`，按新窗口重建完整 μ。产物包括新 profile、canonical/identity/mirror窗口张量、新 A4096/B16384计划、完整 raw μ、新归一化、全部条件槽 mask，以及引用原 P0载荷的新索引。生产器输出实际新增窗口、标签/条件覆盖和逐轮更新数；真实 pilot 数量以生产结果为准。

新窗口覆盖所有对象完整结束，再追加一个完整16拍空窗，曲中空窗保留。持续对象只在声明窗出现，完整时长、等待和分支保留。新增引用继承原每轮 chart/view/occurrence/curriculum；过滤新增后原引用逐项一致，原连续音频组实例不合并或移动。

按用户最新指令重建完整 μ/raw mask/归一化，包括结束后真实空块；P0载荷不重算。新 `target_valid` 与 `condition_valid` 每个真实窗口四槽均有效。重建使用 epoch-060.ckpt（内部epoch59），整曲双向编码后重算 train-only 2:1统计；新空块也可能改变原区域的 μ。新缓存 `rebuilt_from` 保留原 A数据来源，训练 checkpoint 同时保存新目标与统计，不覆盖 A元数据。

`build-decoder-context-cache` 用指定冻结 A90 每唯一音频前向一次，保存 candidate keys、padding mask和拍坐标。P0引用旧载荷，不保存 C0/C1。candidate query依赖 Decoder隐状态，在训练时实时计算并保持输入梯度。训练步不调用音频 encoder，prior 和 Decoder 的每层 P0 K/V 按唯一音频复用。

各生产器仅对新资产进行一次覆盖、状态、有限值及实际 loader读取验收；不另加旧音频、Physical、几何或旧 μ扫描。

## 执行顺序

实验会话通过 SSH16079 上传独立源码，资源空闲后运行：

```bash
bash configs/run-prior-decoder-16079.sh
```

脚本输出固定为 `/root/autodl-tmp/main-new-prior-decoder-20261009/run`。原输入默认位于 `/root/autodl-tmp/main-new-latent-ab-20261008/source-5878f7a541f309d926254aa6461fb5234c41421c/run`，可用 `OLD_RUN` 指定同一已完成 A90/μ/P0的实际保留目录；不替换为其他 A或旧 B。

顺序为新数据/计划、encoder60完整μ重建和P0索引生产及一次验收 → candidate缓存生产及一次验收 → 新 B原逻辑batch CUDA检查和成本记录 → B20 → 末轮固定样本。没有 A CUDA资格或 A训练步骤，不打断其他实验。

正式 B使用 `configs/train-b-decoder20.json`：stop20、horizon50、offset90、FP32、逻辑预算16384、seed20260811、workers12；lr0.0001、wd0.01、clip1、warmup50。完整 chart/view/occurrence 组成4096软预算 Decoder微批；一个逻辑batch只更新一次。L3使用原全逻辑batch的三类有效监督分母。

last每完整轮保存；编号第5/10/15/20轮，最终 `B/ar.ckpt`。无验证集，validation关闭，完整训练诊断只第20轮。异常/轮内不保存。完整轮恢复须读取原模型、optimizer、scheduler、游标及所有 RNG和DataLoader generator；恢复原输入和配方，不重复已通过且条件未变的 CUDA检查，不自动续训。

## 末轮生成对比

`scripts/diagnose-prior-decoder.py` 固定取缓存顺序前3个唯一音频的 identity谱面，以 seed20260811、谱面temperature0、trace关闭生成三组：真实 μ区域仅干预 Decoder、prior默认1/1、prior双温度0/0。GT干预不喂回 prior，缺失尾部沿用同seed prior输出。报告保存全部生成行/字段/时间、窗口状态，以及对象结束后空窗的 EOS/行数行为；这是训练集样本证据，不设质量或性能阈值。

普通与流式生成使用新 checkpoint的尾部条件合同，每窗四槽均可用。共享全曲音频、prior K/V及 RNG，回滚不重新抽 latent。真实音频、Physical边界和 span_beats语义保持。旧模型推理沿用旧覆盖。

## 后续 A90 的完整 μ

本轮不重训 A，但已经切换为按新 profile 完整重建 μ。生产器读取新 `condition_end_tick`，在完整双向 encoder中加入真实空块并整曲重编码；统计包含新有效空块，排除padding。后续新 A90可直接使用同一新 profile、A4096计划及相应 encoder60完整μ，不把它伪装成旧 A计划续训。

旧 μ和两次失败目录均保留，本轮只消费独立新输出。未来 A的配套受影响资产绑定与正式训练另行执行。本轮不重验或重建旧 Physical、音频及几何资产。

实现提交、CPU和独立复核结果记录于 证据（源文档引用：`docs/evidence/prior-decoder.md`，未导入）。正式实验启动、异常、进度和结果只在“6.1sol 训练实验new”报告。
