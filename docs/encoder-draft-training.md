# Encoder＋Draft 阶段一训练

入口为 `python -m maidata.training`，仅训练可学习的 E/D/R/P；不构建主 Decoder，不消费训练 Physical 候选，不读取旧冻结音频隐藏表示。所有默认值统一来自 `entry.CONFIG`，`--config JSON`可覆盖现有键，路径也可覆盖；相对路径相对于命令工作目录。服务器配置见 [Linux配置](../configs/encoder-draft-pilot320-linux.json)，操作交接见 [训练交接](encoder-draft-training-handoff.md)。

## 数据与计算

固定pilot320：训练288首307谱、7924基础窗；验证32首36谱、904基础窗。同曲及同音频标识不跨侧。R/P全部基础窗、有效空窗和追加完整尾窗各监督一次，无镜像、无复杂窗口过采样。

`training.data`使用原仓库同类PyTorch Dataset/Collator/DataLoader：默认12个persistent workers，每worker计算线程数为1。每个Dataset项是一整个audio_variant组；采样器保持既有seed/epoch组间和组内排列，不能跨组找小曲回填。训练与验证直接消费已验收的计划和缓存，不执行输入校验或启动扫描。

Dataset/Collator关闭共享读取器和标签适配器的`validate_input`，不重复扫描Mel/aux32、重放谱面语法、验证token/拍时间表或核对窗口覆盖。Encoder/D/R/P前向和loss适配不再检查输入shape/dtype、有限值、坐标、条件范围、mask一致性或监督计数。正常mask、索引、归约和token预算仍参与计算；R使用动态padding的独立查询，无固定行容量。超本地音频窗GT duration保留完整值，监督包含跨窗CONT后停止；窗内发出的最后非CONT即使落点在窗外也正常监督，仍按原字段计数。离线缓存生产和显式验收保留独立校验；不新增训练校验配置开关。

`training.network.Pretrainer`负责组合，E/D/R/P不持有彼此。E每组计算一次全曲G0，每个音频覆盖窗编码一次，中心P0拼成全曲bank；同音频不同等级/难度各生成完整逐拍Z，共享音频计算。R/P读取所有基础窗；R按真实适用字段创建独立局部查询，三种时间使用385类CONT路径，P独立读取目标全部16拍并预测plan12。无detach或跨参数更新缓存。

## 双重 batch 上限

每批同时满足 `pretrain_audio_groups_per_batch=2`（最多两组）和 `pretrain_token_budget=45056`。梯度累积 `pretrain_gradient_accumulation_steps=1`。预算、两种权重、组数上限及累积均可配置。

```text
A = 音频覆盖窗数 × (before + target + after) × samples_per_beat
    + ceil(actual_audio_end_beat) − first_beat
R_reference = 本组全部基础监督窗数 × pretrain_batch_reference_rows (207)
cost = pretrain_audio_token_weight × A + pretrain_row_token_weight × R_reference
```

两种权重默认各1.0。A计入每窗重复计算的重叠音频上下文和全曲G0拍token；R_reference保留原计划的207行分批proxy，包含同音频各难度、空窗及尾窗；它不限制或填充新R查询。真实query/substep数单独写入训练及CUDA成本报告，不因此改变既有batch边界或实例。此线性指标限制批次规模，不等于精确显存模型；全曲attention和不同长度padding成本仍由CUDA实测确认。预算显式设为`null`可关闭，只保留组数上限。

顺序遍历当轮既定完整组；加入下一组将超出任一上限时，先提交当前batch。整曲不拆、顺序不改、末尾不足量保留。单组超过预算时指出该组并失败，不截断或静默越界。当前训练最大单组cost=34616，验证最大28575，均低于初始上限。

第1轮173批（115个双组批＋58个单组批）；100轮各164–177次更新，总17272步；前50轮8647步。验证保持固定第1轮顺序，共19批。每轮调度计数由同一预算组批函数计算，不能硬编码旧144步，也不能用epoch1顺序代替所有轮。第1轮边界见 [batches.json](../../artifacts/pilot320-encoder-draft-sampling-20261010/train/batches.json)。

## 优化与恢复

五项loss权重1/1/1/1/0.25；AdamW，lr=2e-4，weight decay=0.01，全局裁剪1；CUDA bf16 autocast、FP32参数和损失归约；scratch，seed=20261010。线性warmup200次optimizer更新，100轮余弦调度，绝对完成50轮停止。

`runner`按完整逻辑batch的各项有效计数归约；累积时各物理batch的分子除以共同分母，再分别反传，最后一次裁剪、optimizer.step与scheduler.step。不平均不同大小batch的均值。训练与验证统计也分别累加五项total/count。

验证每5轮及显式末轮，重合一次；全字段准确率与预测示例仅末轮收集。`last.ckpt`每完整轮保存，`best.ckpt`只在轮末验证改善时保存，`epoch-NNN.ckpt`每5轮及末轮保存。best只是损失诊断，不是谱面质量判定。中途异常不保存、不伪造完整轮。

新checkpoint schema为`maidata.encoder-draft-training.v2`；旧R架构的v1不能作为新完整轮resume。checkpoint保存成套E/D/R/P、optimizer、scheduler、完整轮游标、全部Python/NumPy/Torch/CUDA RNG、两个DataLoader RNG及启用状态、各轮步数和训练配方。源码版本与输入路径仅作为来源记录，不计算或比较源码、缓存、目录元数据或计划的哈希；旧checkpoint中的输入绑定不参与恢复判断，无需身份迁移。模型张量/形状、配方、各轮调度步数和完整恢复状态仍检查，训练保留loss/梯度非有限值报错。允许延长绝对停止轮但不改变调度合同；完成40轮从41轮继续，完成50轮不再更新。替换persistent workers后补偿额外base-seed消耗；CPU测试覆盖连续运行与中断恢复权重/RNG逐位一致。

## Linux 输入

已核对 `ssh -p 16079 root@connect.cqa1.seetacloud.com`：单张RTX4090 D，24564 MiB；数据盘约111GB可用。现成路径：

| 配置 | 服务器路径 |
|---|---|
| `pretrain_chart_cache` | `/root/autodl-tmp/chart-cache-original-v2-master-remaster-fields-20261010` |
| `pretrain_audio_cache` | `/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911/audio/encoder-inputs-all-official-20261010` |
| `pretrain_mel_root` | `/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911/audio/features` |

计划与320首源文本准备为 [新增输入包](../../artifacts/pilot320-encoder-draft-inputs-20261010)，约9.8MB，目标为 `/root/autodl-tmp/pilot320-encoder-draft-inputs-20261010`。源文本提供精确等级和必要时的源时间格，副本与既有source_sha256一致。消费端通过source_root和Mel根目录定位，不改写缓存的生产者身份，不重算Mel、aux32或原谱。

## 训练会话执行

新会话上传固定源码和新增输入包，只验证本次传输；已有输入直接读取，不重验。`--cuda-check`从第1轮既定batch中选择加权cost最大的实际batch，以相同12 workers、bf16和完整组进行一次前向/反向/更新，记录峰值显存、耗时、各模块梯度和更新、调用次数。检查模型独立丢弃，正式训练仍scratch。

CUDA验证交给训练会话；CPU通过不代表CUDA通过。若OOM，不自行缩小预算、组数或改累积，先报告实测。在正式训练准备启动时按确切命令和父子树停止当前远端Physical，保留产物和进度；不提前停止，不宽泛终止python。阶段一不等待Physical全部完成。实验进度只在训练会话报告，主会话交接后不主动轮询。

## 批量计算与传输

局部attention沿用原仓库矩形query块，各窗口的块合并为一次SDPA，每块合并局部L与全部S的key，保持联合softmax、绝对拍RoPE及精确半径mask。CPU按坐标准备候选邻域，device上每stage生成精确mask并复用；参数、窗口和采样密度仍由CONFIG决定。

DataLoader worker准备插值/上采样端点及权重、G0槽索引、P0中心组合、R/P的Z窗口映射和Draft有效token索引。模型对值做批量gather/计算，梯度贯通E/D/R/P，不跨更新缓存可学习值。独立调用可省略layout，由公开坐标即时准备CPU几何。layout仅适用于准备它的坐标和mask，切片或更换输入时重新准备。Dense MoE保持原参数与专家定义，仅对有效token执行专家计算。

训练batch按dtype打包为少量连续缓冲区，经IPC、pinned memory及非阻塞H2D后恢复公开张量对象；网络不依赖DataLoader或其他网络实例。批量化改变dropout随机数消费顺序，不改变概率、位置和可配置性。数值/梯度对照关闭dropout；真实速度对照使用原配置。
