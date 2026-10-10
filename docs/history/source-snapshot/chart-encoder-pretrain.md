> 历史参考，迁移于 2026-10-10。原路径：`docs/chart-encoder-pretrain.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 独立谱面 encoder 预训练

本 worktree 从 `main-new@62e6654` 创建；仅训练现有整曲双向谱面 encoder 和独立辅助重建头。输出仍为每4拍128维，可作为后续与 MajRadar 联合使用的固定表征；本阶段不接入雷达 A，不训练音频、主 Decoder、prior，也不计算 Physical 或雷达监督。

## 表征与重建

Encoder 沿用2层、128宽、4头、FFN512和 branch→NOTE→MOMENT 聚合、整曲双向 attention、4拍池化。确定性输出 `local_hidden` 后的 `mu`；`logvar` 冻结且不调用，不采样、不加 KL 或额外噪声。局部 latent 能读取整曲谱面上下文。

每16拍四个 latent 进入2层、128宽、4头、FFN512的 cross-attention＋FFN重建头。查询只来自固定行序号，所有窗口共用从既有长度元数据取得的行上限（包含结束事件）；没有 GT 前序行、GT 时刻、GT NOTE 数量或 encoder 中间特征。没有 query self-attention 或音频旁路。

离散输出包括事件、种类、位置、modifier和Route71，时间输出包括 MOMENT delta、Hold及分支时长。首个 MOMENT 的 delta 是窗口内偏移，后续是相邻 MOMENT 间隔；跨窗对象保留完整时长。有效空窗重建结束事件，部分尾窗保留，不存在的4拍槽与标签 padding单独屏蔽。相同 MOMENT 内 NOTE及附属分支采用稳定语义排序，避免要求集合聚合的 encoder 恢复随机行排列。

目标为离散 CE（label smoothing0.05）加时间 SmoothL1（beta1）。时间目标是 `log1p(ticks/96)`，两个分量系数均为1，分别按有效监督坐标取均值，损失在FP32计算。GT 仅供目标标签和适用字段 mask；重建头不使用 teacher forcing 的字段嵌入。

## 输入与执行

复用当前 A90 绑定的 pilot261 固定计划及谱面资产，不重新抽样、扩增或打包。计划提供 curriculum、identity/mirror、occurrence、顺序和 batch 边界。每batch唯一谱面视图完整编码一次，损失只覆盖原抽中的窗口。

```bash
PYTHONPATH="$PWD/src" python -m maimai_baseline train-chart-encoder \
  --dataset-root /data/maidata \
  --song-chart-cache /data/song-chart \
  --epoch-plan-cache /data/plans/a-plan \
  --experiment-config configs/train-chart-encoder.json \
  --device cuda --output /runs/chart-encoder/pretrain
```

`--dataset-manifest` 支持既有路径映射，显式 CLI 路径覆盖清单；清单相对路径基于数据根，显式相对路径基于当前工作目录。只读取谱面、整曲GT输入、固定计划、词表及 encoder 所需的只读 Route71 prototype；不要求音频、BPM特征、Physical或雷达路径。复用资产用既有元数据和正常读取检查，不新增载荷扫描或重建。

配方固定 stop30／horizon155／offset0、seed20260811、workers12、BF16、token budget4096，AdamW学习率0.000565685424949238、weight decay0.01、clip1、warmup79。预算描述原计划，不能用于重新打包。scheduler 总更新数取固定计划的155轮实际更新数，停止30轮不缩短调度。原数据无验证集，保持 validation disabled。

`--cuda-check` 仅对原计划首个batch执行本次模型的 CUDA 前向、反向及成本记录，不进入正式epoch或保存训练 checkpoint。受影响 CPU 验证、提交和独立复核后，由“6.1sol 训练实验new”在 SSH16079资源空闲时执行该检查并试训；不打断已交接A90，不将可选旧诊断加入队列。

## 保存、恢复与诊断

独立 checkpoint 保存完整模型（encoder＋重建头）、optimizer、scheduler、完整轮游标、Python／NumPy／Torch／CUDA及 DataLoader generator RNG。last每完整轮保存，编号每5轮及末轮保存；无best，异常或轮内不保存。恢复上个完整轮，允许增加绝对停止轮并沿用155轮horizon，保持实际配方、输入和计划语义绑定；源码SHA只记录出处。

末轮导出可独立加载的 encoder 权重与必要结构信息，不自动生产 latent 缓存。该文件和预训练 checkpoint 不交给雷达 A checkpoint loader。

```python
from maimai_baseline.training.chart_encoder_checkpoint import load_encoder
from maimai_baseline.model.chart_encoder_pretrain import encode_pretrained

encoder, metadata = load_encoder("/runs/chart-encoder/pretrain/encoder.pt", device="cpu")
latents, valid, stats = encode_pretrained(encoder, songs)
```

`songs` 使用既有完整谱面 packed 格式；返回 `[unique_charts, max_four_beat_slots, 128]` 和对应有效 mask，padding 为零。导出 loader 使用 eval 模式，冻结 `logvar`；后续把 encoder 当固定表征使用时可显式 `encoder.requires_grad_(False)`。

完整诊断只在显式末轮收集：固定训练样本用真实、置零和错配latent重建，记录事件与字段指标、时间误差、结束位置及重建样例。它们是训练集重建证据，没有预设性能或loss阈值，也不代表生成质量或泛化验收。实验进度和结果仅在实验会话内报告。
