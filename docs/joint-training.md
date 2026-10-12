# 从头联合训练：共享 Encoder、Z4、GT plan

入口为 `python -m maidata.joint_training`，唯一配置来源是 `maidata.entry.CONFIG`。旧阶段一和阶段二入口保留；本入口随机初始化唯一一套 E/D/R/P/M，第一个更新起联合训练，不加载教师或神经缓存。

## 固定默认配方

绝对停止90轮，cosine horizon150轮；峰值学习率4e-4，warmup200更新。AdamW、CUDA fused、WD0.01、clip1、BF16、workers12、seed20261010、累积1。Z为D末层归一化的256维隐藏，每4拍一枚，开启M的Z attention。训练及验证使用GT plan，P仍独立接受监督；M在该模式下不通过P反传。

目标为 `L_M + L_R + L_P`，三个外层系数均为1。内部字段平均、空/非空数量、正计数、Planner和主M损失归约沿用当前实现，Physical/Judge/Trajectory系数为0.8/0.5/0.5。M、R、P的分母分别在整个逻辑更新内累计，不按块平均。

## 数据与共享计算图

阶段一计划决定完整音频组顺序与更新边界，最多2组、预算45056、分批proxy207；M纳入对应组的全部既有occurrence。复杂窗追加实例只增加M监督，R/P每基础窗一次。既有原谱、Mel/aux32、GT plan、Physical、v5窗口和采样计划直接读取，不扫描或重建旧资产。

每曲只准备一次G0；coverage窗与M监督所需局部窗取并集，包含完整空尾窗，统一执行一次E。M读取同一24拍窗的真实局部P0，同曲同起点共享；D只读取coverage中心组成的全曲P0。同音频多难度共享E结果，D分别消费相应谱面条件。

梯度路径是 `M→Z→D→E`、`M→局部P0→E` 和 `R/P→D→E`。8192行只限制M内部计算块，保留全部实例与完整历史，块间不更新参数。单块合并总损失直接反传；多块对音频K/V与实时Z建立叶梯度桥，FP32累积M贡献，最后连同R/P一次多输出反传真实E/D图，依据[PyTorch多输出梯度累加机制](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.backward.html)。没有重复E/D前向、反复retain_graph、activation checkpoint或跨更新旧P0/Z复用。

worker准备窗口映射、R字段选择、Encoder及M attention分组；热路径使用这些索引，指标在设备累计后回读。条件mask每个M块首次确认一次，所有层及字段复用确认结果；Z关闭或mask改变时保留精确单key语义。保留mask、dropout、坐标与BF16语义。[SDPA后端取决于实际输入条件](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)，由显式成本诊断记录。

## 命令

从仓库根运行并设置 `PYTHONPATH=src`。Linux路径配方只覆盖输入/输出路径，训练数值仍来自CONFIG：

```bash
python -m maidata.joint_training --show-config
python -m maidata.joint_training --config configs/joint-z4-gt-linux.json --show-config
python -m maidata.joint_training --config configs/joint-z4-gt-linux.json
python -m maidata.joint_training --config configs/joint-z4-gt-linux.json --resume /path/to/checkpoints/last.ckpt
```

后两条是正式训练/恢复命令；本次实施仅执行下面的短CUDA诊断，不启动90轮训练。默认诊断选择第一轮原组批proxy最大的完整更新；`--check-step`可显式指定从0开始的更新索引，仍保留该更新的完整batch。

```bash
python -m maidata.joint_training --config configs/joint-z4-gt-linux.json --cuda-check
python -m maidata.joint_training --config configs/joint-z4-gt-linux.json --cost-check
```

正确性审计记录五模块梯度、更新及真实前反向调用次数。成本测量关闭参数审计和完整诊断，每模式预热2次、测量7次；每次计时前恢复同一模型、optimizer、scheduler及全部RNG，恢复开销不计入。模式为完整joint、仅R/P辅助分支、固定Z的M＋音频E基线（不执行D/R/P），不再测切断M→Z对照。正式训练始终保留M→Z→D→E梯度。GT plan固定；各模式保留原音频组与M实例。报告记录数据等待、H2D、E/D/辅助前向、M前向损失/M分块反向/共享反向、clip/optimizer、显存、工作量及实际attention算子。各分段有包含关系，不能直接相加。

报告在输出目录同级 `diagnostics/cuda-audit.json` 和 `diagnostics/cuda-cost.json`。这些是显式诊断产物，不是普通启动前置条件。

## 完整轮恢复与推理

联合checkpoint格式为 `maidata.joint-training.v1`，保存五模块、optimizer、scheduler、完整轮游标、Python/NumPy/Torch/CUDA RNG和DataLoader generator状态。严格检查架构及表示、权重形状与完整轮游标；旧阶段一/阶段二checkpoint不能完整续训。不新增来源哈希或旧资产验收凭据。

每完整轮保存last；每5轮及绝对末轮验证、保存编号checkpoint，重合时只执行一次。best仅随轮末验证改善更新。完整诊断仅绝对末轮；轮内及异常退出不保存，从上个完整轮恢复。

用户明确取消验证集时，将两套 validation 计划设为同语义空计划；联合循环跳过所有验证，不写 validation 指标或 best checkpoint。完整轮 last 和每5轮／末轮编号保存照常执行。测试集只记录独立清单，不传入训练或验证 loader。

每次更新及完整轮指标的 `losses` 单列 `main_decoder` 和 `drafter`；后者包含重建的 `drafter_fields`（有效字段等权分类）、`drafter_moment`（MOMENT96）、`drafter_count`（空／非空与正计数）以及单列的 `planner`。各项已包含原内层及外层权重，可相加还原总 loss；仅改变报告，不改变归约、目标或反传。

单个联合checkpoint整曲推理：

```bash
python -m maidata.inference \
  --joint-checkpoint /path/to/last.ckpt \
  --audio /path/to/song.wav --bpm 120 --first-seconds 0 \
  --grid /path/to/grid.npz --mel-root /path/to/referenced-mel \
  --geometry /path/to/path-catalog --slide-data /path/to/slide_data.json \
  --difficulty 5 --level 14.5 --device cuda --precision bf16 \
  --output /path/to/generated/song
```

推理其余音频、grid和导出参数沿用[推理说明](maidata-inference.md)。E只加载一次，coverage与M局部窗口取并集，实际局部输出供D/M复用；M保持window范围。推理使用P预测plan，报告明确记录训练GT与推理predicted的差异。详细trace默认关闭。

实现、测试与实测数据见[实施记录](joint-training-implementation.md)。短测不代表谱面质量，也不外推正式训练时长。
