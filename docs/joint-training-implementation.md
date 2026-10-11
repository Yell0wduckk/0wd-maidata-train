# 联合训练实施记录

实施基线为 `02eaa75`，新入口与旧两阶段入口并存。源码、CPU测试和独立16079短CUDA测试的结果在此记录；不启动90轮正式训练，不替换已有实验的绑定源码。

## 已实现

- `joint_training/data.py`：阶段一完整音频组＋对应M occurrence，原局部窗并集及worker布局；R/P基础窗一次。
- `network.py`：唯一E/D/R/P/M和实时Z；M读取真实原始局部P0。
- `shared_gradients.py`、`update.py`：共同分母、音频KV＋Z FP32梯度桥、一次共享反向及统一更新。
- `recipe.py`、`checkpoint.py`、`runner.py`、CLI：CONFIG配方、完整轮保存/恢复、GT验证、显式正确性/成本诊断。
- 单checkpoint推理：共享E及真实局部输出，预测P plan，保留双checkpoint入口。
- 重建字段索引提前到worker；条件key的实际mask每个M块只确认一次、跨层复用，避免逐字段GPU筛选，保留动态mask及精确单key语义。

## CPU验证

438个唯一用例完成分批回归（合并回归及修复后受影响74项、阶段一43项；重复用例不重复计数）。覆盖真实小型五模块更新，单/双/三M块与完整单块真实图的全部参数梯度及更新对照，共同分母、梯度归属、复杂追加实例隔离、Z4、空尾窗、多难度和独立24拍局部P0 oracle。

独立P0对照使用同一prepared G0，逐窗重建原始WindowInputs与Encoder layout，再调用encode_windows，与M实际读取的values/beats/valid比较；没有用同一张量作为oracle。多难度同窗复用也在该测试内覆盖。

非零E/D/R/P/M dropout的6轮完整训练，与第3轮中断后从完整第2轮恢复逐位对齐：权重、AdamW槽、scheduler、Python/NumPy/Torch RNG、loader generator/started及metrics一致。按每epoch历史dropout重选实例，验证5轮/末轮重合一次、异常不保存、旧schema拒绝和参数形状相同但Z间隔不同仍拒绝。

单联合checkpoint完成CPU FP32/BF16整曲推理到IR及MiaCode流程；共享E与尾窗局部P0复用、加载严格性及训练GT/推理predicted来源均覆盖。1个旧raw-audio提取用例因本机Numba0.61.0要求NumPy≤2.1、实际NumPy2.4.6而排除；本次不修改环境。已验收grid推理路径正常。

合并回归曾发现关闭Z后实际单key被误当多key，产生约1e-14的Q/K梯度；已修复适用条件并补上6项动态mask/dropout/cache/inference_mode回归。成本快照恢复另修复CPU optimizer tensor别名，恢复时深复制，开销在计时外。

## 16079原batch CUDA实测

实现提交 `b58ba7f`，mask兼容修复及独立P0 oracle提交 `cd1abe278f919da4ead77cf6c5635b5c43400bdc`。远端独立源码在 `/root/autodl-tmp/maidata-joint-z4-gt-20261011/source-cd1abe2`，报告在同级 `diagnostics`。完整源码归档SHA256 `22ee44b54b1d9a37c21bd3ab0fa36799e9f4ffc159ad022ca7000f5b8d294041`及679个新上传文件通过传输核对；修复归档SHA256 `5686a55a95e3a8f4a002a56ebab2b0b0541ed097af92b714a10db8cfc4c81bc1`及其中4个改变文件通过核对。没有递归重验旧输入。

GPU为4090 D（nvidia-smi 24564 MiB），Python3.12.14、Torch2.11.0+cu130/CUDA13.0。命令使用 `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，DataLoader仍为12 workers。运行前GPU空闲，没有中断其他进程；两轮短测均退出0，结束后GPU显存0 MiB、无compute进程。没有启动正式90轮训练。

第一轮第33个更新（`--check-step 32`）是最大阶段一proxy的原始完整batch：2个音频组、proxy44356/45056、57基础窗、59 M occurrence、57唯一局部E窗（55 coverage＋2完整尾窗）、4516实际R查询、3427有效/5369 padded M行，自然为1个M块。没有改变精度、预算、实例或workers。

正确性审计中E/D/R/P/M各前向、反向一次；五模块梯度均非零，分别有169/346/38/16/385个参数张量改变。CPU已验证GT plan下M不经过P。审计损失14.4204026仅作为数值诊断；其耗时不计为吞吐。

修复后每模式预热2次、测量7次，参数审计及完整诊断关闭，模型/optimizer/scheduler/RNG每次在计时外恢复。下表为墙钟中位数，显存为7次最大分配值；原始样本及reserved值见JSON。

| 模式 | 中位秒/更新 | 最小–最大秒 | peak allocated GiB |
|---|---:|---:|---:|
| 完整joint | 0.270450 | 0.231569–0.413256 | 6.4741 |
| 只切断M→Z | 0.231294 | 0.229211–0.234752 | 6.4739 |
| 仅R/P辅助 | 0.172294 | 0.169649–0.178089 | 5.5108 |
| 固定Z的M＋E，无D/R/P执行 | 0.197816 | 0.173949–0.199323 | 6.1946 |

该同批对照中完整joint比切断M→Z中位数增加16.93%，显存差约0.15 MiB。保留了样本波动，没有删除首个计时样本。固定Z基线仍编码相同局部窗并保留M→音频梯度；共享E使独立分支耗时不能简单相加。初次及修复后报告均保留，不据短测推断全程训练时间或谱面质量。

首次取batch含12个worker启动的数据等待为1.7509秒，单列于更新计时之外。完整joint各CUDA Event分段中位数：H2D3.814ms、E前向31.946ms、D前向13.209ms、R/P前向6.985ms、共享前向52.854ms、音频投影2.584ms、M前向/loss34.284ms、单块总图反向165.253ms、clip5.911ms、optimizer3.732ms。共享前向包含E/D/R/P，分段有包含关系，不能相加；单块按计划一次直接总loss反传，无法将该次真实反向拆成两次测量。

额外profiler步在计时外，实际SDPA为102次Memory-Efficient及30次Flash，反向次数分别相同，没有math后端。未发现E/D重复执行或activation重算。profiler中24次item含CPU准备逻辑；两个is_nonzero为更新末尾损失有限值检查和clip有限值检查，未出现aten::nonzero。条件mask每M块首次一次小计数回读，后续层/字段复用；没有逐字段设备回读指标。源码没有强制某一种attention后端。

原始证据：[正确性审计](evidence/joint-training/cuda-audit.json)、[修复后成本](evidence/joint-training/cuda-cost.json)、[初次审计](evidence/joint-training/initial-cuda-audit.json)、[初次成本](evidence/joint-training/initial-cuda-cost.json)。

## 自然双M块原batch

只读取既有计划与窗口manifest长度元数据，选择第一轮自然M分块最多、总padded行最多的原始完整更新：`--check-step 107`（第108次更新），2音频组、proxy42738/45056、75基础窗、90 M occurrence、75唯一M窗、48局部E窗（46 coverage＋2尾窗）、7949实际R查询、7135有效M行。原8192预算下自然分成8084及5719行两个M块，未缩预算或丢实例。

两块正确性审计通过：E/D/R/P各前向及反向一次，M各两次；五模块均有梯度和更新。逻辑batch的旧统计oversized标记为true只表示15480行整体不能作为单M块，联合入口按原音频组保留并内部分块；不因此拆分optimizer更新。

完整joint中位0.358847秒/更新（0.357315–0.369168），peak allocated5.9210 GiB；辅助分支0.159275秒，固定Z的M＋E基线0.288270秒。此前按原计划完成的detach_z对照为0.359145秒，该批两者差异处于样本波动内。没有把该结果作为速度阈值或质量结论。

完整joint CUDA Event分段中位数：H2D4.296ms、E26.990ms、D14.452ms、R/P6.945ms、共享前向49.169ms、音频投影2.542ms、M前向/loss合计67.746ms、两M块反向合计113.770ms、唯一共享反向99.675ms、clip9.749ms、optimizer5.684ms。首次取batch等待2.4877秒（含worker启动）。实际SDPA为162次Memory-Efficient和66次Flash，反向各同数，无math后端或aten::nonzero。两条诊断命令均退出0。

证据：[多块选择元数据](evidence/joint-training/multichunk-selection.json)、[选择脚本](evidence/joint-training/select-multichunk.py)、[双块审计](evidence/joint-training/multi-cuda-audit.json)、[双块成本](evidence/joint-training/multi-cuda-cost.json)。这些仅是实施证据，不是启动凭据或验收门禁。

用户随后取消M→Z切断对照，默认成本诊断已移除该模式。已完成的原始记录保留；此后不再运行测试或CUDA测量。正式联合入口从始至终保留M→Z梯度，不受诊断对照影响。最终此项修改仅改变显式成本诊断的模式列表。
