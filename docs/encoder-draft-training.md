# Encoder＋Draft 阶段一执行状态

目标为pilot320上的新E/D/R/P联合预训练，主Decoder不参与。架构、损失和已确认配方见[Draft接口](draft.md#阶段一训练默认值)及[训练计划](../../0wd-v3-refactor-docs/training-plan.md)。本页记录可执行条件，不增加旧资产重验或额外审批。

## 已落实

- 独立E/D/R/P、全曲P0和Z组合、原谱R/Planner标签及五项损失接口；D全曲一拍一个Z，R每16拍目标窗容量207。
- 固定训练288首307谱、验证32首36谱；R/P全部基础窗各监督一次，训练7924窗、验证904窗，含空窗及追加尾窗。
- `pretrain_audio_groups_per_batch=2`、`pretrain_gradient_accumulation_steps=1`，由独立batch函数消费，保持完整音频组和顺序。第1轮batch边界已保存，144批/144次更新；验证16批。实际batch及累积步数可配置。
- 五项权重1/1/1/1/0.25，AdamW、lr=2e-4、weight decay=0.01、全局裁剪1、bf16、scratch、seed=20261010、workers12。warmup200次更新，100轮调度/绝对50轮停止；当前batch对应调度14400步、停止7200步。

## 尚缺的执行接入

| 部分 | 当前状态与需要完成的工作 |
|---|---|
| 训练入口和数据批次 | `python -m maidata`只构建组件；`maidata.training`只有配方与batch函数。尚需独立Dataset/Collator和workers12的DataLoader，将音频输入、完整组、条件、全曲P0/Z与R/P监督接起来；按各轮计划的实际组顺序形成固定batch，不重抽歌曲、不做过采样 |
| 完整训练/验证循环 | 尚需实现E/D/R/P的模式切换、前向、五项total/count归约、反传、统一裁剪及一次optimizer/scheduler更新；验证每5轮及显式末轮、重合一次，完整诊断仅末轮。不能调用旧冻结音频训练器替代新E的联合反传 |
| 完整轮保存/恢复 | 尚需保存成套E/D/R/P、optimizer、scheduler、完整轮游标、全部RNG及源码/架构/输入/计划绑定；last每完整轮、best只在轮末验证改善时、编号每5轮及末轮保存。轮内/异常不保存，从上一完整轮恢复，绝对补到50轮 |
| Linux输入接入 | 音频grid旧Mel引用和原谱source_path仍可能是Windows绝对路径。已读样本185:5的grid引用本机旧Mel，源记录无level_text且source_path指向本机maidata.txt；`read_condition`会从该源读取等级。尚需消费端路径映射或独立的精确条件元数据，保留原生产者身份；Planner缺render_end_beat时也要能读取对应源时间格或显式元数据。不要求重算Mel/aux32或重建原谱缓存 |
| 默认batch的CUDA执行 | 尚无新E/D/R/P、bf16、2个完整组的完整更新峰值/梯度/计算复用证据。训练接入完成后核对该实际batch，覆盖E和D/R/P更新与mask；不拿CPU合成或旧冻结音频路径的显存实测代替，也不擅自缩小batch或改变累积 |

## 远端启动范围

目标为 `ssh -p 16079 root@connect.cqa1.seetacloud.com`，单张RTX4090 D约24GB。用户授权在上述执行接入齐备后新建 `gpt-6.1-sol` 训练会话，上传固定源码及必要新增资产并启动pilot320；实验状态只在该训练会话报告。

停止当前Physical计算的时机为正式训练实际准备启动时，不在本地补齐代码期间提前停止。目标是远端 `0wd-v3-physical-attribution-f372ec2-20261010/run-production.py` 及其 `maidata.chart_cache.rebuild_supports` 子进程；执行时重新核对该进程树，保留已产出文件和进度，不按宽泛python进程名终止。阶段一不消费主Decoder的训练Physical候选，因此不以它全部计算完成作为E/D/R/P训练前提。

当前仍有上述实现缺口，未创建训练会话、未上传本次源码、未启动训练，也未停止Physical计算。
