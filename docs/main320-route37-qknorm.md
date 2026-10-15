# Main扩容、Q/K归一化与Route37原配方

本次将整数Draft worktree提交 `86136a5` 中的Q/K归一化移植到main，并整理提交main已有的Route37、P0/Z归一化及M侧Plan输入开关。main保持Z4、单Plan16和原R监督，移除Plan4实验实现。

原batch配置为 `configs/joint-main320-z4-plan16-qknorm-route37-gt.json`。它由原Route37完整运行的保存配置生成，除了扩容、四组件Q/K归一化开关、M chunk和新输出目录，其余配置逐项保持。配套大batch配置为 `configs/joint-main320-z4-plan16-qknorm-route37-large-gt.json`，仅在此基础上调整曲目数、总预算、峰值学习率及输出目录。

用户已授权两组服务器目标，配方已在本地准备：

| SSH端口 | 配方 | 完整音频组上限 | 总预算 | 峰值学习率 | M chunk |
|---|---|---:|---:|---:|---:|
| 24705 | `joint-main320-z4-plan16-qknorm-route37-gt.json` | 2 | 45056 | 4e-4 | 32768 |
| 37012 | `joint-main320-z4-plan16-qknorm-route37-large-gt.json` | 3 | 80000 | 6e-4 | 32768 |

两组均完全从头初始化，绝对停止90轮、150轮cosine调度、warmup200、BF16、workers12；M的 `decoder_train_plan_input` 默认并显式保持 `true`。两份配置的模型结构、损失、数据计划、输入路径及其他参数相同。大batch配置的三处学习率别名同步为6e-4，三处输出别名共同指向独立的 `../artifacts/main320-route37-qknorm-large-20261015/training`。

| 项目 | 共同设置及两组差异 |
|---|---|
| E/D/M | 320维、8头、每头40维 |
| E/D/M深度 | E为1+1+2，D为8层，M为8层、前2层Router |
| FFN | E/M前段1280，D/M每专家640 |
| P/R | P128维、4头、MLP256；R FFN512，保持原专属宽度 |
| Z | 旧块中心Z4，中心2、6、10…，每枚重建对应四拍 |
| Plan | 每窗单枚Plan16；P读取窗内4枚Z，M读取6枚Z+1枚无位置Plan |
| Q/K | E/D/P/M全部Transformer attention，投影拆头后、RoPE前逐头无仿射RMSNorm |
| Q/K数值 | FP32计算，epsilon1e-6，恢复原dtype，保留1/sqrt(head_dim)，V保持 |
| R监督 | Route37 acceptance37、真实字段及MOMENT96；数量监督关闭 |
| 学习率 | 原batch峰值4e-4，大batch峰值6e-4；warmup200，AdamW/fused、weight decay0.01、clip1 |
| Batch | 原batch为2/45056，大batch为3/80000；M chunk均为32768 |
| 轮次 | scheduler horizon150，绝对停止90轮，完全从头初始化 |
| 运行 | BF16、workers12、seed20261010，原计划、occurrence及顺序保持 |
| 参数 | 56,178,272，Q/K归一化新增0参数；相对Route37增加23,431,846 |

单窗Plan16保持原统计范围和现有GT缓存，不引入Z2或逐拍起音存在头。原P0和M侧Z归一化保持开启。稠密、packed、有效索引投影与增量KV使用同一Q/K处理；K只在生成时归一化一次。

归一化开关分别为 `qk_normalization`、`draft_qk_normalization`、`planner_qk_normalization`、`decoder_qk_normalization`。默认均关闭，独立新配方全部开启。旧checkpoint缺字段时明确按关闭恢复，旧缓存identity保持；开启时单独缓存身份。无参数改动也通过保存配方区分，完整续训拒绝静默切换归一化。

当前源码及两份配方完成本地准备与CPU验证，远端启动记录由后续实验步骤保存。部署时按上述服务器目标核对实际路径与资源，并设置独立远端输出目录。输入路径沿用Route37保存配置，旧输入不重新扫描、重建或归因于新消费者。

main原有未提交源码已备份到 `../artifacts/main320-route37-qknorm-20261015/main-before.tar.gz`。main已有改动按功能分步提交；Plan4实现已丢弃，保留单窗Plan16及M侧Plan输入开关。配置逐项差异、CPU验证和分步提交记录位于同目录。
