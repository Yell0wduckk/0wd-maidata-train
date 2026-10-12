# 数量监督改为 MSE

2026-10-11 用户要求停止 full 联合训练，修改数量目标后从头训练。

新 CONFIG 默认 `reconstruction_count_loss="mse"`、`reconstruction_count_weight=1`。
R 的 COUNT 查询输出一个线性标量，直接拟合每拍真实 MOMENT 数量（起音组数量，
不是同时按键数量）。有效拍含零数量和完整空尾；目标没有 log、标准化或截断，
训练输出没有 sigmoid、取整或截断。MSE 在 FP32 中累计平方误差，除以同一次逻辑
更新所有物理 batch 的有效拍数。R/P 仍仅监督基础窗一次，M 的复杂 occurrence
不重复增加辅助监督。字段、MOMENT、P、M 的归约及实时共享梯度路径保持原样。

四项辅助 loss：

| 项 | 定义 | 权重 |
|---|---|---|
| fields | 每个有效字段各自 CE 均值，再对有监督字段等权平均 | 1 |
| MOMENT | 有效起音查询的拍内 96 类 CE 均值，无 smoothing | 1 |
| count | 全部有效拍的原始数量 MSE 均值，含空拍 | 1 |
| Planner | 8 个连续目标 MSE 均值 + 0.25 × 配置类别 CE 均值 | 整体 1 |

总目标为 `L_M + L_fields + L_moment + L_count + L_planner`。
`L_M = L_main + 0.8 L_physical + 0.5 L_judge + 0.5 L_trajectory`，
Physical 的既有四类内部权重 `(1,1,4,1)` / 4 不变。
权重相同不表示数值或梯度贡献相同；MSE 与原 CE 的数值不能直接比较。

原数量目标是空/非空 CE（全部有效拍）加正数量 96 类 CE（仅非空拍），两项各权重 1。
为显式旧配方保留 `ce` 构造路径；新 MSE 模型只构建 scalar count_head，
不保留无损失的旧分类参数。数量架构和权重进入 checkpoint recipe，
旧 CE 完整 checkpoint 不能当作 MSE 续训，必须遵守本次从头初始化要求。
旧联合 checkpoint 的推理不构建 R，仍可使用其 E/D/P/M。

末轮数量诊断报告原始输出 MSE/MAE、四舍五入且下界为零后的 exact accuracy，
以及真实和预测数量分布。取整和下界仅用于诊断，不进入训练计算图。
