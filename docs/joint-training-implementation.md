# 联合训练实施记录

实施基线为 `02eaa75`，新入口与旧两阶段入口并存。源码、CPU测试和独立16079短CUDA测试的结果在此记录；不启动90轮正式训练，不替换已有实验的绑定源码。

## 已实现

- `joint_training/data.py`：阶段一完整音频组＋对应M occurrence，原局部窗并集及worker布局；R/P基础窗一次。
- `network.py`：唯一E/D/R/P/M和实时Z；M读取真实原始局部P0。
- `shared_gradients.py`、`update.py`：共同分母、音频KV＋Z FP32梯度桥、一次共享反向及统一更新。
- `recipe.py`、`checkpoint.py`、`runner.py`、CLI：CONFIG配方、完整轮保存/恢复、GT验证、显式正确性/成本诊断。
- 单checkpoint推理：共享E及真实局部输出，预测P plan，保留双checkpoint入口。
- 重建字段索引提前到worker；已分组的2+条件key直接使用SDPA，避免热路径重复GPU筛选，保留动态mask调用兼容路径。

## 验证记录

CPU与CUDA最终结果将在完成实测后写入本节。检查范围仅涉及新图、数据传递、损失/梯度、恢复、推理和原batch成本；旧输入直接复用。
