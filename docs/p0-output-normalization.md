# FPN 最终 P0 输出归一化

2026-10-11 用户授权：在 FPN 输出与 M 音频 K/V 输入两处中的一处补归一化，
重放通过后关闭 COUNT、保留 MOMENT96，从头启动 full 联合训练。

选定唯一位置为 `LocalPyramid` 完成全部多尺度融合后的最终 P0 输出。
使用逐 token、沿隐藏通道的 `LayerNorm(d_model, elementwise_affine=False)`，
eps 沿用 Encoder LayerNorm 的 1e-5；不新增可学习参数或 optimizer 状态，
避免输出增益重新放大。归一化后再次清零无效 token。P1/P2、坐标、mask、
音频窗口、完整历史、M 分块与逻辑分母均保持原样。
D 和 M 共用这一 P0 边界；M 的 K/V 入口不重复添加 LN。

唯一默认配置 `entry.CONFIG['fpn_p0_normalization']=True`。
归一化改变模型数学语义，进入 Encoder 配方身份。缺少此字段的历史配方
明确按 False 构造，保证旧 checkpoint 推理和冻结资产仍使用原数学路径；
不得把旧 P0 缓存标为新归一化输出。本次不生产任何神经缓存。

隔离重放先严格恢复失败实验第7轮完整 checkpoint 的五模块、optimizer、scheduler、
游标和 RNG，再只替换 P0 边界；原投影 Parameter 对象和 optimizer 状态不变。
保留旧 COUNT MSE 监督，单独检验归一化修复。重放不保存训练 checkpoint、
不修改旧实验绑定源码。通过范围为第8轮前220次原配方更新，不能据此承诺长期稳定。

新的正式训练使用独立输出目录，五模块、optimizer、scheduler、RNG、epoch/step
全部从零初始化；开启 P0 归一化，关闭 COUNT，保留 MOMENT96 和 Planner。
其余原 full 计划、GT plan、Z attention、M→Z→D→E、90轮停止/150轮调度、
workers12、BF16、45056预算/最多两组、M8192行分块、累积1及内部损失权重不变。
