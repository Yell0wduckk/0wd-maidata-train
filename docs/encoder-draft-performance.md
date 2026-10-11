# Encoder / Draft 批量化修复验证（2026-10-10）

本文是旧结构的历史测量，不代表2026-10-11拆分式Drafter的CUDA速度或恢复资格；新结构见[Draft](draft.md)，v3不兼容旧完整轮checkpoint。

对照基线为 `1f6a76288daa017c2ce2711302fc7c7bad41f325`。保留全部网络参数名称与形状、配置值、L/S联合softmax、G0共享计算图、绝对拍RoPE、逐拍全曲Draft、R/P监督及模块边界。

## 实现

- 局部attention采用原仓库的矩形query块，将所有窗口的块合并为每Conformer block一次SDPA；不再逐query复制邻域K/V。候选布局在CPU准备，按实际device坐标生成精确半径mask，并在stage内复用。
- CPU collator准备插值、slot、P0拼接和Z窗口索引；GPU批量gather保留值的梯度。布局是输入坐标/mask的临时派生物，不缓存可学习表示。
- Dense MoE沿用原版的有效token计算方式；batch按dtype合并IPC/H2D缓冲区。参数和构造器仍由唯一CONFIG控制。
- 取消源码文件哈希生成与恢复校验。源码版本只记日志；旧checkpoint可直接恢复。模型张量、配方、数据计划、优化器、调度器、完整轮游标及RNG仍按原合同恢复。

## 同一真实batch

选取pilot320第1轮第1批：2个完整音频组、40个Encoder窗、42个R/P监督窗。两版均严格加载第5轮、第863步的同一checkpoint。准备使用12 workers；输入、权重、优化器及初始RNG相同。临时挂起原训练隔离GPU计时，完成后恢复；未修改其权重或输入。

| 项目 | 修复前 | 修复后 |
|---|---:|---:|
| bf16训练更新中位数 | 1.154047秒 | 0.207878秒 |
| 峰值已分配显存 | 7.0426 GiB | 4.0207 GiB |
| 峰值保留显存 | 7.5391 GiB | 4.4824 GiB |

原配置dropout开启，各预热2次、计时7次，包含H2D、forward、backward、clip、AdamW和scheduler；不包含DataLoader准备。该batch实测加速5.55倍，已分配显存下降42.9%；不是整轮吞吐保证。E/D/R/P均有有限非零梯度并执行更新。

批量化改变dropout随机数消费顺序；训练轨迹不承诺逐位一致。关闭dropout的bf16对照loss为3.6574730873与3.6578090191；部分Draft attention的bf16梯度对舍入敏感，不能据此宣称梯度逐位相等。

为区分实现差异与低精度数值路径，同一完整batch另做FP32数学attention参考前反向。验证期间将保存的激活卸载到CPU，以便原训练仍保留显存；该诊断不改变正式训练精度或后端。结果：

- 两版loss均为 **3.654989004135132**，监督计数、mask和拍坐标一致。
- 所有公开输出最大绝对差 **2.14875e-5**；最大相对L2差 **4.37701e-7**。
- 全部参数梯度最大绝对差 **3.56510e-6**。除数学上接近零的重建attention key bias外，最大相对L2差约 **9.01e-5**，对应余弦约0.999999996。接近零项的梯度范数约1e-9，其绝对差约1e-9，单独的相对误差不具有解释力。

## CPU与复现

141项受影响CPU用例通过，覆盖联合attention对dense参考的输出/梯度、非默认参数、负拍位、缺失/空音频、非整数FPN采样率、公开接口、跨模块梯度、packed IPC、prepared/fallback一致及更换源码记录后的完整轮恢复。未重做未变化的真实音频特征提取；该integration用例需要单独指定资产。

`scripts/compare_encoder_draft.py`提供prepare、run和compare子命令；通过PYTHONPATH选择待比较的独立源码目录。准备一次真实batch，分别执行两版eval与train；FP32参考使用`--precision fp32 --math-attention --offload-saved-tensors`。工具是独立诊断，不加入训练启动依赖。

原始JSON证据位于`../artifacts/encoder-draft-performance-20261010/`；远端同名证据目录为`/root/autodl-tmp/pilot320-encoder-draft-perf-20261010/`，保留实际batch和对照张量。
