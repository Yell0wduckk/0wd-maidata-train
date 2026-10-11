# 独立 CONT 序列实施合同

本合同替代主 Decoder 的 FieldReader/临时字段查询设计；Draft R 不变。

主 M 删除额外 FieldReader 的全部参数、attention 和接口。五类 Router 与原条件内容小头保留；三类时间仍为 1..384+CONT。MOMENT_CONT/HOLD_CONT/SLIDE_CONT 是独立输入 token，进入主因果 KV；续接锁定原时间类型，不另计 Router 损失。只有最终短值完成真实时间字段与 Physical 动作。

普通 NOTE/ROUTE 直接使用主干隐藏状态。首次 Slide duration 仅在 Router 后改变音频/Z cross-attention Q；self 坐标不变。CONT 输入坐标来自已知累计时间；duration 使用 NOTE written BPM 和实际秒数反算绝对拍。主干前段不得接收待预测类型或时长。

精确原谱 v2 和完整整数 Physical v2 保留。新增模型谱面序列缓存与窗口 v5，分别记录真实 token 归属、累计值、旧组批长度及实际行数。长 duration 完整链保留在 teacher forcing 输入中，监督止于越窗 CONT。时间 CE 沿 GT 路径求和、按原字段归约；Physical 在受监督且有选择的步骤内平均，再按原字段归约。CONT 每类别处罚表示尾部不存在该类别合法完成值；硬 mask 要求联合合法完成值。首次归责及窗尾零罚语义保留。

pilot320 迁移优先读取现成 v4 标签并重映射，仅重算无法复用的窗口支持。原谱、旧 Physical、音频及几何不重建。新产物在独立目录原子发布，可继续未完成的迁移。本地迁移16 worker、每 worker一个计算线程；正式训练仍12 worker。

保留原 occurrence、shuffle、history dropout、逻辑 batch 和 scheduler 更新次数。M 按实际扩展行数拆计算块，复用同一次 E_M 与音频投影计算图。新 checkpoint 不接受旧 FieldReader 架构续训。交付包括增量/从零一致性、真实动作往返、梯度与增量一致性、原 batch CUDA 成本测量；不自动启动正式训练。
