# Decoder 模块与接口

主 M 使用六层主干、五类 Router、原内容小头及独立 CONT 输入序列。架构配置仍集中在 `entry.CONFIG`；Draft R 独立局部重建不受本次修改影响。实施合同见[独立 CONT](decoder-cont-token-plan.md)。

## 主干与内容头

前两层只消费已知前缀并预测下一目标类型，后四层使用对应五类 MMoE gate。NOTE 的 kind→pad→flags、ROUTE→track_break 直接读取主干最终隐藏状态。保留原条件 embedding/残差小头，不添加字段 attention、字段角色 embedding 或进度 MLP；原 `FieldReader` 模块和接口已删除。

`begin(tokens,memory,cache=...)` 不接受目标类型。`complete(routed,next_types)` 才选择后段分支；同类型内容重试复用输出，改选类型重新计算后段。`prepare_audio` 按同一参数版本投影音频一次，保留梯度。plan 无位置评分，Z 与音频采用绝对拍 RoPE。`decoder_z_cross_attention=false` 跳过 Z 适配，plan 与音频仍正常参与。

条件 memory 的 plan MLP 输出加 source embedding 后通过独立 LayerNorm（使用 decoder_norm_eps），再进入各层 K/V 投影。GT 与 predicted plan 使用同一路径，Z 适配保持原样。条件 attention 对单有效 key 直接读取 V，训练 dropout 按 query/head 作用于标量权重；不执行 Q/K softmax，Q/K 无该分支损失梯度。零有效 key 输出零，多个有效 key 保留原联合 attention。dense、packed 和增量使用相同语义。

## 模型序列与位置

五类目标为 MOMENT、NOTE、ROUTE、HOLD_DURATION、SLIDE_DURATION。输入额外包含 MOMENT_CONT、HOLD_CONT、SLIDE_CONT；三种 CONT 各占一个主序列及因果 KV 位置。Router 不增加类别；续接锁定原时间类型，不重复类型损失。

三个时间头仍输出 385 类：类别 0..383 表示 1..384 tick，384 表示 CONT。正整数 d 唯一编码为若干 CONT 加一个正余数，384 的整倍数以末余数384结束。Hold 完整值至少2 tick，续接后的余数1合法。输入的时间数值编码表示该次输出的块/余数，累计值由数据状态跟踪。边界 MOMENT 为输入控制，不展开。

`TokenBatch.beats` 是已知输入的 self-attention 坐标。CONT 使用推进后的已知游标；最终 MOMENT 推进真实起音，最终 duration 回到所属 NOTE 起音。因果关系由序列顺序决定，不要求拍坐标单调。

`TokenBatch.slide_beats` 是仅由已知 NOTE 起音及 BPM 推得的启动位置。前段与 Router 不使用它。后段仅当目标类型为 SLIDE_DURATION 时，将音频/Z 的 cross-attention Q 放到该位置，self Q/K 保持输入坐标。Slide CONT 已处于续接坐标，不再次加启动等待。时长按 NOTE written BPM 换算实际秒数，再反算全局拍位；不可跨 BPM 直接加四拍。

## 监督与 Physical

训练窗口 v5 将实际 GT CONT 链展开为主序列。普通字段的支持与首次归责继续复用原谱/Physical v2；CONT 不生成真实动作或新的归责事件。时间 CE 沿实际 GT 路径求和，分母按原字段计算；右删失末步对兼容类别集合求 NLL。

时间 Physical 在每个实际受监督前缀计算局部合法补集损失，先在该原字段有选择的步骤内取平均，再按原字段归约。不计算 GT 没有经过的续接主干。CONT 的类别处罚表示所有可达完成叶都携带该类别的首次处罚；存在合法完成即不处罚。窗尾关闭叶为零罚。推理 hard mask 仍要求存在联合满足四类的完整值，不能拼接各类别分别合法的候选。

四类权重保留 `(1,1,4,1)/4`，NOTE kind/pad 各0.5，flags 不接 Physical 梯度；几何项按 ROUTE 数归约。所有 M 分块使用整个逻辑 batch 的共同分母。

长 duration 完整 GT 链仍进入 teacher forcing，并在结束时提交完整真实动作；时间监督止于跨出本地音频窗的 CONT。窗内发出的最终短值即使落点在窗外，仍正常监督。无监督的后续 token 不改变这一边界。自由生成沿用已有完整 duration 的局部合法域。

## 推理、缓存与兼容

每次 CONT 调用主干追加 KV，锁定时间类型；选择最终短值后组装完整 Moment/HoldDuration/SlideDuration，才调用真实规则接受。CONT 搜索使用不可变 KV 分支，失败可退回前缀，不污染 Physical。跨窗从真实谱面历史重新展开并 prefill；物化只接收真实谱面，不包含 CONT。

新训练 checkpoint 为 `maidata.decoder-training.v4`，模型架构为 `maidata.decoder-causal-cont-tick384.v2`；旧 FieldReader 及未归一化 CONT v1 checkpoint 不支持完整续训，不迁移 optimizer。精确源谱和完整 Physical 仍为 v2；模型序列缓存为 `maidata.model-chart-sequence.v1`，窗口为 `maidata.decoder-window-tensors.v5`。迁移见[缓存说明](time384-cache-migration.md)。
