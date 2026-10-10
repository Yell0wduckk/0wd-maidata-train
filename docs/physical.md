# 行内条件预测与 Physical

这是复制源码保留的字段、L3 和 Physical 语义参考。新 encoder 尚未实现，不改变这些规则。当前路线与执行边界见 [current-runtime](current-runtime.md)；版本兼容只由中央 runtime-versions.toml 维护。

## 行内条件读出

每行主 Decoder 运行一次，使用 256 维隐状态和轻量两层残差适配器，按 `event → kind → pad/start_pad 与 flags 并行 → route → duration` 读出；完整行随后进入历史。branch 使用所属 NOTE 的起点，不生成独立 kind 或额外 head。

训练后续头使用真实前序字段；生成使用已选字段。目标字段不能进入主 Decoder 或更早阶段。不适用字段跳过计算和损失，不监督 NONE；适用 flags 为独立二分类。Route-71 和 duration 分布保持当前形式。位置与 short bit 的耦合用小型联合支持，不枚举全部 flag 组合。

每阶段从硬 mask 前的概率计算非法质量 `q`，损失为 `-log(1-q)`。无合法后续才提前归责，否则由后续阶段承担具体错误。Physical 保留四类权重 `(1,1,4,1)`、固定四类归约及基础系数 `1.0`；实际 warmup 由实验配置提供。Tap/Touch 保留位置与 short flag 的联合支持接口；两列同禁令分析边缘化，仅向位置头发送梯度。短 Hold 等价修正不引入额外 short flag 梯度；break、EX、firework、track_break 预测头不直接承担 Physical 梯度。

## L3 监督预算

监督目标为 `0.65 note_semantic + 0.15 delta + 0.4 router + 0.2 conditional_event`，不归一化总系数。为保留组接口，`structure = mean(2 × router CE + conditional event CE)`，外层系数仍为 0.2；两项共用全部有效事件行的分母，语法强制行仍计入分母。time/note conditional event 不分别分配顶层预算。各项按自身有效支持平均，保留既有事件区间加权、难度加权与 specialized supervision；缺失组贡献零，不重新分配预算。

诊断中 `field_losses.event`、`routing.event` 和 `supervision.event` 仍记录未加倍的 router + conditional event，`conditional_event` 单列内部事件损失；原始 CE 口径不随预算改变。`budgets.event:router` 的贡献使用 0.4，其余 `event:conditional_event`、`event:time_event`、`event:note_event` 使用 0.2，均以全部事件行为共同分母。router + conditional，或 router + time + note 的贡献等于 structure 贡献；这些分项不再次加入总损失。`component:structure` 和 `budgets.structure` 的值则包含 router 的两倍因子，与未加倍的 event 字段诊断区分。

普通 NOTE 每行是 `0.45 kind + 0.55 F`，内部字段 F 如下：

| 类型 | F |
|---|---|
| Tap | `7/15 pad + 4/15 break + 2/15 EX + 2/15 short_hold` |
| Touch | `0.6 pad + 0.2 firework + 0.2 short_touch_hold` |
| Hold | `0.3 pad + 0.2 break + 0.1 EX + 0.4 duration` |
| Touch_hold | `0.4 pad + 0.2 firework + 0.4 duration` |
| Slide / Wi-Fi | `7/30 start_pad + 2/15 head_break + 1/15 head_EX + 1/15 track_break + 0.5 route + 0.5 duration` |

L3 的数值预算以 `src/maimai_baseline/loss/common.py` 为准；身份和兼容范围仅维护于中央版本文件。本文不维护版本白名单。

Slide 内部合计仍为 1.5，不归一化、不再额外乘 1.5、不额外提高 slide kind。branch 每行仍为 `0.4 route + 0.4 duration + 0.2 track_break`，不乘 0.55、不预测独立 kind/start_pad。note_semantic 将 NOTE 与 branch 按实际有效行数合并平均；没有 residual 项。

flags 使用上述逐字段系数，不再作为等权平均字段参与目标；不适用字段不监督，缺失字段不将份额转给其他字段。位置保留原区间权重，因此预算系数守恒不意味着实际 loss 数值守恒。可选 specialized supervision 保留原类别/字段倍率，默认关闭。route 仍为 `CE + 0.5 Judge + 0.5 Trajectory`，作为一个字段，不按 route_informative_mass 拆份额。

诊断中的 `budgets` 保存原始 loss、加权 loss、有效行数、难度权重和实际目标贡献。`note:<kind>:modifier_break` 等逐 flag 项取代原 `note:<kind>:flags`，branch 使用 `note:branch:modifier_track_break`；这些明细与位置、kind、route、duration 的贡献相加得到 note_semantic。字段项贡献以所有 NOTE/branch 的支持为分母；epoch 汇总累加分子与分母后归约，不平均 batch 均值。

`budget_summaries` 单独保存 `note:<kind>:flags` 摘要（批次键为 `budget_summary:note:<kind>:flags`），仅供诊断，不参与总 loss 或明细求和。摘要 raw_loss/loss 为按固定 flag 系数加权的均值，缺失字段不重新分配份额；count/weight_sum 使用适用 flag 的行支持并集，contribution 等于对应逐 flag 贡献之和，分母仍为全部 NOTE/branch。`timing` 对应 delta。Physical 的诊断、梯度算法和缓存身份不受本次调整影响。

## 时间和占用

一拍 96 tick。位置、默认等待、保护与释放容差沿全局拍轴。Hold/Slide duration 固定为写下时 BPM 对应的秒数。设出现 tick 为 q、写下 BPM 为 B、全局秒映射为 T：Hold 实际结束为 `T(q)+d*60/(96*B)`；Slide 启动为 `T(q+96)`，实际结束为 `T(q+96)+d*60/(96*B)`。跨 BPM 的结束位置不提前取整。统一换算覆盖分桶、音频上界、动作、导出和诊断。

- Tap、Touch、Slide_head 占用 `[q,q+2 tick)`，不叠加旧短音释放尾部。
- 1 tick Hold／Touch_hold（含 short flags）分别等价 Tap／Touch，占用 `[q,q+2 tick)`，不加释放尾，保留谱面类型和监督。长 Hold、长 Touch_hold 与每条 Slide 轨道在实际结束后延长 22 个全局 tick 释放。实际运动速度、duration 和音频结束判断不加此容差。
- 普通 Slide 与每条 branch 各占一手，Wi-Fi 两手，同头组不封顶。每个 owner 最多一个 branch；Wi-Fi 不能添加 branch。每条轨道分别释放。
- Slide 起点从启动保护到第 70 tick（包含），第 71 tick 恢复。它是位置限制，不额外占手。
- 任意时刻所有活动 Touch／短 Touch_hold／长 Touch_hold 合计一手，不要求同拍开始。每个音符分别检查同位置叠键与 250ms 重复限制。
- 同位置、同一时刻有 Slide 启动时，Tap／短 Hold／Slide_head 可合并一个启动组。Touch 池存在匹配启动的合格短 Touch／短 Touch_hold 时，抵消整池的一手，仅持续该短音的两 tick；池最多抵消一手，启动组不能重复给其他短音抵消。长 Touch_hold 不触发抵消；短音结束后活动长 Touch_hold 恢复池占用。长普通 Hold 独立占手。
- 所有 Touch 类排除 `tap_on_slide` 的起点、中途和尾区；占手、叠键及重复规则继续有效。
- 快滑的完整覆盖为有向关系：Wi-Fi 覆盖同起点中心与相邻两条直线（例如 `1w5 → 1-4/1-5/1-6`），普通直线不反向覆盖 Wi-Fi。Wi-Fi 同路径、既有等待/端点及形状豁免继续保留，几何 alias 不代表覆盖关系。
- 普通轨道终点等于下一轨起点，实际结束 E 与下一轨启动 S 满足 `E <= S <= advance_global_ticks(E,22)` 时，于 S 消除前轨剩余释放尾。逐轨确定性一对一匹配，branch 参与、分叉余轨单独占手；Wi-Fi 排除。轨道可同时具有前驱与后继，形成链；超界只取消优惠。比较实际秒数，跨 BPM 不提前取整，只消费已接受前缀和当前候选。
- 路径经过保护使用 200ms。同头同路径在前 Slide 出现到启动期间再次出现（包含启动端点）明确禁止，不受快滑形状豁免影响。
- 同 pad 普通 Touch、短 Touch_hold、长 Touch_hold 任意组合，起点间隔小于等于 250ms 为叠键，使用绝对秒数且不额外延长占手。训练只归责 pad 的 overlap；位置×short bit 两列同禁令，禁止向 short flag、event、kind、router、duration 转移此规则的梯度。历史和跨窗口基态保留最近 250ms Touch 类动作。

## 紧凑阶段缓存

缓存身份包含图、视图、时刻、可见历史和 owner，不含 epoch、occurrence 或 batch 位置。identity/mirror-lr 各自变换位置和 Route-71，保存自身结果。

阶段只保存实际已知条件下的结果：event/kind 可完成性；真实 kind 下位置支持；真实位置和 flags 下的一维 route 向量；真实 route 下单组 duration 边界。当前阶段仍检查全部候选，不能只检查真值答案。位置支持不读目标 route，route 支持不读目标 duration。构建其它位置的可完成性可检查路径并提前终止，但不保存完整位置×路径结果。

1/2/3 个 NOTE 块枚举 1/2/6 种顺序，覆盖 branch 进度和插入决策，并共享物理等价状态；4+ 只保留区间与关系，按实际前缀在线归并，不缓存指数子集。短生命周期查询上下文共享活动区间、启动组、位置禁令及换算。结果通过 offset/shape 定位到只读 mmap 连续数组；阶段结束释放上下文，不随 occurrence 增长。

训练与推理共用 StageEngine。推理按已选字段查询，提交/回滚使上下文失效。CPU oracle 与 `query_counts_legacy` 保留作独立参考/资产工具；当前训练不再依赖旧稠密位平面损失和 CUDA 计时接线。

规则实现由 `intervals.reasons/hand_segments` 独立区间参考与增量 `IntervalQueryContext` 对照。几何 key 区分 Touch 与非 Touch，状态 key 保留轨道终点/实际结束等规则字段；等价短 Hold 可共享事实。历史面板的活动组、需求与启动匹配边预计算复用；普通短音增量查询，轨道仅按受影响释放端点缓存面板，时间顺序前缀的衔接匹配跨 duration 二分复用。

历史裁剪先从已接受前缀锁定已经发生的逐轨交接，保留后轨的 incoming 已使用、仍在历史内的前轨 outgoing 已使用及其实际缩短释放点。过期前轨可删除，但后轨不能重新领取另一条前轨的尾部抵消。未来交接不提前锁定，同拍未发生的交接仍能按可见候选确定性匹配；不读取未来 GT。匹配采用按启动顺序的活动终点候选池，不在每个候选查询中重新扫描全曲。小表与大组在线基态都保存/加载同一匹配摘要；状态 key 包含摘要，提交/回滚按原接口恢复可见前缀。

当前规则与支持表合同已更新，旧规则表加载时报 semantics mismatch。载荷布局重打包不能升级旧规则；repack 仅接受规则语义已经相同的输入，不把旧表标记为新结果。复制来源的端到端训练包含 delta 候选可行性查询和非法时间概率损失，与上述配置规则共同参与梯度；正式训练应使用已准备的新规则表，本次仓库复制未生产正式资产。Mel、音频几何和固定计划不因 Physical 规则修正失效。不添加启动重建或扫描步骤。默认 MiaCode 独立运行时保持原状，适配器与回滚继续使用已有检查路径。
