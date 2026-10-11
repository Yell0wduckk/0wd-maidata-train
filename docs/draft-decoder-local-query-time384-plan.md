# Draft 局部条件重建、Decoder 字段查询与 384 tick 时间编码计划

> 历史方案：主 Decoder 的 FieldReader、逐字段读取和旧窗口监督描述已由[独立 CONT 序列合同](decoder-cont-token-plan.md)取代；下文保留作为历史记录。Draft R 的独立局部查询设计继续有效。

日期：2026-10-10（America/Los_Angeles）。状态：已授权实施，模型、监督、推理和缓存编译已接入；具体验证证据另见本次实施记录。

适用代码为独立仓库 `repos-0wd-v3/src` 下的 `maidata`。设计起点为提交 `41efda0f7e3d1d4e6af591dfbb88218755055a76` 及已有采样工作树。原文第2节记录改造前状态，第10节列出实施覆盖接口；当前实际接口见[Draft](draft.md)、[Decoder](decoder.md)、[缓存迁移](time384-cache-migration.md)。未启动正式训练、整库迁移或替换实验；超本地窗GT duration监督按下述最新补充实施。

## 最新补充：超音频窗 duration 监督

用户追加确定：保留完整 GT 组织、精确时长、真实动作和后续历史；只监督到跨到音频窗尾或其外的最后一个 CONT。如果最后非 CONT 从窗内发出，落点在窗外也正常监督。以相对边界 500 tick 为例，700 监督 `CONT,316`，769 监督 `CONT,CONT`，不把真实 769 改成 768。

本补充覆盖下文原“延期边界”和训练侧局部 cap 投影设计。训练读取完整原 Physical 支持，跨窗 CONT 是保留概率质量的零罚未完成叶，窗外短值仍使用原 Physical 处罚；不因局部查询边界重投影 type/WiFi。原谱及 Physical v2 源资产、迁移和上传无需等待监督修改。监督窗口单独升为 `maidata.decoder-window-tensors.v4`，旧 v3 窗口标签通过不变的精确源重新编译，不能只换版本名。推理的既有局部 cap 暂不改变。当前行为以 [Draft](draft.md)、[Decoder](decoder.md) 和 [迁移说明](time384-cache-migration.md) 为准。

## 1. 目标与范围

将 Draft 的整窗固定行序号重建改为局部、按字段条件化的辅助预测；主 Decoder 同步采用字段查询位置、条件传递和离散时间预测。Planner 保持读取目标窗全部 16 拍 Z，不随重建头缩小为三个近邻 Z。

首版约定：

- D 仍从音频及既有难度/等级条件生成全曲 Z；保持一拍一个、128 维，不同时改成四拍一个 Z。
- GT 只安排重建查询及监督索引，不把 GT 总行数或完整 GT 类型序列作为网络输入；不预测 padding/空槽。
- 重建内容头局部读取；时间头使用覆盖其预测跨度的有限读取范围。
- MOMENT、HOLD_DURATION、SLIDE_DURATION 均改为 `1..384` 加 CONTINUE 的离散分类，共 385 类。
- 三种时间共享编码算法和边界适配，但各自保留字段角色、条件与预测参数；不默认强制共享三套输出头权重。
- CONTINUE 是一个时间字段内部的续段选择，不新增真实谱面事件或 Physical 动作。
- 真实谱面、96 tick 拍轴、written-BPM 时长、Route-71、链/分支和四类 Physical 的物理定义保持。duration 可生成范围按第 6 节改为本地音频窗上限；它对候选支持的影响单独迁移。
- 优先复用原谱 v2、Physical v2、音频输入、几何、采样计划；只重新编译改变的监督表示，必要时补算受影响的候选支持。

本计划不自动更换 Encoder 结构、不宣布冻结共享 Encoder 已可用，也不决定新训练轮数、学习率或启动实验。既有双 Encoder 路径可继续使用；共享/微调策略另由主任务结果决定。

## 2. 现状与此次变化

| 层次 | 当前实现 | 计划 |
|---|---|---|
| D / Z | 全曲双向 D，一拍一个 128 维 Z | 保留粒度、宽度和音频条件边界 |
| R | 48 拍 Z，固定 207 行，按序号预测类型、pad、其他字段及绝对时间 | 独立 GT 对齐内容查询；不预测行类型、pad、空槽；时间用条件化相对 tick |
| P | 目标 16 拍 Z，输出 plan12 | 保持目标 16 拍全部 Z；不使用 R 的局部读取 mask |
| M | AR 类型/字段，连续时间混合分布；字段共用当前前缀位置 | 保留 AR 结构和 pad；增加独立字段查询坐标，三种时间改为 385 类续段预测 |
| Physical 源 | 完整事件 GT 前缀上的整数 tick 区间与归责见证 | 复用；按新候选域做投影或小范围修正，不为 CONT 新建动作 |
| 窗口监督 | 旧连续分布区间，duration 为 4 tick cell | 从精确源字段和整数支持重新编译逐 tick 监督 |

R 的准确率以后是“给定合法条件时的字段预测能力”，不能和旧盲重建准确率直接横向比较，也不能据此宣称主 Decoder 已学会自由生成。

## 3. Draft 查询与 Planner

### 3.1 GT 只决定训练查询，不输入总行数

按真实字段创建独立查询。物理 batch 可以 padding、压紧或打包，但其有效位仅用于计算和 loss，不成为其他查询能读取的长度条件。GT 查询不回流到 D、音频 Encoder 或 Planner 的输入。

不再要求 207 个固定行槽重建整个 16 拍序列。每条查询只带字段角色、自身允许的历史条件、局部序号和位置，不接收整窗 GT 行数或后续类型清单。查询间不加能够读取未来 GT 类型/位置的 self-attention。

| 任务 | 内容头可见的条件 | 初始查询位置 | 监督 |
|---|---|---|---|
| MOMENT | 前一有效时间锚、已知历史；续段时加累计 tick | 前一锚点 | 下一起音 delta 的离散编码 |
| NOTE | 当前 MOMENT 位置、同刻 NOTE 序号 | 当前 MOMENT | kind 与适用 flags；不监督 pad |
| ROUTE | 当前 MOMENT、轨道/链内序号及已知链前缀 | 当前 MOMENT | route_id、适用的 track_break |
| HOLD_DURATION | 当前 MOMENT、已知 NOTE 种类 | 当前 MOMENT | 完整 hold 时长的离散编码 |
| SLIDE_DURATION | 当前 MOMENT、当前轨道已知的完整 route 链 | slide 启动位置 | 完整轨道总时长的离散编码 |

这里的“字段角色”是要回答哪个任务，不等于提供待预测的 NOTE.kind、route_id 或时间值。Slide 的条件必须覆盖当前轨道完整链，不能只给最后一个 route，也不能混入其他并行轨道的答案。

同刻多个 NOTE 使用可区分的局部序号；ROUTE 使用轨道与链内序号。它们不是总行数输入。若多条查询的全部输入完全相同，不能期待其产生不同的确定性输出。

对 MOMENT，选 Z 和定位 query 均不能使用当前待预测起音。NOTE 等任务允许使用已给定的当前 MOMENT，但这些条件不能通过共享 query hidden/mask 泄漏给 MOMENT 预测。

### 3.2 空拍、首音和尾部

保留独立逐拍 MOMENT count 监督，`count=0` 承担空/非空监督；不再另造逐 padding 槽的空行任务，也不把 GT count 输入内容头。空拍依然属于有效 Z 和有效训练范围。训练内容 loss 按 GT 适用字段计算，不用模型预测的空/非空结果屏蔽 GT 内容。

R 与 M 复用窗口边界约定：先有固定窗首锚，窗首恰有起音时直接安排 NOTE 查询，不为它生成 delta=0 类别。窗内其他起音由正 delta 预测；末尾安排独立终止查询。旧 R 在窗首额外创建零位置 MOMENT 的做法由此替换，GT count 仍把窗首真实起音计入。

终止查询的网络角色仍是普通 MOMENT，条件仍只来自已知前缀。`terminal/exact/censored/closed-by-continue` 仅供目标、loss 和控制结果解释，不能作为输入提前告诉时间头“本次应当结束”。

窗首没有更早真实音符、连续空窗、历史 dropout 均沿用显式边界角色，不依据数值 0 推断“真实起音”。终端查询不更新最后真实起音。

### 3.3 局部读取范围

对一拍 Z，中心坐标为 `j+0.5`。普通内容查询在拍位 q 时，首版读取包含 q 的拍及前后一拍，即索引 `floor(q)-1, floor(q), floor(q)+1`。缺失槽使用 mask，不循环补齐、不按 GT 长度补 Z。

MOMENT 的一次短值覆盖未来 384 tick＝4 拍。首版使用与已知查询游标绑定的有限前向读取：从 `floor(q)` 到 `floor(q+4)` 的 Z，共不超过 5 个槽，覆盖 `(q,q+4]` 内可能的起音拍。CONT 后按新游标重新选择。不能为了限制为三个 token 而提前使用 GT 下一起音选择邻域。

duration 续段在累计实际结束时间对应的拍位读取近邻三个 Z，保留 NOTE 起点和 route 等固定条件。D 可含全曲上下文，但重建头不能绕过局部选择再读取全曲 Z/P0。

局部化约束施加到字段查询；主 M 的既有因果历史与音频窗口仍保留，不因 R 的三个 Z 限制而裁掉整个主干的音频/历史条件。所有局部查询只能读取当前消费者已提供的音频/Z，不能静默扩大现有 24 拍音频输入。

### 3.4 Planner 必须读取全部目标 16 拍

P 读取 `[window_start, window_start+16)` 的全部有效 Z：首版为 16 个 Z，不含前后余量。P 使用自己的窗口选择和 mask；不得复用 R 的三个近邻 mask，也不接收 GT 内容查询、行数、位置列表或 R 输出。

plan12 的 8 个连续统计和 4 类概率、空窗标签及 GT 统计定义先保持不变，因此现有 GT plan12 缓存可复用。这里是明确保留当前目标窗级接口，不是将 Planner 局部化后再扩大。

## 4. 主 Decoder 的字段查询

### 4.1 保留生成完整谱面所需的结构

主 M 仍自回归预测类型，保留 NOTE 的 kind→pad→flags、ROUTE、两种 duration、边界与 Physical。只有 R 不监督 pad；M 仍须预测和监督 pad，否则不能生成可播放谱面。

主 M 训练的“给定条件”来自 GT 已知前缀；推理时来自已接受的生成前缀。构造函数与条件可见性相同，不能在训练时直接输入尚未生成的当前/下一 MOMENT 答案。

### 4.2 Router、历史和查询位置分开

沿用 `Decoder.begin()` 不接收目标类型、后续内容分支才使用类型条件的边界。GT next-type 只能选择被监督的分支，不能通过位置、邻域选择、mask 或类型 embedding 提前进入 Router。

分开保存：

- `history_beats`：已提交真实 token 的坐标，用于因果 self-attention/KV；不会随某个字段的 CONT 移动。
- `field_query_beats`：当前内容或时间查询的临时位置，用于读取音频/Z。
- `origin_tick`、`elapsed_ticks`：当前时间字段原点与累计量；它们不等于真实谱面游标。

首版采用独立字段读取模块：从主 M 当前前缀得到内容基态，加入允许的字段条件，按字段位置读取音频/局部 Z。时间续段复用同一主前缀基态，输入角色、原点、累计量和当前查询位置；不反复向主 self-KV 追加同一前缀。

查询 Q 与音频/Z 的 K 使用同一拍坐标系，推荐继续使用绝对拍位。窗内位置作为额外条件时显式加回 window_start；不能将窗内 Q 与全曲 K 直接配对。plan token 仍保持无时间 RoPE。

在窗内预测新 MOMENT 时，初始查询锚为当前有效时间游标；它可能是已接受的真实 MOMENT，也可能是显式窗首边界。窗首边界输入仍保留到最后真实起音的历史间隔，不能把这两个原点混为一谈。

## 5. 384 tick＋CONTINUE 的共同编码

### 5.1 词表与唯一分解

保持 96 tick／拍；384 tick 是长度为四拍的一个时间块，不是提高到 384 tick／拍。词表固定预定义为 `1..384` 和 `CONTINUE`，不按训练集出现频率收集。

对正整数时间 d：

```text
n = (d - 1) // 384
r = (d - 1) % 384 + 1
encode(d) = [CONTINUE] * n + [r]
d = 384 * n + r
```

| d | 唯一编码 |
|---:|---|
| 1 | 1 |
| 48 | 48 |
| 96 | 96 |
| 384 | 384 |
| 385 | CONTINUE, 1 |
| 768 | CONTINUE, 384 |
| 769 | CONTINUE, CONTINUE, 1 |

CONT 表示剩余量大于 384；它令查询累计量增加 384，但并未确定完整时长。累计为 a 时，短值 r 对应完整值 `a+r`，CONT 对应完整值域 `d >= a+385`。

生成 MOMENT 至少 1 tick；Hold 完整时长至少 2 tick；Slide 至少 1 tick。Hold 的续段余数 1 可以合法，如 385＝CONT,1；合法性检查完整总量，不对每个余数重复应用完整时长最小值。外部零边界不进入此正值词表。

### 5.2 字段内部循环与提交

主 M 对原始时间字段仅做一次类型决策。CONT 后锁定当前字段类型，不重新让 Router 选择 NOTE/ROUTE。记录 `field_kind/origin/elapsed/route_context/query_position`，只更新临时查询状态。

普通短值完成后，合成一个完整时间字段再交给原 Grammar/Physical。MOMENT 更新真实游标；duration 完成动作但不推进原 NOTE 起音。CONT 不调用 `stage_and_accept`，不修改 owner、track、最后真实起音、活动动作或首次归责账本。

字段循环状态与候选栈、主 KV 引用、Physical 快照一起恢复。跨窗只保留真实历史；不把扫描经过的 CONT 位置写入谱面或下一窗历史。

首版按用户描述逐子步生成：温度为 0 时，每一步在当前合法短值/CONT 中取 argmax；非零温度按每步分布采样。此处采用分层贪心，不能宣称等价于对所有完整时间值的 `P(d)` 取众数：CONT 汇集多个长值，其概率可以大于任一短值，但继续后的每个完整长值概率仍较小。这是相对旧完整候选众数选择的生成语义变化，应记录在推理结果中；本计划不自动加入完整值 MAP 搜索。

规范语义 token 仍采用完整 MOMENT / HoldDuration / SlideDuration。CONT 只出现在模型内部查询/监督表示中，MiaCode 导出先合成完整值，因此原谱语义缓存无需插入 CONT。

### 5.3 CE 与 loss 分母

精确时间值的 loss 是整条规范编码路径的 NLL 之和。一个原始时间字段仍只贡献一次原始字段计数，不能改成按 CONT 子步数平均；类型 loss 也只计算一次。R 的时间项按原始有效时间字段归约，M 继续使用原始有效决策的归约口径。

子步数影响计算量，须另行计入查询成本；不能用 loss 分母隐藏计算量，也不能因展开子步而重复数据 occurrence。三头、R 和 M 共用分解/合成及标签集合函数，不复制独立编码公式。

CE 直接学习 tick 类别，但仍允许 47、49 等细格，不能宣称必然消除相位抖动。原 duration 的 4 tick cell 改为逐 tick 分类属于精度变化，后续评估要与单纯替换分布形式区分。

## 6. 时间原点、窗尾与 duration 上限

### 6.1 MOMENT

设固定谱面目标窗为 `[s,b)`，音频后余量上限 `U=b+audio_after_beats*96`，本次完整 MOMENT 的原点为 o。初始查询在 o，累计 a 后在 `o+a`。

- 短值 r 得到 `o+a+r < b`：合成真实 MOMENT，继续生成其内容。
- 短值 r 得到 `o+a+r >= b`：关闭本窗，不提交真实起音。允许从窗内直接预测跨窗值，不把候选裁到 b。
- CONT 后 `o+a+384 >= b`：立即关闭本窗，不继续预测剩余短值。此分支没有最终 r，不能硬套完整值公式生成一个假起音。
- CONT 后仍在窗内：累计 a 增加 384，再查询一次。

恰好到 b 的起音属于下一窗。CONT 跨窗的结果记录为终止控制及其下界：选择 CONT 时已知完整值至少 `a_old+385`，而查询游标只推进了 384。控制下界、临时查询位置和真实下一起音必须分别表示。

窗口结束后保持既有真实历史和活动动作；下一窗重新注入边界，边界间隔仍从最后真实起音计算。

### 6.2 终止训练与右删失

精确 GT 下一起音存在时，先得到规范编码，再在与推理相同的首次跨 b 位置停止监督。若 CONT 已跨 b，即使源谱知道更远的确切位置，也不再监督剩余编码；这是新的终止监督语义，不冒称保留了全部旧窗外距离信息。

旧标签“到 U 仍未观察到起音”是右删失，不是恰好在 U 起音。迁移时保留该标志和阈值。若当前残量下界为 L：

- L > 384 时，下一步确定为 CONT；若此步跨 b，监督至此结束。
- 1 <= L <= 384 时，目标集合为短值 `r >= L` 加 CONT，对兼容且 basic 合法的类别概率求和后取负对数。

这仍是 385 类概率上的集合 NLL，不新增虚构时间类；恰好 U 的未观察起音与更晚起音同属尾部。新增表示必须保留 `exact/censored/closed-by-continue` 的区别，不能用一个 tick 同时冒充三者。

### 6.3 Hold / Slide 的 tick 和 RoPE

保留现有 written-BPM 时长定义。设 NOTE 起音 tick 为 o，完整时长整数为 d，`T(x)` 是全局 tick 到秒的映射，`BPM(o)` 是该 NOTE 写入 BPM：

```text
unit_seconds = 60 / (96 * BPM(o))
hold_start   = T(o)
slide_start  = T(o + 96)
hold_end     = hold_start  + d * unit_seconds
slide_end    = slide_start + d * unit_seconds
```

Slide 初始 query 对齐 `o+96`，即启动位置。累计 a 后的 query 为 `second_to_beat(start_seconds + a*unit_seconds)`。跨变 BPM 时不能直接在 RoPE 上累加 `a/96`，也不能对每个 CONT 重新取当前位置 BPM。主输出仍是原 written-BPM 整数时长。

### 6.4 duration 的本地音频窗边界

本方案按用户要求，duration 可生成至当前本地音频窗末尾 U；同时保留整首有效音频末尾的原有合法性限制。因此候选域取两者交集，用上面的真实秒数公式求最大完整 duration tick，不能直接用 `U-o` 替代。

duration 的 CONT 若已经没有任何合法更长完整值就不可选；最后一个合法整数值仍须由短值完成，不在 cap 处制造一个未知长度的真实动作。MOMENT 的窗尾 CONT 出口不受这一规则误伤，因为它不提交音符。

超出本地音频窗的 GT duration 按用户要求暂不设计自动延窗、跨窗续接或其他特殊生成机制。迁移必须保留源精确值及旧支持，不截短、不伪造已迁移的可训练标签、不默默从训练计划删除相关实例；本次处理到的这类记录列出受影响窗口，留作实施时的边界未覆盖项。不能因为该局部问题升级为整库重建。

## 7. Physical 复用与离散概率

### 7.1 保留完整事件作为物理单位

Route 几何、动作轨迹、占手、交接、释放、四类原因及已有冲突见证继续以完整事件计算。CONT 不建立新的 Physical 阶段，也不重复记录一次相同冲突。

原 `maidata.training-physical-original.v2` 保存的是整数起音/时长区间，每段包含 basic、完整 physical、首次 penalty、deferred、conflicts/new_conflicts；它并非 4 tick 的损失桶，可以复用为新适配器的输入。

累计为 a 时：

- 短值 r 查询完整总量 `a+r` 所属区间。
- CONT 的 basic 支持为剩余域 `[a+385, upper]` 中是否存在 basic 合法完成；MOMENT 的窗尾控制作为合法出口保留。
- 生成时启用 Physical hard mask，CONT 必须存在一个同时满足 basic 和全部硬约束的完整完成，不能逐类分别做存在性再拼接。
- 训练概率只按 basic mask 归一化；不能先用 Physical mask 删除违规概率，再声称训练了违规质量惩罚。

不能将 CONT 的四类位简单 OR/AND 为一个普通候选。特别是不同完整值分别违反不同类时，逐类 AND 可能全零，但不存在任何完全合法的完成。

### 7.2 对完整时间分布计算 Physical loss

以同一个原始时间字段为单位，设第 k 个 CONT 前缀后的分布为 `p_k`，短值 r 的完整值为 `d=384*k+r`：

```text
S_0 = 1
S_k = product(p_j(CONT), j < k)
P(d) = S_k * p_k(r)
P(close_by_CONT_at_k) = S_k * p_k(CONT)
q_c = sum(P(d) for d whose original first-attribution penalty contains category c)
L_c = -log(1 - q_c)
```

`close_by_CONT` 概率只在该 CONT 首次跨越谱面窗尾时形成终止叶；它纳入总概率和 Physical 合法补集，不再展开后续层，不能又把同一质量计入完整长值。由短值跨窗形成的终止叶也保留其概率。

保持四类权重 `(1,1,4,1)` 与固定四类归约。窗口终止叶子的 Physical 罚为零，不提交真实 NOTE。一个原始字段只计算一次 Physical 分母，CONT 不修改 `charged`；完整字段结束后沿用原首次归责流程。上述 L_c 是数学表达；实现沿用现 `physical_categories` 的稳定合法补集 `logsumexp` 和 FP32 epsilon floor，不直接浮点计算 `1-q_c`，也不在所有 basic 候选均违规时产生无穷 loss/NaN 梯度。

精确计算 q_c 要覆盖整个有限合法域的 CONT 单链及每层短值，或在已形成窗尾终止叶处停止，不能只计算 GT 走过的层数、把未计算长尾概率当零。这里每层只有一个继续分支，不是对 384 个短值分别递归；允许批量构造各深度已知的 CONT 前缀查询，但必须与逐步推理等价。

当前范围内可复用原整数区间对完整值做索引/切片；无需重新求几何交点。生成时 basic/raw hard mask 与训练 penalty 概率保持分开。

### 7.3 本地 cap 会影响哪些上游支持

收紧 duration 上限后，某些早期选择的存在性判断也会改变，例如“现在结束链并选择 SLIDE_DURATION”的类型候选，以及不能继续接段的 WiFi route 候选。

优先用旧完整 duration 区间裁到新 cap 后重新投影。若某个旧 stage 只有已投影的 type/route 结论，没有相应反事实候选的完整时长区间，则不能从一个 bitmask 反推新结论；只对该窗口、该前缀、该候选调用现有 Physical 查询，复用已载入几何和已缓存真实动作。

候选域缩小可能改变首次归责位置。受影响前缀之后应重新推进归责账本，使用已有冲突见证重算 penalty；仅在缺少候选证据时补查询。可复用条件必须包含 grammar、活动动作、charged 和新候选域一致，不能仅比较 token_index。

窗口边界与 terminal 原本就可能需要单独查询，继续沿用现有 `ChartSupports.window` 的等价前缀复用思路。这些局部计算是本次语义变化所需，不扩展为全曲几何重建或全部候选重算。

## 8. 现有资产的迁移矩阵

| 现有资产 | 首选处理 | 不可混淆的边界 |
|---|---|---|
| 音频、Mel、aux32、BPM/first、归一化元数据 | 原样复用 | 不重提音频、不重新估计统计 |
| Route-71、path catalog、slide_data、已存真实动作 | 原样复用 | 不因 CE 或 CONT 重建几何/轨迹 |
| 原谱 `full-chart-original.v2` | 原样作为精确语义源 | tokens/token_map 保留精确 tick；CONT 不写入原谱 |
| 完整 `training-physical-original.v2` | 复用整数区间和见证；按窗口派生 cap 投影 | 缺少反事实 duration 支持的少量前缀按第 7.3 节补算 |
| 旧 Decoder window-tensors v2 | 从精确原谱＋完整支持重新编译新目录 | 不从旧 4 tick duration target 猜精确值 |
| 旧 compact-window 积分区间 | 留给旧消费者 | 它只压缩旧连续分布标签，不能充当此次 CE 迁移器 |
| Draft 标签/容量 | 从已选原谱窗口派生新查询索引、时间标签及查询成本 | 207 行上限不再作为新语义容量；空有效窗继续存在 |
| 固定 P0 | E 权重和输出接口不变时复用 | E 更新后需重新前向导出；原 Mel/aux32 不变 |
| 固定 Z / 预测 plan12 | 生产者与接口不变时可作旧教师对照 | D/P 更新后重新推理导出；不能 reshape/插值旧 Z 冒充新训练结果 |
| GT plan12 | 16 拍窗口与统计定义不变则复用 | Planner 读取全部 Z 不要求重算 GT 统计 |
| split、window_id、occurrence、镜像/抽取规则 | 原样复用 | 不把 CONT 计为起音，不重抽数据或加倍实例 |
| 旧 checkpoint | 可明确选择兼容组件作初始化 | 新头不能等同旧模型完整轮 resume；不伪造 optimizer/调度兼容 |

旧窗口 tensor 的 duration 已经使用 `(ticks+2)//4` 标签，并按桶代表值投影 Physical。仅靠这些标签不能恢复源精确 tick 或桶内规则变化。必须回读已经存在的原谱 v2 和完整 Physical v2；这属于监督格式重编译，不是从源音频/源谱开始全量重建。

旧神经特征没有一般性的“换个格式就得到新权重输出”的迁移方法。只有生产者保持不变才能直接复用；必要的 E/D/P 前向导出与昂贵的音频/几何重建要明确区分。当前阶段二在线训练 E_M，不应把重新导出固定 P0 加成每次迁移的必经步骤。

## 9. 增量迁移顺序与新缓存内容

### 9.1 首选迁移流程

1. 读取当前任务明确指定的原谱、Physical、计划和相关小型 manifest，保留原资产身份；不递归重哈希已验收载荷。
2. 按既有 window_id 读取所需原谱 v2，恢复精确起音、duration、route 链、边界及源索引。
3. 对同一真实 GT 前缀复用 Physical 整数支持。按本地 cap 派生窗口域，必要时只补第 7.3 节缺少的支持及后续归责。
4. 使用同一个共享时间编译器，生成 R 查询、M 字段查询、规范 CONT 路径、删失集合标签和临时 RoPE 坐标。
5. 生成新的监督缓存与 manifest；原谱、旧支持和旧窗口缓存保持只读。每项新产物保存时验收一次，按既有方式原子发布。
6. 复用 GT plan、split 和 occurrence。分别记录真实主序列长度与新增 query/substep 成本；按已有组批合同处理新增计算，不悄悄改变实例集合、batch 边界、workers 或预算。
7. 新模型训练及相应特征导出属于后续实施/实验阶段，本次不执行。只在 E/D/P 实际变化后生成对应的新神经特征。

### 9.2 新监督缓存的必要字段

新表示至少保留：

- `chart_id/window_id/source_token_index` 与真实语义字段归属；一个完整时间字段到多个内部查询的映射。
- 独立的真实主序列、历史位置、边界角色、最后真实起音与字段查询记录。
- 查询角色、原点、累计量、query 拍位、同刻/轨道/链内序号，以及该字段允许读取的条件索引。
- 精确 GT 整数 tick、规范短值/CONT 路径、右删失下界、首次关闭位置；不只保存分桶标签。
- 完整支持区间的引用或窗口切片、basic/raw/penalty 的区分、需要修改的 cap 投影及来源映射。
- 原始字段 loss 计数、主序列长度和内部查询成本，分别统计。

实际新窗口 schema 为 `maidata.decoder-window-tensors.v3`，消费者记录 `relative-tick384-continue` 表示、局部查询规则及 cap 语义。旧tensor必须从精确原谱v2与完整支持重新编译，不能只改schema字符串。

可以将新增查询索引/窗口域支持存成引用旧原谱和旧支持的 sidecar；不强制复制所有旧载荷，也不建立仅为证明旧输入有效的新凭据。

主 self-attention 序列不因 CONT 展开而自动增长；增加的是字段内部查询。因此不能笼统将每个 CONT 加进旧 `lengths` 的主 AR 行数。若实现选择不同的历史展开方式，须先改清此合同，而不是静默改变 batch 和历史语义。

## 10. 从零生产与增量迁移共用实现

本节为实施覆盖接口清单。实际可执行参数见[缓存迁移](time384-cache-migration.md)，新监督生产和增量迁移均调用唯一`compile_window`。

| 模块/入口 | 实施职责 |
|---|---|
| `maidata.time` 的新增离散编码适配模块 | 唯一实现分解/合成、合法最小值、query 坐标、窗尾/删失及时间目标集合 |
| `draft/targets.py`、`draft/supervision.py`、`training/data.py` | 创建独立局部查询和 count 标签；调用共享时间适配；不传 GT 总行数 |
| `draft/export_targets.py`、`draft/capacity.py` | 导出新查询可核对标签，容量改为 query/substep 成本；与训练用同一目标构造函数 |
| `draft/reconstruction.py`、`draft/losses.py` | 条件查询、局部读取、无 pad/类型/空槽重建，时间 CE/集合 NLL |
| `draft/planner.py`、`draft/plan_targets.py` | 保持独立目标 16 拍选择及现 plan12 统计；检查不会继承 R 局部 mask |
| `decoder/network.py`、`heads.py`、`attention.py` | 分离 Router、历史 KV 与字段 query；三时间头替换为 385 类，并保留 M pad |
| `decoder/windows.py`、`decoder_training/supervision.py` | 相同前缀构造字段条件和时间循环标签；新分母、Physical 完整值概率 |
| `decoder_training/support_cache.py`、`physical/attribution.py` 的适配 | 复用同义前缀，增加窗口候选域处理与局部缺项补算；CONT 不新增归责阶段 |
| `decoder_training/export_windows.py` | 从精确原谱和完整支持调用统一新 `compile_window`，同时供增量迁移和全新生产使用 |
| `decoder_training/window_cache.py`、`lengths.py`、`ordering.py` | 新 schema/pack/unpack/索引，分开主序列与查询成本，保留 history dropout 的真实物理状态 |
| 拟新增的监督迁移入口 | 接收既有原谱/Physical/计划与新输出目录，调用同一编译器；不另写第二套时间/Physical 算法 |
| `chart_cache/produce.py`、`training.py`、`rebuild_supports.py` | 从零产生精确原谱与完整整数支持后进入相同编译链；让新窗口 cap 所需支持可由同一 helper 获得 |
| `chart_cache/migrate.py`、`rebuild_fields.py` | 旧 v1 来源仍先做既有 v1→v2 语义迁移，再进入同一新编译链；已有 v2 不重新走该路线 |
| `decoder_training/compact_windows.py` | 明确保留为旧连续分布缓存的等义压缩工具，不冒用为新 CE 迁移器 |
| `inference/sampling.py`、`generation.py`、`decoder/session.py` | 字段内部 CONT、重定位、窗口关闭、完整值支持、快照/回退与最终一次提交 |
| `inference/materialize.py`、chart codec 的消费适配 | 只接受合成后的完整事件，验证时间和 route 链原义保持；不把 CONT 导出为音符 |
| `p0_cache.py`、`dp_cache.py`、`producer.py`、checkpoint loader | 区分可复用生产者与新权重导出；新头的架构身份和初始化/续训边界明确 |
| `entry.py` 和实际 launcher | 实施时集中设置真实消费者需要的配置；默认 workers12；从零/增量路径都选择新编译器 |

从零生产路径为“源谱解析→精确原谱→完整 Physical 支持→统一查询/时间编译→新监督缓存”。增量路径从已有精确原谱/支持接入同一链路。音频生产入口算法不因这一时间头方案重写；原有音频和几何资源继续复用。

采样计划始终按真实 NOTE 起音统计，CONT、边界、终端、duration 不计入复杂节奏数量。新的查询成本只影响计算编排，不授权额外 occurrence、改变镜像分配或重新划分训练/验证。

## 11. 实施顺序与针对性验证

建议按以下顺序实施，避免把监督、压缩粒度和 Encoder 共享策略同时改变：

1. 实现共同时间表示、边界/删失、written-BPM 坐标与字段查询协议。
2. 实现 Draft 局部内容查询，保留一拍 Z、独立 count 和目标 16 拍 Planner。
3. 实现主 M 字段查询及内部 CONT 循环，保持结构预测、pad 和真实历史。
4. 接入完整值 Physical 概率、局部 cap 支持与缓存编译/迁移；同步从零生产入口。
5. 在明确选定的新实验中训练和比较；不替换正在运行任务的源码、输入或 checkpoint。

实施时只验证此次影响的合同：

- 唯一时间编码：1/2、383/384/385、768/769；窗口边界首音、空窗和跨窗 CONT。
- 变 BPM 的 MOMENT 与两种 duration 逐值还原；Slide 等待一拍、链/并行分支和原始结束秒数一致。
- GT 条件隔离：改变未来 GT 行数/类型/位置，不改变对应前缀 Router 或较早时间查询；同刻序号可区分。
- R 局部读取、MOMENT 有限前向读取、P 目标全部 16 拍读取各自独立；空有效窗与 padding 不混淆。
- 字段循环批量训练与逐步推理一致；history/KV/Physical 不随 CONT 假推进，回退可恢复。
- 新旧表示映射到同一完整 tick 时，规则结果一致；跨非单调支持的 CONT 做联合合法完成判断。
- 完整值概率之和、删失集合概率及 Physical q 的正确性；长尾不因只计算 GT 链而消失；分母按原始字段计。
- 对本次选择的少量真实窗口比较“迁移编译”与“同一精确记录直接新编译”，核对语义、标签和支持；只验收新产物，不全库重验旧资产。
- cap 改变的 type/WiFi 存在性投影及后续归责，覆盖“唯一合法完成被 cap 排除”的具体例子。

新增网络/训练实现按既有合同验证受影响的梯度、mask、共享计算、冻结范围；本次CPU验证与少量明确真实窗口迁移对照独立记录。当前本机为CPU Torch，未执行CUDA原batch检查/成本测量，未启动正式训练或整库迁移。

正式训练继续保留 workers12、完整轮 checkpoint、每5轮及末轮 validation、完整诊断仅末轮和默认 trace 关闭。质量由自由生成谱面与用户评估判断，不设新的 loss/性能阈值或资产验收凭据。

## 12. 现状依据

- [Draft 当前模块与监督](draft.md)、[阶段一训练](encoder-draft-training.md)。
- [Planner 的目标窗选择](../src/maidata/draft/planner.py)、[plan12 GT 定义](../src/maidata/draft/plan_targets.py)。
- [主 Decoder 与类型/内容分离](../src/maidata/decoder/network.py)、[当前训练接口](decoder-training.md)。
- [原谱与完整 Physical 缓存](chart-cache.md)、[字段/时长/窗尾规则](chart-fields-physical.md)。
- [精确时长定义](../src/maidata/time/timing.py)、[原窗口终止标签](../src/maidata/chart/windows.py)。
- [Physical 区间来源](../src/maidata/physical/support.py)、[首次归责与见证](../src/maidata/physical/attribution.py)。
- [完整支持的保存与读取](../src/maidata/chart_cache/training.py)、[等价前缀复用](../src/maidata/decoder_training/support_cache.py)。
- [旧时间区间/4 tick 标签](../src/maidata/decoder_training/intervals.py)、[窗口编译](../src/maidata/decoder_training/export_windows.py)、[旧窗口格式](../src/maidata/decoder_training/window_cache.py)。
- [固定 P0](frozen-p0-cache.md)、[固定 Z/plan](frozen-dp-cache.md)、[采样计划](sampling.md)。
