# 拆分式 Draft：直接 Z 解析接口

本次采用[已批准的拆分合同](../../0wd-v3-refactor-docs/drafter-split-execution-plan.md)。`maidata.draft`保留全曲非自回归D、辅助解析R和目标窗Planner P；主M的独立CONT、385类时间头及Physical语义继续由Decoder维护。合并方案仍为讨论记录。

## D、Z与直接解析

默认值集中在`entry.CONFIG`。D保留8层、256维、8头、4个FFN512专家，以及音频、难度、等级、拍中心坐标和主干RoPE。每拍Z直接取最终LayerNorm后隐藏，默认256维；`draft_latent_dim == draft_dim`，没有压缩投影或独立latent normalization。`BeatMemory(values, beats, valid)`不变，无效占位清零，有效空拍和完整空尾窗仍生成Z。

R直接索引查询所属拍Z，不读邻拍，没有cross-attention或查询间attention。各角色共享残差MLP：

```text
x = Z[所属拍] + role_embedding + allowed_conditions
h = LayerNorm(x + Linear(512→256)(Dropout(GELU(Linear(256→512)(LayerNorm(x))))))
logits = field_linear(h)
```

宽度随D推导；默认隐藏宽度512、dropout0.1、epsilon1e-5。序号使用无固定上限的连续编码，route前缀保留顺序。只对有效查询执行解析，batch padding不成为空槽预测。COUNT共享解析模块，只读本拍Z和固定COUNT角色。

| 角色 | 允许条件 | 监督 |
|---|---|---|
| MOMENT | 拍内起音序号、上一真实起音相对本拍起点的位置及有效位 | 拍内0…95 |
| NOTE_KIND | GT拍内起音、MOMENT序号、组内NOTE序号 | kind |
| NOTE_FLAGS | NOTE相同上游条件，加GT kind | 适用flags |
| ROUTE_ID | NOTE起音、MOMENT/NOTE/轨道/段序号、已知kind、此前route前缀 | route_id |
| TRACK_BREAK | ROUTE相同上游条件，加GT当前route_id | 首段适用track_break |
| COUNT | 无字段或数量条件 | 空/非空与正计数 |

kind不进入NOTE_KIND，当前route不进入ROUTE_ID，下级使用独立查询。GT总行数、未来类型表、pad、duration数值不进入解析；不预测type、pad、空行或结束。GT数量仅安排有效查询。

## 拍内标签与轨道

对真实起音整数tick t，所属拍 b=floor(t/96)，类别 u=t−96b。同刻多NOTE只生成一个MOMENT，按拍内真实时间排序并从0编号。窗口按左闭右开归属事件，窗首真实起音监督0；下一拍边界归下一拍0。上一真实起音可以来自前拍或前窗，全曲首次使用无历史标记；当前目标u不输入MOMENT查询。使用完整96类CE，不按历史裁候选、排序修正或去重。

Draft不再创建终端、窗外起音、CONT、删失或两个duration查询。原谱duration保持原样；适配器仍以SLIDE_DURATION关闭当前轨道、清空route前缀并推进轨道序号，保留拼接链、并行轨道及首段track_break适用性。主M及Physical的duration不变。

COUNT空/非空头2类（0为空、1为非空），监督所有有效拍；正计数头96类对应1…96，仅监督GT非空拍。有效空拍、空窗和完整空尾窗参与空/非空监督，batch占位不参与。P仍独立读取目标全部16拍Z，原plan12统计、池化结构与损失保持原样。

## 张量、loss与训练

E按音频准备G0并编码覆盖窗，唯一中心P0拼接成全曲bank；同音频及难度/等级的D只计算一次，全曲Z供所有监督窗、R及P复用计算图，不detach或跨更新复用旧表示。

`ReconstructionTargets`分别持有字段标签、`moment_tick [B,Q]`、`moment_counts [B,T]`、窗口有效位及独立queries。`ReconstructionOutput`为`field_logits`、`moment_logits [B,Q,96]`、`empty_logits [B,T,2]`、`positive_count_logits [B,T,96]`、窗口/query/count有效位。Q只是batch补齐宽度，不是网络容量或可见总行数。

分类kind、route_id、track_break及每个flag各有独立CE总和/计数，各自归一后对逻辑更新中有监督的字段等权平均。总loss为字段均值＋MOMENT96＋空/非空＋正计数＋Planner连续项＋0.25×Planner类别项。MOMENT无smoothing，其余分类和数量头smoothing0.05；无标签项贡献有限零。累积先取得整个逻辑更新共同分母，再反传各microbatch分子；epoch及validation也按累计分子/分母报告。

训练继续复用pilot320计划、完整组、顺序和batch边界。`pretrain_batch_reference_rows=207`仅为既有成本proxy，真实查询数量另记；workers12、BF16、scratch、学习率/调度、完整轮checkpoint及每5轮/末轮validation不变。末轮诊断包括条件MOMENT准确率、96类真值/预测分布、数量两头准确率、GT非空条件正计数与合成数量准确率，以及分类混淆和少数类召回。

## 导出与兼容性

```powershell
python -m maidata.draft.export_targets --chart /path/to/charts/10/5.json.gz --start-beat 0 --start-beat 16 --output /path/to/new-targets.json
python -m maidata.draft.capacity --plan /path/to/plan.json --split /path/to/split.json --output /path/to/query-cost
```

标签导出标识为`beat-local-tick96`。原谱v2、Mel/aux32、Physical、采样计划、GT plan及主M窗口缓存直接复用，不迁移整库。新阶段一checkpoint为`maidata.encoder-draft-training.v3`，严格记录结构和损失配方；旧checkpoint不作为完整续训输入，不隐式部分加载。

旧128维Z不能补零扩为新缓存。取得新教师权重后须在另目录前向导出Z/预测plan；新DP缓存schema为`maidata.frozen-draft-planner.v2`，记录`z_representation=final-normalized-beat-z`及`z_width`，P0格式不变。导出与消费代码支持新宽度，教师配置与M配置分别构造。M启用Z时投影须匹配生产者宽度；关闭Z且GT/无plan时不加载教师。

本次验收范围为受影响CPU测试。CUDA原batch、正式训练、真实新教师导出及谱面重建质量尚未验证；实际测试结果见[实施记录](drafter-split-implementation.md)。
