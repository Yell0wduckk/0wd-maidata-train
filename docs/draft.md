# 拆分式 Draft：直接 Z 解析接口

本次采用[已批准的拆分合同](../../0wd-v3-refactor-docs/drafter-split-execution-plan.md)。`maidata.draft`保留全曲非自回归D、辅助解析R和目标窗Planner P；主M的独立CONT、385类时间头及Physical语义继续由Decoder维护。合并方案仍为讨论记录。

## D、Z与直接解析

默认值集中在`entry.CONFIG`。D保留8层、256维、8头、4个FFN512专家，以及音频、难度、等级、拍中心坐标和主干RoPE。默认每拍Z直接取最终LayerNorm后隐藏，默认256维；`draft_latent_dim == draft_dim`，没有压缩投影或独立latent normalization。`BeatMemory(values, beats, valid)`不变，无效占位清零，有效空拍和完整空尾窗仍生成Z。

R直接索引查询所属拍块Z，不读其他块，没有cross-attention或查询间attention。各角色共享残差MLP：

```text
x = Z[所属拍块] + role_embedding + allowed_conditions + block_beat_position
h = LayerNorm(x + Linear(512→256)(Dropout(GELU(Linear(256→512)(LayerNorm(x))))))
logits = field_linear(h)
```

宽度随D推导；默认隐藏宽度512、dropout0.1、epsilon1e-5。序号使用无固定上限的连续编码，route前缀保留顺序。只对有效查询执行解析，batch padding不成为空槽预测。COUNT共享解析模块，只读所属拍块Z、固定COUNT角色及块内拍偏移。间隔为1时偏移编码为零，保持原计算与权重形状。

| 角色 | 允许条件 | 监督 |
|---|---|---|
| MOMENT | 拍内起音序号、上一真实起音相对本拍起点的位置及有效位 | 拍内0…95 |
| NOTE_KIND | GT拍内起音、MOMENT序号、组内NOTE序号 | kind |
| NOTE_FLAGS | NOTE相同上游条件，加GT kind | 适用flags |
| ROUTE_ID | NOTE起音、MOMENT/NOTE/轨道/段序号、已知kind、此前route前缀 | route_id |
| TRACK_BREAK | ROUTE相同上游条件，加GT当前route_id | 首段适用track_break |
| COUNT | 无字段或数量条件 | 空/非空与正计数 |

kind不进入NOTE_KIND，当前route不进入ROUTE_ID，下级使用独立查询。GT总行数、未来类型表、pad、duration数值不进入解析；不预测type、pad、空行或结束。GT数量仅安排有效查询。

## 可配置的 Z 拍间隔

统一参数 `draft_z_stride_beats` 默认 `1`，只接受 `1 / 2 / 4 / 8 / 16`；同时要求它整除目标窗口长度。阶段一同步更新 Encoder 或读取冻结 P0 的模式均可使用：

```bash
python -m maidata.training --config run.json --z-stride-beats 4
python -m maidata --component draft --z-stride-beats 4
```

也可在 JSON 配置写入 `"draft_z_stride_beats": 4`。训练入口的命令行值覆盖 JSON，省略时沿用配置。

设间隔为 N，每个音频/条件组以音频覆盖的 `first_beat` 为块起点，D 的查询坐标是 `first_beat + k*N + N/2`。直接减少 D 的全曲查询数量，输出仍是各查询最终 LayerNorm 的隐藏，不做事后平均或抽样。负拍起点照常处理；每个16拍目标窗及完整空尾窗有 `16/N` 枚 Z，batch 占位无效并归零。音频 Encoder 的逐拍 G0/S、Mel/P0 密度和音频窗口范围不变。

真实起音的拍归属、MOMENT96标签、拍内序号、上一真实起音条件及逐拍数量两头均不变。R读取所属块的一枚Z；所有角色额外使用**已知的块内拍偏移**，编码为 `sinusoidal(offset)-sinusoidal(0)`，让同一块内不同拍可得到不同预测。COUNT仍每拍监督，不把N拍的数量相加；当前待预测tick、GT数量和未来事件不进入查询。没有新增参数、attention或独立生成流程。

P池化目标16拍内全部 `16/N` 枚Z。M的Z上下文取与请求区间相交的完整块，使用块中心绝对拍坐标，边界外占位无效，不把Z复制回逐拍。例如N=8、目标[16,32)、前后各4拍时，选中块中心12、20、28、36；音频仍读取原来的[12,36)局部窗口。较粗Z覆盖整块是条件粒度的改变，不改变采样、音频余量或主M监督。

checkpoint配方和新DP缓存都记录间隔。没有该字段的既有拆分式checkpoint/DP缓存明确解释为1，默认路径的网络权重形状和计算保持兼容。完整阶段一续训不能更改间隔；新间隔应训练相应D后另目录导出Z/预测plan，不能靠修改manifest或抽取旧Z实现。主M训练/适配从缓存自动采用宽度与间隔；在线推理检查教师间隔与M所训练的条件一致。Z关闭且使用预测plan时，由教师独立决定其间隔。

原谱、Physical、Mel/aux32、冻结P0、采样计划、GT plan及M窗口标签继续复用；采样顺序、batch边界及207行成本proxy不随N改变。此次仅修改入口与衔接代码，不改变已绑定实验的源码或配置。

## 拍内标签与轨道

对真实起音整数tick t，所属拍 b=floor(t/96)，类别 u=t−96b。同刻多NOTE只生成一个MOMENT，按拍内真实时间排序并从0编号。窗口按左闭右开归属事件，窗首真实起音监督0；下一拍边界归下一拍0。上一真实起音可以来自前拍或前窗，全曲首次使用无历史标记；当前目标u不输入MOMENT查询。使用完整96类CE，不按历史裁候选、排序修正或去重。

Draft不再创建终端、窗外起音、CONT、删失或两个duration查询。原谱duration保持原样；适配器仍以SLIDE_DURATION关闭当前轨道、清空route前缀并推进轨道序号，保留拼接链、并行轨道及首段track_break适用性。主M及Physical的duration不变。

COUNT空/非空头2类（0为空、1为非空），监督所有有效拍；正计数头96类对应1…96，仅监督GT非空拍。有效空拍、空窗和完整空尾窗参与空/非空监督，batch占位不参与。P仍独立读取目标全部16拍所对应的Z，原plan12统计、池化结构与损失保持原样。

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

旧128维Z不能补零扩为新缓存。取得新教师权重后须在另目录前向导出Z/预测plan；新DP缓存schema为`maidata.frozen-draft-planner.v2`，记录`z_representation=final-normalized-beat-z`、`z_width`及`z_stride_beats`，P0格式不变。导出与消费代码支持新宽度，教师配置与M配置分别构造。M启用Z时投影须匹配生产者宽度；关闭Z且GT/无plan时不加载教师。

本次验收范围为受影响CPU测试。CUDA原batch、正式训练、真实新教师导出及谱面重建质量尚未验证；实际测试结果见[实施记录](drafter-split-implementation.md)。

## Z 拍间隔实现验证（2026-10-11）

本地受影响CPU用例累计125项通过，其中 `test_draft_z_stride.py` 新增20项：五种间隔的FP32/BF16真实E/D/R/P更新、负拍坐标和不等长padding、完整空尾窗、逐拍标签与数量监督、所属块梯度、P全目标读取、缓存模式E冻结、P0/DP离线数据集与读写、M消费粗粒度Z的前后向、完整状态恢复、旧教师加载及CLI/采样边界。其余105项回归覆盖既有D/R/P、训练与恢复、Decoder适配和Z/plan选择路径。旧DP测试桩已补齐现有 `validate_input=False` 参数；不同查询数量造成的CPU矩阵舍入差异约1e-7，不属于条件泄漏。

代码与CPU验证完成；没有上传、CUDA检查、正式训练或真实谱面质量验证。更改N需要对应训练与新的Z/预测plan产物，不能据此宣称现有教师已学会另一种间隔。
