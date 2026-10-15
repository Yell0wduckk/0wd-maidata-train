# 六枚独立 Plan4 与 Z4 对齐

`configs/joint-main320-z4-gt.json` 选择 `planner_span_beats=4`：每个16拍目标窗保留前后各4拍，M分别读取6枚Z4和6枚Plan4。目标起点为`s`时，两组中心都为`s + [-2, 2, 6, 10, 14, 18]`。两类条件分别投影、归一化并使用source embedding；全部条件按绝对拍RoPE进入同一个softmax。Plan不拼入Z，也不改变Z的数值路径。

Plan与Z各自保留mask；首窗之前和完整尾窗之后缺失的槽无效。有效的无音符段与完整空尾窗仍有效，不能按零音符数屏蔽。GT条件按同chart、绝对中心从相邻16拍bank行中读取，不能复制边缘Plan冒充前后文。

P共享原query、attention pool、MLP和8/4读出参数，每4拍独立监督。每个基础16拍窗只监督4段，避免上下文重复加权；M的第1/6枚由相邻基础窗提供。loss以有效4拍段累计，P外层权重不变。Z4下每段恰有一个Z，pool的单key softmax退化为读取V，query/key梯度可为0；V、输出投影、MLP和读出继续向Z/D/E反传。此变化不增加参数。

训练M读取GT Plan4，P通过自己的监督更新；推理M读取P预测的6枚Plan4。M→Z→D→E的原梯度桥保留，GT Plan不建立M→P路径。推理GT JSON需显式`plan_span_beats=4`，每窗`plan12`为`[6,12]`、`center_beats`为上述6中心、`valid`为6个布尔值；none仅屏蔽Plan。在线、预编码/预解码缓存及交互窗口都保留相同槽语义。

已验收GT资产为`/root/autodl-tmp/plan4-statistics-20261014/gt-plan4`，schema为`maidata.gt-plan4x12.v1`。生产记录与统计位于同级`artifacts/plan4-statistics-20261014`；训练不重建或扫描旧音频、Physical、几何和窗口输入。DP预测缓存显式记录Plan跨度及中心，旧Plan16不能冒充Plan4。

默认CONFIG继续使用Plan16；历史checkpoint缺少`planner_span_beats`时按16解释，保持原无位置Plan评分。新完整轮恢复绑定Plan跨度，即使参数shape相同也不能把Plan16伪装为Plan4续训。新配方从头训练。

当前新配方还包括：E/D/M宽320、8头，每头40维；E/M前段FFN1280、D/M专家640；M为8层、router前2层；3完整音频组/80000、M chunk32768、LR6e-4、workers12、BF16。沿用既有完整组、occurrence及150轮scheduler/90轮停止计划，完整轮保存规则不变。已有validation为空，不能产生验证质量结论。

CPU验证、CUDA对照和启动证据将记录于同级`artifacts/plan4-cuda-20261014`。CUDA对照只改变Plan16与Plan4，采用相同320×8结构、LR6e-4、chunk32768和既有普通/极端自然batch，不重新统计旧数据或给谱面质量设损失阈值。
