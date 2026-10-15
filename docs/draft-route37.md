# Draft 的 Route-37 接受度监督

`entry.CONFIG.reconstruction_route_vocabulary = "axis15-37"` 仅改变 Draft 的 R
route 输出头与监督：71 个精确 route 沿起点—对侧中心线合并为 37 类。
以起点1为例，轴为1—5，固定1和5，交换2/8、3/7、4/6，并交换
`<`/`>`、`p`/`q`、`pp`/`qq`、`s`/`z`。34对镜像合并，`1-5`、`1v1`、`1w5`单独保留。
新类别按组内最小原ID递增编号，完整映射在 `maidata.draft.route_vocabulary`。

原谱、基础窗、缓存及 R 的已知 route 前缀条件保持原71类。loss适配时将
有效原 route 标签转换为37类，保留忽略位、实际支持数、字段等权归约、
MOMENT、COUNT开关及 Planner。普通与 prepared loss 共用转换和损失计算；
准确率/混淆矩阵仍按精确37类GT统计，并报告原ID分组，不改为接受度加权准确率。
阶段一和联合训练共用这条路径，不重建输入资产。

## 接受度矩阵与损失

默认 `entry.CONFIG.reconstruction_route_loss = "acceptance37"`。固定矩阵
`A [37,37]` 的行是GT类别，列是预测类别，元素为该预测对GT的接受度。
矩阵来自已填好的 `route37-接受度填写表-反向联动.xlsx`：空白为0、对角为1，
保留非对称关系，不做行归一化。源码资源位于
`src/maidata/draft/route_acceptance.py`；训练直接读取固定资源，不读取Excel。
原表位置为工作区的
`outputs/01a1298c-a591-7cb1-a4f2-9132c65e0a6c/route-acceptance/route37-接受度填写表-反向联动.xlsx`。

对GT类别 `y` 和模型37类概率 `p`，ROUTE损失为：

\[
L_{\mathrm{route}}=-\log\sum_{j=0}^{36} A_{y,j}p_j.
\]

这替换ROUTE的硬标签CE和均匀label smoothing；接受度不作为归一化软标签。
使用FP32的 `log_softmax` 和 `logsumexp` 计算，接受度0对应项不参与概率求和。
WiFi行只有自身接受度为1，因此自然退化为硬标签NLL。单位矩阵也逐项退化为NLL。
空监督贡献有限零；有效位、支持计数及逻辑更新、epoch、validation的归约保持原规则。
其他字段继续使用各自原有损失和smoothing设置。

显式 `reconstruction_route_loss = "ce"` 保留旧CE模式及其原有smoothing。
`acceptance37` 与 `axis15-37` 配套；使用 `route71` 时需同时选择 `ce`。

主 Decoder 的71类输出、训练标签、几何距离、Physical和推理精确route不变。
现有 checkpoint recipe 的 `reconstruction_route_loss` 保存损失模式；接受度模式下
`reconstruction_route_acceptance` 保存从源码资源生成的实际37×37矩阵，不是可覆盖的
配置项。完整续训不能静默切换损失模式或矩阵。旧配方缺少
`reconstruction_route_loss` 时按旧CE解释。
损失选择不增加网络参数或state_dict项，不影响相同输出头的权重形状。
37类 R 不能直接加载旧71类 R权重或冒充完整续训。明确配置 `route71` 和 `ce`
可构建原头；旧配方缺少词表项也按原71类解释。
仅使用 E/D/P/M 的旧 checkpoint 推理不依赖 R 分类头。

Decoder软标签仅记录于[未来方向](decoder-route-soft-label-future.md)，尚未实施。
本次只修改本地实现和文档并进行定向CPU验证，不启动训练、上传或重建缓存。
