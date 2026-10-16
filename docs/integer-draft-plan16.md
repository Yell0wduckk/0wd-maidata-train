# 整数拍 Draft 与中心重叠 Plan16

本实现位于隔离 worktree。覆盖配置为 `configs/joint-main320-integer-plan16-gt.json`；默认 CONFIG 及旧 checkpoint 保持历史语义。

## 坐标合同

设目标窗口为 `[s,s+16)`，既有完整覆盖（含一个完整空尾窗）的终点为 T。

| 对象 | 坐标或范围 |
|---|---|
| D 的 Z | 整数拍0…T，包含T，每拍一枚320维连续向量 |
| Z中心t的R任务 | `[t-2,t+2)`四个完整拍 |
| Plan中心c | 0、4、8…T |
| Plan精确GT | 原谱事件位于`[c-8,c+8)`，固定16拍 |
| P池化 | Z中心`c-8…c+8`，包含两端，17槽 |
| M直接读取Z | `s-3…s+19`，23槽 |
| M读取Plan | `s+[0,4,8,12,16]`，5槽 |

五个Plan需要P读取33个Z（`s-8…s+24`），不扩大M的直接Z或原24拍音频范围。全曲D本身仍可读取完整音频bank并双向交互。

覆盖之外的节点保留固定槽位，独立mask为false；不复制边缘、不把有效空段当成缺失。Plan锚点存在时保持有效，统计范围越界部分无源谱音符，秒长仍按完整TimingMap计算。尾窗靠前的Plan可包含此前真实音符；不能假定所有尾窗Plan恒为零。

## 配方、接口与监督

新配方设置`draft_z_anchor_mode=integer`、`draft_z_stride_beats=1`、`reconstruction_span_beats=4`、`planner_layout=centered-overlap`、`planner_stride_beats=4`、`planner_span_beats=16`。布局与统计跨度独立，跨度16不再隐含使用旧的单Plan无位置路径。

R查询新增独立的`owner_z_beats`，真实目标beat、MOMENT拍内tick96和原字段允许条件保持。父窗负责本窗的整数Z锚点，最后父窗另负责T；标签从全谱取，跨父窗边界仍完整。不同Z对同一真实拍的监督是预期重叠，仅去除相同owner/query被父窗重复安排的情况。按实际有效query/owner配对数归约，保持原外层权重；数量监督继续关闭。

P每个父窗输出5槽，前四槽参与辅助监督，只有最后父窗的第五槽参与监督。这样每个`(chart,center)`只监督一次，M窗口共享边界Plan不会重复加权。

Plan接口新增独立`plan_beats`，M消费Z `[B,23,320]`、Plan `[B,5,12]`及各自位置/mask。分别投影和归一化后组成28个条件key，均按绝对拍RoPE进入同一个softmax。Z关闭或Plan关闭各自屏蔽对应key，不依赖另一类条件的长度或有效位。

GT缓存使用`maidata.gt-plan16-centered-stride4.v1`，按`(chart_id,center_beats)`保存唯一Plan；DP预测缓存记录对应布局与坐标。新GT从原谱精确统计，不能平均旧Plan4值代替。生产入口：

```text
python -m maidata.decoder_training.gt_plan_cache --config RUN_CONFIG --output NEW_GT_DIRECTORY --workers 12
```

旧Plan16单token及旧Z块中心路径继续按保存配方加载。main不启用已弃用的Plan4配方，Plan统计范围保持16拍。缺失新字段解释为历史默认值；参数形状相同也不能将旧语义作为新语义完整续训。

`configs/joint-main320-z4-plan16-gt.json`单独保留扩容后的320维、M八层和3/80000预算，但使用旧Z4块中心与旧Plan16单token，明确设置`planner_layout=legacy`、`draft_z_anchor_mode=block-center`，两个新跨度/间隔字段保持null。D中心为2、6、10…，R解析各Z所属四拍块；P仅池化当前16拍内四枚Z。M读取六枚Z（相对中心−2、2、6、10、14、18）和一枚Plan，`plan_index=6`。Plan key不旋转，Plan评分使用未旋转query；Z评分仍使用绝对拍RoPE，两者保留旧单softmax数学。普通推理、预计算缓存与增量推理按checkpoint配方选择该历史路径，无需新增生产源码分支。

该配置从头训练150轮、调度150轮，最大LR6e-4、M chunk32768、BF16、workers12。GT引用既有旧Plan16资产；部署到不同输入根时仅覆盖运行路径，不改其布局身份。兼容测试`tests/test_main320_plan16_compat.py`检查实际320维完整权重严格加载、旧字段缺省、单Plan无位置评分，以及小模型普通/冷缓存/热缓存和增量一致性；CPU测试不代表CUDA性能或生成谱面质量验收。

## 整数Draft保留，改用单窗无位置Plan16

独立模式`planner_layout=window-single`由`configs/joint-main320-integer-single-plan16-gt.json`选定。它保留整数Z0…T、Z1输出间隔、每枚Z的四拍重叠R任务与M的23枚Z；每个16拍窗口仅有一枚Plan，GT统计范围为`[s,s+16)`。P读取整数节点`s…s+16`共17槽，内部池化query位于`s+8`且保留P自身的绝对拍RoPE。输出为`[B,12]`及独立`[B]`有效位，`plan_beats=None`。

M的23枚Z与单Plan共24个key，`plan_index=23`。Z的评分使用旋转后的query/key，Plan的评分使用未旋转的query/key，共用原单softmax计算；Plan不获得位置编码。该选择与旧Z4/单Plan16的`legacy`布局是两个不同配方身份，不能通过把layout改为legacy而丢失整数Draft几何。普通、GT/预测条件、预计算缓存、交互及增量推理保留同一语义。

GT直接复用既有`maidata.gt-plan12.v1`窗口Plan16缓存，统计与归一化不变。消费者接受其历史`legacy`统计布局，但不改写缓存manifest、schema或生产者身份，也不生产新统计资产。P的辅助监督每个基础窗一次，对应相同的`[s,s+16)`统计；R监督及去重规则不变。预测D/P缓存和新checkpoint另行记录`window-single`，不能复用其他布局的预测Plan。

从`centered-overlap`完整轮checkpoint迁移须显式执行：

```text
python -m maidata.joint_training --config NEW_RUN_CONFIG --resume COMPLETE_EPOCH_CHECKPOINT --upgrade-single-plan16
```

迁移只允许`centered-overlap/stride4`改为`window-single/stride=null`及GT缓存输入路径变化。全部其他配方字段、每轮更新数和其他输入路径必须一致；不能同时升级Z归一化。五模块权重严格加载，完整optimizer、scheduler、完整轮游标、DataLoader和全部RNG照常恢复；不修改原checkpoint。新输出记录`single-plan16-upgrade.json`来源及变更，此后新checkpoint按`window-single`正常续训，不再传迁移标志。普通resume仍拒绝跨布局。

受影响CPU测试包括`test_window_single_plan_runtime.py`与`test_window_single_plan_resume.py`：P的17节点/中心与原centered第三槽数值相等，窗外Z不影响P；M24-key无位置Plan评分，dense/packed/增量一致；旧GT读取不修改资产；单块/多块E/D各一次共享反向；严格状态迁移和非法变更拒绝；普通/冷缓存/热缓存的GT、预测、独立条件开关一致。它们不作为真实CUDA训练或谱面质量证据。

## 参数与运行范围

E/D/M保持320维、8头、每头40维；E/M前段FFN1280，D/M专家640，M为8层/router前2层。P与R结构不变。参数总数56,178,272，相对当前Plan4增加0；相对最初256×6结构增加23,431,846。相同seed下各模块初始权重逐项完全相同。

保持学习率6e-4、3完整音频组/80000预算、M chunk32768、BF16、workers12及已验收采样/occurrence/batch边界。新增表示和重叠R会增加实际计算，原分批proxy不是显存上限。

验证覆盖整数锚点/终点、重叠监督与跨父窗归属、P17节点/GT16拍、23Z+5Plan独立mask、dense/packed/增量路径、单块/多块梯度及E/D单次共享反向、缓存与旧语义兼容。CUDA以原五类自然batch对照当前Plan4/Z4；最长D为800拍，已包含其中。每case独立进程，2次预热、5次计时，DataLoader等待单列；OOM只记录，不自动修改配方。

只有完成实现、CPU验证、新GT准备及上传后才停止旧运行；允许轮内停止，保留最近完整轮checkpoint，不另存半轮状态。停止并释放GPU后执行CUDA测速。本轮不启动新正式训练。

执行证据及最终测量报告位于`repos-0wd-v3/artifacts/integer-draft-plan16-20261014`，不把CPU结果或测速结果当作谱面质量结论。
