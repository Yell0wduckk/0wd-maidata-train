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

旧Plan16单token、旧Plan4六token及旧Z块中心路径继续按保存配方加载。缺失新字段解释为历史默认值；参数形状相同也不能将旧语义作为新语义完整续训。

## 参数与运行范围

E/D/M保持320维、8头、每头40维；E/M前段FFN1280，D/M专家640，M为8层/router前2层。P与R结构不变。参数总数56,178,272，相对当前Plan4增加0；相对最初256×6结构增加23,431,846。相同seed下各模块初始权重逐项完全相同。

保持学习率6e-4、3完整音频组/80000预算、M chunk32768、BF16、workers12及已验收采样/occurrence/batch边界。新增表示和重叠R会增加实际计算，原分批proxy不是显存上限。

验证覆盖整数锚点/终点、重叠监督与跨父窗归属、P17节点/GT16拍、23Z+5Plan独立mask、dense/packed/增量路径、单块/多块梯度及E/D单次共享反向、缓存与旧语义兼容。CUDA以原五类自然batch对照当前Plan4/Z4；最长D为800拍，已包含其中。每case独立进程，2次预热、5次计时，DataLoader等待单列；OOM只记录，不自动修改配方。

只有完成实现、CPU验证、新GT准备及上传后才停止旧运行；允许轮内停止，保留最近完整轮checkpoint，不另存半轮状态。停止并释放GPU后执行CUDA测速。本轮不启动新正式训练。

执行证据及最终测量报告位于`repos-0wd-v3/artifacts/integer-draft-plan16-20261014`，不把CPU结果或测速结果当作谱面质量结论。
