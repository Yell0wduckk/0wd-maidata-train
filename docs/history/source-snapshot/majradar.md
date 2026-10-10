> 历史参考，迁移于 2026-10-10。原路径：`docs/majradar.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 每16拍 MajRadar 条件

A 的 GT 雷达替代原 latent，prior 在同一 A optimizer 内学习雷达。谱面 encoder 保留冻结且不执行；不采样 posterior、不计算 KL。MOMENT、时间和配置继续由原模型预测，Physical 规则与损失不变。执行和恢复见 [训练](../../luna-training.md)。

## 数值与时间

字段顺序固定为 `note, peak, sweep, slide_tricky, slide_sequence, jack, fitted_constant`。前六项取 MajRadar 公共 Scores，范围0–250，除以250；定数范围0–18，除以18。内部 raw 与 `slide_cumulate` 不是训练目标。不重估 normalization。局部窗口应用上游整谱统计公式；拟合定数不能直接解释为整谱难度。

使用原 segment 的16拍区间，左闭右开；Tap/Hold/Touch/TouchHold 按起点、Slide 按声明拍归属，保留完整等待、时长、轨道、head/group 关系及分支，不纳入上一窗持续对象、不裁短尾部。拍、秒、路径段时间减去窗口起点。窗口零点补有效 BPM；有效统计时长仍为上游的 `max(window-end, last-owned-event-end)`。密度网格和模式链从窗口零点重新开始。

既有正时长 segment 均是有效标签，不因音频 mask、空谱面或处于最后边界被删。原生正时长空谱面的六轴为零，但拟合定数约10.931399414465353；完整保留，不伪造七维零标签。batch padding 无效。解析异常或 partial 保留具体错误，不能作为零值参与训练。

## 缓存生产

生产器读取选中谱面的 canonical note-view，按原空间变换生成 identity/mirror。完整视图仅渲染和适配一次，随后裁取所有既有窗口并调用未修改的 MajRadar Runtime。MiaCode 与上游 MajRadar 仅用于离线生产；训练读取缓存，不加载 .NET 或重新分析谱面。

本次 pilot 选择为261首/280谱/6658个既有 segment，两个视图共13316条标签。独立输出 `D:/01YD/01code/codex/maimai-ai-charting/data/majradar-pilot261-main-new-v1`。同一 `(segment_id, spatial_view_id)` 在轮次、occurrence 及 NOTE 顺序增强间共享；mirror 单独分析。

从本 worktree 执行：

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m maimai_baseline build-majradar-cache `
  --profile ../repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/profile `
  --subset-config configs/pilot261-mmoe-decoders-a90.json `
  --miacode-exe ../tools/MiaCode/build/Release/miacode_chart_ir.exe `
  --majradar-project ../repos/MajRadar/MajRadar.csproj `
  --build-root .test-temp/majradar-20261008/cache/host `
  --output ../data/majradar-pilot261-main-new-v1
```

宿主为持久 .NET9 JSONL 进程。缓存生产与训练启动分开；本次允许新雷达分析，不重算 Mel/BPM/Physical 资产。新缓存只做一次覆盖、状态、有限值及 loader 读取验收。来源提交是出处；实际标签数据及布局用于恢复绑定，不把源码/生产者哈希作为准入。

## A 与推理

新增输入 `--majradar-cache`，或数据清单 `inputs.majradar_cache`。路径解析同其它训练输入；不根据缓存重新选择记录或改变计划。每窗7维 GT 生成七个128维语义轴 token，router 与八层条件 cross 不对轴使用时间 RoPE。

八层自回归 GMM4 的输入为上一窗规范化 GT，BOS为7维，查询中心为 `16j+8`；读取同一次编码的完整 P0。全序列前向只为因果前缀与复用，NLL 仅取本次原 batch 的 target windows，保留重复 occurrence，分母为有效窗口数乘7。A 总目标为 L3 + Physical + Radar NLL（系数1）；无新增 warmup。NLL 在 FP32 中按先加维度 log-prob、再混合四个分量计算。

Decoder 使用 GT，不把谱面损失回传 prior；雷达 NLL 经共享 P0 更新音频 encoder。训练采样条件与生成预测条件存在既定差异，质量以实际评谱为准。

推理默认 AR，仅需 A checkpoint；不需要 GT 缓存、参考谱或 B。每窗采样一个联合7维向量，裁到规范化[0,1]，同一结果用于反馈和 Decoder。保留 logvar[-6,2] 及两种温度默认1；温度1下最小标准差约为雷达轴12.45、定数0.896，这是当前采样合同。NLL 不改成截断密度。普通和流式共用一次全曲 prior K/V、独立 seed 和连续 RNG；暂停或 Decoder 回滚不重复抽样。
