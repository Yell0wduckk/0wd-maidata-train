# 推理：Hold 与 Slide 最终判定区

本规则只用于 `integer-draft-plan16` 的整曲生成、常驻续写和范围生成。
不修改训练 Physical、既有标签/缓存、checkpoint 配方或用户配置。

## 判定

每条 Slide 分支仅取最终 route 的最后一个几何判定区及实际进入秒数 `t`。
中间 route 的末判定区、route 终点坐标及 Slide 结束秒数均不代替该数据。
若 Hold / Touch Hold 的 pad 属于该区，且真实按住区间 `[start, end)` 与
`[Timing(t - 96 tick), t]` 相交，即触发本规则，包括这一拍内才开始的 Hold。
等价条件为 `start <= t` 且 `end > Timing(t - 96 tick)`；到区瞬间起按会触发，
恰在窗口左端松手不触发。96 tick 按全曲 timing 反算再前移一拍，
支持变 BPM；Hold 的 `end` 不含动作层附加的 22 tick 松手保护。

只在 duration 完整选定后检查，不在前面的 pad / route 采样阶段提前 hard-mask。
原有起点保护、经过判定区后的 150/200 ms 修复和其余 Physical 继续生效。
原谱前缀允许照常读取；本规则限制新生成的完整动作，不重写人工前缀。

## 概率与修改

- 捕获当前 pad / route 头经过既有合法性 mask、重复惩罚、温度和 top-k/top-p
  后的原概率及支持，连同字段之前的规则/KV快照保留。
- Slide 后生成：最后 route 仅考虑终点不同的候选；同终点的所有形状排除。
  候选只需存在合法 duration，不要求与原 duration 搭配合法。
- Hold 后生成：pad 候选必须与原 duration、原 kind 和 flags 组成合法动作。
  不额外改 flags，不通过已有 EX/MOMENT 修复去补救非法替代 pad。
- duration 候选概率为每级 CONT 与最后短值概率的乘积。合法区间由既有
  Physical 分界与新规则的解析时间交点合并生成，不逐 tick 重跑几何检查。
- 两个字段的所有合法替代合并，按原概率总量归一化，无放回尝试。
  不预先把两个字段各自合法部分归一化成 1，不把温度重复作用于已得到的概率。
- 温度 0 使用原筛选支持内的温度 1 softmax 概率确定性比较；同分按位置字段、
  duration 字段、候选编号的顺序选取。top-k/top-p 不因修复失败扩大支持。
- 选中 route 后用新 route 重新推理 duration；选中 pad 后保留原 duration。
  改 duration 时保留原位置字段。字段改变后重放相应 KV/CONT，未变前缀复用。
- 无合法替代时回到原搜索及预算恢复机制。即使预算恢复放宽旧 Physical，
  本规则仍不直接放行冲突动作。若原筛选支持及搜索前沿本身没有可行出口，
  仍保留原推理错误，不伪造时长或静默改用 Touch。

修复前不发布冲突的完整动作。字段回滚沿用流式 rollback/token 协议，文本与
IR 同版本更新；取消、拒绝和前缀改变不会把候选或重复历史带到下一请求。
报告中的 `hold_slide_tail_repairs` 独立于原 `tail_repairs`，记录双方身份、
末判定区、进入/检查时间、修改字段、修改前后值及额外模型前向对数。
详细 trace 默认关闭。每对额外前向包含一次 `begin` 和一次 `complete`。

## 验证证据（2026-10-15）

专项测试 `tests/test_inference_hold_tail.py` 覆盖两种 Hold、半开边界、最终判定区、
多分支、变 BPM 下每 tick 对照、CONT 完整概率及缓存复用、原概率合并、
同终点 route 排除、route 后重新推理 duration、pad 后保留 duration、实际 Decoder
KV 重建对照、跨窗动作、回退报告撤销、流式文本/IR配对及交互拒绝/取消/前缀修改。

可控小模型前向计数：无冲突 Hold/Slide 分别为 2/3 对；本组 pad/route 修复
分别为 4/5 对，duration 单独修复为 3/4 对。新增成本包括一层尚未访问的 CONT
候选展开；位置修改另有一对 KV 重放。实际成本随候选支持和 CONT 深度变化，
此计数不作为真实大模型的固定耗时或性能阈值。

使用本机 `draft-new-single-plan16/epoch-050.ckpt`，Windows CPU FP32、4线程，
运行 8 秒真实音频：预编码 5.77 秒，预解码冷/热分别 2.79/2.51 秒，整曲任务
7.59 秒（含启动加载，复用预计算）；三个窗口均为单枚 `[12]` Plan16。
常驻续写 229 ms、范围生成 481 ms。此次真实短段未触发本规则，额外前向为 0；
它证明入口兼容，不代替可控冲突测试，也不构成谱面质量或 CUDA 验收。

完整输出、命令、配对快照及耗时保存在工作区
`artifacts/plan16-hold-tail-20261015/`，不作为运行门禁。
