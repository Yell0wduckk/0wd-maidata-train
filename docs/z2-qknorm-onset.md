# Z2、逐头 Q/K RMSNorm 与逐拍起音存在监督

配方为 `configs/joint-main320-z2-single-plan16-qknorm-onset-gt.json`，在既有整数 Draft worktree 直接实现。E/D/M 320维、8头、每头40维；P每头32维。M八层、E/M前段FFN1280、D/M专家640保持。峰值LR6e-4、150轮cosine/绝对停止150轮、3完整音频组/80000预算、M chunk32768、BF16、workers12保持。

## 表示与 attention

整数 Z 锚点以 coverage.first_beat 为相位，间隔两拍并包含完整空尾窗终点T；长度从T+1变为T/2+1（这里T指覆盖拍数）。每枚中心t仍重建 `[t-2,t+2)`。对目标窗 `[s,s+16)`，P读取s到s+16的9枚Z，M读取原锚点请求范围内s-2到s+18的11枚Z及单枚Plan，共12个key。无效外缘不复制，空段有效。

单Plan的统计保持 `[s,s+16)`，旧窗口Plan16 GT缓存直接复用。P自身的query位于s+8并保留RoPE；M的Plan key及对应query保持无位置编码，与旋转后的Z评分进入同一个softmax。

E/D/P/M所有Transformer点积attention均在Q/K线性投影拆头后、RoPE前执行无仿射RMSNorm：FP32 `x*rsqrt(mean(x²)+1e-6)`，恢复原dtype，保留 `1/sqrt(head_dim)`。V与频率tokenizer池化保持。KV只在生产投影时归一化一次，dense、packed和增量共享数学。四个组件开关保存到recipe，旧recipe缺字段时为false；旧checkpoint不会因无参数变更而静默切换数学。

## 起音存在监督

`reconstruction_onset_presence_supervision=true`启用独立 `Linear(320,1)`；`reconstruction_onset_presence_weight=1.0`。使用已有COUNT角色、四拍内相对位置及R共享MLP，不输入起音数、标签或未来字段。输出和标签与普通真实事件查询分开。

每枚Z对四拍范围内每个有效拍预测是否存在至少一个真实起音；同拍多个起音仍为1，持续音符不产生新起音。有效空拍/空尾为0，覆盖之外不监督。普通父窗拥有8枚Z，最后父窗额外拥有T；各owner任务只安排一次，整曲每个真实拍恰有两个owner。

起音存在项使用FP32 BCEWithLogits，分母为逻辑batch实际有效(owner Z,beat)任务数，完整覆盖每个chart为2T。不做类别加权。字段与MOMENT96仍按各自真实有效查询归约，不做拍内平均；数值数量监督继续关闭。新增321参数，总数56,178,593，相对当前Z1单Plan模型增加321。

每轮日志单列起音损失、正负支持及实际任务数。完整诊断仅绝对末轮：从该轮实际训练前向收集，无额外数据遍历，标记teacher-conditioned训练样本，报告owner/beat混淆矩阵、precision/recall/F1。既有验证集为空，不新建验证拆分。

## 兼容与执行

默认配置保留旧数学；新实验通过独立配方开启全部改动。新旧Z间隔、Q/K开关及起音头开关完整续训必须匹配。推理按保存配方自动构造，旧无归一化缓存key保持，新表示使用独立身份。音频、Physical、窗口、原谱和采样计划不重建；外层batch proxy不依赖Z间隔，更新次数及batch边界保持。

CPU覆盖几何/端点/去重/空尾/标签隔离、损失公共分母和梯度、Q/K数值及RoPE顺序、旧checkpoint、普通/缓存/增量、单块/多块与E/D单次共享前后向。实现上传完成后停止16079旧Z1训练，保留最近完整轮checkpoint；GPU释放后对照旧Z1单Plan与新方案，每个自然batch/profile独立进程，2次预热和5次完整更新计时，DataLoader等待单列。复用普通/P98/最大M/最大E局部与全局/最长D样本，重复样本合并；OOM或非有限值阻止正式启动，不自动改变预算。验证完成后从头训练150轮。
