# M Z 条件适配器归一化与 ckpt55 继续训练

用户已用 Still 自由生成确认 epoch60 相比 epoch55 明显退化，并授权停止旧运行、补齐归一化、观察梯度后从完整 ckpt55 继续。

仅在 `Linear(Z) + source_embedding[0]` 之后使用 parameter-free LayerNorm，eps 沿用 `decoder_norm_eps`。不增加参数或 optimizer state；Z 坐标、有效位、plan 分支、Q/K/V 参数与条件 gate 不改。D 末层及 E P0 既有归一化保留，M→Z→D→E 梯度不 detach。

唯一配置 `decoder_z_normalization` 新训练默认 true；缺少此字段的旧 checkpoint 在推理中解释为 false，不会因读取新源码改变旧模型行为。新 checkpoint recipe 显式保存该标记。

首次从旧联合 checkpoint 升级需同时使用 `--resume .../epoch-055.ckpt --upgrade-z-normalization`。该开关只允许 false→true 的这一项变化；其他表示/架构检查和严格 state_dict 加载保留，五模块参数、optimizer、scheduler、完整轮游标和全部 RNG 都恢复。不会把修改后的图称为原图的逐位续训。后续从本次新 checkpoint 恢复不再传升级开关。

继续沿用 stop90 / horizon150、COUNT off / MOMENT96 on、GT plan、Z attention on、BF16、workers12、原预算/逻辑 batch/窗口与 loss。旧运行独立保存，续训输出另设目录；不重验旧资产、不产冻结 P0/Z 缓存。

正式恢复前仅对受影响部分进行 CPU 检查和原配方 epoch56/57 重放，记录每个更新的非有限值检查、preclip 梯度范数、模块梯度、Z 适配器输出尺度及末层 condition Q/K 尺度。重放不保存训练 checkpoint，正式恢复再次读取原 epoch55 的完整状态。有限重放通过不保证后续永久稳定。
