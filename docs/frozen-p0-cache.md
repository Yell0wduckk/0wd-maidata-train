# 固定 Encoder 的 P0 导出

`python -m maidata.decoder_training.p0_cache --config cache-config.json --output NEW_CACHE`
只从明确指定的阶段一完整轮 checkpoint 导出公开 P0，不启动 Decoder 训练。

输入复用两个既有 Decoder plan.json 的完整音频组、现成 Mel/aux32、缓存生产元数据与阶段一 checkpoint。配置沿用 entry.CONFIG 的架构和 decoder_train_* 路径；本命令不消费学习率、调度或辅助损失系数。默认运行约定为 CUDA bf16、12 workers，每次处理一个完整音频组，覆盖所有训练与验证音频。独立 FrozenProducer 持有 E/D/P，D/P只用于CPU上的配套初始化合同核对，不执行前向、不传入GPU；不构建M，仅E执行eval/no_grad前向。

每曲保存唯一目标中心拼接的全曲P0：values保留实际输出dtype，beats/valid逐项保留，包含真实音频边界处无效padding；不保存Encoder内部G0/S，也不把额外空尾监督窗制造成音频。每曲文件内保存生产者checkpoint SHA、variant、coverage与schema。manifest记录精确生产者、归一化元数据、split、资产/时间映射、shape/dtype/有效数、本次生成文件的SHA和大小。旧输入不重扫、不重哈希载荷。

写完每曲后记录进度，全部完成才置complete=true；首曲重读逐位核对values/beats/valid。目录必须独立；失败保留部分产物和日志，不自动覆盖或冒充完成。本缓存供固定阶段一 D/P 的离线导出使用。阶段二训练与验证从既有 Mel/aux32 网格在线计算新 Encoder 的 P0，不读取 `decoder_train_p0_cache`；该路径在训练中可留空。固定 Z/plan 仍通过 `decoder_train_dp_cache` 读取，见[训练接入](decoder-training.md)。
