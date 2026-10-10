# model

来源模型的音频处理、router、decoder、音频草稿及相关组件。来源的一拍草稿路线读取外部预计算的原生音频特征，训练步不运行该外部音频 encoder；MajRadar、GMM prior 和谱面 encoder 等模块属于保留的其他路线。

计划中的全曲/局部联合音频 encoder 尚未在本副本实现。已有全曲与局部模块不能据此描述为新架构已经完成。

仓库定位见 [README](../../../README.md)，当前范围见 [current-runtime](../../../docs/current-runtime.md)。
