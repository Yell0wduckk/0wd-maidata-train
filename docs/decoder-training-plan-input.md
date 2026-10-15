# Decoder 训练 plan 输入开关

唯一默认值位于 `src/maidata/entry.py` 的 CONFIG：

```python
"decoder_train_plan_input": True,
```

将该项设为 `False` 可关闭 M 的 plan 输入；JSON 配置使用 `false`。
适用于阶段二训练、冻结 Encoder/Drafter 的 SFT 和从头联合训练，训练及验证使用相同设置。
`decoder_train_plan_source` 仍选择 GT 或预测 plan 来源。该开关只控制 M 是否接收它，不改变 Z 输入、音频输入、采样、loss 权重或联合训练中 P 自身的监督。

关闭时在生成 packed attention 布局之前，将 plan 值置零、plan_valid 置 false，移除有效 plan key；不是把零 plan 当成有效条件。
已有 plan 缓存仍按原路径加载，不因此重建或更换缓存。
开关随训练 recipe 保存进 checkpoint。旧配置未指定该项时保持开启；推理仍使用推理入口的 plan-source 参数。

此项为启动时配置。修改本地配置不会热更新正在运行的训练；需要在后续启动时明确选择。
