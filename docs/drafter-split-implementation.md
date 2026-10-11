# 拆分式 Drafter 实施记录

实施合同：[drafter-split-execution-plan.md](../../0wd-v3-refactor-docs/drafter-split-execution-plan.md)，确认日期2026-10-11。源码Git根为`repos-0wd-v3/src`，本次仅做源码、受影响CPU验收和本地提交。

## 实施范围

- D最终LayerNorm隐藏直接作为每拍Z；默认256维，无压缩及额外latent归一化。
- 六种角色直接索引所属拍Z，通过共享残差MLP解析；kind/flags与route/track_break分别查询。
- 真实起音拍内96类标签，包含窗首0及跨窗历史；Draft无terminal/CONT/duration监督，轨道分隔仍保留。
- 每字段独立归一、逻辑更新共同分母；空/非空及正计数分别监督，末轮诊断同步。
- 阶段一checkpoint v3、标签导出、新教师Z/plan生产消费及M宽度适配；教师与M配置独立，关闭Z的GT/无plan路径不加载教师。

正式默认值、pilot320计划、207成本proxy、主M的CONT/385类及Physical保持现有合同。复用已验收资产，不上传、不运行CUDA、不启动或干预实验，不扫描或重建旧资产。

## 验证状态

代码及本次合同要求的CPU合成项目已完成。环境为Windows、Python 3.13、Torch 2.11.0+cpu、pytest 8.3.4。测试临时文件均位于本源码D盘`.test-temp`。

| 范围 | 实际结果 |
|---|---|
| Draft标签/网络/导出/共享图及入口 | 49 passed，8.17s |
| 阶段一配方、分字段归约、实际CE梯度、FP32/BF16更新、完整轮恢复及输入消费 | 38项唯一CPU测试通过 |
| 统一Draft/pretrain/frozen/Decoder/inference回归 | 357 passed、1 failed、1 deselected，182.62s |
| v3真实路径provenance导出修复后定向复核 | 4 passed，1.08s |

通过项目覆盖tick0/95/96、负拍、变BPM、同刻多押、跨窗历史、空拍/空尾窗、源duration轨道分隔、kind/route下级条件隔离、无GT pad及总数输入、仅所属拍Z梯度、末层LayerNorm与Z精确一致、不同microbatch共同分母/梯度一致、FP32/BF16合成更新、完整轮恢复、教师导出加载、P全部16拍、M宽度接入和旧无Z的GT/none路径。解析头仍仅用于辅助监督与诊断。

统一命令（复现本次选择与已知环境排除）：

```powershell
$env:TEMP = Join-Path (Get-Location) '.test-temp'
$env:TMP = $env:TEMP
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$taskTestFiles = @(rg --files tests | Where-Object { $_ -match 'test_(draft|pretrain|frozen|decoder|inference|split_drafter_consumers|time384_integration_review|maidata_entry|cont_sequence).*\.py$' })
python -m pytest @taskTestFiles --basetemp .test-temp/split-final --deselect tests/test_inference_cli.py::test_raw_audio_uses_training_features_and_partial_first_beat -m 'not cuda and not tooling and not integration' -q -o addopts=''
```

已知未通过项如实保留：

- `tests/test_decoder_training_performance.py::test_decoder_hot_path_has_no_dynamic_boolean_gathers_and_flags_run_once`断言没有`aten::nonzero`，但已有`decoder/condition_attention.py`的单/多key分支使用`nonzero`。未改动主M或该测试；在独立`git archive b4d6dfe`源码快照、隔离PYTHONPATH下复现同一失败（1 failed，4.90s），确认是实施前既有问题，不作本次性能结论。
- `tests/test_inference_cli.py::test_raw_audio_uses_training_features_and_partial_first_beat`曾实际运行失败：`acoustic18 → librosa → numba`导入报告Numba要求NumPy≤2.1，环境安装NumPy2.4。统一验收仅明确排除这一已知环境失败，未修改环境或音频算法。既有grid入口与FP32/BF16推理测试通过。

独立复核发现真实v3训练使用`binding={paths:...}`，旧导出器索引历史哈希字段会触发KeyError；现按v3路径provenance恢复教师，不重建旧资产哈希，仍严格检查架构和模型状态。测试使用不存在的已验收输入路径完成教师加载与新DP roundtrip，确认无隐含旧资产扫描。修复后执行：

```powershell
python -m pytest tests/test_split_drafter_consumers.py::test_export_builds_teacher_recipe_instead_of_consumer_recipe_and_roundtrips tests/test_decoder_training.py::test_producer_load_is_paired_strict_and_stage_two_resume_is_separate tests/test_decoder_supervision.py::test_encoder_draft_planner_to_real_decoder_gradients --basetemp .test-temp/split-producer-fix -q
```

配置差异核对仅包含本次批准的Z宽度、移除R attention/上下文配置及替换loss键；正式workers12、BF16/scratch、学习率、调度、完整组/顺序、batch边界和207proxy未改。`git diff --check`通过。仓库外Draft设计正本、训练说明和讨论/合同状态同步更新，不属于本独立源码Git提交的文件。

CUDA原batch、正式训练、真实新教师缓存、速度和重建/谱面质量尚未验证。本次没有上传或轮询/干预其他实验。

DP缓存升级`maidata.frozen-draft-planner.v2`，记录末层Z表示与宽度，P0格式保持v1。旧阶段一checkpoint不能恢复新完整训练；旧128维Z不能伪装新Z。既有实验继续使用绑定源码。原谱、音频、Physical、采样计划、GT plan和主M窗口缓存继续复用。

## 复核修复：共享音频grid的输入规格

独立复核发现，教师和M恢复不同Encoder配置后，在线推理仍只接收一份grid；教师每拍6个采样点、M每拍5个采样点的合法权重配对能通过加载，却在教师前向触发矩阵维度错误。当前单grid接口要求两侧`samples_per_beat`和`mel_bins`一致，现于checkpoint加载、网络构造之前明确报告不匹配字段及两侧值；直接调用`prepare_conditions`也使用同一规则。只比较配方中这两个输入维度，不扫描grid或旧资产，不将网络宽度、深度或音频余量重新绑成相同配置，也不自动插值或重建输入。该修复明确拒绝不兼容的单grid配对，未增加双grid推理功能。

Z开启和仅预测plan都执行该检查；Z关闭且GT/无plan时继续只加载M，不读取或约束未使用的教师。原有不同隐藏维度/上下文范围的FP32、CPU BF16正向路径继续通过。

定向CPU回归：**37 passed、1 deselected，7.81s**。新增4项参数化测试覆盖两个输入维度、Z开启/关闭的预测plan路径、加载前与直接前向前拒绝，以及无教师路径。唯一排除项仍为上述NumPy/Numba环境问题的原音频提取测试；本次未运行CUDA、训练或上传。

```powershell
$env:TEMP = Join-Path (Get-Location) '.test-temp'
$env:TMP = $env:TEMP
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -m pytest tests/test_split_drafter_consumers.py tests/test_inference_joint_encoder.py tests/test_inference.py tests/test_inference_plan_source.py tests/test_inference_cli.py --deselect tests/test_inference_cli.py::test_raw_audio_uses_training_features_and_partial_first_beat --basetemp .test-temp/shared-grid-config-fix -q -o addopts=''
```
