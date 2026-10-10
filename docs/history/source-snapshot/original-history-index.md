> 历史参考，迁移于 2026-10-10。原路径：`docs/history/index.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

## 纯清理对照（2026-10-07）

本轮七份退役脚本和三个无效测试可从 `01f977d` 的同名路径恢复；`git show 01f977d:<路径>` 可只读查看。四个无消费者定义也保存在该提交。不要覆盖当前源码来执行历史实验。

删除脚本：

- `scripts/infer-decoder8-extra-five.py`
- `scripts/infer-decoder8-five.py`
- `scripts/infer-song-dense-five.py`
- `scripts/materialize-decoder8-extra-five.py`
- `scripts/materialize-decoder8-five.py`
- `scripts/materialize-song-dense-five.py`
- `scripts/run-decoder468-native-a5.py`

删除测试：

- `tests/test_abc451515_contract.py`
- `tests/test_full_structure_runner.py`
- `tests/test_full_no_validation.py`

# 历史恢复索引

## 原生 AR／六层 decoder 退役清理（2026-10-01）

本实验基于 `abeb2c6`。diffusion、ControlNet、读出器、S1 的专用模型、训练入口与测试可从该提交恢复；共享 AR、原始资产工具和有效 A fixture 保留在当前源码中。历史实验须使用原绑定源码，不将恢复文件覆盖 `repos/ar-decoder6`。可用 `git archive --format=tar --output=historical-abeb2c6.tar abeb2c6` 导出独立目录。

六层实验合同见 [原生 AR／六层 decoder](../experiments/ar-decoder6.md)，历史验证见 实现证据（源文档引用：`docs/evidence/ar-decoder6-implementation.md`，未导入）。当前 worktree 的八层合同见 [decoder8](../experiments/decoder8.md)。

## 更早清理

本轮清理以 `147922b` 为对照。旧报告、已完成讨论计划、退役 C/S0/旧 full 入口及仓库内源码归档均可从该提交恢复，不改写 Git 历史。

只读查看：

```sh
git show 147922b:docs/luna-training.md
git ls-tree -r --name-only 147922b
git show 147922b:src/maimai_baseline/training/style_abc.py
```

恢复到独立目录供历史实验使用：

```sh
git archive --format=tar --output=historical-147922b.tar 147922b
```

不要将恢复文件覆盖当前训练源码或改变旧实验绑定。更早节点：`c6683fd` 为行内预测改造前 main，`04a2f99` 为 Physical 准备优化前，`a76fc95` 为 router/逐轨/Touch 修复合入节点。具体文件是否存在以对应 Git tree 为准。

当前入口、规则与配方见 [README](../../../README.md)；本次实测和清理依据见 清理报告（源文档引用：`docs/evidence/runtime-cleanup-20260926.md`，未导入）。


## joint-rp 联合训练回滚（2026-09-27）

根据用户对生成结果与诊断证据的判断，当前代码已恢复到引入 joint-rp 前的 `7ffb504`，移除无谱面 encoder 的 R/P 联合训练入口、模型、数据路径、checkpoint 加载分支和专属测试，继续维护 A → B → 读出器 → S1。joint-rp 不再是当前训练或推理功能。

历史实验保留于 Git：`5da4958` 为初始实现，`f7ccf05` 为 pilot 音频路径修复，`cabb49d` 为条件快速路径/BF16 对照，`2ce7db8` 为恢复原 FP32 的联合训练实现。旧 checkpoint 的训练、诊断及推理须使用其实际绑定源码，不能用当前 A/B/S1 加载器或改写身份冒充兼容。

可用 `git archive --format=tar --output=historical-joint-rp-2ce7db8.tar 2ce7db8` 导出独立历史源码。远端源码、checkpoint、诊断及生成产物未在本次回滚中改动；未跟踪目录保留。

历史记录：初始验证（源文档引用：`docs/evidence/joint-rp-20260926.md`，未导入）、pilot 计时（源文档引用：`docs/evidence/joint-rp-pilot256-timing-20260926.md`，未导入）、执行优化交付（源文档引用：`docs/evidence/joint-rp-execution-20260926.md`，未导入）、N=1/N=4 实测（源文档引用：`docs/evidence/joint-rp-execution-N1-N4-20260926.md`，未导入）。这些报告中的入口和命令只适用于各自的历史源码。


本次本地验证：`src/`、`configs/`、`scripts/`、`tests/` 与 `7ffb504` 的 Git diff 为空；CLI 不再注册 joint-rp，普通推理 API 默认采样步数为 50。定向运行 master_experiment、full_no_validation、song_diffusion、song_checkpoint、row_field_intervals、dual_inference、inference_cache、style_boundaries 八组测试，结果 **165 passed、13 failed、1 skipped**。失败为旧测试接口未同步：1 项仍导入已不存在的 `scripts/abc_response_v10.py`，1 项访问已不存在的 `_common_cache`，11 项仍断言旧 16-slot 而当前为 4-slot。上述源码和测试均已核对与回滚目标一致；本次未另建历史 checkout 重跑，不将这些失败记为通过，也不在回滚中追加测试迁移。未启动远端 CUDA 检查或训练。

## main-new 端到端恢复（2026-10-08）

当前 worktree 从 `d044593` 保留 Physical 规则和连续全曲 P0，撤销公开 GT 节奏骨架，恢复 MOMENT 与 delta 预测。旧骨架脚本、模块和专属测试可从 `d044593` 同名路径恢复到独立历史目录；它们不适用于当前端到端权重。本次保留的全曲音频、Physical、冻结、推理回滚和完整轮恢复测试已迁为无公开骨架夹具，见 当前实现记录（源文档引用：`docs/evidence/main-new-end-to-end.md`，未导入）。
