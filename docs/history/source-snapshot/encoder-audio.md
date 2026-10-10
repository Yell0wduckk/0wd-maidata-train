> 历史参考，迁移于 2026-10-10。原路径：`docs/encoder-audio.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 冻结 full 原生音频的 encoder60 → A90 → B20

本实现基于 `e1a8afd`，完整训练配方与表示见 [current-runtime](../../current-runtime.md)。音频生产器留在仓库外；训练源码只消费其输出。现有旧 encoder60、A90、B 与 μ/P0 不作为本次学习权重，也不覆盖它们。

## 外部生产依赖

本地固定源：`D:/01YD/01code/codex/maimai-ai-charting/artifacts/local-inference/full-mmoe-decoders-five-20260929/source-3d0bc68`。本地 full A90：`D:/01YD/01code/codex/maimai-ai-charting/artifacts/remote-training/full-mmoe-decoders-migration-20260929/A-last.ckpt`，内部 epoch89、step27892。绑定 run `maidata-full-mmoe-decoders-a90-b50-3d0bc68-20260929`，源提交 `3d0bc6829bdb877be04abbc7bb05f07f516bc49a`。已有传输记录 SHA256 为 `ebe72d2c47d7629badd678e98138964423bb7f782618cdc6cfa5ce8a0183cb77`；日常使用不重新遍历旧资产。

外部包装：`D:/01YD/01code/codex/maimai-ai-charting/artifacts/native-audio-export-20261009/export_native_audio.py`。它在独立 Python 进程导入旧源码及旧 checkpoint loader，整曲模块每唯一音频一次，局部模块每唯一音频/窗口一次。新仓库的 `TeacherAudioWriter` 只定义数据交换格式。若已有 Mel 使用不同 affine 统计，包装显式反标准化后应用 checkpoint 原统计；不重新估统计，不重算 Mel，不接受非线性预处理合同变更。

```bash
python external/export_native_audio.py \
  --source-snapshot external/source-3d0bc68 --checkpoint external/A-last.ckpt \
  --route-artifact "$ROUTE" --cache-writer "$PWD/src/maimai_baseline/data/teacher_audio.py" \
  --requests "$DATA_PREP/export_requests.json" --output "$RUN/teacher-audio" --device cuda
```

原生缓存保存176×256的22拍 P0、528×256 C0、264×256 C1及 mask/绝对坐标/插值索引，identity/mirror 共享。P1/P2 不落盘。全曲 bank 是各窗口目标中心 `P0[32:160]` 的原生窗口拼接。新 A 时间候选 key/query/fusion/gate 从头训练，绝不导入旧候选头。

## 数据与入口

`prepare-encoder-audio-data --dataset-manifest ... --epoch-plan-cache ... --output ...` 不要求旧 μ/A90/P0；生成完整对象尾部、曲中空窗与结束后16拍空窗、窗口张量、原引用派生的 A4096/B16384 计划和外部导出 requests。Physical/Mel/已有几何载荷只引用或硬链接；新窗口只补索引和必要裁窗几何。已准备的兼容完整数据可用 `PREPARED_DATA` 直接复用；队列按清单读取实际计划路径，只补缺少的导出 requests。音频导出覆盖谱面条件末端与真实音频推理末端的并集；只扩音频缓存，不增加训练引用。新产物由实际 loader 一次验收，旧输入不额外扫描。

统一 `--teacher-audio-cache` 支持 encoder、A、B、context 生产及推理；路径可以在 dataset manifest 的 `inputs.teacher_audio_cache` 提供。`build-latent-cache` 接收新版 encoder60 checkpoint 或末轮 export（completed_epochs60），只编码干净 LN μ，包含有效空块；不会生产后验或噪声缓存。

`build-song-context` 对 native 模型只绑定原生拼接 P0；`build-decoder-context-cache` 只用冻结新 A 处理缓存 C0/C1，训练步无需音频网络。B 的 NLL 使用干净规范化 μ，Decoder 用同次预测样本。普通/流式共用缓存与 AR 会话，回滚复用已抽 latent。

新曲原始音频必须由外部旧脚本按同一 producer 合同先准备 native cache，再传 `infer --teacher-audio-cache ...`。缺少缓存明确报错；不回退当前 continuous 编码器。推理校验原生 producer/表示合同，训练数据清单身份不限制新曲。

## 实验交接

固定源码、外部包装、旧源码快照和 full A90 上传至 SSH16079 的独立目录；实验会话使用 `configs/run-encoder-audio-16079.sh`。默认路径：

- 外部依赖：`/root/autodl-tmp/main-new-encoder-audio-20261009/external`。
- 输出：`/root/autodl-tmp/main-new-encoder-audio-20261009/run`。
- 源码：独立提交快照，不能替换正在运行实验的绑定源码。

资源空闲后依次执行数据/计划 → 外部原生音频缓存 → E原batch CUDA/成本 → E60 → 干净μ → A原batch CUDA/成本 → A90 → 原生P0映射/新A candidate → B原逻辑batch CUDA/成本 → B20。worker12，三阶段完整轮 checkpoint，末轮诊断，无 validation，无自动续训。所有新上传文件只做本次传输完整性核对，新缓存生产后验收一次。

Encoder 末轮固定样本包含真实/置零/错配/扰动重建；B末轮固定样本包含 GT条件、prior默认采样和两温度为零，单列尾窗 EOS。它们是训练集证据，未设置新增质量或性能阈值。CUDA及训练进度/结果仅由“6.1sol 训练实验new”报告。

本次实现的 CPU 与独立复核证据见 encoder-audio-validation（源文档引用：`docs/encoder-audio-validation.md`，未导入）。
