> 历史参考，迁移于 2026-10-10。原路径：`docs/latent-ab.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# Encoder60 μ 缓存与 A90 → B50

分支从 `b97019b` 起步，默认 `full-song-frozen-mu128-v1`。实验会话通过 SSH16079 上传独立源码，资源空闲后依次执行 μ 缓存、A 原 batch CUDA 检查与成本记录、A90、新 A 的 P0 缓存、B 原 batch CUDA 检查与成本记录、B50。不打断已有实验，不自动续训。

encoder60 源为 `/root/autodl-tmp/main-new-encoder-pretrain-20261008/pretrain/checkpoints/epoch-060.ckpt`，内部必须 `epoch=59`。缺失、结构不符或不同轮次时报错，不替换为第90轮 `encoder.pt`。当前 SSH 连接关闭，现场文件确认待实验执行时完成。

以下 `RUN` 是新独立实验输出目录。`DATA`、`GT`、`A_PLAN`、`SONG_AUDIO`、`BPM`、`GEOMETRY`、`ROUTE` 复用已绑定旧资产。完整 A 调用还须保留现有空间视图、note-view、Physical path/source/interval 映射。

```bash
export PYTHONPATH="$PWD/src"
python -m maimai_baseline build-latent-cache \
  --dataset-root "$DATA" --song-chart-cache "$GT" --epoch-plan-cache "$A_PLAN" \
  --chart-encoder-checkpoint "$ENCODER60" --output "$RUN/mu" --device cuda
python -m maimai_baseline build-b-plan \
  --dataset-root "$DATA" --epoch-plan-cache "$A_PLAN" --output "$RUN/b-plan"
```

μ 按唯一完整谱面 identity/mirror 视图编码一次，保存最终 raw μ、块有效 mask、16拍窗口映射和 train-only 归一化。有效空块、尾块及完整跨窗对象保留，padding 排除。不保存投影前特征、logvar 或重建头。生产后一次核对本次覆盖、状态、有限值、写回与 loader 读取，不重验旧资产。

B 计划逐轮保留原 flattened references，仅调整 batch offsets。源计划有多少轮就生成多少轮，不延长；正式 B 使用 offset90 后50轮。

```bash
python -m maimai_baseline train --stage A --experiment-config configs/train-a.json \
  --dataset-root "$DATA" --song-chart-cache "$GT" --song-audio-cache "$SONG_AUDIO" \
  --epoch-plan-cache "$A_PLAN" --latent-cache "$RUN/mu" \
  --chart-encoder-checkpoint "$ENCODER60" --output "$RUN/A-cuda" --cuda-check
# 同样输入，加既有 A 的显式资产路径，去掉 --cuda-check，输出改为 $RUN/A。
```

检查用一次更新不作为正式训练状态，A 从新初始化正式训练90轮。

```bash
python -m maimai_baseline build-song-context \
  --stage-a-checkpoint "$RUN/A/last.ckpt" --latent-cache "$RUN/mu" \
  --song-chart-cache "$GT" --song-audio-cache "$SONG_AUDIO" \
  --bpm-audio-manifest "$BPM" --interpolation-geometry "$GEOMETRY" \
  --route-artifact "$ROUTE" --output "$RUN/p0" --device cuda
python -m maimai_baseline train --stage B --experiment-config configs/train-b.json \
  --stage-a-checkpoint "$RUN/A/last.ckpt" --latent-cache "$RUN/mu" \
  --song-context-cache "$RUN/p0" --epoch-plan-cache "$RUN/b-plan" \
  --route-artifact "$ROUTE" --output "$RUN/B-cuda" --cuda-check
# 同样输入，去掉 --cuda-check，输出改为 $RUN/B，正式训练50轮。
```

P0 从本次冻结 A 的连续全曲音频图生成，唯一音频一次；保存最终256宽 P0、mask、绝对拍坐标和谱面映射。新 P0 完成后一次核对覆盖、状态、有限值及 B loader 读取。不沿用旧 A learned P0 或窗口拼接产物。

完成 B 后推理反归一化 latent，每窗四个 token，普通与流式共用曲目 P0、K/V、RNG，回滚不重抽。不施加 Radar [0,1] 裁剪，默认不传 `--trace`。
