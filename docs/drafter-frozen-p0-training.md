# 拆分 Drafter：冻结主 Decoder ckpt45 的 Encoder，训练 50/100 轮

用户于2026-10-11指定：使用主 Decoder 阶段第45轮 checkpoint 的 Encoder 冻结产出 P0，训练新拆分式 D/R/P 至第50轮，调度周期100轮；本阶段结束后停止。训练会话为“主 Decoder 训练：关闭 Z、GT plan”。本页是新实验交接，不修改旧实验的绑定源码或续训合同。

## 两种训练输入

`python -m maidata.training` 支持统一CONFIG中的 `pretrain_audio_mode`，也可用 `--audio-mode` 显式覆盖：

- `joint`（原默认）：读取既有Mel/aux32，Encoder与D/R/P同步更新，全曲P0保留计算图。
- `cached`：`pretrain_p0_cache`或`--p0-cache`指定新P0目录。Dataset仅读取P0及原谱标签，不加载Mel，不运行Encoder；只更新D/R/P。Encoder从缓存携带的快照严格加载、冻结并保持eval，以便完整轮checkpoint携带自足E/D/R/P，后续教师导出和推理不依赖原P0目录或原阶段二权重路径。

缓存模式只从生产者恢复Encoder配置，不覆盖D/R/P配置。共享的目标拍窗须与既有计划一致。仍采用原完整音频组、顺序、207成本proxy和budget组批，不按缓存后变少的计算量扩大batch。AdamW只接收可训练参数；CUDA原batch检查分别记录E零调用/零梯度/零更新、D/R/P实际梯度及更新。

原谱、Mel/aux32、Physical、GT plan和采样计划直接复用，不迁移或重建。新缓存使用公开P0中心拼接语义：每个音频覆盖窗口仅取目标中心，padding保持无效；完整空尾窗有Draft Z，但没有新增真实P0。源E_M可能在主Decoder训练时读取局部P0；此导出按D所需的全曲音频覆盖逐窗执行同一个E_M，再拼接公开中心P0，不使用主Decoder隐藏或旧Draft Z。

## 缓存生产与来源

新入口 `python -m maidata.training.p0_cache` 只加载阶段二v4/CONT v2的`encoder.*`参数，严格按其Encoder配方构建，设为eval且requires_grad=False；不构建M或旧D/R/P。生产时对所选checkpoint记录一次SHA256、轮数、step、来源；对本次新增bank写出形状/坐标/有限值验证、文件哈希及首bank精确回读。旧音频输入正常读取，不额外全量扫描或重哈希。

输出含 `manifest.json`、`encoder.pt`和`banks/<audio_variant>.pt`。P0 bank延用`maidata.frozen-p0-bank.v1`；manifest增加Encoder配方和快照引用。缺快照的旧导出不能直接用于此训练模式。训练只核对这些已有生产者/manifest和计划元数据，按需读取bank，不重新计算输入哈希。模式、生产者身份、精度和manifest已记录的bank哈希进入完整轮恢复合同，不能将换E或换P0伪装成同一运行续训。目录搬迁不改变内容身份。

## 本次运行

SSH：`root@connect.cqa1.seetacloud.com:16079`；Python：`/usr/local/bin/python`。

指定ckpt45：

```text
/root/autodl-tmp/decoder-cont-v2-no-z-gt-20261011/training/checkpoints/epoch-045.ckpt
```

这是当前独立CONT、plan归一化、关闭Z、GT plan主M实验的第45轮，不使用旧`decoder-no-z-gt-20261011`或阶段一第45轮。训练会话先读取所选文件的schema/architecture/completed_epochs确认来源，不替换仍在运行的旧实验源码；若主M原第90轮任务尚未结束，先让它按原授权完成，再使用空闲资源。

新实验根：`/root/autodl-tmp/drafter-split-p0-m45-20261011`；新源码放`source`，P0放`p0`，训练放`training`。使用本次最终提交的独立归档并记录SOURCE_COMMIT，只校验本次上传文件。默认D8层256维、直接末层Z、96类MOMENT、不监督duration、全16拍Planner。D/R/P从零开始；lr2e-4、WD.01、clip1、warmup200、seed20261010、BF16、workers12、完整组上限2、cost budget45056、累积1保持。配置见[Linux配置](../configs/drafter-split-frozen-p0-m45-linux.json)。

```bash
export PYTHONPATH=src
/usr/local/bin/python -u -m maidata.training.p0_cache \
  --config configs/drafter-split-frozen-p0-m45-linux.json \
  --decoder-checkpoint /root/autodl-tmp/decoder-cont-v2-no-z-gt-20261011/training/checkpoints/epoch-045.ckpt \
  --output /root/autodl-tmp/drafter-split-p0-m45-20261011/p0

/usr/local/bin/python -u -m maidata.training \
  --config configs/drafter-split-frozen-p0-m45-linux.json --cuda-check

/usr/local/bin/python -u -m maidata.training \
  --config configs/drafter-split-frozen-p0-m45-linux.json
```

缓存只生产一次；成功后恢复或重启训练不重新导出。仅对新的缓存训练路径做一次既定真实原batch CUDA检查，保留全部组/预算/workers12，不复做无关E/M资格或添加性能、loss门槛。使用可靠后台launcher记录PID、日志、exit；首个实际optimizer更新、配置、GPU及进程证据确认启动。实验进度和问题只在训练会话报告，主会话交接后不主动轮询。

checkpoint只在完整轮保存；每5轮及末轮验证，完整诊断仅第50轮。中断从本实验最后完整轮恢复全部权重、optimizer、scheduler及RNG，最多回退一轮。不自动导出新Z/plan，不启动主Decoder、推理或其他对照。

## 本地验证边界

定向CPU回归 **81 passed，17.79s**，覆盖所有`test_pretrain*.py`、旧P0接口、新教师消费、独立E/M推理、采样计划和两个Decoder生产者/绑定用例。CPU FP32/BF16合成导出和读回、负拍/不等长bank的mask/padding、冻结E零前向及D/R/P梯度和更新、保留组批、完整轮恢复、完成checkpoint的教师加载均已执行。原joint同步更新回归保留。真实ckpt45导出、CUDA成本、50轮训练与谱面质量由训练会话实际运行后报告，CPU验证不代表已训练成功。
