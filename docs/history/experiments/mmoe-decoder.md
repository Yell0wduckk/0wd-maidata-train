> 历史参考，迁移于 2026-10-10。原路径：`docs/history/experiments/mmoe-decoder.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 四层 MMoE decoder 实验

从本地 main `2cc0974` 派生，分支 `experiment/mmoe-decoder`，工作目录 `repos/mmoe-decoder`。旧 audio256-query4 训练和产物保持原绑定源码。本实验没有复用旧 decoder 权重。

## 架构和兼容

四层均为256维/8头共享因果self-attention、音频cross-attention和latent cross-attention。每层共享各子层归一化及一个latent缩放参数；FFN替换为四个独立256→512→256专家，GELU及dropout沿用原配置。Time/Note各用Linear256→4和FP32 softmax，密集加权全部专家；gate零初始化，专家独立初始化，新增构造隔离RNG。无Top-k、负载均衡、熵正则或新增监督目标。

事件router保持独立128维；原Time/Note最终归一化、条件事件头和Note行内kind→位置/flag→route→duration读出不变。任务family仅选择gate和读出头，当前目标字段不进入router/decoder/gate特征。训练和推理沿用原family选择语义。

每层保留单份已选任务路径self-KV，候选切换恢复快照并重算受影响路径；音频/latent条件KV复用。接口与物化行语义不变。模型、checkpoint和父模型使用MMoE身份，版本兼容范围仅维护在 `src/maimai_baseline/runtime-versions.toml`；旧双decoder训练/推理权重拒绝，不增加来源SHA准入锁。

音频256维/动态4×64 query、谱面128维latent和GMM4保持原结构。普通无style模型参数16,712,975，比来源模型减少653,024；参数量减少不表示计算减少。四个512维FFN密集计算约为原单路1024维FFN的两倍；全模型实际成本由CUDA真实更新记录。

## 固定数据和配方

使用来源实验261首/280谱及既有排除记录、seed20260811，无验证集。每谱每轮一次，occurrence=0，identity/mirror-lr按2:1分配一个视图；复用原curriculum、歌曲顺序、音频组和batch边界，不采用272首方案。

配置为 `configs/pilot261-mmoe-decoder-a90.json`、`-ar-b50.json`、`-whole-song-50.json`。A配置保留通用配置格式中的历史阶段字段，实际队列只消费A及本次AR配置，不启动通用B35/readout/S1。

A从头初始化，4096行预算，停止90轮、scheduler horizon155轮，LR0.000565685424949238，LR warmup79步，Physical warmup988步。监督为0.65 Note + 0.15 Delta + 0.4 Router + 0.2 Conditional event；break预算及Physical合同不变。

队列只执行A90→后验/归一化→P0→song-context→B计划→GMM4 B50。所有学习缓存由本次完整 `A/last.ckpt` 生成；原始Mel、GT、几何和Physical输入继续复用，不改写旧生产者、不全量重验。B冻结全部新A，重新初始化原GMM4，绝对计划轮91–140，整曲16384行软预算，同歌不拆分；停止50轮、horizon取真实50轮更新总数，warmup50步，FP32 AdamW LR1e-4/WD0.01/clip1。

workers12。last每完整轮，编号每5轮及末轮，无best/validation；完整诊断仅显式末轮，额外记录每层/任务专家平均权重与gate熵，以有效任务行数汇总，不进入loss。异常不保存轮内checkpoint；恢复optimizer、scheduler、完整轮游标及全部RNG。

## Sol执行交接

使用SSH22857现有环境和已验收原始资产。在 `/root/autodl-tmp/maidata-pilot261-mmoe-decoder-a90-b50-<commit>-20260929` 下隔离源码、计划、缓存与输出。上传固定提交归档并核对本次传输；不重新扫描原始输入。路径参数从该聊天上一次audio256-query4启动记录读取，保持相同资产。

先用原计划真实batch执行本次批准的CUDA A前反向更新、mask/梯度/计算复用检查，B在新A和缓存就绪后检查冻结/更新范围及成本。记录batch规模、耗时、峰值显存，不增加独立benchmark或性能阈值；资格更新不得使正式A90跳过首batch或改变初始化RNG。

已提供 `scripts/launch-mmoe-a90-b50.sh`，固定资产路径来自上一实验实际launch.sh：原计划为 `/root/autodl-tmp/maidata-pilot261-l3-v3-a90-20260927/plans`。入口相对自身定位源码，自动设置PYTHONPATH及每worker计算线程为1；DataLoader仍12 workers。调用 `bash "$SRC/scripts/launch-mmoe-a90-b50.sh" "$RUN_ROOT"`，新计划写入RUN_ROOT/a-plan，训练及学习缓存写入RUN_ROOT/training。仅首次派生计划，续训复用。

下列等价展开示例中，SRC是本次源码，ORIGINAL_PLAN和其他缓存变量取上述launcher里的固定资产；RUN是隔离目录：

```bash
cd "$SRC"
python scripts/prepare-query6-plan.py --original-plan "$ORIGINAL_PLAN" --output "$RUN/plan"
python scripts/run-query6-ar.py \
  --dataset-root "$DATASET_ROOT" --epoch-plan-cache "$RUN/plan" \
  --physical-interval-cache "$PHYSICAL_CACHE" --song-chart-cache "$CHART_CACHE" \
  --song-audio-cache "$AUDIO_CACHE" --output "$RUN" --execute
```

通过现有 `scripts/experiment-queue.py append/run` 接入队列，保留其他jobs和运行。OOM、资源不足或真实合同不匹配在实验聊天报告，不自行修改workers/batch/数据/权重；失败A不进入依赖B。实现交接后主聊天不主动轮询。

B50后推理11231、11749、11739、11176、438，沿用上一实验MASTER/DX输入、音频、BPM与timing；字段seed20260811/temperature0，latent seed20260812，latent和mixture temperature均1，trace、Physical hard mask及MiaCode slide检查保持原设置。保留generation/materialization、rows/IR、日志、退出码、截断、Physical bypass及语义round-trip，下载并物化。质量由用户看谱评价，不以loss或专家利用率决定优劣。

## 验证记录

2026-09-29，Windows CPU，D:/Anaconda/python.exe：受影响测试去重共211项，207通过、4项CUDA测试因本机无CUDA跳过。组合包括MMoE/双任务/Physical回退、字段顺序及损失预算、A真实L3+Physical梯度、训练态A完整轮下一步恢复、B冻结与完整轮恢复、学习缓存、计划/队列、版本边界、音频初始化及mask、流式推理。MMoE诊断开启/关闭的loss和全部梯度一致。

分组执行记录：MMoE初始11通过；decoder/字段/预算回归56通过；运行与缓存组最初76通过及2项旧参数量断言失败，修正后这2项通过；恢复/音频组29通过、4跳过及1项旧参数量断言失败，修正后该项通过；最终Physical/诊断/边界组37通过（含重复的音频KV用例）。测试临时目录因Windows权限需以已批准的提升权限运行，未改变正式数据或训练配方。

原首层共享/硬分工专属测试由MMoE独立公式和行为测试替代。删除一项引用已删除 `abc_response_v10.py` 的历史诊断测试；保留现役router因果、实际Physical回退和新增末轮诊断测试。同步旧router系数断言到main已有的0.4预算。入口help、中央版本输出、launcher的bash -n、git diff --check通过。初始化额外复核2项通过，新增模块仅播种CPU默认generator，不重设CUDA RNG。

CUDA真实更新、正式训练和五首谱面由“6sol训练与实验”聊天执行，CPU fixture不代表CUDA或谱面验收。原始资产未扫描、未重建；未启动远端训练。
