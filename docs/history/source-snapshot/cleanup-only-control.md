> 历史参考，迁移于 2026-10-10。原路径：`docs/cleanup-only-control.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 纯清理对照：旧训练逻辑 A90

## 起点与范围

起点 `01f977db728aedbd61f152d9e39050cb872b8323`；分支 `maintenance/cleanup-only-control`；worktree `D:/01YD/01code/codex/maimai-ai-charting/repos/maidata-cleanup-only`。运行源码与 `6f533ad` 相同，包含 BF16 和 GT 多分支修复。仅删除四个无消费者定义、七份退役脚本和三份无效/重复测试，整理目录 README 与九份历史实验文档；删除清单见 [历史索引](../index.md)。

不采用任何现役清理版训练入口、校验、音频时间表、Physical、骨架格式化、初始化顺序或重复计算修复。保留版本35.02、全部配置、依赖、参数和执行逻辑；保留 a_parent、谱面 encoder、posterior、prior、SFT 和有效测试/fixture。main 与其他 worktree 不修改。

## 本地验证

只验证批准的删除及文档，并运行受影响 CPU 测试。临时文件全部在本 worktree `.test-temp`。不运行独立 CUDA、replay 或 profiler，不增加训练计时代码，不把验证报告变成启动依赖。实际验证：20项CPU测试通过、0失败/0跳过，耗时30.910秒。覆盖 frequency_pooling 全文件、gt_scaffold_bf16 全文件，以及 gt_scaffold 的初始化/RNG、完整轮恢复下一次更新和正式 fit 保存/恢复三项。证据 `.test-temp/pytest-control.xml`。

比对526个保留的非Markdown文件：除四个定义的精确删除外，与起点内容一致（仅统一CRLF/LF比较）；配置、版本、依赖、夹具及两份骨架源码不变。删除脚本/测试无剩余运行消费者。85份Markdown本地链接检查无缺失，所有保留tracked目录有README，`git diff --check`通过。证据 `.test-temp/scope-verification.json`。这些证据是本地一次检查，不是远端启动依赖。

未执行：CUDA诊断、单batch重放、profiler、全套历史测试、旧资产扫描；A90用时和远端运行结果由训练会话独立记录，CPU通过不代表性能已恢复。

## 正式运行

接收会话：6.1sol 清理版训练与 loss 对照（`01a1144c-68ff-74d3-932d-210eef664b56`）。目标 `ssh -p 47604 root@connect.cqa1.seetacloud.com`，独立根 `/root/autodl-tmp/maidata-cleanup-only-control-20261007`，固定源码 `source-<完整提交>`。

若当前训练还在运行，等下一完整轮 last checkpoint 保存完成再停止。若已进入下一轮，丢弃未完成更新，保留完整轮恢复点及停止原因；不保存半轮，不把主动停止写成正常 A90 完成。确认训练及 worker 退出、GPU可用后启动对照，不并行抢占。不修改旧运行源码。

从上传的固定源码目录直接执行 A-only Python 入口（外层 shell 会自动生成五首，因此不用）：

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -Sn 65536
/opt/maimai-runtime/bin/python -u scripts/run-gt-scaffold-a90.py /root/autodl-tmp/maidata-cleanup-only-control-20261007
```

新初始化 A90，seed20260811；父 `/root/autodl-tmp/maidata-decoder8-20261003/a-plan`、pilot261首/280谱、原引用/视图/occurrence/顺序/batch边界。A4096、BF16、workers12 persistent；LR0.000565685424949238、LR warmup79、Physical warmup988；stop90/horizon155。无validation/best；last每完整轮、编号每5轮和末轮，完整诊断仅90轮。不执行B、缓存生产或自动生成。

同服务器、同运行环境；仅核对本次上传，不重验旧资产。遇OOM或真实错误保留具体轮次/batch，不静默改预算、精度、workers、计划或源码。

## 用时观察

复用训练会话既有监测，不新增重复自动化。记录现有 train/save/epoch_seconds、updates、峰值显存；首轮和第90轮单列。先看前10轮，再按相同轮次比较21–35、41–58、61–75、76–89，覆盖长期变慢区间。优先对照同服务器旧6f533ad复跑及36aa002运行；缺失指标标未记录，worker并行时间不加到主进程墙钟。

完整跑完A90，不因前10轮变快就判定解决，不自动重启或扩展实验。进度、异常和结果仅在训练会话报告；本会话交接后不主动轮询。
