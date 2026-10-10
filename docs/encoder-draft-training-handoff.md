# pilot320 Encoder＋Draft 训练交接

## 授权与范围

用户已授权新建 `gpt-6.1-sol` 会话，上传本次源码并启动pilot320阶段一，启动训练时停止当前远端Physical计算。新会话负责训练前真实CUDA batch检查。仅训练E/D/R/P，不训练主Decoder、不跑后续阶段、不重建旧资产。本任务使用独立仓库 `D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/src`，不要转到根AGENTS默认的maidata-baseline。使用交接消息指定提交，不复制共享目录的未提交工作。

目标为 `ssh -p 16079 root@connect.cqa1.seetacloud.com`。现有 `/usr/local/bin/python` CUDA Torch环境可用；执行前读取当前GPU、目标目录、磁盘和相关进程状态即可。不增加旧载荷全量哈希、重新提取或验收凭据。

## 固定输入和配置

源码归档包括指定提交的 `src/maidata`（含chart/assets）、Linux配置和训练文档，上传到独立 `/root/autodl-tmp/0wd-v3-encoder-draft-<commit>-20261010`，记录SOURCE_COMMIT。运行使用该目录的 `PYTHONPATH=src`，不依赖旧包、全局安装或冻结音频模型。

新增输入：`D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/artifacts/pilot320-encoder-draft-inputs-20261010`，包括两侧base-only计划、320份源文本、selection/split及upload-files.json；上传到同名Linux目录。只校验本次新增传输与源码归档，不重哈希引用的旧缓存。路径见 `configs/encoder-draft-pilot320-linux.json`，路径变更用显式JSON覆盖。

固定配方：288首307谱训练，32首36谱验证；全部基础窗各一次；R容量207；最多两组完整音频＋加权token上限45056，音频/row权重各1，累积1，workers12；E/D/R/P全部scratch；bf16，AdamW lr2e-4，weight decay0.01，clip1，五项loss权重1/1/1/1/0.25，seed20261010；warmup200、100轮调度、绝对50轮停止。100轮17272步，前50轮8647步，第1轮173批，验证固定19批。预算已计入音频窗口重叠和固定R query，不能改成只数GT有效行。

## 执行顺序

1. 上传固定源码与新增小输入包，校验本次传输。记录环境、提交、实际配置和输入路径。
2. 在独立源码根运行CUDA检查。入口选第1轮加权cost最大的真实batch，保留整组和12 workers，记录实际音频组、R窗数、E/D/R/P梯度与更新、峰值显存和耗时。不设性能或loss通过阈值。

```bash
export PYTHONPATH=src
/usr/local/bin/python -u -m maidata.training --config configs/encoder-draft-pilot320-linux.json --cuda-check
```

3. 真实实现错误可最小修复并复核受影响部分，固定修复源码再执行；不能隐式改预算、组数、累积、workers或精度。若OOM，将失败batch和显存证据留在本实验会话，等用户调整，不用小batch冒充通过。
4. 正式训练准备好后，重新识别 `/root/autodl-tmp/0wd-v3-physical-attribution-f372ec2-20261010/run-production.py` launcher及其 `maidata.chart_cache.rebuild_supports` 子树：源为 `chart-cache-original-v2-master-remaster-fields-20261010`，输出为 `training-physical-v2-original-master-remaster-20261010`。先停止会重启子任务的launcher，再温和停止已确认子树，确认退出。不能用旧PID或`pkill python`；保留全部Physical产物和进度。
5. 用独立后台launcher记录命令、PID、日志和exit状态后启动正式训练。输出目录必须独立；如已存在运行先核对是否同一请求，不覆盖、不重复启动。

```bash
/usr/local/bin/python -u -m maidata.training --config configs/encoder-draft-pilot320-linux.json
```

6. 在本实验会话核对实际命令、提交、配置、日志首个完整optimizer更新、GPU占用与进程树后说明启动成功，不能只凭PID。进度只在实验会话报告，不向主会话发送消息或要求确认。

## 完整轮续训

```bash
/usr/local/bin/python -u -m maidata.training --config configs/encoder-draft-pilot320-linux.json --resume /root/autodl-tmp/pilot320-encoder-draft-training-20261010/checkpoints/last.ckpt
```

保持绑定源码和计划，恢复所有状态，不宽松加载。仅完整轮保存last；best只在轮末验证改善，编号每5轮及末轮；验证每5轮及末轮，完整诊断仅末轮。中断最多回退一轮，不保存轮内checkpoint。完成绝对50轮停止，不自动执行Decoder、推理、Physical重启或其他实验。validation loss仅为诊断，谱面质量由用户实际评谱。
