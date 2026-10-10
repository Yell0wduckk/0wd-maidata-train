# Predicted／GT plan 训练对照

两次训练仅切换主Decoder的plan12来源，比较准确统计条件对谱面生成的作用。全曲Z始终来自同一份固定阶段一D/P缓存；E_M与M都从同seed重新随机初始化，各自建立新的optimizer、scheduler和输出目录。共享base config保留相同模型、数据计划、采样、历史dropout、预算、学习率与调度。

## 条件表示

| 模式 | 前8维 | 后4维 |
|---|---|---|
| `predicted`（默认） | 固定P的sigmoid预测 | 固定P的softmax类别概率 |
| `gt` | MusicPlanTargets的归一化原谱统计 | `single/chord/slide/touch` one-hot |

连续字段顺序为：

1. `note_count`：`min(1, log1p(count)/log1p(64))`。
2. `note_density`：`min(1, count/窗口秒数/16)`。
3. `chord_ratio`：多音起音组数／起音组数。
4. `slide_ratio`：slide数／音符数。
5. `last_position`：最后真实起音在目标窗内的相对位置。
6. `touch_ratio`：touch与touch_hold数／音符数。
7. `hold_ratio`：hold与touch_hold数／音符数。
8. `relative_intensity`：窗口workload相对全曲reference的比例乘0.5，再截到`[0,1]`。

类别按窗口内各起音组的配置频次选择，顺序为`single/chord/slide/touch`，规则复用`draft.plan_targets.MusicPlanTargets`。空窗及完整空尾窗仍有效，连续值全0，类别为`single`，不是NULL。GT统计不经sigmoid或softmax再次变换，也不作为Z的输入或监督目标。

## 一次离线准备

在既有完整base config中填入独立GT缓存目录，其他训练配方沿用选定值：

```json
{
  "decoder_train_plan_source": "predicted",
  "decoder_train_gt_plan_cache": "/path/to/gt-plan-cache"
}
```

显式运行一次小缓存生产：

```bash
export PYTHONPATH=src
python -m maidata.decoder_training.gt_plan_cache \
  --config /path/to/base.json --output /path/to/gt-plan-cache --workers 12
```

工具仅读取所选train/validation基础窗对应的原谱和源结束拍信息，每份谱复用一次MusicPlanTargets，覆盖空窗与额外完整尾窗；不计算音频、Physical或网络。输出`maidata.gt-plan12.v1` manifest和各audio_variant的`banks/<variant>.pt`，包含`window_ids`、float32 `[N,12]` plan及bool `[N]` valid。只验收本次新增缓存的落盘往返，不重建或扫描旧资产。

训练worker仅读取GT小bank，并按window_id选择；不解全谱或在线计算统计。`predicted`模式无需GT缓存路径；`gt`模式需要`decoder_train_gt_plan_cache`，也可使用CLI的`--gt-plan-cache PATH`覆盖。

## 单次或串行运行

```bash
python -m maidata.decoder_training --config /path/to/base.json \
  --plan-source predicted --output /path/to/predicted-run
python -m maidata.decoder_training --config /path/to/base.json \
  --plan-source gt --gt-plan-cache /path/to/gt-plan-cache --output /path/to/gt-run
```

串行launcher接同一base config和两个独立输出目录，先predicted、结束后再GT：

```bash
PYTHON=/path/to/venv/bin/python bash scripts/run-decoder-plan-comparison.sh \
  /path/to/base.json /path/to/predicted-run /path/to/gt-run /path/to/comparison-logs
```

GT缓存路径从base config读取。第四个日志目录可省略，默认为`<PREDICTED_OUTPUT>.plan-comparison-logs`；两份日志分别为`predicted.log`和`gt.log`，以追加方式保存。建议使用绝对路径；相对路径从源码仓库根解析。`PYTHON`省略时使用`python`，脚本自动设置本仓库`src`的PYTHONPATH。

脚本分别启动两个fresh Python进程，不传`--resume`，不复制第一项权重；同seed来自同一base config。predicted退出非0也继续执行GT，分别记录两项退出码；两项结束后，只有均成功时整体返回0，任一失败则整体返回非0。脚本只负责训练调用和日志，不生产缓存、不做CUDA检查、不上传或改变workers/预算/验证频率。启动此脚本会执行正式训练；本次交付仅做shell mock验证，未运行真实训练。

## 记录与比较

训练与validation都使用当前选定plan来源，GT validation仅作准确条件下的oracle诊断。v2 checkpoint recipe记录来源，输入记录保留对应缓存；完整轮恢复继续加载E_M/M、optimizer、scheduler与全部RNG，不新增plan来源身份门禁。该对照使用两个独立fresh运行，不把来源切换当作第一项的续训。

GT条件衡量的是准确谱面统计可带来的帮助；GT训练的checkpoint在自动推理时仍使用固定E_D/D/P的预测plan，不读取GT谱面。推理provenance分别记录`training_plan_source`和`inference_plan_source`，因此GT对照权重正常推理会记录前者为`gt`、后者为`predicted`。比较时明确生成所用plan来源、checkpoint、温度和seed，以实际自由生成谱面的听感及用户评谱为主；各自validation loss作为诊断，不能将两种条件信息不同的loss直接视为部署质量结论。
