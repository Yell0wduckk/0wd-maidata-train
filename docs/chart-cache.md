# 全官谱原谱全曲缓存

`maidata.chart_cache` 是独立离线编排和存储层，消费新 `maidata.chart`、
`maidata.time`、`maidata.physical`，通过 MiaCode 可执行程序解析源谱。
不导入旧模型包，不计算音频或神经特征，不生成镜像，不确定训练切窗配方。

## 生产命令

在本独立 Git 根运行（使用既有 `.venv-audio`）：

```powershell
$env:PYTHONPATH = 'src'
$env:TEMP = "$PWD/.test-temp"
$env:TMP = $env:TEMP
./.venv-audio/Scripts/python.exe -m maidata.chart_cache.produce --source ../../data/extracted --audio-cache ../artifacts/encoder-audio-cache-all-official-20261010 --parser ../../tools/MiaCode/build/Release/miacode_chart_ir.exe --geometry ../../repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/physical/path-catalog --slide-data ../../tools/MiaCode/assets/reference/slide_data.json --output ../artifacts/chart-cache-all-official-original-20261010 --workers 4
```

`--song-id`（可重复）、`--limit` 仅用于真实小样。生产范围从源库全部非空
`inote_*` 正文枚举，必须与音频 catalog 的当前 `songs[].charts` 集合对应；
不使用 `legacy_full_variants`、旧 full/pilot 选择、版本或难度过滤。
MiaCode `--difficulty` 支持数字 ID（1..7），此处传源字段后缀，避免固定 MAS。

完成状态按谱原子发布；同命令重启复用已有本任务状态。来源和生产代码身份
不匹配直接报错，不覆盖另一配方。执行异常标为 failed，下次同身份重试；
不可表示输入保留为 unrepresentable，不能把它计作成功。
输出在仓库外 artifacts，不提交大缓存。恢复不重复扫描旧音频、几何载荷。

## 存储和读取

`catalog.json` 保存完整源谱清单、版本、难度、原始文件/正文 SHA、正文头行号、
当前音频 variant/asset、时序、音频终点与几何来源。`views=['original']` 固定为
唯一视图，没有增强开关。ID429 通过 `audio_only` 记录，不创建虚构谱面。

每份成功谱为 `charts/<ID>/<difficulty>.json.gz`，schema 为
`maidata.full-chart-original.v2`：

- `tokens`：完整真实序列（首枚外部边界 MOMENT，绝对起点保存在start_tick），NA 保存 null，有效零保存 0。
- `metadata`：IR 头、完整 controls、来源及诊断；IR 的 180Hz tick 不作模型拍 tick。
- `token_map`：token_index、96tick绝对拍位、秒数、moment、owner、track、segment，
  并保留原始 note ID、IR位置与源正文行列/raw。ROUTE 逐段映射，duration 归整轨。
- `actions`：按完成位置保存真实 Physical 动作、owner/track、精确 duration_tick、
  连续链分配后的各段时间、路径接触和释放。所有已确定动作均保留。
- `decisions`：与 tokens 等长的基础错误、四类 Physical 原因、deferred 标志；
  `physical_categories`、`basic_errors` 是可读取计数。

```python
from maidata.chart_cache import load_chart
from maidata.chart.codec import EncodedChart, token_from_dict

record = load_chart(cache_root / 'charts' / '10070' / '5.json.gz')
chart = EncodedChart(tuple(token_from_dict(t) for t in record['tokens']), record['metadata'], record['start_tick'])
```

读取验证 schema、唯一视图、字段/语法、拍坐标、动作归属及段时序；不重算 Physical。
首次写入完成压缩读回、字段/映射检查和载荷 SHA/字节记录。

MiaCode maidata CLI 默认使用全局 `first`，而已验收音频采用适用的
`first_<difficulty>`；缓存编码前显式选用音频 catalog 的该难度源锚。
旧 IR 秒投影不是监督来源，时间统一由完整 BPM/源锚推导。

## GT 前缀与不可表示输入

`PhysicalState.accept_source(token)` 是显式 teacher-prefix 入口，语法仍严格。
它只在完整动作时计算真实四类原因和音频范围，提交动作即使违规。
pending hold/chain 保持 deferred；不存在占位单段、假 duration 或穷举范围。
正常 `query/accept` 的拒绝语义保持原样。flags 不参与四类规则。

源谱中未知 flag、非默认 head/等待、非96tick网格、绝对时长或 per_segment 等
按现有字段合同报告，不静默转换。`status/<ID>-<difficulty>.json` 记录原因；
能解析的失败谱保存在 `failures/*.ir.json.gz`，逐音符定位记录全部受影响音符，
不从成功谱中删音符。解析器错误保存 stderr、退出码和源正文位置。

`production-summary.json` 汇总实际处理范围、成功/不可表示/异常、字节、耗时、
Physical 计数和逐谱结果。这些是生产证据，不新增训练启动凭据。

## 针对性验证

```powershell
./.venv-audio/Scripts/python.exe -m pytest tests/test_chart_fields.py tests/test_chart_physical.py tests/test_chart_cache.py --basetemp=.test-temp/chart-cache
```

覆盖候选拒绝与GT保留的分离、语法失败原子性、deferred连续链、链/分支映射、
音频范围违规保留、存储读回和归属损坏拒绝。真实小样覆盖全部五类NOTE和WiFi路线、
多BPM、WiFi、并行分支及6段连续链。未运行 CUDA、训练或 Decoder。

## 最新设计迁移与训练 Physical 缓存

`python -m maidata.chart_cache.migrate` 读取已验收、已上传的原谱v1缓存；显式将旧END锚移入start_tick、将旧wifi NOTE转为slide/Route-70，然后重新生成v2动作与训练支持。不重算音频、几何，不生成镜像。不可表示源谱读取原有failure IR重试最新编码，仍不可表示的保留原因与IR，不能当成功谱。

每个训练产物schema为 `maidata.training-physical-original.v1`，同样使用 `charts/<ID>/<difficulty>.json.gz`。`stages`与真实目标位置逐一对应，首枚输入边界不预测，另保留末尾时间查询支持。所有支持严格来自目标前的GT前缀：kind对位置作存在性投影，pad使用teacher kind，route使用既有连续段/branch，duration使用完整已选轨道。flags不参与Physical；ROUTE的语法错误与基础合法性独立于四类Physical mask。源码只产生候选支持，模型概率q、−log(1−q)、梯度及训练循环不在本次范围。

时间支持保存绝对起音tick或duration整数tick的完整区间分区，按规则比较的交点计算，不逐tick穷举整曲，不假设单调safe_upper。Duration基础域由真实音频末端及written BPM确定；释放交点使用完整TimingMap的22tick逆映射。每段分别保存basic错误、四类位mask及deferred，允许合法支持多段不连续。`load_supports`正常读回检查schema、索引和区间覆盖；`time_decision(..., window_end=..., censored=...)`对窗外/尾部查询绕过真实NOTE完成性约束。4tick duration cell通过既有duration_cells适配，不缓存模型概率。窗口长度、stride、历史预算与数据计划均由后续消费者指定。

远端入口（原谱旧包已在该目录，输出使用新目录，不覆盖旧包）：

```bash
PYTHONPATH=src python3 -m maidata.chart_cache.migrate \
  --source /root/autodl-tmp/chart-cache-all-official-original-20261010 \
  --output /root/autodl-tmp/chart-cache-original-v2-20261010 \
  --physical-output /root/autodl-tmp/training-physical-original-20261010 \
  --geometry /root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911/physical/path-catalog \
  --slide-data /root/autodl-tmp/0wd-v3-physical-20261010/resources/slide_data.json --workers 12
```

这里12是独立离线生产进程数；训练DataLoader workers默认也保持12。新产物首次写入压缩读回、结构验收及SHA记录一次。状态绑定生产代码与输入身份，同身份续跑复用已有结果；不增加旧音频/几何验收凭据。
