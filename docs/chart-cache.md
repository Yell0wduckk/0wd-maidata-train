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
`maidata.full-chart-original.v1`：

- `tokens`：完整真实序列（一次全曲 END 起点），NA 保存 null，有效零保存 0。
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
chart = EncodedChart(tuple(token_from_dict(t) for t in record['tokens']), record['metadata'])
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
音频范围违规保留、存储读回和归属损坏拒绝。真实小样覆盖全部六类NOTE、
多BPM、WiFi、并行分支及6段连续链。未运行 CUDA、训练或 Decoder。
