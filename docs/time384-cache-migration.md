# 精确 tick384 窗口编译与增量迁移

`maidata.decoder_training.export_windows.compile_window` 是从零生产和增量迁移共用的唯一窗口编译器。输入是 `maidata.full-chart-original.v2` 的精确完整字段、`maidata.training-physical-original.v2` 的完整整数区间，以及既有采样计划中的窗口。原谱、旧 Physical、Mel/aux32、几何、GT plan12 和旧窗口目录只读。旧 duration 的四 tick 标签不参与新 tick 的恢复。

输出监督 schema 为 `maidata.decoder-window-tensors.v4`，时间表示是 `relative-tick384-continue`。保存真实主序列、历史与边界、源 token 索引、精确完整 GT 字段、删失终端，以及 `[field, depth, 385]` 时间支持和标签。内部 CONT 不进入原谱、真实 self-KV、occurrence 或首次归责次数。旧 v1/v2 连续窗口及 v3 局部 cap 标签可沿用原 manifest 的窗口选择，从不变的原谱/Physical v2 重新编译监督；不能只改 schema 名称。完整源资产的版本和迁移不受该监督变更影响。

`lengths=[target_rows,history_rows]` 仍只计真实主序列。manifest 的 `field_query_cost`、运行 batch 的 `field_queries` 与 `Chunk.query_cost` 单独计内部查询。当前保持既有行预算、batch 边界、history dropout、split、window_id 和 occurrence；没有通过扩展 CONT 行来重新组批。

## 增量迁移

在本 Git 根运行，配置 JSON 沿用实际输入路径和已批准计划。输出必须是独立新目录。`--source-windows` 只读取旧小型 manifest，以核对窗口身份；不会读取旧四 tick tensor 载荷。重复 `--window-id` 可明确选择少量窗口；省略它则编译旧 manifest 中的全部实例，应只在获准执行整库迁移时使用。

```powershell
$env:PYTHONPATH = 'src'
python -m maidata.decoder_training.migrate_windows --config D:/runs/decoder-cache-config.json --source-windows D:/cache/decoder-windows-v2 --output D:/cache/decoder-windows-v4-selected --window-id 10146:5:0 --window-id 10146:5:1 --workers 12
```

配置必须包含下面全部实际路径；其余架构/窗口值从 `entry.CONFIG` 读取，也可显式填写同一原计划的值。模型/训练参数无需为缓存编译另选学习率、轮数或 checkpoint。输出路径由 `--output` 覆盖 `decoder_train_window_cache`。

```json
{
  "decoder_train_train_plan": "D:/plans/train",
  "decoder_train_validation_plan": "D:/plans/validation",
  "decoder_train_chart_cache": "D:/cache/charts-v2",
  "decoder_train_physical_cache": "D:/cache/physical-v2",
  "decoder_train_audio_cache": "D:/cache/audio-inputs",
  "decoder_train_geometry": "D:/cache/path-catalog",
  "decoder_train_slide_data": "D:/tools/slide_data.json",
  "decoder_train_route_distances": "D:/cache/route-artifact/route_geometry.npz",
  "target_beats": 16,
  "audio_after_beats": 4,
  "chart_history_beats": 4,
  "samples_per_beat": 24
}
```

原绑定消费者会读取两个计划的小型文件及各源缓存的 `catalog.json`、`production-summary.json`，并沿用几何和 route-distance 元数据哈希；`decoder_train_route_distances` 不能是 null。它不遍历既有音频/谱面/Physical载荷。实际本地路径已填入 `../artifacts/time384-local-query-verification-20261010/selected-export-config.json`，可以据此替换完整 Physical 根。

每个新窗口在原子发布前只验收本次存储 roundtrip，并记录该新文件的长度和 SHA-256。既有载荷不递归重哈希，不重提音频、不重估统计、不重建几何。生产者未变的 GT plan12 可以直接复用；E/D/P 权重改变后的神经 P0/Z/预测 plan 必须另行重新前向导出，格式迁移不能替代它。

## 从零生产

源谱首次生产仍先得到精确原谱 v2，再生成完整 Physical v2，最后调用相同的 `export_windows` 编译器。已有 v2 则直接从最后一步接入。下面的路径应替换为已明确选择的真实路径；各输出目录必须独立。

```powershell
python -m maidata.chart_cache.produce --source D:/data/extracted --audio-cache D:/cache/audio-inputs --parser D:/tools/miacode_chart_ir.exe --geometry D:/cache/path-catalog --slide-data D:/tools/slide_data.json --output D:/cache/charts-v2-new --workers 12
python -m maidata.chart_cache.rebuild_supports --source D:/cache/charts-v2-new --geometry D:/cache/path-catalog --slide-data D:/tools/slide_data.json --output D:/cache/physical-v2-new --workers 12
python -m maidata.decoder_training.export_windows --config D:/runs/decoder-cache-config.json --output D:/cache/decoder-windows-v4-new --workers 12
```

最后一步的配置须指向前两步产物和绑定这些精确原谱的采样计划。`chart_cache.produce` 同步写入实际 `producer_recipe`，供采样和编译绑定；不把消费者源码身份写成旧资产生产者。v1 来源先使用既有 `chart_cache.migrate` / `rebuild_fields` 的 v1→v2 语义迁移，再进入上述共同链路。`compact_windows` 明确保留为旧连续 v1/v2 的积分区间等义压缩，不支持本次 CE 迁移。

## duration 监督边界与完整源资产

原谱 v2 和 Physical v2 保存完整精确 duration 及支持；本次监督规则不改变这些源资产，不要求重新迁移、重算几何或删改实例。窗口编译复用完整支持与首次归责记录，保留完整 GT 动作。局部音频边界只控制时间头查询链，不再用它裁掉训练的完整 duration 候选或重投影 type/WiFi。

GT 监督包含跨到音频窗尾或窗外的最后一个 CONT 后停止；最后一个非 CONT 从窗内发出时，即使落点在窗外，也正常监督。例如相对边界为 500 tick：GT 700 监督 `CONT,316`，GT 769 监督 `CONT,CONT`，完整原值仍保留。边界由本地窗尾与真实音频尾取较早者，按 NOTE written BPM、Slide 启动时间和实际秒数计算，384.25 tick 的边界不会提前删掉 384 处查询。

编译器正常输出训练 labels，`metadata.duration_window_crossings` 只记录完整 GT 跨窗信息，不再代表拒训。CONT 终止叶保留概率质量且不新增 Physical 处罚，已完成短值叶仍读原完整区间。旧原谱/Physical 缓存无需重建；已有旧监督窗口则通过同一编译器更新标签，不能复用旧 v3 标签或 `labels=None` 作为新语义。

## 本次验证

CPU 定向测试覆盖 exact385 编码、缓存打包/读取、query 与主行成本分离、history dropout、超窗完整 GT 与监督前缀、窗外短值监督，以及 Physical 叶概率、FP32/BF16 梯度和变 BPM/小数音频尾边界。原缓存前缀复用与推理局部 cap 路径单独保留。下述三个真实窗口及 exporter CLI 记录是此前 v3 实现的证据，不冒称已执行新的整库迁移。

明确选择真实 `10146:5:0`、`10146:5:1`、空尾 `10146:5:22`，从既有原谱 v2 和完整 Physical v2 编译，与同一精确记录的独立直接规则编译逐张量相等；新产物存储 roundtrip 相等。真实主行/历史分别为 `[28,0]`、`[48,9]`、`[1,0]`，内部 query 成本为 `50`、`84`、`4`（NOTE 与 ROUTE 各一次，时间按完整合法 CONT 单链深度；pad/flags 复用 NOTE 查询）。证据与可复执行脚本位于 `../artifacts/time384-local-query-verification-20261010/selected-window-equivalence.json` 和 `verify_selected_windows.py`。

也实际尝试了下列三窗 exporter CLI：

```powershell
python -m maidata.decoder_training.export_windows --config ../artifacts/time384-local-query-verification-20261010/selected-export-config.json --output ../artifacts/time384-local-query-verification-20261010/selected-cli-export --window-id 10146:5:0 --window-id 10146:5:1 --window-id 10146:5:22 --workers 12
```

该 CLI 在原绑定步骤停止：本地 `training-physical-v2-original-remote-completed-20261010` 下载目录已有选定载荷、原 catalog 和 `download-summary.json`，缺原 `production-summary.json`。现有完整 pilot 目录保存 summary，但选定载荷位于下载目录；本次未伪造或合并旧元数据。实际 `migrate_windows` CLI 还需要本地未提供的真实旧 window manifest。因此目前已验证共同编译器和新产物，未声称完整 exporter/migrator CLI 已在本地资产上成功发布。提供含原 catalog、production summary 和载荷的完整 Physical 根即可重用上述命令；迁移另需真实 `--source-windows` 目录。

仅执行上述选定窗口和 CPU 针对性验证；未执行整库迁移、全曲特征导出、CUDA 原 batch 检查、正式训练或远端部署。训练默认 GT plan 与原谱对应，自由推理仍读 Planner 预测；训练 workers12、完整轮 checkpoint、每五轮及末轮 validation、末轮完整诊断及详细 trace 默认关闭均不因本工具改变。
