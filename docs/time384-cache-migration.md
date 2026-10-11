# 独立 CONT 缓存生产与迁移

当前窗口为 `maidata.decoder-window-tensors.v5`，模型谱面序列为 `maidata.model-chart-sequence.v1`。精确原谱 v2、完整整数 Physical v2、Mel/aux32、几何及 GT plan12 保持原生产者身份并只读复用。

## 当前 v4 的低成本迁移

在独立源码仓库运行：

```powershell
$env:PYTHONPATH='src'
python -m maidata.decoder_training.migrate_cont --config ../artifacts/decoder-cont-token-20261011/migration-config.json --source-windows ../artifacts/time384-supervision-v4-fb5ef9b-20261010/windows-v4 --output ../artifacts/decoder-cont-token-20261011/windows-v5 --workers 16
```

本次仅覆盖当前 pilot320 train/validation 计划。每个 worker 设置一个 Torch 计算线程；16 是离线迁移 worker 数，不改变正式训练的12 workers。迁移按谱面分配任务，读取原谱一次，复用现成 v4 精确标签及完整可达支持，展开输入 token、重映射监督行、投影 GT 前缀 Physical。无需运行 Encoder、几何重建或全曲 Physical 计算。

每谱面另存模型序列及原谱 token 索引、累计时间。窗口 manifest 的 `logical_lengths` 保留旧组批长度，`lengths` 记录实际展开长度；训练用前者保持优化更新边界，用后者分割 M 计算块。源 token/字段及精确 GT 元数据保留；CONT 没有独立真实动作。

输出必须是独立目录。每个新增载荷在写入时执行一次存储 roundtrip 与 SHA-256，原子替换临时文件；谱面完成记录保存已验收产物。相同配置重跑只继续未完成谱面，不重新读取完成谱面的载荷。全部任务成功后才发布 `complete=true`。已有资产不递归重哈希。

## 从零编译

`export_windows.compile_window` 从精确原谱与完整 Physical 构造窗口支持，再调用 `sequence_cache.expand_window`；增量迁移也调用同一展开及投影函数。原谱和支持从零生产仍用既有 `chart_cache.produce` 与 `chart_cache.rebuild_supports`，已有 v2 不重复生产。

```powershell
python -m maidata.decoder_training.export_windows --config D:/runs/cache-config.json --output D:/cache/windows-v5-new --workers 12
```

配置沿用真实的 train/validation plan、chart、physical、audio、geometry、slide_data、route_distances 路径，以及原窗口参数。旧 v1/v2/v3 窗口不能恢复精确时间；使用 `migrate_windows` 从精确 v2 源重新编译，不能只修改 schema。当前完整 v4 采用上面的快速迁移入口。

## 时间与 Physical 语义

三类时间共用 1..384+CONT，每次 CONT 占主序列位置。输入保留完整长 duration 链，监督止于越窗 CONT；右删失保持集合 NLL，不伪造终点。主损失沿 GT 步骤求和、按原字段计数；Physical 在同字段有选择的受监督步骤平均，再按原字段计数。CONT 的类别处罚从可达尾部完成叶投影，关闭叶零罚。无需在 GPU 计算其他 CONT 分支。

GT plan12、采样与源资产均不改动。实际迁移源码快照、配置、运行结果及成本证据位于 `../artifacts/decoder-cont-token-20261011`。旧 FieldReader checkpoint 不迁移为可续训 checkpoint。
