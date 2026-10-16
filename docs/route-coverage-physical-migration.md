# ROUTE 首段覆盖与 Slide 启动占手归责迁移

本入口现同时合并 [Slide 一拍后启动占手修复](slide-launch-physical.md)。从零生产与两级迁移均覆盖两项规则；Hold／Slide 末判定区仍仅推理。

训练 `AttributionState` 现在与推理共享首段 `path_coverage` 查询。第一段的出现时间、启动时间和路径在 ROUTE 选定时已经确定；无论追加多少段或改变总 duration，都不能消除的覆盖冲突在该 ROUTE 候选处计罚。

见证仍使用完整动作的 `(track owner/index/tick, old track owner/index/tick, new segment 0, old segment index)`。GT 选择该 ROUTE 后记入首次归责集合，随后追加 ROUTE、SLIDE_DURATION 类型和 duration 数值保留 raw 冲突，但不重复处罚这条见证。其他动作或后续段产生的新冲突仍独立计罚。后续段的时间依赖总 duration，继续在完整轨道检查；不提前猜测 duration、不改变分支规则、L 路径豁免、EPS 边界或四类损失权重。

## 兼容与执行范围

- Physical 保持 `maidata.training-physical-original.v2`，窗口保持 `maidata.decoder-window-tensors.v5`。不拒绝旧 v2 Physical 或旧 v5 窗口，也不在训练启动时自动迁移。
- 新生产的 Physical 带描述字段 `route_coverage_attribution=first-route-path-coverage.v1`；窗口编译记录所用支持的该字段。它不是普通缓存读取的最低版本门槛。
- 使用旧窗口训练，仍使用其中的旧罚项；仅更新源码不会自动修正已经固化在窗口里的标签。完整使用新归责需要先迁移 Physical，再迁移对应训练窗口，最后由后续实验显式选择新目录。
- 原谱、精确 duration、动作归属、音频/Mel/aux32、P0/Z、GT plan、采样计划与 occurrence 不变。不得替换正在运行实验的绑定源码或输入。
- 本次只交付代码、CPU 合成测试和以下命令说明，未执行真实缓存迁移、上传、CUDA 或训练。

## 1. 迁移整谱 Physical

在仓库根设置 `PYTHONPATH=src`。以下路径均由实际环境填入，输出必须是独立的新目录：

```text
python -m maidata.chart_cache.migrate_route_coverage --fields <original-v2-fields> --source-physical <old-physical-v2> --output <new-physical-v2> --geometry <existing-path-catalog> --slide-data <existing-slide-data.json> --workers 12
```

可重复传 `--chart-id ID:difficulty` 只迁移所选成功谱面；不传则处理旧 Physical 成功清单。相同命令可以续跑，已完成谱面从各自 status 恢复。失败直接报错，已完成的新文件保留，源目录只读。

脚本从现有动作与语法前缀定位受影响的类型、NOTE 和 ROUTE，按真实 type→kind→pad/route 顺序比较归责，复用其他 stage。从首个受影响 stage 开始重新计算各字段的首次归责，直到包含下一个重置归责的真实 MOMENT；末组同时更新终端 stage。已有精确 duration 区间作为 raw 冲突来源复用，仅缺少的反事实查询按规则计算，不重建几何资产，也不逐 tick 穷举。迁移记录包含 `changed_target_indices`、重算和复用 stage 数。

新 catalog/summary 保留字段源身份并记录此次 Physical 生产者。来源 SHA 使用已有元数据；每个新 Physical 文件保存时验收一次。不会预扫描或重新计算旧载荷的哈希。原谱、时间和动作文件不改写。

从零生成 Physical 的 `chart_cache.training.build_supports` 和 `chart_cache.rebuild_supports` 已使用相同新规则，无需额外开关。

## 2. 迁移包含 Physical 的训练窗口

整谱 Physical 迁移完成后：

```text
python -m maidata.decoder_training.migrate_physical --config <existing-training-config.json> --source-windows <old-v5-windows> --physical-cache <new-physical-v2> --output <new-v5-windows> --workers 12
```

配置中的 `decoder_train_chart_cache`、几何路径、目标窗、音频后余量和谱面历史长度必须与原窗口一致。`--physical-cache` 仅覆盖本次离线迁移的 Physical 来源。入口只读取原窗口清单，不创建或重新排列训练计划。

可重复传 `--window-id ID`；不传则迁移旧清单的全部窗口。每谱完成后可续跑，最终 manifest 的 `complete=true` 才表示所选范围完成。若只迁移一部分，输出仅包含这一部分，不能直接替代需要全计划的缓存根。

窗口使用新 Physical 支持通过现有编译器产生候选罚项，逐窗对照原始值，只替换各组 `penalty` 张量和描述字段。token、GT、CONT 展开、坐标、基础 allowed、flags、loss 分母、历史及长度必须完全不变；具体不匹配报错，不扩大为其他标签迁移。原模型序列文件直接复制，保留已有大小/哈希，不重新生成或重哈希。

此入口专门迁移 v5 的 Physical。v1–v4 的结构/时间迁移仍使用 [原有工具](time384-cache-migration.md)，不能把旧格式重标为 v5。

从零生产窗口继续使用 `maidata.decoder_training.export_windows`；在配置中选择新 Physical 目录即可。选择旧 Physical 时仍允许按旧支持生产，不暗中改写原缓存。

## 验证

累计 82 项受影响 CPU 测试通过；两个 CLI 的 `--help` 可用，`git diff --check` 通过。

定向 CPU 测试覆盖 ROUTE 正向处罚梯度、具体见证消重、同类新增冲突、变 BPM、等号边界与路径豁免、拼接/并行轨道/WiFi、原归责快照、非单调时间支持、窗口边界/空尾窗，以及迁移结果与新鲜规则生产结果一致。

两个入口在合成目录中验证了完整产出、相同命令续跑、输出身份不匹配报错、原文件字节不变和旧格式正常读取。真实数据迁移成本、CUDA 和重新训练后的谱面质量尚未验证；本次不设置新性能或质量阈值。
