# Slide 一拍后启动占手与 Physical 迁移

选择 Slide NOTE 时，星星头之外还检查全曲 timing 上出现位置＋96 tick 的
启动占手。所有合法 Slide 至少滑动一个 written tick，之后还有沿全局拍轴
换算的 22 tick 释放后缀；这段必然存在的区间在 NOTE 阶段即可判定。
等待的一拍不算持续占手，不读取未来 GT route 或 duration。

查询使用独立的占手资源，不查询占位路径几何。已知起点允许合法 incoming
接力；未知终点不能授予 outgoing 接力。普通分支按一只手、WiFi 按两只手，
沿用 Touch 池和启动共享规则。分支及 WiFi 的额外需求在 ROUTE 确定时检查。
完整 duration 仍负责检查超过最短区间的后续冲突。

推理与训练共享 capacity 见证，保持原来的四类分类与权重。kind/pad/route
按现有存在性投影和首次归责计罚；完整动作中的同一见证保留 raw 标记但不
重复处罚。其他动作或后续时间形成的新见证仍独立计罚。

例如 `{8}1-3[8:1]*-4-2-8-6-8-2-4-6-8<3[2:1],1-6-2[8:3],`，
第二个星星的一拍后启动与短分支的释放后缀重叠，此时长分支仍在滑动，
应在第二个 Slide 的位置候选处提前拦截。

## 推理预算

`multitouch` 始终禁止，包括 `physical_hard_mask=False`、预算兜底、CONT
候选投影和最终提交。其他 Physical 类别保留原有预算放行行为。
CONT 必须存在至少一个不多押的完整叶子才可进入，不能只在余数头过滤。

无法完成动作时恢复最近完整前缀，回滚文本/IR/KV/规则状态后推进合法
MOMENT；到输出边界正常结束。不会扩大 top-k/top-p 支持或伪造 duration。
报告保留 `physical_fallback`/`physical_bypass`，新增 mandatory 类别说明，
recoveries 记录回滚数量、phase、原因及 minimum_tick。基础输入错误仍报错。

## 合并迁移

从零生产使用相同规则。新支持带描述字段
`slide_launch_attribution=one-beat-slide-launch-capacity.v1`，不新增普通旧缓存
读取门槛。训练不会自动迁移已有标签。

现有入口同时迁移此前的 ROUTE 首段覆盖归责及本次启动占手：

```text
python -m maidata.chart_cache.migrate_route_coverage --fields <original-v2-fields> --source-physical <old-physical-v2> --output <new-physical-v2> --geometry <existing-path-catalog> --slide-data <existing-slide-data.json> --workers 12
python -m maidata.decoder_training.migrate_physical --config <existing-training-config.json> --source-windows <old-v5-windows> --physical-cache <new-physical-v2> --output <new-v5-windows> --workers 12
```

输出使用独立目录；两个入口均可续跑，规则身份参与输出绑定。迁移按
type→kind→pad/route 的真实归责顺序发现变化，从受影响 stage 重放至下一
MOMENT，复用语义未变的 raw duration 支持；不复用旧罚项冒充新归责。
窗口只替换 Physical penalty 与描述字段，保留全部模型/GT/CONT/基础合法
标签、顺序和分母。详见 [两级迁移合同](route-coverage-physical-migration.md)。

上次 Hold／Slide 末判定区规则仍仅用于推理，本次不加入训练标签。
本次交付代码与合成 CPU 验证，不执行真实资产迁移、CUDA 或训练。

验证：350 项受影响回归通过，3 项缺少外部原生解析器而跳过；6 项旧 CONT 窗口格式测试在未修改 HEAD 上同样失败，单独记录并排除，另有 2 项 tooling/integration 标记测试未选入。原片段通过本机实际几何复现了 NOTE、训练与完整 duration 三处多押判定。未执行真实缓存迁移或模型质量验收。

最终补充复核：26 项 CLI/规则/迁移测试通过（包含主回归中的重复用例），2 项 CLI 测试因旧参考几何路径不可用而跳过；实际几何复现已另行通过。
