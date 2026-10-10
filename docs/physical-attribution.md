# Physical 候选与首次归责

本接口只实现数据、规则、候选支持与离线存储，不接模型概率、loss归约、训练或推理调度。

## 模块边界

- `physical.conflicts` 为四类规则生成具体冲突证据：规则名、动作owner、track、起音、pad，以及适用的segment/接触位置。完整动作的 `rules.reasons` 从同一套证据归约类别，不复制另一套判定。
- `PhysicalState` 保管语法、pending字段与完整动作。`query/accept` 保留完整候选拒绝，`accept_source` 保留违规GT；`query_types()` 提供已知前缀下的类型候选支持。
- `AttributionState` 在独立对象中维护已归责冲突。`stage_and_accept(token)` 按类型→kind→pad或类型→route/时间的顺序生成各阶段支持，再沿GT提交。flags不进入四类规则。
- `chart_cache.training` 保存、读取候选支持，不计算模型概率。字段编解码不依赖归责结果，也不据此删谱。

## 候选与归责

类型候选只依赖当前前缀，不按GT下一类型裁剪。NOTE类型、kind对后续合法内容作存在性投影；pad只条件化已选kind。Touch重复限制仍只在pad查询。普通ROUTE前缀可继续拼接，不能把暂时单段当最终路径；未知补全保持deferred，不枚举任意最大链长。

SLIDE_DURATION类型表示结束当前链，通过全部基础合法整数duration区间检查是否仍有合法完成方式。可用交点划分非单调区间，不逐tick穷举。WiFi只能单段，因此选择WiFi时即可检查其全部duration；普通分支后选WiFi的确定性占手违规归责route_id。已有两条轨道或已有WiFi轨道后再选分支，类型ROUTE已确定违反分支规则，归责类型。

存在合法补全时，前一阶段不计罚。所有补全都违规时，优先记录每种补全均包含的具体冲突；没有共同证据时，记录覆盖全部不合法补全的确定性证据集合。已有不可避免冲突不顺带领取某些较长duration才新增的冲突。该过程只确定候选责任，不改变四类权重或阶段loss归约。

同一冲突只在首次负责阶段计罚。消重使用具体证据，不按四类bitmask整体相减；较长时值与其他动作形成新的multitouch时仍计罚。占手证据使用最小过载资源集合，Touch仍为一个池，启动时的短按/轨道匹配保持原规则。GT历史自身已有的过载不会仅因候选时间与其重叠就成为候选的新责任。WiFi/超量分支限制与其必然造成的同owner轨道过载作为同一违规，外部动作参与的新过载独立保留。

`snapshot/restore` 同时保存Physical快照与归责集合。查询不推进状态；非法语法提交恢复整个快照。跨窗注入边界只重置前缀归责集合，活动动作仍由Physical按真实时间保留。窗口关闭MOMENT和截顶尾部没有NOTE完成性约束；历史dropout不清空活动动作。

## 缓存合同

训练支持schema为 `maidata.training-physical-original.v2`，与完整字段缓存 `maidata.full-chart-original.v2` 是不同合同。旧训练支持v1缺少类型与首次归责信息，不能改标签冒充v2。

每个stage保存全部语法允许类型的 `types`，再保存当前目标需要的kind、pad、route或时间区间支持。每个候选包含：

| 字段 | 含义 |
|---|---|
| `basic` | 独立的语法/基础范围错误 |
| `physical`、`conflicts` | 完整Physical类别mask与具体证据，用于完整检查/硬约束 |
| `penalty`、`new_conflicts` | 当前阶段新增归责的类别mask与证据，用于训练非法质量 |
| `deferred`、`penalty_deferred` | 两种查询各自的未确定标志；未知不当作非法 |
| stage的 `attribution` | GT在每个阶段实际选中的首次归责证据 |

`candidate_decision/type_decision/time_decision` 默认读取penalty；`penalty=False`读取完整Physical。`time_decision`仍接受当前 `window_end/censored`，窗外/尾部查询放行。缓存不固化目标stride、历史长度、dropout或网络容量。外部边界及重新切窗后的前缀由运行时适配，不能用旧位置的类型mask替代不同语法状态。

## 仅重建支持

已有验收通过的原谱v2字段时使用：

```text
python -m maidata.chart_cache.rebuild_supports --source <v2-fields-root> --output <new-physical-root> --geometry <existing-path-catalog> --slide-data <existing-slide-data.json> --workers 12
```

可重复指定 `--chart-id ID:difficulty` 只处理受影响谱面。默认只处理源清单中实际成功的原谱，保留源范围与隔离结果，不把不可表示谱计入成功训练索引。输入字段文件只读；来源SHA复用已有生产记录，新的Physical文件写入后按本次合同读回一次。输出记录字段原生产者与本次支持生产者，不重建音频或几何，不修改字段/owner/track/segment映射。运行命令是入口说明，不代表已启动重建。

验证覆盖类型归责、WiFi分支、链的继续/关闭选择、重复hold违规、新增同类冲突、既有GT过载、窗口终端、快照与回退、GT未来类型/时长独立、缓存读回及只重建支持时字段字节保持不变。
