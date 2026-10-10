# 新字段、时间与 Physical

当前交付仅为数据与规则层。没有 embedding、字段预测头、分布网络、Decoder、Router/MMoE、训练循环或推理调度。设计正本见 [new-decoder](../../0wd-v3-refactor-docs/new-decoder.md)。

## 模块与接口

| 模块 | 接口与职责 |
|---|---|
| `maidata.chart` | `Moment/Note/Route/HoldDuration/SlideDuration`、`Grammar`、`encode_chart/decode_chart`、JSON转换、镜像、窗口标签 |
| `maidata.time` | `TimingMap`、written-BPM持续时间、全局等待/释放、链段时间分配、duration标签cell |
| `maidata.physical` | `Geometry`读取既有几何、动作物化、四类规则、`PhysicalState`候选查询/提交/快照/恢复 |

三个模块不导入旧 `maimai_*` 包、Torch、音频Encoder或音频缓存生产。字段和时间模块不导入Physical；IR物化只消费传入几何对象的路径长度，不调用规则。Physical只接收已知字段、时间与几何，不读取隐藏状态或未来标签。

## 字段与编码

- 仅五种 token；没有 END。外部调用 `inject_boundary(Moment(delta), start_tick=a, history_present=..., window_end=b)` 注入边界；有历史时 delta 从最后真实起音计算，无历史或历史 dropout 时为0，绝对锚仍是a。边界角色由运行时指定，不从数值零推断。生成 MOMENT 必须前进至少1 tick，直接 NOTE 才将边界登记为真实起音。连续空窗与终端查询不更新最后真实起音。
- `Note.make(kind, pad, **flags)`将适用的未指定flags设为有效0，不适用字段保持None。按钮为A1..A8；Touch为A/B/D/E各8区加C。Python的`break_`在JSON中为`break`。
- NOTE.kind只有tap、hold、touch、touch_hold、slide；WiFi由ROUTE=70恢复。每条轨道首段允许全部71类；普通链的连续段排除70，WiFi路线后只接SLIDE_DURATION；完成duration后允许普通/WiFi并行分支。
- `Route(route_id, track_break)`沿用Route-71；0..69普通路线，70 WiFi。首段必须提供0/1，后续段必须None。新分支重新从head起点开始。
- `HoldDuration(ticks)`最小2；`SlideDuration(ticks)`最小1。0..1 tick源hold转换为short flag，恢复为1 tick。时间token不改变起音游标。
- 数据编码不设置链段或branch条数上限，也不进行Physical筛谱；WiFi在各自轨道内保持单段，允许参与并行分支组合。字段集合封闭，不存在OOV输出，非法输入报具体错误。
- `token_to_dict/token_from_dict`显式保存`type`与字段，NA为null，数字0保持有效。`mirror_tokens`对按钮、Touch和每个连续/分支路线分别变换，D/E与A/B使用各自镜像映射。

`encode_chart(ir)`读取`maimai.chart_ir.v1`有理拍字段，返回`EncodedChart(tokens, metadata, start_tick)`。不使用IR的180 Hz投影tick当作拍坐标。IR头元数据与控制事件保留；NOTE顺序保留，同一MOMENT组装一个each组，each从实际NOTE数派生。只接受star head、默认一拍等待、末尾总时长链；不可表示输入不静默改写。`decode_chart(encoded, geometry)`重建IR及180 Hz投影，连续链按MiaCode samples折线长度分配秒数，后续段起点从前段终点继承。IR内的逐段duration为派生投影，不作为独立速度监督。

`Grammar.accept(token)`返回完成动作的`Completion`或None，包含owner、moment、起音tick、track_index与完成位置token_index。它不代表完整模型的自回归解码器。

## 窗口与时间标签

`window_sequence(tokens, timing, start_tick=..., end_tick=..., audio_after_beats=..., audio_end_seconds=...)`消费完整真实序列，返回本窗`tokens`及独立`terminal`。窗口长度和后余量由调用方传入，不引入训练默认配置。

终端为`TimeTarget(delta_ms, absolute_tick, censored, closes_window)`：首枚窗外起音在`[b,U)`时保留精确标签；到U未观察到起音则censored=True，后续模型应监督尾部概率。恰好在b的起音归下一窗；恰好在U属于未观察尾部。U取窗尾加音频后余量，不以音频有效末端截短；额外空尾窗仍监督完整终端尾部。音频外不得生成真实 NOTE。

`Grammar.close_window(target)`或`PhysicalState.close_window(target)`消费终端目标，不把它加入真实谱面，不替换最后真实起音，不提交动作。未完成NOTE/轨道禁止关闭；快照支持回到关闭之前。精确、整数格点的窗外MOMENT也可经带`window_end`的语法关闭窗口。

`duration_cells(kind, lower, upper, timing=..., onset_tick=...)`返回4 tick桶的代表tick及积分边界：hold为毫秒，slide为tick；首尾按合法边界裁剪。`duration_target_bucket`使用half-up标签。原始整数时长保留，只有候选/标签层分桶；不实现概率头、CDF或loss。

## Physical 查询与状态

```python
from maidata.chart import Moment, Note
from maidata.time import TimingMap
from maidata.physical import Geometry, PhysicalState

geometry = Geometry.from_resources(path_catalog_root, slide_data_path)
state = PhysicalState(TimingMap([(0, 120)]), geometry, audio_end_seconds=120.)
state.inject_boundary(Moment(0), start_tick=0, window_end=16 * 96)
support = state.query_note_positions('tap')
saved = state.snapshot()
state.accept(Note.make('tap', 'A1'))
state.restore(saved)
```

`path_catalog_root`指向既有目录中的`paths.json`、`runtime-path-semantics.json`和`runtime-path-rules.json`；`slide_data_path`指向MiaCode原有`slide_data.json`。正常读取时检查实际结构，不重建、不改写或全量重哈希旧缓存。Route-71的小型静态映射作为包资源独立携带。

- `query(token)`返回`Decision(basic_errors, physical, deferred)`；只读，不推进语法或动作。kind、位置分别由`query_note_kinds()`与`query_note_positions(kind)`查询。
- `allowed`表示当前未发现禁令；deferred=True表示尚需后续字段，不保证任意duration都合法。未完成链不使用占位route或占位整链时间误剪候选。
- `query_duration_ticks(ticks)`接受明确、递增的候选tick，返回逐候选结果与可不连续的合法整数区间。只计算传入候选，不扫描音频、全库或穷举未来路线。
- `accept(token)`先检查再原子提交。短音符在NOTE完成时提交，长hold在duration完成时提交，每条轨道在SDUR完成时提交一次。失败抛出携带Decision的PhysicalError，状态保持原样。
- `snapshot/restore`包含pending链、owner/track编号、语法游标和活动动作及交接摘要。窗口切换不会清空长hold/轨道；按真实时间裁剪时先锁定已发生的一对一交接。

四类规则为multitouch、overlap、slide_too_fast、tap_on_slide，沿用Touch池、250ms重复限制、2 tick短按、22 tick释放、启动后70 tick保护与200ms路径经过保护。Touch重复只归责位置；flags不参与规则。短hold的1 tick音频范围检查单独返回`duration_outside_audio`，不冒充multitouch。

连续链共享一次head、启动等待、占手和末尾释放；快滑逐段使用实际运动时间与旧覆盖/豁免。内部接点不生成新head或等待，也不调用独立轨道handoff。并行分支保持独立轨道占手，最多一个附加分支；WiFi轨道按两手占用计算，包含WiFi的分支组合可编码；完整动作按原有Physical规则判断并保留违规GT。权重(1,1,4,1)与固定四类归约仅导出为后续loss常量，当前不计算模型损失或梯度。

训练候选通过独立 `AttributionState` 维护首次归责，包含Router类型选择；完整动作拒绝与训练新增归责分别读取。`PhysicalState.query_types()`提供类型支持。具体冲突证据、快照、缓存v2及仅重建支持的接口见[Physical首次归责](physical-attribution.md)。

## 验证

```powershell
$env:TEMP = Join-Path (Get-Location) '.test-temp'
$env:TMP = $env:TEMP
.\.venv-audio\Scripts\python.exe -m pytest tests/test_chart_fields.py tests/test_chart_physical.py --basetemp=.test-temp/fields-physical
# 显式读取少量既有几何资源，与保留的旧单段区间实现对照：
.\.venv-audio\Scripts\python.exe -m pytest tests/test_chart_physical.py -m integration --basetemp=.test-temp/fields-physical-real
```

覆盖五类NOTE与统一slide/WiFi路线、NA/零、IR往返、Route-71镜像、短音符、变BPM、4tick标签、终端尾部、长链/分支、flags等价、非单调duration支持、接点占用、交接裁剪和回退。仅这些有标记的对照测试导入旧包；生产模块不依赖旧包。未执行训练、CUDA或几何重建；全库缓存生产以实际产物摘要为准；旧MiaCode额外运行态、Decoder K/V和全模型回滚尚未接入。
