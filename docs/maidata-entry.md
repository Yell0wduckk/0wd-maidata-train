# Maidata Encoder 入口与接口

`src/maidata` 已实现独立音频 Encoder：frontend、geometry、attention、blocks、pyramid、encoder 和 types。它消费准备好的 Mel、aux32、绝对拍坐标与有效位，输出 P0/P1/P2 三路局部表示。独立输入生产及真实音频适配见[全官谱音频输入缓存](encoder-audio-cache.md)。新Draft D/R/P及辅助loss已实现，见[Draft接口](draft.md)；Decoder网络及主/Physical/几何loss见[Decoder接口](decoder.md)，阶段一和阶段二分别由`maidata.training`与`maidata.decoder_training`接入；旧 `maimai_baseline` 保留作历史实现参考。

从独立仓库根目录运行：

```powershell
Set-Location D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/src
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m maidata --help
python -m maidata --show-config
python -m maidata
```

帮助和 `--show-config` 仅依赖标准库，后者显示入口顶部的统一配置（含18项Encoder配置及Draft配置）。默认命令延迟导入 Torch，成功构建 `AudioEncoder`，输出已构建组件和训练/推理入口并正常退出；不创建训练目录或产物。`--component`可单独选择draft、planner、reconstruction或decoder，详见[Draft接口](draft.md)。

Draft新增 `--z-stride-beats {1,2,4,8,16}`（统一配置 `draft_z_stride_beats`，默认1），阶段一训练入口同样支持，同步更新与冻结P0输入两种模式通用。坐标、逐拍监督及缓存衔接见[Draft拍间隔](draft.md#可配置的-z-拍间隔)。

## 配置与构建

- [entry.py](../src/maidata/entry.py) 顶部普通 `CONFIG` 是唯一人工参数位置，默认值及含义见[设计正本](../../0wd-v3-refactor-docs/new-audio-encoder.md#21-当前默认值)。构造器显式接收参数，不另设同名默认值。
- [model.py](../src/maidata/model.py) 的 `build_model(config)` 从统一配置选择Encoder字段，构建 `AudioEncoder`。模型宽度、heads、FFN、各stage深度、频率query、窗口和输出密度均由这份配置传入；Draft字段由其独立工厂消费。
- `entry.main` 负责帮助、配置展示和构建。阶段一配方、数据路径和训练/恢复由[训练入口](encoder-draft-training.md)维护；拆分D/R与checkpoint v3见[Draft](draft.md)。

容量变化不要求重算内容未变的 Mel/aux32；修改 Mel 频带数或采样密度时需提供匹配输入。只保留真实数学与张量约束，例如 `d_model/heads` 为偶数、三 stage 参数长度一致和输出网格长度可表示，不固定为默认宽度、层数或参数总数。

## 张量接口与按曲复用

完整字段、位置约定和 aux32 离线生产约定见[实施文档第4节](../../0wd-v3-refactor-docs/encoder-execution-plan.md#4-张量协议)及[第8节](../../0wd-v3-refactor-docs/encoder-execution-plan.md#8-aux32-离线输入约定)。其中 B 为不同曲目数，J 为批内最大拍数，W 为窗口数，R/M/d 来自配置，`C=A+T+Z`、`N=C×R`。布尔位统一 True=有效，坐标为绝对小数拍，索引为整数。

| 容器 | 字段与形状 |
|---|---|
| `SongInputs` | `mel_patches[B,J,R,M]`、`sample_valid[B,J,R]`、`beat_centers[B,J]`、`aux32_centers[B,J,32]` |
| `WindowInputs` | `mel[W,N,M]`、`aux32[W,N,32]`、`beats[W,N]`、`valid[W,N]`、`target_start_beats[W]`、`window_to_song[W]` |
| `EncoderOutput` | 仅 `P0/P1/P2`，每路 `ScaleMemory` 含 `values[W,C×r,d]`、`beats[W,C×r]`、`valid[W,C×r]`；r 来自对应 fpn_rates |

以下张量由输入生产方准备，示例展示调用方式，不读取音频文件或启动训练：

```python
from maidata.entry import CONFIG
from maidata.model import build_model
from maidata.audio.types import SongInputs, WindowInputs

encoder = build_model(CONFIG)
song_inputs = SongInputs(
    mel_patches=mel_patches,
    sample_valid=sample_valid,
    beat_centers=beat_centers,
    aux32_centers=aux32_centers,
)
window_inputs = WindowInputs(
    mel=window_mel,
    aux32=window_aux32,
    beats=window_beats,
    valid=window_valid,
    target_start_beats=target_start_beats,
    window_to_song=window_to_song,
)
memory = encoder.prepare_song(song_inputs)
output = encoder.encode_windows(memory, window_inputs)
p0_values = output.P0.values
# 单次组合调用：output = encoder(song_inputs, window_inputs)
```

`prepare_song` 对每曲计算一份 G0 及各非末块的 G0 K/V；`encode_windows` 按 window_to_song 路由，每窗单独初始化和更新 S。SongMemory 保留 autograd 图，训练中同次前向的多窗可共享；参数更新后重新 prepare，不永久缓存或 detach。最终块省去 G0 K/V 与无消费者的 S 更新；输出不含 C0/C1/C2、最终 S 或旧 fine_memories。

aux32 按全曲统一离线准备，网络只做融合。拍中心位于音频范围外时由生产方清零中心 aux32；部分有效 Mel 拍仍可生成 G0，整拍无效则输出零。不能用 GT 谱面长度推断音频有效位。

## CPU 合成验证

从仓库根使用现有 CPU 环境执行：

```powershell
New-Item -ItemType Directory -Force .test-temp | Out-Null
$env:TEMP = (Join-Path (Get-Location) '.test-temp')
$env:TMP = $env:TEMP
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
D:/Anaconda/python.exe -m pytest tests --basetemp .test-temp/encoder -q
```

小型 CPU 合成验证已通过，覆盖默认及非默认容量、前向/反向、mask、联合 attention 数学参考、同步读取、末块省算、按曲复用和坐标/FPN。BPM/相位测试仅验证已准备输入的逐位置消费，未运行真实时间映射或特征提取。最终命令结果以本轮交接为准，不把旧测试结果当作本仓库证据；该验证不包含 CUDA、真实音频资产、输入批量生产、训练或整模型生成。
