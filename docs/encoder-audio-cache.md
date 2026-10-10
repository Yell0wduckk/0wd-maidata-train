# 全官谱音频输入缓存

`maidata.preparation` 是离线输入模块，独立于网络与旧训练系统。生成的是音频、aux32、绝对拍坐标、PCM 中心及有效位；G0/S/L/P 仍在 Encoder 前向中计算。

本轮目录为 `D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/artifacts/encoder-audio-cache-all-official-20261010`。源清单为现有 `data/extracted` 的1804个官谱目录/ID、27个 version 值、7087份非空谱面正文；这是当前源库范围，不声称穷尽互联网最新目录。648个ID的version为FESTiVAL及以后。ID数量不能解释为互不重复的录音数量。

本轮生产已完成：1813份acoustic18基础缓存、1850份aux32时序缓存，零失败/缺失ID。直接引用旧full的1045个Mel变体，共9,251,293个有效位置，旧Mel覆盖缺口与坐标冲突均为0；新提取805个变体，共7,356,397个有效位置。768个新增ID对应787个变体，另18个当前变体来自既有ID中实际不同的BPM映射或音源。新NPZ载荷合计6,142,731,936字节（约5.72GiB）；4进程正式生产耗时551.30秒。旧WAV/Mel没有复制或重建。

## 增量资产与来源

旧 full 的1036个ID/1045个音频时序变体直接引用原始Mel及坐标，保留生产者路径、manifest和SHA元数据；另768个ID及确有差异的当前音源/时序位置补算。旧22拍窗口不会限制全曲缓存，新24拍窗口只是读取范围改变。pilot的均匀10ms Mel不作为本次直接FFT的来源。

当前官谱对应1804份WAV，另保留旧full所用9份历史WAV：505、544、550、562、575、576、600、604、605的 `before-master-e-trim`。`catalog.json` 的 `songs[].charts` 指向当前源的各难度时序，`legacy_full_variants` 明确指向历史音源，不能互换。总计1813份已归一化WAV/1850个时序变体。归一化WAV的已登记SHA在本清单中均不同，未据此宣称录音内容无重复。

ID429没有inote正文；保留其WAV，以声明的 `wholebpm=70` 和默认beat0锚生产音频输入，映射在 `audio_only_variant`，不把它计作有正文的训练谱面。旧full中837的original MP3元数据与归一化manifest记录不同，但实际消费的归一化WAV SHA一致；两份来源信息都保留，不改写生产者身份。

WAV均引用 `artifacts/audio-normalized-momentary-p50-20260905` 中的已验收文件，全曲固定增益P50目标为 -8.092632368636558 LUFS。float32读入、不再增益、不限幅、不覆盖MP3或WAV；幅值大于1保留。旧资产没有重哈希、再归一化或重新验收；为新增aux18而正常解码音频不等于重做音频验收。

## 配方与坐标

Mel由24kHz单声道PCM在 `b=k/R`（默认R24）处独立提取：完整BPM表/beat0音频锚映射到float64秒数，`floor(t*24000+0.5)` 得PCM中心，中心偏移[-512,511]，重复反射边界，periodic Hann，FFT1024、power、HTK三角Mel128（30..12000Hz，无面积归一化）、log1p、既有full的固定float32 mean/std。每个位置分析窗固定1024样本，不随拍长拉伸。

有效位同时要求真实秒数位于 `[0,WAV_duration)` 且量化PCM中心位于 `[0,N)`；真实静音有效。连续整数拍覆盖完整音频，包括负拍；首尾部分拍允许。L坐标是 `j+i/R`，G0/S坐标仍为 `j+0.5`，FPN目标单元中心由网络几何模块处理。

固定统计来自 `repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/audio/feature-stats.json`，SHA256为 `7718be2a059bb3e018519d71343c106513f1f270298c971a1468ba7b2ce30ddf`。不重新估计统计、不混入pilot统计。旧full复用按WAV/BPM/锚/R/统计元数据匹配，消费时核对grid_k、PCM中心、秒数和valid；只对真实缺口补算FFT。

aux32按18+4+10顺序（准确字段名见catalog.aux_keys）生产：

- acoustic18：22050Hz、FFT2048、hop441，Chroma12/onset/RMS/flux先按全曲median/IQR裁剪[-8,8]；relative RMS、1秒intensity slope和前后2秒Chroma novelty保留参考尺度。50Hz时间轴使用float64，不裁窗口，不做自动tempo检测。
- rhythm4：完整50Hz时间轴上按同一BPM映射计算onset的±0.25/±0.5拍、flux/RMS的±0.25拍均值。局部拍时长采用beat导数和BPM fallback；BPM跳变时独立查找每个邻域端点，避免单向指针漏算。
- context10：音频进度/时长、BPM、beat/bar4/phrase16相位及beat进度直接在目标拍位计算。周期和BPM不插值；音频之外全部置零。18+4维在音频内部首尾使用最近可用基础帧。

## 磁盘字段与读取

`catalog.json` 记录当前ID、别名、版本、难度→variant、WAV及旧Mel来源、真实时长、BPM/锚与配方。`coverage.json` 是元数据计数；`production-summary.json` 是本次实际生成结果，`production.log`/`production.jsonl` 是生产记录。它们不是额外的日常启动凭据。

| 容器 | 字段 | 布局 |
|---|---|---|
| base/asset.npz | times_sec, acoustic18 | [F] float64, [F,18] float32；仅音频相关 |
| grids/variant.npz | beat_centers, first_beat | [J] float32绝对拍中心，int64起始整数拍 |
| 同上 | grid_k, sample_times_sec, sample_indices, sample_valid | [J,R] int64/float64/int64/bool |
| 同上 | aux32_grid, aux32_centers, rhythm4 | [J,R,32]、[J,32]、[F,4] float32 |
| 新Mel变体 | mel_patches | [J,R,128] float32，无效位置清零 |
| 引用旧Mel变体 | mel_source_indices | [J,R] int64；旧producer帧索引，-1表示未引用 |
| 引用旧Mel变体 | mel_new_flat_indices, mel_new_values | [Q] int64、[Q,128] float32；仅实际缺口，Q可为0 |
| base/grid旁的JSON | fields、sha256、input_key、mel_reference等 | 本次产物说明、来源及必要缓存身份 |

旧Mel引用的path/SHA/producer variant存在grid旁的JSON，旧payload保持只读。`load_grid` 按引用和少量补丁恢复逻辑 `mel_patches`，不重新提取或重哈希旧音频。新文件写入时重读校验shape/dtype/有限值与round-trip并计算本次产物SHA；共享读取器默认保留离线检查，Encoder＋Draft训练显式使用`validate_input=False`，读取和batch组装均不校验输入。

```python
from maidata.preparation.cache import load_grid, song_inputs, window_inputs

grid = load_grid(cache_root / 'grids' / f'{variant_id}.npz')
songs = song_inputs([grid])
windows = window_inputs([grid], [0, 16], [0, 0],
                       target_beats=16, before_beats=4, after_beats=4)
output = encoder(songs, windows)
```

批内曲目按最大J补齐，mask明确区分padding与真实静音；窗口重叠处逐位置输入相同。读取接口由缓存实际R/M与窗口范围驱动。更改d/heads/FFN无需重做输入；更改BPM/锚/R才更新相关Mel中心和aux派生网格。参考音源基础acoustic18不依赖模型容量和拍网格。

## 生产与验证

安装独立输入依赖可用 `python -m pip install -e ".[audio-inputs]"`。本次使用仓库本地 `.venv-audio`、Torch2.11 CPU/librosa0.11/soundfile0.14；离线生产workers与训练DataLoader workers是不同参数。

```powershell
$env:PYTHONPATH = 'src'
python -m maidata.preparation.produce catalog --output ../artifacts/encoder-audio-cache-all-official-20261010 --source ../../data/extracted --wav-root ../../artifacts/audio-normalized-momentary-p50-20260905 --stats ../../repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/audio/feature-stats.json --full-manifest ../../repos/maidata-baseline/training-dataset/full-dataset/runtime-lr-repaired/audio/bpm-audio-manifest.json
python -m maidata.preparation.produce build --output ../artifacts/encoder-audio-cache-all-official-20261010 --workers 4
```

catalog只读取maidata及已有manifest，不扫描音频载荷；生产存在时不替换其清单。build支持同任务中断继续，复用已经写完且身份匹配的本次文件。默认全曲范围，不使用PRE_FESTIVAL、MASTER-only、pilot、旧full选择或GT谱面末尾裁剪。`--song-id` / `--limit`仅供小型真实输入测试，不是训练配方。

CPU验证命令：`python -m pytest --basetemp .test-temp/encoder-audio`。真实样本测试需显式设置 `MAIDATA_SAMPLE_CACHE` 为本次产物目录，然后执行 `python -m pytest tests/test_audio_preparation.py -m 'not tooling' --basetemp .test-temp/encoder-audio-real`。测试覆盖固定增益、字段/尺度、直接FFT对照、k/R相位、变BPM/负拍/首尾、旧Mel引用与缺口补算、跨窗一致、可配置R及Encoder前向。6个真实变体的31位置FFT抽样对照最大绝对差为0，默认输出192/96/48；分批FFT/GEMM的一般float32舍入差约1e-6，按数量级记录。未执行CUDA、训练、Decoder或checkpoint任务。

本次仓库CPU测试70项通过（真实集成测试默认不选）；显式真实样本命令15项通过。所有新增NPZ均在生产中完成一次保存后验证与哈希记录。最后只核对本次生产记录的asset/variant集合覆盖，不追加旧音频或旧缓存载荷扫描。准确源码commit与结果见产物目录中的 `REPORT.md` / `result.json`。
