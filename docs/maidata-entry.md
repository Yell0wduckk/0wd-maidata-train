# Maidata 最小入口

新包 `src/maidata` 目前只有入口和空模型位置，架构仍在讨论。没有迁入 Encoder、Decoder、草稿、数据管线、训练循环或 checkpoint；旧 `maimai_baseline` 保留作现有实现参考。

从独立仓库根目录运行：

```powershell
Set-Location D:/01YD/01code/codex/maimai-ai-charting/repos-0wd-v3/src
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m maidata --help
python -m maidata --show-config
python -m maidata
```

前两条命令成功退出；`--show-config` 当前显示 `{}`。默认调用明确提示模型与训练尚未实现，返回非零状态，不创建输出目录或训练产物。新包仅依赖 Python 标准库。

## 修改与扩展

- [entry.py](../src/maidata/entry.py) 顶部的 `CONFIG = {}` 是唯一参数编辑位置。当前不预填未确定的架构或训练参数；后续只添加实现实际使用的参数。
- [model.py](../src/maidata/model.py) 的 `build_model(config)` 是模型构建位置，目前只抛出明确的 `NotImplementedError`，不约定模型基类或网络形态。
- `entry.main` 负责参数展示和调用模型构建；后续确定训练行为时再扩展，不预建空训练器、插件、数据接口或 checkpoint 框架。

协作与默认操作规则由 `AGENTS.md` 和维护文档约束，不转换为入口中的固定参数白名单、来源哈希锁或验收凭据。未来的必要检查放在实际输入读取、运算和恢复位置，按真实表示和状态解释错误。

本轮验证范围是三个入口命令的输出、退出状态与标准库独立启动；不代表模型、CUDA、训练或推理验证。
