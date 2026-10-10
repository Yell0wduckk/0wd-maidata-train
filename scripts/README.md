# scripts

数据准备、特征与缓存生产、训练入口包装、诊断及谱面物化等来源脚本。各脚本的输入和用途以其参数及实际消费者为准。

本目录按来源快照保留，启动脚本可能绑定原实验的远端目录、输入或输出。它们不是新工作区的默认运行命令，也不代表本次复制已安排远端执行；实际使用仍需相应外部工具与输入。

本副本排除依赖测试材料或临时实验产物的 `benchmark-inference-cache.py`、`diagnose-batches.py`、`compare-decoder8-initialization.py`。其余诊断脚本按来源保留，具体用途由实际调用流程决定。

仓库定位见 [README](../README.md)，当前范围见 [current-runtime](../docs/current-runtime.md)。
