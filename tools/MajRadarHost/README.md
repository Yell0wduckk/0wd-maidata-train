# MajRadarHost

来源代码中可选的离线 MajRadar 标签生产工具，使用 .NET 9。stdin/stdout 按行交换 JSON；请求包含整谱 `inote`、`record_id`、`view` 及十六拍 `windows`，结果用于窗口级雷达缓存。训练步不运行这个 C# host。

本副本只包含 host 源码，不包含外部 MajRadar 项目或已构建 DLL。项目文件保留来源的相对 `ProjectReference`，新目录下未自动适配；构建时可显式传入 `-p:MajRadarProject=/path/to/MajRadar.csproj`。Python 的 `build-majradar-cache` 路径也可通过 `--host-dll` 使用已有构建。

来源的一拍音频草稿路线不使用 MajRadar/GMM prior；保留此工具不表示它是新联合音频 encoder 的训练依赖。

仓库定位见 [README](../../README.md)，当前范围见 [current-runtime](../../docs/current-runtime.md)。
