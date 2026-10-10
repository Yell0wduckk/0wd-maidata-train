> 历史参考，迁移于 2026-10-10。原路径：`docs/plans/song-first-batching.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# A 阶段整首优先打包实验

实现与独立复核已完成，并已交接训练会话。实现提交 `0c7b137` 已合并到 main，当前维护目录为 `repos/maidata-baseline`，原 worktree 已退役；远端实验继续使用自己的绑定源码。实现验证详见 复核记录（源文档引用：`docs/evidence/song-first-batching-20260930.md`，未导入）。下文保留原批准计划，其中 worktree 路径为历史记录。

## 目标与隔离

从 main@42cafc7 创建独立 worktree `repos/song-first-batching`，分支 `experiment/song-first-batching`。本聊天负责计划与复核，新建 GPT-6.1-sol 会话负责实现。保留 main、其他实验和未跟踪文件。所有编辑与导入使用此 worktree；继承文档中指向 main checkout 的工作目录说明须改为此 worktree。

同时维护 pilot/full 入口，本次只训练 pilot：261 首／280 谱，A90→自身学习缓存→B50；full 不自动启动。
服务器：`ssh -p 47604 root@connect.cqa1.seetacloud.com`，资产路径沿用克隆前路径。

## 打包规则

仅改变 A 的 batch 边界，预算保持4096目标行；模型、损失、序列窗口和 B 整曲打包逻辑不变。

- 按 song_id 汇总该轮所有已选谱面窗口，多难度共同计入歌曲成本；成本沿用每个窗口 max(1,target_rows)。
- 保留原歌曲顺序、歌曲内部引用顺序、完整音频窗口组、seed20260811、occurrence=0 和 identity/mirror 视图分配。新策略标识不得改变原歌曲排序随机键。
- 歌曲总量≤4096：完整放入当前batch；空间不足则结束当前batch，再完整放入下一batch。多首完整歌曲能放下时合批。
- 歌曲总量>4096：先结束前一个batch，从空batch开始，按原顺序以完整音频窗口组拆分。末尾batch可继续容纳后续完整歌曲。
- 单个不可拆音频窗口组若超过4096，单独形成超预算batch并记录，不拆组、不丢样本。

2026-09-30通过新服务器已有profile/selection元数据只读统计：pilot261首/280谱，6658窗口，总347411行，单首中位1266、P95=2106、最大3113，261首均≤4096，最大音频组255行；full995首/1089谱，24454窗口，总1255383行，单首中位1173、P95=2264、最大4532，994首≤4096，最大音频组712行。所有歌曲≤16384。本次pilot每首歌每轮应恰好参与一次A optimizer batch。

B保持16384行整曲软预算，不引入latent序列分段。

## 计划、接口与兼容性

- pilot从原已验收A计划读取引用、顺序和视图，仅重算batch offsets；full用同一打包函数生成计划。输出新实验目录，不覆盖旧计划。
- 更新每轮batch数、审计和实验绑定；scheduler horizon更新步数按新计划求和。保留A90/155轮horizon、79步LR warmup、988步Physical warmup及原LR。
- 为新打包增加明确计划策略元数据，新实验入口要求匹配。保留CLI参数；pilot计划准备入口增加--dataset-root读取长度和音频分组元数据，启动脚本同步传入。
- 中央runtime-versions.toml升21.10，训练/推理保留21.00起同架构兼容，物化保留15.00起行语义兼容。不增加源码SHA门禁，不迁移权重；版本值只在中央文件维护。
- 旧checkpoint仍可按既有兼容规则推理，但不可伪装为新batch计划完整轮续训。新训练从头初始化；恢复匹配本次计划，恢复optimizer、scheduler、完整轮游标和全部RNG。
- B冻结本次A90，用其后验/归一化、P0、song-context缓存，重新初始化prior。B50、绝对计划轮91–140、自身50轮horizon/50步warmup不变。
- workers12，无验证集；last每完整轮、编号每5轮及末轮，完整诊断仅末轮。原始Mel、GT、几何、Physical资产复用。

## 验证与复核

- 独立参考：整首换batch、多首合批、恰好4096、超长歌曲拆分、末尾合批、超大音频组独立放置、空输入。
- 新旧计划逐轮引用展平完全一致：顺序、视图、occurrence、音频组完整性不变，仅batch边界变化。
- pilot每首每轮仅一个A batch；full可容纳歌曲不拆，超长歌曲连续且无重复遗漏。
- scheduler采用新计划步数；同计划完整轮恢复成功、跨计划续训拒绝；pilot/full入口、A父模型和学习缓存绑定正确。
- 新旧A计划经B整曲重打包后引用与batch边界相同；覆盖真实A更新、B冻结A/更新prior、缓存生产和相关恢复回归。
- 计划生成记录新旧batch数、填充率、歌曲参与更新次数、超预算组，不设性能或质量阈值。

主聊天独立审查差异、不变量、入口和恢复边界，需要时交回实现会话修正。CUDA交给训练会话：新计划真实A batch前反向、更新范围、音频/谱面条件复用，记录耗时/峰值显存，不改变正式初始化或跳过batch；B未改，不追加无关资格或独立benchmark。不扫描已验收旧载荷，不新增完成凭据或SHA门禁。

## 会话交接与训练

1. 新建GPT-6.1-sol实现会话，仅在本worktree实现、验证并提交；禁止自行上传或训练。完成提供提交、测试、未覆盖项。主聊天等待并独立复核。
2. 主聊天复核通过，交给既有“6.1sol训练与实验”`01a0dede-fc12-7850-8583-6127d5aa6a7d`：绝对路径、最终提交、启动命令、复核证据。
3. 训练会话上传到47604独立目录，仅核对本次传输，保留已有任务，执行pilot完整流程。资产缺失/合同不匹配/OOM在实验会话报告，不缩减batch/workers或替换数据。
4. B50后生成11231、11749、11739、11176、438，原难度/音频/BPM/时间；字段seed20260811/温度0，latent seed20260812，latent及mixture温度1。下载、物化，记录退出码、截断、Physical绕过和MiaCode语义重解析。
5. 质量由用户评价。训练交接后主聊天不主动轮询或转述训练进度，不自动合并/删除worktree。

## 现有入口与资产定位

继承pilot启动脚本 `scripts/launch-mmoe-decoders-a90-b50.sh`、full脚本 `scripts/launch-full-mmoe-decoders-a90-b50.sh`，需更新为新计划准备流程。pilot原计划：`/root/autodl-tmp/maidata-pilot261-l3-v3-a90-20260927/plans`；数据：`/root/autodl-tmp/maimai-full-dataset-lr-fixed-20260911`。其余输入路径见脚本，保留原值。

本地Python：D:/Anaconda/python.exe；Git Bash：E:/Program Files/Git/bin/bash.exe。测试临时目录使用D盘，PYTHONPATH明确指向本worktree/src。Windows Git按需使用单命令safe.directory，不改全局配置。应用worktree工具因外层非Git仓库返回Not a git repository，已按授权fallback创建本worktree。
