> 历史参考，迁移于 2026-10-10。原路径：`docs/history/experiments/decoder8-song-dense-plan.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# 八层 decoder 全曲 dense attention 对照

## 用户目标与交付

用户要求：从原八层 decoder worktree 出发，新开会话，直接将 decoder 的音频交叉注意力和掩码自注意力扩展到全曲，不做其他调整，作为对照组测量开销。本任务是实现与成本诊断，不启动 A90/B50 正式训练，不自动交接训练。

- 唯一源码、导入、测试、Git 目录：`D:/01YD/01code/codex/maimai-ai-charting/repos/decoder8-song-dense`。
- 分支：`experiment/decoder8-song-dense`。
- 起点：原 `repos/decoder8` 的干净提交 `45a860d8db9752d6c264c1db9392a2146ce74d54`，不是 sparse 分支。
- 主会话：`01a10311-739d-7642-8c1d-6cfe48ae64e1`。实现完成后回报一次准确提交、证据位置、验证结果、未完成与阻断项，供主会话独立复核。用户已授权本次委派及实现回报。
- 不修改 main、decoder8、song-sparse 或其他 worktree；原 sparse 实现任务继续独立进行。

先读本 worktree 的 AGENTS.md、README.md、docs/current-runtime.md；其中写死的原 checkout 路径不覆盖本任务明确指定的新目录。测试 TEMP、TMP、basetemp、日志、缓存、checkpoint 均放本 D 盘 worktree。可用共享 Python 解释器，但 PYTHONPATH 必须指向本 worktree，验证实际导入位置。

## 最小实现边界

1. 八层 decoder 的谱面自注意力读取当前 query 之前全部合法谱面输入行，保留严格 causal mask；音频交叉注意力读取同一音频变体全曲完整分辨率 P0，包括未来音频。使用标准 dense attention/SDPA，不加 top-k、摘要、池化降采样、三路 gate、linear attention 或自定义 CUDA/Triton。
2. 八层 Time/Note MMoE、encoder、128 维 latent、八层原注册 AR、宽度、头数、dropout、损失权重、优化器、精度、随机种子和数据配方保持原样。latent 交叉注意力继续读取所属原窗口四个 latent；delta candidate 保留原局部特征与查找语义。独立 router 的原有注意力范围保持原样，不附带扩展；报告这一边界，以及它与 sparse 方案的差异。
3. 必须做的整曲输入、坐标、mask、窗口监督映射与缓存接口适配属于本次范围；不要把 sparse 分支全部移植过来。优先整曲/分组批量执行，避免照搬逐窗口、逐 query 的 Python 调度。不得为了成本表现变更数学可见范围。
4. 绝对拍坐标和输入行序共同决定因果关系，同拍 MOMENT/NOTE/SLIDE_BRANCH 保留真实顺序。预测下一 MOMENT 时只使用已接受输入的 anchor，不能用目标未来坐标。历史来自 decoder 实际隐藏状态，保留梯度，不换成双向 encoder 表示、不 detach。
5. 按同谱面/视图/occurrence/音频变体组织全曲输入，去除重复历史副本，保留必要控制行、各窗口 TARGET_START 与原目标，SEGMENT_END 不新增跨窗口损失。历史用所属目标窗口本轮的 NOTE 重排；显式映射回原监督。字段、mask、权重、branch ownership、Physical source-row 和 KL 分母不变。
6. 当前可训练音频 encoder 编码所需完整窗口集合，同一音频窗口同一步只编码一次。中央目标 16 拍 P0 按绝对坐标拼 canonical 全曲 bank，保留真实有效 mask；原局部 halo 仍服务 delta candidate。每拍 P0 密度与原实现一致，不借旧冻结 P0、不跨 optimizer 更新缓存激活。
7. 全曲历史缓存跨窗口保留，窗口局部条件/latent 正确切换；回滚与 replay_last 撤销相应 K/V 和坐标/行序。截断丢弃行不得进入下一窗口，已接受末行应在所属窗口条件下入缓存。训练与增量推理可见集合一致。
8. 不增加模型参数或无消费者配置标签，不引入 checkpointing、编译、优化器替换或打包优化等额外调整来改善数字。若原配置 OOM，保留原始失败证据并报告边界；不要静默缩小 batch、截断历史、降低精度、启用重计算或调整 worker。任何另行探索须清楚分开，不作为本次默认实现。
9. 只做必要的兼容性更新，中央 TOML 使用与原八层及 sparse 版本区分的版本，建议 27.00；不迁移旧 checkpoint 作为正式续训，不恢复已退役源码 SHA 门禁。参数结构相同可在独立诊断中复制同一初始化张量用于公平比较，并明确不是续训。

## 验证

- 监督等价：真实多窗口、非零历史、NOTE 重排、branch、Physical 映射和全部统计分母。
- 固定音频/latent 后，改变未来谱面行及标签不影响此前输出；同拍行、跨窗口和 padding 都覆盖。
- 小规模 dense 参考对照输出、loss、梯度；确认所有远端有效 key 可见、未来谱面 key 不可见。
- 八层有效梯度及参数更新，远历史与全曲音频梯度不断；A prior 冻结且不执行，B 原有行为不改。
- dropout 关闭后 teacher forcing/逐行增量等价，跨窗口条件切换、快照回滚、replay_last 和截断后续接一致。
- 编码和 K/V 投影复用次数、有效行数/音频 token 数可核对。诊断关闭时不额外执行统计同步，诊断开启时统计覆盖整批，不能仅留最后窗口。
- 运行受影响测试，不重验未改旧资产。代码与证据提交，明确已通过、未执行、阻断项。

## 同原始 batch 的 CUDA 成本对照

SSH：`ssh -p 47604 root@connect.cqa1.seetacloud.com`；运行环境 `/opt/maimai-runtime/bin/python`。独立诊断根 `/root/autodl-tmp/maidata-decoder8-song-dense-20261003`，源码 `source-<完整提交>`。上传本次文件并只验传输完整性，不扫描/重哈希旧资产。开始前读进程和资源状态，不覆盖或停止任何其他会话任务；旧的“GPU 空闲”不是现在的保证。

原八层基线使用上述起点源码，模型实现来源为 `005cc6fee713ff03808dedb01dd39bbf3f200b6c`。可只读参考 sparse 对照的诊断脚本、原 batch 定位和旧资产路径：`repos/decoder8-song-sparse/docs/evidence/decoder8-song-sparse-implementation.md` 与其 CUDA JSON；不能将其模型实现导入本 worktree。稀疏会话可能同时使用 GPU，需要避免重叠测量；资源冲突时在本会话报告，不抢占。

1. 主要比较原局部八层与本全曲 dense，使用相同原始 A 首 batch、原最大 padding batch、相同目标/数据增强/posterior 随机样本与原始初始化张量。保持原 references、views、occurrence、batch 边界和监督总量；缺失全曲上下文只作为新增输入，不增加损失。
2. workers=12、原正式 BF16 autocast、完整 L3/Physical/KL、backward、梯度裁剪、AdamW 更新和 optimizer 状态。不能仅比较 attention forward 后声称训练成本。计时路径的完整诊断关闭，诊断开启的额外成本单独列出。
3. 预热后交替重复计时，报告每次原始结果、中位数/范围、峰值 allocated/reserved、目标行/实际上下文行/音频 token 数。首次准备与稳定更新分开，输入等待与模型成本分开。原配置 OOM 也是有效对照结果，记录阶段和已知内存状态。
4. CPU/CUDA profiler 分解编码、输入组织/索引、self/cross attention、L3/Physical、backward、optimizer，记录实际 SDPA kernel 和 backend。调用 SDPA 不等于使用 Flash；不把 CUDA event 时间称为纯 kernel 忙碌时间。
5. 可将已存在的 sparse 证据并列展示，但标清其不同执行组织、router 范围和 activation checkpointing；不能把三者所有差异归因于 attention 密度。无需改动或重新启动 sparse 任务。
6. 不调整最终训练计划、不启动正式长训、不新增性能通过阈值。完整更新成功则据实报告增幅；失败据实报告 OOM/资源/资产阻断，不以缩小后的结果替代。

## 完成回报

在本会话保留详细成本结果；向主会话发送一次实现完成通知，含准确提交、worktree、本计划与证据路径、测量摘要、实际作用范围、CPU/CUDA 已通过及未执行项。主会话之后复核关键点；不自行启动正式训练或向训练会话派发任务。
