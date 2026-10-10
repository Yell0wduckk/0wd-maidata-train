> 历史参考，迁移于 2026-10-10。原路径：`docs/main-cleanup-plan.md`。本文中的“当前”、配方、授权、提交和实测均属于原仓库当时的记录，不作为本独立副本的运行说明或验证结果。旧 Git 对象未导入；当前状态见 [current-runtime](../../current-runtime.md)。

# Main 清理与通用训练入口

## 起点和交付边界

从 `main@01f977db728aedbd61f152d9e39050cb872b8323` 建立 `maintenance/main-cleanup`，唯一工作目录为 `D:/01YD/01code/codex/maimai-ai-charting/repos/maidata-cleanup`。本次整理文档、死代码、退役测试和可读性，不修改 main、历史 worktree、原始/学习资产或远端运行，不上传或启动训练。交付代码提交、简洁变更记录及实际验证/未执行项。

## 已批准的修改

- 保留当前 GT 骨架模型行为、参数注册及固定种子初始化、监督、Physical 和回滚。谱面 encoder/posterior/prior 是暂时停用能力，连同有用途的缓存与测试保留；SFT 保留，不假称已完成当前骨架模型适配。
- 维护一个 `train --stage A|B|sft` 单阶段入口。数据根、固定计划、阶段配方、输出显式传入，资产路径可显式覆盖；复用既有 runner，不另建通用调度框架。pilot/full 不再是代码分支、固定歌曲数或配置白名单。
- 去掉只服务 readout/S1 的合同、旧实验包装和重复配置；阶段轮数、计划起点、horizon、学习率、warmup 和种子取自实际配方，不暗用全局 B50/A90 常量。默认 workers12，真实计划引用与 batch 边界不变。
- 大幅减少重复身份/配方/版本/报告验证。旧缓存和计划按实际数据读取，版本、producer、source/hash、旧实验名字保留为元数据，不是准入条件；不重建、改写或扫描旧资产。
- 保留真实结构、所需形状/mask/索引/有限值及完整轮模型、optimizer、scheduler、游标和 RNG 恢复。缺失必要数据就报告，不以放宽张量加载或丢弃状态掩盖错误。
- 按职责整理可读性差的代码，避免为组织目录改变公共行为或参数键。清理确无消费者的功能时同步处理其专用测试；保留独立参考、有效 fixture 和仍服务保留能力的测试。
- 九份旧实验说明移入 `docs/history/experiments`，历史证据原文保留，只补档案说明或修复链接；现役文档消除旧 A→缓存→B、自动 CUDA、固定 full/pilot 等错误描述。每个有 tracked 内容的目录补简短 README。

## 接口与兼容

`python -m maimai_baseline train --stage A|B|sft --dataset-root <目录> --experiment-config <配方> --epoch-plan-cache <计划> --output <输出>`。B/SFT 的父模型及缓存等输入继续显式提供。B 以完整 song-context 为输入时不要求独立 P0 sidecar/profile。旧 `fit`、`sft` 和底层 B 调用保留兼容入口；SFT 仍有旧目标未适配限制，本次不自动串联阶段或重新启动历史队列。

已有布局的数据集可更换歌曲、难度、子集及根路径；不承诺接收任意未经准备的文件格式。现有 plan 的引用、视图、occurrence、offsets 与顺序直接沿用。旧配方中未消费的历史阶段字段不驱动执行；不以整份字典必须相等阻止读取。

## 必要验证

仅运行受影响 CPU 测试和静态/文档检查，临时产物放本 D 盘 worktree。验证参数与输出不变、当前骨架监督与梯度/冻结/回滚、通用入口的路径覆盖和阶段派发、非固定数据子集与轮数、B 实际配方执行，以及完整轮保存恢复。对旧输入证明直接读取且不重建，计划引用和边界保持。

不执行训练、独立 CUDA 诊断、单 batch 重放、profiler、全历史测试或旧资产扫描。文档只检查受影响链接；真实结果据实报告，不设额外性能、loss 或报告凭据门槛。
