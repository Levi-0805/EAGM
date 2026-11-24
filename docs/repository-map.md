# EAGM 仓库导航

## 从哪里开始

- [`main`](https://github.com/Levi-0805/EAGM/tree/main) 是 GitHub 默认分支，保留早期环境审计、基线复现和初版实验。
- [`EAGM-2.0-full-validation`](https://github.com/Levi-0805/EAGM/tree/EAGM-2.0-full-validation) 是当前最完整的研究快照，包含 Stage 7–8.3 的证据和 EAGM 2.0 的计划、审计与验证。查看结果时以该分支对应报告为入口；分支名中的 full-validation 描述工作范围。
- [Stage 8.3 研究决策](https://github.com/Levi-0805/EAGM/blob/EAGM-2.0-full-validation/reports/stage8_3_research_decision.md) 记录联合验证门槛未通过。[EAGM 2.0 审计](https://github.com/Levi-0805/EAGM/tree/EAGM-2.0-full-validation/reports/eagm2_audit)与[研究计划](https://github.com/Levi-0805/EAGM/tree/EAGM-2.0-full-validation/reports/eagm2_plan)可分别查阅。

## 文件职责

| 路径 | 内容 |
| --- | --- |
| `models/`、`controllers/`、`training/` | 模型、控制器和训练代码 |
| `scripts/`、`configs/`、`tests/` | 运行入口、实验配置、测试 |
| `reports/`、`runs/` | 研究决策、报告、实验产物和复现记录 |
| `vendor/`、`third_party/`、`assets/` | 旧基线复现材料、外部依赖和资产 |
| `analysis/`、`paper_eagm/`、`artifacts/` | 后续分支上的分析脚本、独立论文草稿和存档数据 |

## 分支关系

截至本次盘点，下面各分支的提交均已包含在 `EAGM-2.0-full-validation` 的历史中。表格按相对 `main` 的领先提交数排序。

| 研究分支 | 领先 `main` |
| --- | ---: |
| [`stage3-protocol-repair`](https://github.com/Levi-0805/EAGM/tree/stage3-protocol-repair) | 3 |
| [`stage5-confirm-budget`](https://github.com/Levi-0805/EAGM/tree/stage5-confirm-budget) | 4 |
| [`stage6-exec-generalization`](https://github.com/Levi-0805/EAGM/tree/stage6-exec-generalization) | 8 |
| [`stage6c-path-aware-saturation`](https://github.com/Levi-0805/EAGM/tree/stage6c-path-aware-saturation) | 9 |
| [`stage7-cross-task-generalization`](https://github.com/Levi-0805/EAGM/tree/stage7-cross-task-generalization) | 11 |
| [`stage8-2-dynamics-jacobian-audit`](https://github.com/Levi-0805/EAGM/tree/stage8-2-dynamics-jacobian-audit) | 12 |
| [`EAGM-2.0-part0-part1-audit`](https://github.com/Levi-0805/EAGM/tree/EAGM-2.0-part0-part1-audit) | 16 |
| [`EAGM-2.0-research-plan`](https://github.com/Levi-0805/EAGM/tree/EAGM-2.0-research-plan) | 17 |
| [`EAGM-2.0-full-validation`](https://github.com/Levi-0805/EAGM/tree/EAGM-2.0-full-validation) | 22 |

继续实验时，从包含所需代码与协议的提交开工作分支，并在报告中记录基准提交、配置和产物路径。历史分支上的实验入口与报告可能被引用。归档前先核对这些引用，并为需要长期复跑的节点保留稳定的提交链接。
