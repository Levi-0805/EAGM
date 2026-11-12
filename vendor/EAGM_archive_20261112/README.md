# EAGM 后续方法研究归档（2026-11-12）

本目录将 EAGM（Execution-Aware Gradient-consistent World Models）探索阶段的脚本、报告和运行记录与 CVPR 2027 诊断性论文的实验记录分开保存。**这些内容不属于当前投稿，未写入论文正文、补充材料或 LLM Usage Statement。**

## 结论状态

EAGM Stage 2 已触发预先设定的停止条件：在论文头号预算（3,000 updates、五个种子）下，方向监督模型 $M_u$+Dir 相对 $M_u$ 的最终回报差为 $-1.85\times10^{-6}$（按 test start 聚类 95% CI $[-5.01,+1.42]\times10^{-6}$，1/5 种子为正），未观察到可靠策略训练收益。按约定，当前投稿保留诊断性论文定位，EAGM 作为后续独立方法研究，不进入主稿。

1,000 updates、五种子时方向监督确有提升（$M_u$+Dir $-$ $M_u$ $=+1.867\times10^{-4}$，约 +8.7%，CI $[+1.414,+2.333]\times10^{-4}$，4/5 种子为正），但该收益在 3,000 updates 消失，且未缩小与 $M_c$ 的差距。完整分析与审计见 `report/EAGM_阶段1_2报告.md`。

## 目录

- `scripts/`：EAGM 预实验、Stage 2、3,000-update 对照、策略评估及分析脚本（6 个）。
- `report/`：阶段 1–2 报告。
- `runs/`：EAGM 全部运行记录的副本，共 7 个目录：
  - `eagm_pilot_smoke_20261112/`
  - `eagm_pilot_v2_s0_20261112/`
  - `eagm_pilot_full_s0_20261112/`
  - `eagm_stage2_20261112/`
  - `eagm_stage2_1k5_20261112/`
  - `eagm_mudir3k_20261112/`
  - `eagm_policy3k_20261112/`
- `MANIFEST.sha256`：本归档内全部文件的 SHA-256，共 44 条（不含清单自身），路径相对本目录。

## 说明

- 原始文件仍保留在 `本地仿真预实验/` 和 `本地仿真预实验/runs/` 的原位置，未移动、未修改；本目录是独立副本。
- 论文视觉实验（RGB aliasing、camera-conditioned state estimation）不属于 EAGM，其记录仍在 `runs/visual_aliasing_20261112_v2/` 和 `runs/visual_state_20261112/`，并已纳入论文哈希清单。
- 本归档不参与论文编译，也不被 `CVPR2027/qa/` 下的两个清单引用。
