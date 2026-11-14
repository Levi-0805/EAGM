# PART 0 — 工作区盘点与审计范围

日期：2026-11-14。本轮授权范围：PART 0 与 PART 1；交付后推送新的 GitHub 分支。没有启动 PART 2–9，没有实现新方法或运行训练。

## 版本与路径

- 用户指定 `e:\CVPR论文\CVPR\11月论文\CVPR2027_EAGM_Final\` 与 `e:\CVPR论文\EAGM-branches\stage8-2-dynamics-jacobian-audit\`，当前 Linux 无对应挂载；不能声称读过这些目录。
- 实际仓库：`/home/aaa/桌面/汇报/11月论文/eagm_gpu`。
- 只读审计基准：`9416b92190b5d4d97232a1850eb27ee83953fa39`，原分支 `stage8-3-optimizer-data-alignment`，开始时工作树干净。
- GitHub：`https://github.com/Levi-0805/EAGM`。远端历史分支 `stage8-2-dynamics-jacobian-audit` 为 `348d0b262f6a9b445211e00051e4e024619c3881`，远端 main 为 `2fd27a8b6fd6abbcce4f8eefc72739191d80b2d7`。
- 交付工作树：`/home/aaa/workspace/EAGM-2.0-part0-part1-audit`；新分支 `EAGM-2.0-part0-part1-audit`。这只是审计交付分支，不是获批的第四阶段实验实现分支。
- 新分支继承本地 Stage8.3 已有提交和研究稿；相较远端 Stage8.2 包含此前未推送的 Stage8.3 历史。**本轮新增变更仅在 `reports/eagm2_audit/`。** 不覆盖 main、历史分支或原始实验。

## 资料完整性

| 请求资料 | 实际状态 | 本轮处理 |
|---|---|---|
| Final main.pdf 与全部 LaTeX | 指定版本不可得；存在 `paper_eagm/main.pdf`, `main.tex`, 八个 sections 与 tables | 审计现有 Stage8.3 研究稿；不得外推到 Final |
| 正确 supplementary.pdf 与源文件 | `paper_eagm/` 内不存在 | 标为缺失，无法核查主补一致性 |
| experiment_evidence_ledger.md | 未找到同名文件 | 参考 `reports/stage8_evidence_ledger.md`，非同一文件 |
| claim_evidence_mapping.md | 未找到同名文件 | 参考 `paper_eagm/claims_ledger.md`，非同一文件 |
| formula_code_audit.md | 未找到 | 直接对照实际代码和论文，不能称复核该审计 |
| research_audit.md | 未找到 | 参考 `reports/stage8_historical_audit.md`，非同一文件 |
| reviewer_self_assessment.md | 未找到 | 本轮独立审稿，不假设原评价 |
| method_pivot_change_log.md | 未找到 | 参考历史 decision/protocol，不替代指定文件 |
| final_submission_checklist.md | 未找到 | 不能认定提交完整性 |
| 原始 JSON、配置、训练/控制代码 | 本地大量存在 | 全部 Git 跟踪文件建立哈希清单；按关键主张针对性阅读，未逐行语义审计所有历史文件 |

旁目录 `/home/aaa/桌面/汇报/11月论文/CVPR2027/` 有 main 与 supplementary PDF/TeX，但其内容属于原有 Execution Memory 诊断论文，不能当作 EAGM Final 的正确补充材料。任务书提醒的 legacy supplement 也没有被拿来替代。已异步询问作者指定 Final 的可访问位置，未收到位置前以当前版本作有范围限制的审计。

## 证据保全与检查

- `00_source_inventory.csv`：684 个 Git 跟踪文件的路径、大小、SHA-256；包含代码、配置、论文、实验和归档分片。哈希覆盖不等于每个文件均完成科学复核。
- `00_provenance_checks.json`：论文 source_manifest 的 43 项源文件哈希全部匹配。只验证对应关系与文件完整性，不证明统计或方法正确。
- `00_reviewed_main_pdf.txt`：对现有 main.pdf 用 pdftotext 提取，供 PDF/TeX 内容交叉检查；没有重新编译或改写 PDF。
- 原始种子、失败实验、数据与配置保持原样；未重新训练，未添加确认性种子，未改变确认性/探索性标识。
- Reviewer A/B/C 为独立上下文的审稿角色，各自从同一冻结版本出发。它们是同一系统的模拟独立检查，不是外部人类同行评议；没有将角色票数当成统计证据。

## 阶段状态与下一步边界

PART 0：现有工作区盘点已完成，但指定 Final 资料完整性检查未通过。
PART 1：对现有 Stage8.3 稿完成有范围限制的审稿；Final 与正确补充材料仍需到位后复核。
PART 2–9：未执行。后续方法方案需按用户阶段制由作者确认后，才建立实验实现分支并启动实现/训练。本报告不将任何候选方法、建议实验或验收标准写成已完成结果。
