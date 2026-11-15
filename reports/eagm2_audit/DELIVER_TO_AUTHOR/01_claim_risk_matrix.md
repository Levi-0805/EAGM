# PART 1 — Claim–Risk Matrix

审计基准：9416b92190b5d4d97232a1850eb27ee83953fa39，Stage8.3 `paper_eagm/`；指定 Final / 正确补充材料缺失。下列含现稿真实主张与未来可能主张，明确区分；“不支持”不表示现稿已经作出该主张。所有实验均为历史记录，不是本轮新实验。

| 主张及性质 | 论文定位 | 证据定位 | 审计判断与风险 | 允许的表达 / 解除条件 | 问题ID |
|---|---|---|---|---|---|
| 当前可认证指定Final与主补一致（验收要求） | paper_eagm/README.md:1–32；07_generalization.tex:16 | 00_scope_and_workspace.md | 不支持，P0版本阻断 | 仅认证所见Stage8.3局部证据；取得指定主稿/正确补充后复核 | S00 |
| 在固定B跟踪协议下Dir优于Base（现稿） | 06_results.tex:6 | runs/stage5a_confirm/seed5..9/per_start.csv；00_readonly_checks.log | 支持受限经验改善；5种子不能支撑泛化或普遍显著 | 均值差0.0005141553，5/5同向；独立种子是5而非60起点；不与Stage4混池 | C07,A06 |
| 优势超出一种额外动力学数据解释（现稿） | 06_results.tex:8 | runs/stage5b_budget/seed5..9/per_start.csv；01_reviewer_a_verification.json | 支持特定control-step预算对照；非wall-time等价 | 均值约0.0002591776；保留预算口径，补直接FD/SPSA等替代用途比较 | A05 |
| Gradient-aware model learning 本身原创（现稿未声称） | 02_related_work.tex:3–20 | PAML Sec3 Eq7–9；GAMPS Sec4.1 Eq3；VaGraM Sec3 Eq7，见独立审稿链接 | 不支持；执行特有贡献尚待证明 | 明确执行命令瓶颈/监督接口的具体差异；公平适配与执行不可观测交互消融 | A01,A02 |
| U0等于延迟下nominal或实际执行命令（潜在误读） | 04_method.tex:2–7 | models/eagm.py:14–16；controllers/partial_unknown.py:165–178 | d>0不成立；d=0也不保证等于B实际输出 | U0为delay-free滤波先验；明确d0/d1/d2名义索引及B残差。先报告，后验证新先验 | B03 |
| Base已完整恢复/传播隐藏执行状态（潜在扩写） | 03_formulation.tex:2–8；04_method.tex:5–7 | models/eagm.py:60–85；training/rollout.py:100–145 | 不支持，未传播B完整隐状态 | 仅称部分观测条件预测；完整状态oracle与部署估计分列 | B01,B04,B05 |
| 仓库尚无GRU或多步想象（任务书潜在预设） | 04_method.tex:26；05_setup.tex:6 | models/eagm.py:19–57,89–185；training/rollout.py:100–145 | 错误；已有有限窗口GRU、H8想象及50策略更新 | 新研究应说明比既有History/HistC/rollout增加了什么能力 | A07,B04 |
| HistC失败证明执行历史无价值（现稿表述可能被过度解读） | 07_generalization.tex:5–6 | training/dataset.py:129–134；rollout.py:128–132；reports/stage6c_protocol.md:80–85 | 不支持；时序偏差污染阴性解释，P0证据用途阻断 | 保留失败事实并披露缺陷；修复后的预注册对照才能判断模块价值 | B09 |
| exec-state probe独立验证隐状态恢复（历史报告可能用途） | reports/stage6b2_results.md probe部分 | training/exec_state_probe.py:67–95；runs/stage6b2_histc/seed10/results.json probes | 不支持，训练/所谓holdout重用；不等于主模型全局泄漏 | episode级隔离重新拟合probe并单独报告 | B09 |
| Dir仅监督方向，其收益必由方向提高导致（潜在扩写） | 04_method.tex:16–26 | training/dir_loss.py:24–33；training/paired_train.py:73–90 | 纯方向不成立；现稿已准确说明含模长 | 方向、模长、归一化分离消融；回报与梯度同步验证 | A04 |
| 高前向精度保证梯度准确；高梯度准确保证回报提高（现稿明确否定） | 03_formulation.tex:16–32；08_analysis.tex:3–23 | runs/stage8_2_summary/aggregates.json；runs/stage8_3_alignment/decision.json | 不支持；标准条件smoothness不是EAGM收敛证明 | 保留三种误差/效用分离；新定理需假设证明，更新效用需同分布及未见分布实测 | A03,A07,B06 |
| Stage8.2 oracle证明部署方法成功（现稿未声称） | 08_analysis.tex:3–11 | runs/stage8_2_summary/aggregates.json；single_step.csv | 不支持；oracle替换完整执行/物理导数且回报仍有限 | 仅作冻结checkpoint机制干预；不能归因所有误差于执行隐状态 | B06 |
| 跨控制器普遍优于强memory baseline（潜在扩写） | 07_generalization.tex:2–3 | runs/stage6b1_confirm/results.json paired_seed_stats；protocol.json | 不支持；同代码族Dir−Base为正，Dir−Mc区间含0 | 仅称受限配置泛化；分开nominal delay与extra_lag；新配置独立确认 | B07 |
| Reach跨任务成功（现稿明确否定） | 07_generalization.tex:8–14 | runs/stage7c2_true_ee_current_sites_seed20/results.json | 反证当前seed20成功；不能推断所有Reach必败 | 保留Dir较Base约−0.0002649944与单种子范围 | C04,B08 |
| Stage8.3为成功新算法（现稿明确否定） | 08_analysis.tex:18；reports/stage8_3_research_decision.md | runs/stage8_3_alignment/decision.json | 两个联合gate均未通过 | 保留验证/新测试冲突；不得选择性突出fresh-test赢家 | A07,C07 |
| 已有RGB策略优化和视觉接触泛化（现稿未声称） | 05_setup.tex:3；08_analysis.tex:21 | training/rollout.py:47–65；tasks/reach.py:23–47 | 不支持；旧视觉诊断不等于此证据 | 真RGB部署输入、接触成功条件和独立测评；如不做，限制视觉主张并评估定位 | C01–C05 |
| 仿真physical return代表实机验证（潜在误读） | 05_setup.tex:3；08_analysis.tex:21 | controllers/compatible_arm_env.py；training/rollout.py:179–182 | 不支持；无实机是P2外部有效性限制 | 明确physical指物理模拟器；硬件非必需硬门槛，但硬件主张必须实测 | C06 |
| balanced_dir_acc为balanced sign accuracy（原始字段名称） | 主表未使用；05_setup.tex:20–23为统计语境 | training/eval_metrics.py:18–26；run_stage7c2_true_ee.py:95–97 | 定义不符，实际为普通符号准确率 | 用原始向量/mask另立派生重算或更名；无需默认重训 | C08 |
| fd_mode=central总为关于当前theta的对称差分（部分元数据） | 04_method.tex:14,24 | training/gradients.py:107–137,255–285 | feasible近边界时可非对称；Stage5默认路径不受此项影响 | 分别记录有效左右步长及stencil；先确定受影响训练范围 | A08 |

## 后续研究优先级，而非方法批准

先固定审计对象并修复证据解释，再进入研究方案设计。Delay-aware prior可优先作为有明确数值判据的候选；latent execution state须与已存在的GRU比较，并先消除时序/数据隔离缺陷；视觉接触扩展前要解决28维坐标FD向高维策略扩展的成本问题。最终模块选择必须由独立实验和消融决定，不能由本审稿预定赢家。

本轮没有改论文、改研究实现、改原始日志或重训。上述解除条件不代表已执行。最终Final资料缺失仍然阻止将整个PART0/1标记为“指定版本全面审计通过”。
