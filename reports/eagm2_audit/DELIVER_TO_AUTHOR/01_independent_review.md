# PART 1 — 独立审稿报告

**范围限定：本轮审计 Stage8.3 研究稿（9416b92），不是尚未取得的 CVPR2027_EAGM_Final。** 三位独立角色均给出 Weak Reject；这是同一系统的审稿模拟，不是外部同行评议或接收概率。没有修改论文、研究代码或历史结果，没有开展 PART 2–9。

## 主审综合判断

当前最扎实的证据是固定 Controller B 跟踪任务中 Dir 相对 Base 的配对回报改善。Stage5A 五种子原始 CSV 复算均值为 0.0005141552735794117，43项论文源文件哈希全部匹配。主稿已正确限制 Reach、视觉、oracle 和五种子证据；审计不把已披露的局限误作虚假主张。

优先处理顺序（后续执行仍服从用户阶段制）：

1. **版本阻断 S00**：取得指定 Final 与正确补充材料，再对最终版本下结论。
2. **证据用途阻断 B09**：HistC 收集历史追加旧 c_obs，想象历史追加新 c_obs；probe 的所谓 holdout 与训练样本重叠。不能以其负面消融排除记忆建模，也不能把 probe 当独立恢复证据。这不证明 Base/Dir 主评测泄漏。
3. **度量/定义复核 B03、C08、A08**：U0 忽略 nominal delay；balanced_dir_acc 是普通符号准确率；边界可行 FD 有非对称 secant 仍标 central 的情况。先定位受影响实验，再另立派生修正记录；禁止覆盖旧数据。
4. **研究贡献与证据 P1**：缺公平外部目标基线，执行特有贡献尚待消融；更好梯度不保证分布外更新收益。当前 actor 只有28参数，坐标FD不能直接外推到整个视觉网络。
5. **泛化边界**：既有短时域想象、GRU History/HistC 都已实现，不能作为未有先例的新模块。当前没有RGB策略优化/接触操作成功证据；真实机器人不是会议统一硬门槛。

P0表示阻断特定证据用途或版本认证；P1表示重要科学证据缺口；P2表示局部数值/定义/适用范围问题。条目存在跨角色重叠，不应按数量累加判断严重性。A01–A07、B01–B08、C01–C06逐项覆盖任务书指定问题；A08/B09/C07/C08为额外发现。

下面保留三个角色的原始审稿内容。代码路径均以冻结仓库根目录为基准；旁目录CVPR2027仅作旧诊断材料归属检查，不随此仓库上传，也不作为正确Final supplement。完整问题字段见 `01_critical_issues.csv`；主张用途见 `01_claim_risk_matrix.md`。

---

# Reviewer A — Model-Based Reinforcement Learning（只读审计）

审计对象：本地 EAGM `9416b92190b5d4d97232a1850eb27ee83953fa39` 的 `paper_eagm/main.pdf`、全部八个 section 源文件、相关训练/模型代码及下面列出的原始记录。主稿PDF通过 `pdftotext -layout` 核查。任务书指定的 `CVPR2027_EAGM_Final` 与最终补充材料未提供；此报告不是其最终稿审计，不假装读过指定九份缺失文件。没有运行训练或修改研究代码/原始数据。

**独立建议：Weak Reject（针对所见研究稿；不是 CVPR 官方判定）。** Tracking 的内部对照与诚实报告负面结果值得保留；主要障碍是外部目标基线缺失、执行特有创新尚未建立、梯度保真到实际更新的泛化链不闭合。下列P1表示投稿前的重要科学证据缺口，P2表示局部数值/报告问题；不以每个缺口都必须成为新定理或新模块为要求。

原始核验：`paper_eagm/data/source_manifest.json` 列出的43份文件全部存在且SHA256完全匹配。直接读取Stage5A seeds5–9的per_start.csv，逐seed模型均值与论文+0.0005141553的Dir−Base均值相符。另直接读取Stage5B逐seedCSV、Stage5D原始JSON和Stage8.2 multistep CSV/aggregates JSON。哈希匹配仅证明当前文件与manifest一致，不是重跑复现实验。Stage4 seeds0–4是探索性，Stage5A 5–9才是冻结后的确认性；Stage5B复用5–9，不新增独立种子。未声称穷尽所有原始JSON。

## 七项指定审查及附加数值问题

### A01 Gradient-aware model learning 创新性 — P1

- 论文位置：paper_eagm/sections/01_introduction.tex:7-13; 02_related_work.tex:3-20
- 代码/实验/原论文证据：models/eagm.py:60-86; training/paired_train.py:70-89; PAML Sec.3 Eq.7-8 https://arxiv.org/html/2003.00030
- 判断：现有稿已承认gradient-aware learning不是新概念；独立贡献目前是执行命令瓶颈与查询计费的梯度监督组合，尚无证据说明其收益必须依赖部分可观测执行机制。
- 对论文质量的影响：若只证明已有梯度匹配理念在一个执行接口有效，方法创新可能不足；但不能据此断言它已被先前工作完全覆盖。
- 现有数据可否解决：部分：可精确限定贡献，不能凭现有数据确认独立创新
- 新增代码：需要：共同执行接口上的目标/架构对照
- 重新训练：需要：公平外部适配及独立种子
- 预期验证标准：建立有/无隐藏执行状态 × 有/无命令瓶颈 × 有/无梯度目标的交互对照；验证收益是否随执行不可观测程度出现，并保持预算/容量一致。

### A02 与 PAML、GAMPS、VaGraM 的实质区别 — P1

- 论文位置：paper_eagm/sections/02_related_work.tex:3-12,19-20; 06_results.tex:15
- 代码/实验/原论文证据：training/dir_loss.py:11-33; training/gradients.py:48-61; PAML Eq.7-9; GAMPS Sec.4.1 Eq.3 https://arxiv.org/pdf/1909.04115; VaGraM Sec.3 Eq.7 https://arxiv.org/pdf/2204.01464
- 判断：PAML比较策略梯度泛函并以真实数据critic构造经验目标；GAMPS通过折扣、重要性比和累计score范数加权转移似然；VaGraM以价值函数状态梯度的平方对模型MSE按维加权。EAGM则匹配resettable simulator坐标FD与可微想象回报梯度，并有命令瓶颈。差异成立，性能优势未知。
- 对论文质量的影响：没有同任务、同可观测信息、同重置权限和预算的外部目标基线，无法判断收益来自新的执行处理还是既有decision-aware目标。
- 现有数据可否解决：否：现有内部MLP/辅助head对照不足
- 新增代码：需要：忠实适配或明确标记reimplementation
- 重新训练：需要
- 预期验证标准：固定策略类、输入、command语义、训练步数和查询账本；比较至少最接近且可公平适配的方法，并公开无法适配的理由；不借用原论文数值。

### A03 Forward prediction 与 policy-gradient fidelity 理论关系 — P1

- 论文位置：paper_eagm/sections/03_formulation.tex:16-32; 08_analysis.tex:3-23
- 代码/实验/原论文证据：training/rollout.py:100-145; runs/stage8_2_summary/aggregates.json; runs/stage8_2_summary/multistep.csv
- 判断：完整闭环敏感度递推和条件smoothness不等式是正确的有限解释，但不是EAGM误差界或收敛定理。前向值误差不能单独控制导数；Stage8.2同前向值替换Jacobian支持这一点。oracle H8 cosine=0.9873777474仍未保证held-out改进。
- 对论文质量的影响：不能从低NMSE或高cosine推出策略改进，更不能将标准smoothness不等式当独立理论创新。
- 现有数据可否解决：部分：可用现有数据支持分离现象；新增定理需要明确额外假设和证明
- 新增代码：不必为表述修正；若验证误差界需分析代码
- 重新训练：不必为表述修正；强泛化结论需要新实验
- 预期验证标准：列明导数误差、初态分布、奖励导数、闭环Jacobian范数、步长和非光滑假设；若提出界须完整证明并限定适用域；同时报告相同分布下真实更新增益。

### A04 Directional supervision 是否只是组合；方向与模长归因 — P1

- 论文位置：paper_eagm/sections/04_method.tex:16-26; 01_introduction.tex:9
- 代码/实验/原论文证据：training/dir_loss.py:24-33; training/paired_train.py:73-90
- 判断：实际损失是按|gFD|归一化的Huber discrepancy，监督方向与模长而非纯direction。现稿对此已准确披露，未发现此处公式代码冲突；但现有Base/Dir对比不能分辨方向、模长和归一化的独立作用。
- 对论文质量的影响：若声称改进由梯度方向保真驱动，现有对照仍有解释混淆；Huber、mask、FD和瓶颈的组合本身不构成充分创新证据。
- 现有数据可否解决：部分：可降级归因；不能完成机制区分
- 新增代码：需要：direction-only、norm-only、joint、mask/normalization对照
- 重新训练：需要
- 预期验证标准：保持架构、label数和预算一致，预注册方向/模长单独及组合消融；在新种子同时验证gradient cosine、norm ratio和真实回报。

### A05 有限差分标签成本与可扩展性 — P1

- 论文位置：paper_eagm/sections/04_method.tex:28-29; 06_results.tex:8
- 代码/实验/原论文证据：training/gradients.py:61-68,139-153; training/rollout.py:47-58; configs/stage5_frozen.json; reports/stage5_budget_matched.md:1-31; runs/stage5b_budget/seed5/per_start.csv
- 判断：28参数、3尺度、正负2次、H8使一个标签需1344 control steps；12训练标签需16128，对比基础采样4800。Stage5B有有效control-step预算对照，Dir相对Base+dyn约+2.59e-4，不能否定已有预算证据；但未等价wall time，亦不能外推至视觉网络全参数FD。
- 对论文质量的影响：较小手工策略使查询成本可承受；CNN/Transformer全参数坐标FD线性增长，可能使拟议视觉扩展不可行。
- 现有数据可否解决：部分：现有账本可重算交互成本；新策略规模需要新增测量
- 新增代码：需要：direct FD/SPSA预算基线和计时/规模实验
- 重新训练：需要：若宣称扩展到高维策略
- 预期验证标准：报告总标签/诊断/数据采样steps及CPU/GPU时间；按共同总预算比较直接FD、SPSA和EAGM；预先说明高维策略只监督哪些参数或方向。

### A06 梯度过拟合与标签失配 — P1

- 论文位置：paper_eagm/sections/04_method.tex:9-14,26; 07_generalization.tex:6
- 代码/实验/原论文证据：training/paired_train.py:73-79; training/stage5_common.py:48-55; runs/stage5d_gradpath/results.json; runs/stage5d_gradpath/seed5/eagm_dir_matched_path.json
- 判断：训练确实在label.theta上计算model gradient，不是错配标签。Stage5D初始path cosine差值均值+0.2784(5/5)，最终仅+0.0693(3/5)，bootstrap区间[-0.1122,0.2339]；seed5和9最终变差。因此存在policy-local泛化不足迹象，但不能把它直接定性为训练集过拟合：缺train/test梯度误差随训练曲线的诊断。
- 对论文质量的影响：离开标签策略分布后，回报提升与梯度保真提升不能等同；也不能先验认定周期重标或path-label有效。
- 现有数据可否解决：部分：现有路径结果可支持边界说明；无法确证过拟合原因
- 新增代码：需要：按策略距离/更新次数的独立梯度验证、relabel消融
- 重新训练：需要：用于方法改进结论
- 预期验证标准：独立起点与策略probe上报告train/test gap、mask覆盖率及不同更新距离误差；预固定重标时机，不用test误差挑频率；保留Path失败。

### A07 多步策略优化能力 — P1

- 论文位置：paper_eagm/sections/05_setup.tex:6,18; 07_generalization.tex:9-14; 08_analysis.tex:3-23
- 代码/实验/原论文证据：training/rollout.py:100-145; configs/stage5_frozen.json; runs/stage8_2_summary/aggregates.json; reports/stage8_2_multistep_sensitivity.md:3-20; reports/stage8_3_research_decision.md:3-9
- 判断：代码已有多步想象(H8)和50次策略更新，不能称完全没有多步能力。但证据仅支持tracking短时域；Reach learned H8 cosine A=-0.1321/B=-0.1155，oracle C=0.7947/D=0.9874；Stage8.3联合gate失败。
- 对论文质量的影响：把tracking短期改进扩大为一般长时域优化/跨任务可用性，会超出证据。
- 现有数据可否解决：是：可正确陈述现有短时域结果；否：证明长时域稳健性
- 新增代码：需要：预注册horizon及长期策略路径评估
- 重新训练：需要：推广性结论
- 预期验证标准：不同H下同时报告rollout error、gradient norm/cosine和物理回报；新任务新seed验证，不把单checkpoint六个start当六seed；零真梯度需标记undefined cosine。

### A08 近边界 finite-difference 标签的索引语义 — P2

- 论文位置：paper_eagm/sections/04_method.tex:14,24
- 代码/实验/原论文证据：training/gradients.py:107-137,255-285; metadata fd_mode=central
- 判断：feasible=True且theta位于边界附近但未到边界时，正负点分别clip，使用两端差/span，仍标central；这是非对称secant，不是以当前theta为中心的二阶central stencil。例J(theta)=theta^2, theta=.98, eps=.05，上端1、下端.93得到1.93而真导数1.96。
- 对论文质量的影响：会降低近边界标签精度且fd_mode不揭示非对称性；不能推断所有历史结果因此无效，Stage5默认feasible=False且此问题不适用。
- 现有数据可否解决：部分：可统计近边界标签比例和effective epsilon；已有不全需重算
- 新增代码：需要：数值单测及显式非对称标签元数据；先报告再决定公式
- 重新训练：条件：若查明受影响训练标签用于主张，需重生成/重训；非受影响历史不动
- 预期验证标准：二次/三次解析函数测边界内外，给实际步长与误差阶；分离one-sided、symmetric-central、asymmetric-secant；报告受影响实验清单，不擅自重写历史。

## 本次必要原始文献核查范围

- [PAML 原文](https://arxiv.org/html/2003.00030)：核查Sec.3 Eq.7–9及真实数据critic的经验构造；并非只读摘要。
- [GAMPS 原文](https://arxiv.org/pdf/1909.04115)：核查Sec.4.1 Eq.3加权似然及三种权重含义。AAAI PDF入口被网站阻止，使用作者arXiv全文版本。
- [VaGraM 原文](https://arxiv.org/pdf/2204.01464)：核查Sec.3.1–3.2 Eq.3–7的Taylor近似、value-gradient weighting与按维平方上界。

这些核查支持目标形式的区别，不声称完成PART2全球创新性审查，也不声称证明任何先前方法没有执行信息。

## 可保留与必须限制的主张

可保留：在测试过的固定Controller B tracking协议下，匹配参数的梯度监督相对Base改进回报；有control-step budget对照；真实reward/transition derivative干预区分forward、gradient、update utility。

必须限制：不能声称gradient-aware model learning本身原创；不能把gradient cosine提升说成普遍策略提升保证；不能把Stage8 oracle当部署方法；不能将已实现GRU或新的模块清单写成经过验证的新贡献。


---

# Reviewer B — Robotics and Control 只读审计

审计对象：本地 `eagm_gpu` commit `9416b92190b5d4d97232a1850eb27ee83953fa39` 的 Stage 8.3 研究稿 `paper_eagm/main.pdf`（已用 pdftotext 读取）及对应 LaTeX、控制器、模型、rollout、数据收集与指定原始 JSON。路径均相对仓库。任务书指定的 Windows Final 目录及 final_submission_checklist / formula_code_audit / claim_evidence_mapping 等未在此版本找到；因此这不是对缺失最终稿/正确supplement的认证。未修改源码、论文、实验记录，未训练。

独立判断：**Weak Reject（针对当前可审阅研究稿）**。核心受限 tracking 收益不应抹去，但无法认定为完整部分可观测执行建模方案；方法描述缺少delay先验的精确定义，外部有效性不足。最紧急问题是 HistC 负面消融的实现偏差与 probe 训练/测试重用，须先限定证据解释。当前论文对Reach与oracle限制较诚实；这些不是新发现的造假。P0表示证据用途阻断，不代表全部研究无效；P1表示主要科学/复现缺口；P2表示定义和表达缺口。

## 时间索引复核

以step调用前为t：r_t=qtarget，f_t=filtered，queue=[f_(t-1),f_t]。先clip action，再更新target，随后f_(t+1)=alpha*r_(t+1)+(1-alpha)*f_t，读取旧queue决定nominal，再移队列，最后经过B未知执行链路。legacy源注释把new_filter写f_t，是另一时标；不能据此误认运行顺序错一帧。alpha=.5、r_t=.2、f_t=.15、a=.03的合成合法状态给出U0=.19，而d0/d1/d2 nominal=.19/.15/.10。此为代数诊断，不是新增训练或实验性能。

## B01 [P1] 机器人控制器隐藏状态如何影响执行命令

- **论文位置：** §3 Eqs.(2–4), sections/03_formulation.tex:2–8；§4 Eq.(8), sections/04_method.tex:5–7
- **代码/原始证据：** controllers/partial_unknown.py:199–226；models/eagm.py:60–85；training/rollout.py:100–145
- **发现：** 真实链路为 gain→额外FIFO→相对上次输出的rate limit→smooth→joint clip。Base命令头只读(c_obs,a)，没有传播_extra_queue、_u_prev、_u_smooth；同一名义输入可以对应不同输出。只读提取真实_apply_unknown函数的合成检验：nominal=.2、rate=.03、其余不变时，u_prev=0/.1分别输出.03/.13。该检验是函数反例，不是任务回报实验。
- **影响：** 当前模型是部分观测条件预测器，不是精确执行状态模型；多步梯度会遗漏历史依赖。
- **现有数据能否解决：** 可以确定结构遗漏；不能仅用现有均值指标证明新估计器有效
- **新增代码：** 若升级，需要可部署历史估计与因果测试
- **重新训练：** 新方法需要；当前限制描述不需要
- **验证标准：** 同观测不同隐藏状态的成对干预；比较命令误差、梯度、回报；oracle与历史估计分列。

## B02 [P2] 已知执行状态与未知执行状态是否正确区分

- **论文位置：** §3 sections/03_formulation.tex:2–14；§5 sections/05_setup.tex:3
- **代码/原始证据：** controllers/partial_unknown.py:53–110,228–257；models/eagm.py:214–253；training/dataset.py:83–127；training/rollout.py:101–112,181
- **发现：** 接口明确：c_obs=(target,filter,delay) 29维；Mc的57维含Controller A FIFO，仍不含B隐藏状态；u_true和b_active是标签/诊断。审阅调用未发现Base部署读取privileged字段。snapshot含完整隐藏状态用于物理重放，不等于部署输入。论文正文未给完整维数/字段表，容易把Mc误当B全状态上界。
- **影响：** 主要是复现与公平比较定义不足；不能因snapshot存在就判定泄漏。
- **现有数据能否解决：** 是，可依据现有代码补字段/可用时刻表
- **新增代码：** 可加输入依赖断言；无需变模型
- **重新训练：** 否
- **验证标准：** 逐模型列输入、训练标签、评测专用字段；随机改privileged字段不改变deployable前向输出。

## B03 [P1] FIFO、delay、rate limit、smoothing、gain的数学定义

- **论文位置：** §4 Eq.(8), sections/04_method.tex:2–7；§3 Eq.(2)
- **代码/原始证据：** controllers/partial_unknown.py:165–176,199–226；vendor/legacy_sim/sim_env.py:167–198；models/eagm.py:14–16；training/rollout.py:69–80
- **发现：** 令step前状态为(r_t,f_t)，queue=[f_(t-1),f_t]；step产生r_(t+1),f_(t+1)。名义命令d=0/1/2依次为f_(t+1)/f_t/f_(t-1)。U0始终返回f_(t+1)，未读取delay，也未独立clip action。后者在当前actor已限幅路径内被满足，不能外推任意动作。beta>0时代码是 beta*limited+(1-beta)*old；beta=0特殊分支直接关闭平滑，并非公式的连续极限。
- **影响：** delay>0时先验在非饱和点的瞬时action Jacobian为alpha I，而真实名义命令为0；残差必须学会取消错误直接路径。论文没有明确列这项适用范围，但未明说U0等于真实执行命令，属于定义不足与结构偏差。
- **现有数据能否解决：** 是，可用现有状态/nominal日志核查；新先验效果不能由代数推断
- **新增代码：** delay-aware候选与逐步FIFO测试需新增
- **重新训练：** 比较先验的回报与梯度需重训
- **验证标准：** 逐delay覆盖reset、饱和、常规序列、alpha边界；名义命令数值一致且因果Jacobian正确。f_(t-1)=(f_t-alpha*r_t)/(1-alpha)仅alpha!=1且满足更新不变量，不能泛称任意delay可重建。

## B04 [P1] 是否真正解决部分可观测执行问题

- **论文位置：** §4 sections/04_method.tex:5–7；§7 sections/07_generalization.tex:5–6
- **代码/原始证据：** models/eagm.py:19–57,60–185；training/rollout.py:106–139；runs/stage6b2_histc/seed10/results.json: param_counts, results, probes
- **发现：** 仓库已经实现GRU History/HistC，不能在升级任务书中把GRU说成从未实现的新模块。它们每步重编码有限窗口；Base/Dir主方法没有belief-state更新或可辨识性论证。HistC未显示独立收益，但其实现有B09问题。
- **影响：** 只能支持在受限部分观测设置下的经验收益，不能声称恢复完整隐藏状态、消除POMDP或证明记忆无效。
- **现有数据能否解决：** 可界定现有模型和证据；不足以建立充分统计状态
- **新增代码：** 新belief估计/可辨识性诊断需要
- **重新训练：** 新方法验证需要
- **验证标准：** 在冻结可观测接口下报告历史长度、状态预测误差、容量对照、oracle差距和多步误差；不得用训练内probe相关性代替隐藏状态恢复。

## B05 [P1] 当前模型是否具有足够Markov性质

- **论文位置：** §3 sections/03_formulation.tex:2–6,19–24；§8 sections/08_analysis.tex:3
- **代码/原始证据：** models/eagm.py:60–85；training/rollout.py:69–73,137–144；controllers/partial_unknown.py:53–57,199–226
- **发现：** (x,c_obs)不是一般Controller B的完整Markov状态；c_obs_step仅传播target/filter/delay。固定alpha=.5、d<=2时，名义FIFO可由现有状态不变量恢复，这意味着名义队列不可直接观测不必然造成不可恢复；真正B额外队列、平滑和参数仍无充分状态表示。理论Eq.(6)针对完整增广z，不能直接套给Base状态。
- **影响：** 完整系统链式法则的合理性不能替代对实际learned rollout闭合性的证明。
- **现有数据能否解决：** 是，可澄清理论适用对象；无法从单步拟合证明Markov
- **新增代码：** 如声称充分性，需状态别名与历史条件检验
- **重新训练：** 要验证新闭合模型需重训
- **验证标准：** 明确z与模型输入区别；构造/采样同输入不同未来；比较历史增加是否降低条件残差并检验多步传播。

## B06 [P1] 隐藏状态遗漏是否导致系统性误差及因果归因

- **论文位置：** §8 sections/08_analysis.tex:3–11,21–23
- **代码/原始证据：** runs/stage8_2_summary/aggregates.json:A/B/C/D；runs/stage8_2_summary/single_step.csv；controllers/partial_unknown.py:199–226
- **发现：** Stage8.2对完整转移导数的替换，把H8 cosine由-0.115517提升到0.987378（true reward条件），但同时替换执行与物理转移导数；D的held-out回报变化仍为-7.05373666e-6。它支持完整链路导数失真影响梯度，不能把全部误差归因B隐藏状态；论文目前已说明separate causal shares not fully identified。
- **影响：** 执行状态估计是待检验假设，不应以oracle恢复梯度作为新模块有效性证据。
- **现有数据能否解决：** 现有数据足以保留有限结论；不够分离贡献
- **新增代码：** 需要execution-only、physics-only导数/状态干预
- **重新训练：** 冻结干预可先不训练；方法性能需训练
- **验证标准：** 同forward轨迹的2×2干预，报告逐start梯度及独立策略回报；不得提升seed20机制诊断为多种子方法成功。

## B07 [P1] 能否泛化到不同执行控制器配置

- **论文位置：** §7 sections/07_generalization.tex:2–3；§5 sections/05_setup.tex:20–23
- **代码/原始证据：** runs/stage6b1_confirm/protocol.json: train_ranges/interp/extrap；runs/stage6b1_confirm/results.json: paired_seed_stats；training/dataset.py:188–209
- **发现：** 存在真实配置泛化证据：seeds10–14的Dir−Base插值均值.002342559、CI[.001682825,.002994574]；外推.002444045、CI[.001649452,.003188892]。Dir−Mc相应CI包含0。OOD为预定义同一代码族参数变化（含extra_lag=3），非未知控制器结构；默认policy starts筛选declared delay=2。
- **影响：** 可以主张受限参数变动下优于Base，不能主张普遍controller鲁棒、未见declared delay泛化或优于所有memory基线。
- **现有数据能否解决：** 可按配置和delay复核现有结果
- **新增代码：** 如拓展动态配置切换/不同控制器族需新增协议代码
- **重新训练：** 独立确认需要
- **验证标准：** 预注册新的种子、初态与held-out配置；标明nominal delay与extra lag的区别；报告逐配置配对差值，不混合协议。

## B08 [P1] 结果是否仅适用于特定MuJoCo环境

- **论文位置：** §5 sections/05_setup.tex:3；§7 sections/07_generalization.tex:8–14；§8 sections/08_analysis.tex:21
- **代码/原始证据：** controllers/compatible_arm_env.py:26–54；vendor/legacy_sim/sim_env.py:182–205；training/rollout.py:47–58；paper_eagm/README.md:23–35
- **发现：** 主要证据来自单一OpenArm MuJoCo模型、14维动作与手工结构的28参数actor；Reach是同机器人不同目标且失败。没有第二机器人、接触任务、真实硬件闭环证据。物理return是仿真物理引擎回报，不是实机。
- **影响：** 研究外部有效性不足；可保持局限，不应将主稿命名/摘要扩写成通用机器人执行学习。
- **现有数据能否解决：** 现有数据只能准确界定边界
- **新增代码：** 更广任务/控制器需要
- **重新训练：** 新任务需要
- **验证标准：** 至少独立任务的成功条件与固定测评，真实运行且逐seed报告；真实机器人不是自动硬性要求，但若保留实机有效主张必须有实机证据。

## B09 [P0] HistC阴性消融和隐藏状态probe的证据污染

- **论文位置：** §7 sections/07_generalization.tex:5–6（负面结果的解释）；相关reports/stage6b2_results.md hidden-state probe
- **代码/原始证据：** training/dataset.py:78–81,129–134；training/rollout.py:128–132；training/exec_state_probe.py:67–80,87–95；models/eagm.py:165–173；reports/stage6c_protocol.md:80–85；runs/stage6b2_histc/seed10/results.json:probes
- **发现：** 数据收集在step后append的是旧c_obs_t，但x_buf append新x_(t+1)，导致HistC历史与注释不符；imagination追加新c_obs_(t+1)，构成训练/想象时序差异。probe从全部tr rows采样训练，随后用同一tr尾20%称holdout，属于实证数据泄漏；该泄漏仅已确认于analysis probe，不等于主world model测试泄漏。Shuf对照实际固定time reverse，而非随机shuffle。
- **影响：** P0针对将现有probe作为独立验证或将失败解释成记忆方向已被排除的证据用途；不否定全部Base/Dir结果。保留失败结果正确，但不能把带实现缺陷的阴性实验当成历史建模不必要的结论。
- **现有数据能否解决：** 可以立即发现、标注和限制结论；不能靠重算旧均值修复学习时序
- **新增代码：** 需修复历史对齐、episode级probe拆分及命名；修复应未来独立分支
- **重新训练：** 纠正HistC性能解释需要重训；probe需重新拟合
- **验证标准：** 逐步断言x/c/a时刻；train/val/test episode ID交集为空且归一化只fit train；重新冻结容量和顺序对照；旧结果原样保留为受污染探索证据。

## 审计边界

核查了相关核心源码、主稿全文文本与上述JSON键，未宣称逐字节验证所有历史轨迹/所有训练checkpoint；未取得指定Final与supplement，因此最终版一致性仍未完成。未开展PART2全面文献工作或PART3新方法设计。后续可以先修正文档中的证据等级，任何修复和重训必须等待阶段三研究方案批准。


---

# Reviewer C — Computer Vision and Generalization（只读审计）

审计对象：`/home/aaa/桌面/汇报/11月论文/eagm_gpu`，基准 `9416b92190b5d4d97232a1850eb27ee83953fa39`，Stage 8.3 研究稿。已读取相关 LaTeX、main.pdf 的 pdftotext 内容、策略/任务/评估代码及指定原始结果文件。未修改论文、代码或历史记录，未启动训练。

这不是指定 `CVPR2027_EAGM_Final` 的完整审计。该目录、用户列出的最终审计文档和本研究稿正确补充材料未取得。旁目录 `CVPR2027/supplementary.pdf` 属于独立旧诊断项目，不替代 EAGM 最终补充材料。本审查仅为已定位版本的证据化判断。

结论：对当前证据版本给出 **Weak Reject（审稿模拟，不是会议决定）**。最强依据是实际配对物理回报在受限跟踪设置中改善；决定性不足是面向用户的 CVPR 目标尚无视觉策略贡献、成功跨任务证据及足够独立确认。当前稿诚实保留负面结果和范围限制，不应将这些诚实说明误报为论文宣称成功。真实机器人缺失本身不是自动拒稿理由。

严重度：P0 为阻断可信性/版本识别的致命问题；P1 为目标主张所需的重要证据缺口或度量错误；P2 为应明确的范围限制。下列新增实验仅为未来验证要求，不构成当前阶段启动授权。

## C01 与 CVPR 研究主题的匹配 — P1

- **论文位置**：paper_eagm/sections/01_introduction.tex:7–13（PDF p1）；08_analysis.tex:21（p8）
- **代码/实验依据**：training/rollout.py:47–65、100–145；training/policy_opt.py:73–74
- **发现**：现有贡献是执行链路与策略梯度的受限状态控制研究；稿件明确没有视觉贡献。不能仅凭机器人题材判定适配，也不能声称 CVPR 明文必须有真实机器人或 RGB。
- **影响**：面向用户选择的视觉研究目标，缺少视觉问题、算法及证据之间的联系，当前不支持 Weak Accept 准备度。
- **现有数据能否解决**：只能澄清范围，不能生成缺失视觉证据
- **新增代码**：若保留视觉目标，需要
- **重新训练**：需要
- **验证标准**：先明确视觉观测不足与执行记忆互补的可检验问题，再以真实图像策略及隔离测试、独立测试环境验证；或重新评估投稿定位。

## C02 RGB 是否实际参与策略优化 — P1

- **论文位置**：paper_eagm/sections/05_setup.tex:3（p5）；08_analysis.tex:21（p8）
- **代码/实验依据**：training/rollout.py:47–65 策略为0.55(ref−c)+0.01θ[:14]+0.08θ[14:](ref−q)；policy_opt.py:73–74仅优化28维θ
- **发现**：本版本策略直接使用关节状态、控制目标和参考；没有RGB策略输入或视觉编码器。
- **影响**：不能把已有梯度优化结果称为视觉策略学习。当前稿没有这样误称，因此这是目标证据缺口而非已证实造假。
- **现有数据能否解决**：否
- **新增代码**：是
- **重新训练**：是
- **验证标准**：RGB前向图必须实际连接到动作；部署输入禁用不可获得状态，做输入泄漏测试，独立相机/初态/控制配置测试，并报告配对种子物理回报。

## C03 现有视觉实验是否仅为信息诊断 — P1

- **论文位置**：paper_eagm/sections/01_introduction.tex:13；旁目录 ../CVPR2027/results.tex:57–67、discussion.tex:9（独立旧诊断项目，非本稿）
- **代码/实验依据**：../CVPR2027/results.tex:58–62克隆状态渲染；:65冻结双帧状态估计器；:67明确固定相机、无接触、非视觉世界模型；本仓库rollout.py:47–65
- **发现**：已找到旧项目的视觉混叠与相机条件梯度诊断。它并非简单图像分类器，但也不是RGB闭环策略优化；不能移植为EAGM视觉成功证据。此处仅核对旧项目文本定位，未重算其原始视觉日志。
- **影响**：混淆两个项目会扩大证据范围并掩盖数据/内容重叠。
- **现有数据能否解决**：可澄清归属；无法升级为视觉学习证据
- **新增代码**：若升级，需要
- **重新训练**：若升级，需要
- **验证标准**：明确两个项目的共享资产/数据/贡献；保留旧诊断标签，独立运行RGB策略优化并追溯新日志。

## C04 视觉操作与视觉泛化是否缺失 — P1

- **论文位置**：paper_eagm/sections/05_setup.tex:3；07_generalization.tex:3、9；08_analysis.tex:21
- **代码/实验依据**：tasks/reach.py:3–9、32–47；runs/stage7c2_true_ee_current_sites_seed20/results.json $.dir_random.true_return=-0.015713631265686485，$.base.true_return=-0.015448636835407184；decision.json所有launch字段false
- **发现**：只有状态跟踪与改变奖励/目标的Reach。Reach seed20为负面边界，Dir较Base低0.00026499443；不能作为跨任务成功。没有相机扰动/视觉策略泛化结果。
- **影响**：单平台、负面迁移不能支撑普遍机器人/视觉泛化。
- **现有数据能否解决**：能重新表达负面边界，不能产生正面迁移
- **新增代码**：是
- **重新训练**：是
- **验证标准**：独立任务与测试分布；预注册主比较，种子为统计单位；所有失败保留；相机扰动与执行参数变化分别验证。

## C05 是否应增加接触丰富任务 — P1

- **论文位置**：paper_eagm/sections/05_setup.tex:3（明确无接触操作）
- **代码/实验依据**：tasks/reach.py:23–29、32–47只定义EE位置目标；runs/stage7c2_true_ee_current_sites_seed20/results.json有return、fail_rate，无插入/抓取成功判据
- **发现**：当前任务不能验证接触切换下梯度的可用性。MuJoCo资产包含碰撞几何不等于完成了接触操作任务；当前fail_rate是相对初始策略回报下降，不是任务失败率。
- **影响**：接触任务是检验方法适用边界的合理扩展，但不是靠动画补齐的贡献，也非会议统一硬性要求。
- **现有数据能否解决**：否
- **新增代码**：是
- **重新训练**：是
- **验证标准**：先冻结任务成功定义/时间上限/接触稳定指标，验证真实接触发生；独立训练种子报告成功率、回报、误差、完成时间和交互预算。

## C06 是否需要真实机器人验证 — P2

- **论文位置**：paper_eagm/sections/08_analysis.tex:21
- **代码/实验依据**：paper_eagm/sections/05_setup.tex:3仅MuJoCo；training/rollout.py:179–182依赖env.restore(snapshot)；runs/stage7c2_true_ee_current_sites_seed20/manifest.json
- **发现**：没有物理机器人验证；模拟器精确恢复及有限差分标签获取能力不能直接视为硬件可用。
- **影响**：主要限制外部有效性和实用成本，不应将缺硬件实验单独作为必然拒稿理由。
- **现有数据能否解决**：能界定范围，不能验证硬件
- **新增代码**：若实施硬件方案，需要
- **重新训练**：若声明硬件策略性能，需要
- **验证标准**：明确硬件可获得输入与可重复重置/标签方案；先仿真验证预算与误差，硬件实验另行设计和授权，报告失败与成本。不在只读审计中操控机器人。

## C07 五种子证据及统计强度 — P1

- **论文位置**：paper_eagm/sections/05_setup.tex:20–23（p5）；06_results.tex:6–17（p6）
- **代码/实验依据**：runs/stage5a_confirm/seed5..9/per_start.csv；analysis/stage8_statistics.json；paper_eagm/data/core_seed_differences.csv
- **发现**：独立按(episode_id,frame,delay)配对原始CSV重算，Stage5A Dir−Base五种子差值为[0.0002532010238230608,0.00037674246922168717,0.0010238639529658057,0.0007540029128838964,0.00016296600900260846]，均值0.0005141552735794117，与文稿一致。五个同向差值的双侧精确符号翻转最小p=2/32=0.0625。
- **影响**：不否定观察到的一致改善，但不能把bootstrap区间排零等同于精确检验显著；每种子12起点不能充当60独立种子。
- **现有数据能否解决**：可复核现有数字与敏感性，不能扩大独立样本
- **新增代码**：新实验需协议/调度
- **重新训练**：增强确认性需要
- **验证标准**：冻结主指标、比较和排除规则，注册全新种子，做配对种子差值及CI/多重比较校正；各阶段种子不能混池。

## C08 梯度balanced accuracy名称与实现不符 — P1

- **论文位置**：paper_eagm/sections/05_setup.tex:20–23为统计上下文；主稿未将该指标列为主表结论
- **代码/实验依据**：training/eval_metrics.py:18–26、44–55；training/gradients.py:182；scripts/run_stage7c2_true_ee.py:95–97；runs/stage7c2_true_ee_current_sites_seed20/results.json $.dir_random.theta0_gradient.balanced_dir_acc
- **发现**：balanced_dir_acc实际为掩码内所有坐标符号相等的平均值，并未先按真实正负类别分别计算召回再平均。存在类不平衡时不等于balanced sign accuracy。
- **影响**：若将原始JSON该字段直接当平衡准确率，可能产生误导。对只报告回报/余弦的主结论影响有限；需逐个调用路径追踪，而非宣布全部历史结果失效。
- **现有数据能否解决**：若保留原始向量和mask可重算（Reach gradients.json保存g_model/g_true/mask）
- **新增代码**：需修正度量函数或明确重命名并写测试；本轮仅报告
- **重新训练**：通常不需要
- **验证标准**：构造正负类别不平衡、全单类、空mask测试；对旧结果生成另立版本的派生重算报告，不能覆盖原始记录；注明单类时balanced值未定义。

## 负面边界复核

`runs/stage8_3_alignment/decision.json` 的 H1/H2 均为 false，`stage8_4_authorized` 与 `five_seeds` 均为 false；解释为新测试改善与验证冲突。与论文 p8、`08_analysis.tex:18` 一致。不能挑选新测试赢家宣称新算法。

`runs/stage7c2_true_ee_current_sites_seed20/results.json` 的 Dir 相对初始化变化为 -0.0002574463882852331，初始梯度余弦为 -0.13210224967274187；这同时否定在此单种子任务上已经取得梯度与回报改善的说法，但不证明所有可能 Reach 设置必然失败。

主稿 Section 7 的 supplement draftnote 仍是材料待补说明；在缺失正确补充材料情况下，不给“主稿/补充完全一致”的通过标记。
