# 创新性评估与研究方向决定（方案阶段）

**结论：已有证据尚不能确认 EAGM 具有足以独立成立的新方法贡献；这不等于证明它没有潜力。** Tracking内部对照为真，梯度匹配、历史记忆、延迟接口与可微想象各自已有明确先例。不能靠组合这些模块就宣称超越已有工作。全文定位和12字段对照见02_related_work_deep_review.md与02_method_comparison.csv。

## 最接近的重叠

1. [PAML §3 Eq7–9及官方models.py](https://arxiv.org/html/2003.00030)：直接策略参数梯度匹配已有先例。EAGM区别在resettable simulator FD、执行命令瓶颈与输入权限；这些差异是否有科学必要性须验证。
2. [GAMPS §4.1 Eq3](https://arxiv.org/pdf/1909.04115)与[VaGraM §3 Eq7](https://arxiv.org/pdf/2204.01464)：decision-aware拟合和梯度加权早已有之；不能把“前向误差不够”作为首创。两者监督对象不同，须准确区分。
3. [AWM §3 Eq2–3](https://proceedings.mlr.press/v235/ma24i.html)：改变模型结构与梯度路径已有理论和实证；其论文actor输入的stop-gradient约定与EAGM完整闭环敏感度不同，不能借用定理直接证明当前实现。
4. [Mind the Execution Gap §3 Eq2–5](https://arxiv.org/html/2610.06582v1)：动作语义、执行buffer、applied-action attribution已有直接先例。因此“动作不等于执行命令”不是独立创新。
5. [PlaNet/RSSM](https://proceedings.mlr.press/v97/hafner19a.html)、[Dreamer](https://arxiv.org/abs/1912.01603)：递推隐状态与图像想象学习已有先例；EAGM仓库本身已有GRU。新增encoder、latent或GRU不是贡献证明。
6. [SuperMimic](https://arxiv.org/pdf/2610.28480)、[∇Mimic](https://arxiv.org/html/2610.41654v1)：分别体现learned differentiable model与可微仿真直接actor更新；其特权model输入/可微引擎权限不与EAGM FD访问等同。∇Mimic并非DiffMimic；论文/项目页训练时间版本不同，不能择优引用。

## 建议的独立科学问题

在相同可部署输入与交互预算下，**执行状态别名如何引入闭环导数误差，哪些因果执行信息足以减小这一误差，改善是否能转化为未见策略/初态上的物理更新效用？**

此问题比“更准确的世界模型”具体，也比“给世界模型加GRU”严格。可以由别名反例、名义因果路径数值检验、执行/物理导数干预、D/R/G交互和真实回报共同检验。理论笔记只给条件性标准推导，不预先称新定理；latent充分性和误差界假设在现有模型上仍未证明。

可支持贡献的最低证据：

- 精确区分nominal与executed，delay即刻Jacobian路径与控制器一致；不把此工程一致性单独当重大创新。
- 在同输入/容量下，修复后的窗口GRU、persistent latent、无时间信息MLP及privileged参考形成可解释差异；隐藏执行机制关闭时收益变化能支持机制解释。
- PAML-critic adaptation与VaGraM-inspired adaptation公平实现并计入critic成本；保留最接近目标的对照，不能只和弱MLE比。
- 梯度收益与回报收益在独立确认种子共同验证；如果只有回报上升，应调查容量、数据或优化器解释并限制gradient-fidelity归因。
- 视觉contact若通过开发与确认，提供执行信息和图像互补证据；不把它改成无执行主线的普通视觉操作论文。

## 基线适配决定

PAML是必要的最近目标适配：真实next-state与模型next-state处critic所诱导actor梯度差。VaGraM是第二个不同目标基线。当前actor是确定性的，故GAMPS的score/IS不直接成立；不删除这些因子后冒称原算法。MBPO、Dreamer等作为范式与可扩展性参考，不将其原论文数字搬入EAGM排行榜。若后续选用完整外部agent，需额外冻结策略类/交互权限与预算。

14篇原始方法及所列源码已经核查到相应段落，但存在明确访问限制：GAMPS仓库未获取；Execution Gap未确认官方代码；SuperMimic/NablaMimic核心src未逐函数获取。不能宣称完成这些方法的代码复现或穷尽创新性排查。正式基线开发前固定代码commit与许可。

## 决策边界

先批准开发实现与正确性验证，再冻结确认。若D/R修复后的开发证据仍无回报与梯度共同改善，停止把它包装为EAGM2新算法；可保留严谨的负面/机制研究，并重新评估投稿定位。不能为达Weak Accept评分预定正结论。

原稿references.bib本轮不改：新参考先在本方案以原始来源链接留档，论文改写阶段按实际采用的主张、版本与作者元数据统一更新。这遵守“仅在确认需要时修改”，不把文献建议提前写进原稿。
