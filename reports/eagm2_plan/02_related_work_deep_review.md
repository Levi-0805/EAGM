# EAGM 2.0 相关研究深读合卷

范围：14项原始研究/方法，覆盖任务书指定方法与代表性MBPO、RSSM、POMDP、可微动力学和轨迹优化路线。不是穷尽全球文献的证明。每篇按12字段核查，来源与代码访问限制逐条保留；未运行外部方法。原论文已实现功能、可借鉴机制与EAGM缺失功能分别记录。文献结论须结合 `02_novelty_assessment.md`。

# PART 2 核心文献深读：策略梯度与动力学建模

核查日期：2026-11-14。EAGM 对照版本为 412cf8c（研究代码基底9416b92）。本报告是文献与适配方案，不是复现结果；未改研究代码、论文或历史实验。阅读范围包括正文方法、公式、实验/局限和下列官方代码。没有核查全部依赖，也没有运行这些外部仓库。`未显式`仅表示所核查方法未建立该机制，不是对所有版本的不存在证明。

## PAML — Policy-Aware Model Learning for Policy Gradient Methods

来源：[全文 §3 Eq.(6–9)、§5、附录C](https://arxiv.org/html/2003.00030)。

| 要求字段 | 核查结果 |
|---|---|
| 研究问题 | 模型误差应按策略改进用途衡量。 |
| 数学模型 | Eq.(8) 最小化真实与模型诱导策略梯度的 L2 差；两者共享真实环境价值函数。 |
| 网络输入输出 | 状态/动作→下一状态；另有actor及critic。 |
| 训练监督 | 转移、真实数据训练的critic、策略梯度差。 |
| 策略参数梯度监督 | 是；不是EAGM的多尺度模拟器坐标FD标签。 |
| 执行器状态 | 未显式建模B类隐藏执行链。 |
| 动作延迟 | 所核查方法未专门定义FIFO。 |
| 部分可观测性 | 主建模为MDP；无执行belief推断。 |
| 多步策略优化 | 有模型虚拟样本规划；不同实验horizon不同。 |
| 任务/基线 | 有限状态、LQR、Pendulum、MuJoCo；MLE/模型无关对照。 |
| 局限 | critic近似与策略更新后的模型目标漂移。 |
| 与EAGM差异 | 梯度匹配已有先例；EAGM区别在执行命令结构与可重置FD接口。 |

官方代码核对：[models.py](https://raw.githubusercontent.com/rabachi/paml/master/models.py) L274–331：用真实/模型 next state 分别计算 `-critic(s, actor(s))` 的actor参数梯度，真实梯度detach后取L2差；L286固定此训练路径的learning_horizon=1。因此不能把“预测值准确不保证梯度准确”或“匹配策略参数梯度”本身列为EAGM首创。

## GAMPS — Gradient-Aware Model-Based Policy Search

来源：[全文 §3 Definition 3.1/Theorem 3.2、§4.1 Eq.(3)、§7–8](https://arxiv.org/pdf/1909.04115)。

| 要求字段 | 核查结果 |
|---|---|
| 研究问题 | 受限模型容量下减少model-value gradient偏差。 |
| 数学模型 | 加权转移log-likelihood；权重为折扣×轨迹重要性比×截至t各score范数之和。 |
| 网络输入输出 | (s,a)→转移分布；模型rollout估Q。 |
| 训练监督 | 离线轨迹及策略score加权。 |
| 策略参数梯度监督 | 不直接回归真实PG标签；以PG偏差界导出目标。 |
| 执行器状态 | 未显式。 |
| 动作延迟 | 未显式FIFO。 |
| 部分可观测性 | MDP框架，非隐藏执行状态估计器。 |
| 多步策略优化 | 迭代模型学习、模型Q评估、PG更新。 |
| 任务/基线 | Gridworld、Minigolf、MuJoCo Swimmer；ML、REINFORCE、PGT。 |
| 局限 | 全批数据重复使用、rollout累积误差、策略偏离后的IS方差。 |
| 与EAGM差异 | 似然样本加权并非EAGM可微rollout的FD梯度匹配。 |

[作者页面](https://andreatirinzoni.github.io/publications.html)给出[官方代码地址](https://github.com/proceduralia/gradient-aware-code)，但本次工具获取该仓库及README失败，**代码未核验，不能说已忠实实现**。现有EAGM确定性actor没有GAMPS所需的随机策略密度，直接去掉重要性比/score不是原算法。

## VaGraM — Value Gradient weighted Model-Based Reinforcement Learning

来源：[全文 §3.1–3.2 Eq.(3–7)、§4、Appendix E/F](https://arxiv.org/pdf/2204.01464)。

| 要求字段 | 核查结果 |
|---|---|
| 研究问题 | value-aware模型训练的外插与多解不稳定。 |
| 数学模型 | 用真实next state处∂V/∂s的逐维平方加权预测平方误差。 |
| 网络输入输出 | (s,a)→next-state及reward预测；价值网络给权重。 |
| 训练监督 | 真实转移、价值梯度；非模拟器FD。 |
| 策略参数梯度监督 | 否；状态价值梯度不是actor参数梯度。 |
| 执行器状态 | 未显式隐藏执行状态。 |
| 动作延迟 | 未显式。 |
| 部分可观测性 | 主框架MDP；非belief模型。 |
| 多步策略优化 | 接入MBPO类学习流程。 |
| 任务/基线 | Pendulum分析、MuJoCo；MLE/IterVAML/MBPO及容量、干扰维实验。 |
| 局限 | 价值梯度可靠性、局部近似和坐标缩放。 |
| 与EAGM差异 | 保护价值敏感维的预测精度，不直接校准完整policy gradient。 |

[官方 vaml_mlp.py](https://raw.githubusercontent.com/pairlab/vagram/main/mbrl/models/vaml_mlp.py) L302–310区分“先点积再平方”的Taylor VAML和“逐坐标乘积平方”的scaled MSE；L330–360另处理reward/MSE。适配时不可把这两个损失混为VaGraM。论文首页是ICLR2022，arXiv修订为2023；现有bib的2023应标清版本，不能误报会议年。

## AWM — Action-conditioned World Models

来源：[ICML2024正式入口](https://proceedings.mlr.press/v235/ma24i.html)，[全文 §3–4 Eq.(2–3)、§5、§7](https://arxiv.org/pdf/2402.05290)。

| 要求字段 | 核查结果 |
|---|---|
| 研究问题 | 自回归状态路径损害长期梯度传播。 |
| 数学模型 | ŝ_(t+1)=g(s_1,a_1:t)，减少外部自回归状态路径。 |
| 网络输入输出 | 初始状态和动作序列→未来状态。 |
| 训练监督 | 轨迹状态预测MSE，非FD梯度标签。 |
| 策略参数梯度监督 | 不直接监督；通过模型反传优化actor。 |
| 执行器状态 | 无明确B类执行模块。 |
| 动作延迟 | 未显式队列。 |
| 部分可观测性 | 提到HWM分析可推广；未验证执行belief恢复。 |
| 多步策略优化 | 是，长时域BPO。 |
| 任务/基线 | 双摆、one-bounce、Myriad；Markov/HWM、真实动力学BPO及其他RL对照。 |
| 局限 | 理论确定性假设、主要低维域。 |
| 与EAGM差异 | 改梯度计算图结构，而非执行状态先验或梯度标签。 |

重要细节：论文 Eq.(2),(3) actor输入使用stop-gradient；[官方losses.py](https://raw.githubusercontent.com/micklethepickle/action-conditioned-world-models/main/src/brax/arm/losses.py) L68–97用`bp_discount`控制actor输入梯度，并区分input_observations/detach_obs。因此引用其界必须核对梯度估计器，不能把它无条件当EAGM完整闭环敏感度的理论。已核查[训练入口](https://raw.githubusercontent.com/micklethepickle/action-conditioned-world-models/main/src/brax/arm/train.py)的模型、policy、critic分离更新；未运行。

## MBPO — When to Trust Your Model: Model-Based Policy Optimization

来源：[全文 §4.2、§5 Algorithm 2、§6](https://arxiv.org/pdf/1906.08253)。

| 要求字段 | 核查结果 |
|---|---|
| 研究问题 | 模型样本效率与模型偏差的权衡。 |
| 数学模型 | 概率动力学ensemble；从真实replay状态分支短rollout。 |
| 网络输入输出 | (s,a)→next-state/reward分布；SAC actor/critics。 |
| 训练监督 | 转移似然、真实和模型样本的SAC目标。 |
| 策略参数梯度监督 | 无真实PG标签，非EAGM式模型BPTT训练。 |
| 执行器状态 | 未显式。 |
| 动作延迟 | 未显式。 |
| 部分可观测性 | 主要状态MDP，不处理执行belief。 |
| 多步策略优化 | 多轮actor更新，但模型rollout刻意短。 |
| 任务/基线 | MuJoCo locomotion；模型型与模型无关RL。 |
| 局限 | 模型外推误差与rollout长度选择。 |
| 与EAGM差异 | 改模型使用方式，非PG-fidelity监督。 |

[官方mbpo.py](https://raw.githubusercontent.com/jannerm/mbpo/master/mbpo/algorithms/mbpo.py)可核查ensemble、real_ratio、rollout_schedule及fake_env；它是TensorFlow/SAC体系。引入该完整系统会同时改变策略、critic、数据循环，适合系统级基线，不适合作为单一梯度损失消融。

## SHAC — Accelerated Policy Learning with Parallel Differentiable Simulation

来源：[全文 §3.3 Eq.(5–7)、Algorithm 1、§4与附录A](https://arxiv.org/html/2204.07137)。

| 要求字段 | 核查结果 |
|---|---|
| 研究问题 | 接触动力学长期反传的不稳定及局部极小值。 |
| 数学模型 | 短h步可微模拟回报加γ^hV(s_h)终端价值。 |
| 网络输入输出 | 状态→随机actor动作、critic价值；模拟器提供状态转移。 |
| 训练监督 | 可微物理reward梯度，critic TD(λ)。 |
| 策略参数梯度监督 | 梯度用于actor优化，不是世界模型训练标签。 |
| 执行器状态 | 由可微模拟器定义，无未知执行belief。 |
| 动作延迟 | 未专门FIFO建模。 |
| 部分可观测性 | 所核查实验是状态控制。 |
| 多步策略优化 | 是，短窗反传加critic尾项。 |
| 任务/基线 | CartPole、Hopper、Ant、Humanoid/MTU；model-free及梯度方法。 |
| 局限 | 依赖可微物理与critic近似；不保证任意接触梯度可靠。 |
| 与EAGM差异 | 使用解析模拟器梯度，访问权限强于仅可重置的MuJoCo黑箱。 |

[官方shac.py](https://raw.githubusercontent.com/NVlabs/DiffRL/main/algorithms/shac.py) L152–176初始化截断轨迹、env.step连接actor；L362–377对actor_loss.backward并可裁剪梯度。可借鉴短窗/尾价值，但不能把privileged differentiable simulator比较叫同信息公平基线。

## DiffMimic（补充轨迹优化脉络；不是 ∇Mimic）

来源：[全文 §3.1–3.3 Eq.(3.1–3.3)、Algorithm 1、§4](https://arxiv.org/pdf/2304.03274)。

| 要求字段 | 核查结果 |
|---|---|
| 研究问题 | 物理运动模仿的探索与长时域优化成本。 |
| 数学模型 | 可微物理轨迹与reference的加权状态匹配；demonstration replay截断困难段。 |
| 网络输入输出 | 身体状态/phase→PD target角；非RGB策略。 |
| 训练监督 | 参考运动状态和可微物理梯度。 |
| 策略参数梯度监督 | actor使用解析梯度；非PG标签回归模型。 |
| 执行器状态 | 显式已知PD控制律与gain，非未知执行记忆。 |
| 动作延迟 | 未显式FIFO。 |
| 部分可观测性 | 无执行belief估计。 |
| 多步策略优化 | 轨迹反传与演示回放。 |
| 任务/基线 | Humanoid运动、不同角色；DeepMimic/AMP。 |
| 局限 | 参考状态、可微物理和重置依赖；不等于视觉接触操作。 |
| 与EAGM差异 | 已知物理与PD导数，目标是运动匹配而非学隐藏执行导数。 |

[官方环境代码](https://raw.githubusercontent.com/jiawei-ren/diffmimic/main/diffmimic/mimic_envs/humanoid_mimic.py)使用Brax v1与reference状态误差；[官方仓库](https://github.com/jiawei-ren/diffmimic)可查。项目网页bib年份与论文首页不一致，引用优先论文ICLR2023。不得用DiffMimic代替用户点名的∇Mimic核查。

## 对 EAGM 2.0 的创新性判断（本研究团队推论）

已确认的先例足以排除“首次用梯度训练世界模型”“首次重视多步梯度路径”作为贡献。现版本执行瓶颈+FD监督是具体技术组合，但尚不能证明是独立科学贡献：部分可观测执行机制是否导致现有目标失效、因果delay/隐状态建模能否补上该缺口，均需实验。

应把问题收紧为：**在同一物理系统、相同部署可观测信息和查询预算下，执行历史压缩误差如何改变闭环动作敏感度；显式延迟因果结构与可部署记忆是否改善未见策略/执行配置上的梯度和更新效用？** 这是待检验问题，不是预先成立的答案。新的GRU、Transformer和标准有限差分误差界不是自动创新。

## 公平适配建议（提案，尚未实现）

1. **PAML-critic adaptation 为最接近的必要对照。** 固定EAGM相同命令瓶颈、动力学容量及actor，只替换模型目标为官方DDPG路径启发的critic PG差；critic只用训练轨迹/回报，价值函数输入须与部署历史一致，禁止读真实隐状态。先在LQR解析例验证方向与官方目标一致，再转任务。明确短窗Q、history与原论文的差异。critic拟合/数据/计算成本全部计入。
2. **VaGraM-inspired adaptation 作为第二目标对照。** 复用同一critic及可观测状态/历史接口，detach其next-state梯度，逐维平方加权预测误差；任何MSE floor、normalization或gradient clipping都单列。当前部分观测下价值不一定是物理x的函数，因此不能用privileged完整state critic“帮助”该baseline而不披露。
3. **GAMPS暂作条件基线。** 必须引入有密度的随机策略并记录behavior logprob与IS；否则只能叫GAMPS-inspired weighting，不应进入“原算法完整复现”表。保留无法公平适配原因，不以工程困难证明其劣势。
4. **AWM作为结构基线。** 输入相同可观测历史初始化与动作序列；对比相同horizon/容量。stop-gradient/full-feedback两种估计器明确标识，不能偷偷改变闭环导数再将差异归因架构。
5. **MBPO为系统级候选；SHAC/DiffMimic为privileged机制参考。** 不把不同simulator、访问权限或策略类的原论文数字与本任务拼表。

在实验前冻结每项baseline的原文公式—代码函数—适配差异表、参数量、simulator查询账本与调参预算。确认性阶段不得根据test表现撤销强baseline。原论文未显式执行状态并不证明它们不能利用增强输入；必须同时考察相同可观测历史输入版本。


---

# 执行语义、循环世界模型与接触梯度：原始文献核查

核查日：2026-11-14。仅研究方案阶段，无训练或性能复现。下面区分论文已实现、可借鉴机制和 EAGM 尚未实现能力。全文方法段/公式已经实际查阅；代码核查程度逐项列出。未读取到代码不能写成“无此功能”。网页和主分支会变化，正式复现前仍需锁定版本/commit。当前 EAGM 基准为 412cf8c 所继承的 Stage8.3 研究稿，不是缺失的 Final。

## E1 SuperMimic

来源：[原文PDF](https://arxiv.org/pdf/2610.28480)，§3.1–3.3、Eq.(1)–(3)、Appendix A/C；[官方代码](https://github.com/HybridRobotics/supermimic)。原文19页PDF可读，HTML入口404；官方README已核查，src目录网页未能抓取，未完成逐函数实现复核。

| 字段 | 核查结论 |
|---|---|
| 问题 | 多动作人形全身跟踪的训练成本 |
| 数学模型 | 确定性 surrogate `s'=fω(s,a,ξ)`；Eq.(1) k步拟合；Eq.(2) H步回报+critic尾项 |
| 输入/输出 | 模型读状态、动作、sim context ξ，预测状态及观测/回报/终止；部署actor读本体观测+目标 |
| 监督 | logged transition，多步状态/观测/奖励误差及终止BCE |
| 策略参数梯度监督 | 无EAGM式FD匹配目标；actor通过冻结模型BPTT优化 |
| 执行状态 | sim context含摩擦、标定偏移、COM；不等于从历史恢复未知执行器状态 |
| 延迟 | 所读方法/上下文表未建立显式FIFO/未知延迟估计 |
| 部分观测 | 明确POMDP及带噪actor观测，训练model可获特权context |
| 多步策略 | k=8模型窗口，H=32想象+critic |
| 任务/基线 | Unitree G1、LAFAN1/BONES-SEED；PPO类motion trackers；有硬件演示 |
| 局限 | surrogate Jacobian不自动等于真实导数；输入权限和机器人不同，不能直接比较论文耗时数字 |
| EAGM差异 | 前者用forward多步监督获得可微代理；EAGM用可重置仿真FD约束策略梯度、研究部分执行接口 |

可借鉴：在线数据覆盖、多步拟合、actor/model输入权限表。EAGM尚未验证：同级大规模跟踪、sim-context估计、硬件迁移。不能称“SuperMimic依赖已知完整执行器映射”，原文的context字段不能支持这种扩大判断。

## E2 Mind the Execution Gap: Action-Semantic Mismatch in World-Model Control

来源：[原文](https://arxiv.org/html/2610.06582v1)，§2–4、Eq.(2)–(5)、Appendix C–E。实际核查了未来序列、执行回执和训练–接口交互。精确标题检索及原文链接中未找到可确认的官方代码仓库；此为检索结果，不断言作者未发布代码。

| 字段 | 核查结论 |
|---|---|
| 问题 | 世界模型消费的动作与真正驱动转移的动作不一致 |
| 数学模型 | Eq.(3)按最新已到达计划索引选实际动作；Eq.(5) RSSM以applied action更新 |
| 输入/输出 | TD-MPC2计划序列→执行buffer；DreamerV3观测+执行回执→recurrent state |
| 监督 | 主要冻结既有model/actor、改变接口；另有delay训练交互实验 |
| 策略参数梯度监督 | 未引入FD策略梯度标签 |
| 执行状态 | 明确执行buffer、packet索引及可部署receipt |
| 延迟 | 延迟、丢包、乱序及反馈时延均研究 |
| 部分观测 | 纠正action attribution；不是通用未知gain/rate-limit隐状态辨识 |
| 多步策略 | TD-MPC2多步规划；DreamerV3基于RSSM，接口纠正不等于新梯度训练算法 |
| 任务/基线 | DMC；跨MetaWorld/ManiSkill2/MyoSuite；Hold-Last、Fixed Chunk、Receding Buffer等 |
| 局限 | receipt时效重要；与delay-trained模型组合可退化，非自动叠加收益 |
| EAGM差异 | 接口获取已施加动作/保持计划；EAGM需学习未观测执行链路且直接考察策略梯度 |

这是最接近的执行语义先例之一。可借鉴执行回执的因果时间戳以及issued/applied两种输入对照。EAGM不得把“动作不等于执行命令”本身作为新发现；在相同可观测信息下比较估计器，避免给自己的方法真实u而基线只得issued a。

## E3 ∇Mimic（NablaMimic；不是 DiffMimic）

来源：[原文v1](https://arxiv.org/html/2610.41654v1)，§4、§5.1–5.5、Eq.(4)–(5)、Algorithm 1、§8；[官方项目](https://sailab.ai/nablamimic/)；[官方仓库](https://github.com/sailab-ai/NablaMimic)。原作者OamIcs仓库已重定向到sailab-ai。代码README已读，src网页抓取失败，未逐函数复现。

| 字段 | 核查结论 |
|---|---|
| 问题 | 接触人形任务的仿真梯度极端值及计算成本 |
| 数学模型 | 精确仿真adjoint；逐样本梯度相对batch median裁剪；`y=a+δg/RMS(g)`后多次回归 |
| 输入/输出 | 本体/目标观测→关节目标；NablaEngine状态/动作→下一状态和反向导数 |
| 监督 | 仿真return的action gradient生成actor目标；critic TD(λ) |
| 策略参数梯度监督 | 不是FD监督learned model；以action gradient直接改善actor |
| 执行状态 | 仿真actuator限制存在；未见EAGM式隐藏执行状态估计 |
| 延迟 | 所读主算法不建立未知FIFO延迟模型 |
| 部分观测 | actor观测限制与完整仿真求导须分开，不是belief执行辨识 |
| 多步策略 | H=64真实可微仿真窗口+critic |
| 任务/基线 | G1 LAFAN1接触跟踪；PPO、SHAC、SuperMimic |
| 局限 | NablaEngine专门化G1；论文梯度“exact”须区分非光滑/actuator-clamp模式；无本轮复现 |
| EAGM差异 | 不学习世界模型；梯度来源、算力和可访问物理状态均不同 |

版本提示：arXiv v1与当前项目页报告的到达目标回报时间不同（20分钟与8分钟），不得混用版本数字。可借鉴逐样本异常值处理及action-space更新；EAGM尚未实现该可微引擎/策略回归，更不能将其GPU效率数字纳入EAGM结果。

## E4 PlaNet / RSSM

来源：[ICML原文](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)，§2–4、Eq.(2)–(4)、Algorithm 1–2；[作者项目](https://planetrl.github.io/)；[官方RSSM源码](https://raw.githubusercontent.com/google-research/planet/master/planet/models/rssm.py)。已看模型定义与CEM规划段，未运行旧TensorFlow实现。

| 字段 | 核查结论 |
|---|---|
| 问题 | 从像素学习可用于规划的隐动力学 |
| 数学模型 | deterministic h + stochastic s；`h'=f(h,s,a)`；posterior由h和图像，prior由h |
| 输入/输出 | 图像、历史动作→belief；latent rollout预测奖励；CEM输出动作 |
| 监督 | 图像/奖励重建、KL与latent overshooting |
| 策略参数梯度监督 | 无；原版没有actor/value网络，用CEM优化动作序列 |
| 执行状态 | 通用latent state，非显式执行器状态 |
| 延迟 | 主方法无专门FIFO建模 |
| 部分观测 | 是，因果filtering posterior与RSSM |
| 多步策略 | 是多步在线规划，不能称BPTT actor学习 |
| 任务/基线 | 六个像素控制任务，包括接触/稀疏奖励；D4PG/A3C等 |
| 局限 | latent memory不保证辨识任意隐藏执行参数；CEM预算与EAGM actor更新不同 |
| EAGM差异 | 提供历史belief架构先例，不直接监督policy-gradient fidelity |

可借鉴posterior/prior分离与多步一致性。EAGM若用RSSM，创新须在执行状态信息约束和梯度可用性，不能声称首次循环latent模型。

## E5 Dreamer（v1，ICLR 2020）

来源：[全文](https://arxiv.org/pdf/1912.01603)，§2–3、Eq.(3)、Eq.(6)–(8)、Algorithm 1；[官方源码dreamer.py](https://raw.githubusercontent.com/danijar/dreamer/master/dreamer.py)。核查了像素encoder、RSSM与imagined actor/critic路径。

| 字段 | 核查结论 |
|---|---|
| 问题 | 在latent想象中学习长视野视觉行为 |
| 数学模型 | RSSM + reparameterized action + λ-return；对想象价值反传 |
| 输入/输出 | RGB/历史动作→latent；actor输出连续动作分布、critic输出值 |
| 监督 | 图像/奖励重建与KL；想象λ-return训练actor/critic |
| 策略参数梯度监督 | analytic actor gradient存在，但无真实FD gradient-matching训练模型 |
| 执行状态 | 通用recurrent latent，不显式B控制器状态 |
| 延迟 | 未建立专用FIFO延迟模型 |
| 部分观测 | 是，因果representation model |
| 多步策略 | 是，想象rollout + critic尾项 |
| 任务/基线 | 20个视觉控制任务；PlaNet、D4PG/A3C等；包含接触物理任务 |
| 局限 | model bias及representation选择影响控制；不等于可恢复全部隐执行状态 |
| EAGM差异 | 视觉latent actor学习已有先例；EAGM潜在贡献为执行条件下梯度保真监督 |

可借鉴RGB端到端策略路径；EAGM尚未实现的视觉策略不能用旧视觉诊断替代。

## E6 DreamerV3（版本必须固定）

来源：[2024 arXiv v2方法全文](https://arxiv.org/html/2301.04104v2)，world-model/actor-critic章节；[2025 Nature正式文献](https://doi.org/10.1038/s41586-025-08744-2)元数据和检索方法节可读，但整页直开被认证跳转阻挡；[当前官方agent.py](https://raw.githubusercontent.com/danijar/dreamerv3/main/dreamerv3/agent.py)，`policy`、`loss`、`imag_loss`。当前代码与论文版不假设逐参数相同。

| 字段 | 核查结论 |
|---|---|
| 问题 | 固定超参适应多种视觉/状态控制域 |
| 数学模型 | RSSM、预测损失+KL balance/free bits；想象actor-critic和归一化回报 |
| 输入/输出 | 图像/低维观测+previous action→latent；actor/critic |
| 监督 | 观测、奖励、continuation、KL；想象优势 |
| 策略参数梯度监督 | 不以FD策略梯度拟合world model；当前`imag_loss`使用stop-gradient advantage × logπ |
| 执行状态 | 通用RSSM，不默认提供执行buffer/隐藏B参数 |
| 延迟 | 原始系统无专门执行纠正；Execution Gap另行修改action attribution |
| 部分观测 | 是，RSSM；不可直接推断可观测性充分 |
| 多步策略 | 是，想象轨迹训练actor；不能混称所有版本都用v1的pathwise dynamics gradient |
| 任务/基线 | 多域视觉与状态任务，包括DMC、Minecraft；PPO及专用方法 |
| 局限 | 大模型/软件版本预算差异；通用latent不提供执行状态可辨识保证 |
| EAGM差异 | 是强视觉能力基线/架构先例，不是EAGM FD目标的等价复现 |

源码核查关键点：`agent.py` 的 `policy` 用prevact更新RSSM；`imag_loss`约380–384行将动作与优势stop-gradient后计算logπ目标。因此建议明确用“Dreamer v1-inspired pathwise baseline”或固定DreamerV3版本，不能将两者混合包装成忠实复现。

## E7 DCAC / Random-Delay MDP

来源：[论文全文v2](https://arxiv.org/html/2010.02966v2)，§2.1、Definition 3/5、Theorem 1、§4.1–4.2；[ICLR 2021正式入口](https://openreview.net/pdf?id=QFYnKlBJYR)；[官方代码](https://github.com/rmst/rlrd)、[dcac.py](https://raw.githubusercontent.com/rmst/rlrd/master/rlrd/dcac.py)。README中的ICLR2020标注与正式论文2021不一致，引用采用正式会议年份。

| 字段 | 核查结论 |
|---|---|
| 问题 | 随机动作/观测延迟下off-policy多步value估计 |
| 数学模型 | augmented state=(delayed observation, action buffer, delay metadata)；可重采样轨迹片段 |
| 输入/输出 | 延迟观测、动作历史与delay→actor和value |
| 监督 | replay真实reward、多步soft value target；SAC-style actor |
| 策略参数梯度监督 | 不以FD匹配world model；§4.2在action buffer中BPTT |
| 执行状态 | 显式动作buffer及时间信息；非任意隐藏actuator辨识 |
| 延迟 | 是，随机action/observation delay核心问题 |
| 部分观测 | 以增广状态恢复特定延迟系统Markov性；依赖元信息条件 |
| 多步策略 | 是多步value/actor更新，无learned-dynamics想象 |
| 任务/基线 | delay MuJoCo；SAC、RTAC；六种子及90%CI |
| 局限 | 有界delay/可用delay元数据；不能直接套未知gain/extra smoothing |
| EAGM差异 | delay-aware credit assignment先例，不是执行代理的policy-gradient监督 |

官方wrapper有一帧动作延迟偏移：配置min_action_delay=0实际至少1步。适配时必须先统一时间索引，否则对比不公平。其action buffer梯度不应被误写成“完全无多步梯度”。

## 对研究方案的独立判断

1. 执行链路应拆成**已知队列/动作时间线**与**未知物理/执行状态**。Execution Gap和DCAC已覆盖前者的重要部分；单加delay-aware prior通常是必要结构修复，尚不能自动构成独立创新。
2. 以RSSM/GRU恢复历史信息已有充分先例，且EAGM自己已有GRU窗口模型。建议科学问题改为：在相同可观测历史和查询预算下，执行状态不确定性如何影响局部策略梯度，并能否用因果的状态估计与策略局部监督减小该误差？这是待检验问题，不是已成立贡献。
3. 视觉任务必须让图像真实进入action路径；但不应先把大视觉网络全部参数做坐标FD。先预注册可监督子空间/动作方向，计入标签预算，再判断扩展是否成立。
4. 可借鉴∇Mimic的异常梯度处理，但“更接近真实梯度”不等于“更稳定的更新”。EAGM应同时报告梯度、更新范数/方向和实际收益，保留二者冲突。
5. 比较权限：SuperMimic训练model获得sim context；Execution Gap获得execution receipts；DCAC获得delay元数据。若EAGM不给这些信息，就不能直接用论文数值比较，更不能把信息优势当方法优势。应分别报告相同可观测接口的adaptation与privileged oracle。

未完成项：SuperMimic/NablaMimic核心src逐函数审计、Execution Gap官方代码定位、所有官方commit锁定与本机可运行性验证。以上不影响已核查原文方法的定位，但阻止声称已完成这些方法的忠实代码复现。
