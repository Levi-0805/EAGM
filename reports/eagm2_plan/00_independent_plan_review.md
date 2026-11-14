# EAGM 2.0 研究方案独立复核（Reviewer B）

范围：只读 `reports/eagm2_plan/03_eagm2_method_design.md`、`03_architecture_comparison.md`、`03_ablation_plan.csv`、`03_mathematical_derivation.tex`、`04_experiment_protocol.md`，结合当前控制器/rollout接口核对。尚未实现、未训练、作者未批准；本报告不把方案当结果。未修改这些文件。

总体判断：可进入作者方案讨论，但不宜把当前草案直接当可执行确认性预注册。以下8项是具体的冻结/实现门槛；不要求现在启动新实验。名义FIFO索引、非均匀三点FD权重及误差界、aliasing恒等式、敏感度递推和更新下界经手工代数核对，未发现数学证明错误。尤其赞同明确共同坐标假设，避免把完整状态定理直接套给低维latent。

## P1-01：G因子定义与确认矩阵不一致

位置：03方法§4（78–84行）、§6（104–106行）；04协议63、75、88行。

03的G指旧Huber、direction、norm、联合目标之间的升级；04的八格却是Base/Dir有无旧监督，且“原”明确保留旧损失。若开发选了新联合损失，现有八格既不能确认新G贡献，也无法保持既有Dir参照。03按mean return排序，04先按模块数量再D/R/DR排序，也不是同一个选择器。

修正：用可机读定义固定 G=0/1 具体含义。如八格只检验旧Huber有无，就把G升级降为开发探索；若新G入选，新增固定旧Dir参照并重算方法总数。保留一个唯一candidate roster和排序规则，03直接引用04。每个候选指定task、objective、gradient loss，而不是只用D/R名称。

验证：从experiment matrix自动计算八格因子编码、主候选和reference，检查不存在同名异损失；给虚拟开发指标测试排序器，不使用真实确认数据。

## P1-02：G-only imagination未写出从prior到下一步command的闭合递推

位置：03方法63–70行。

定义command读posterior h_t，G输出prior h^-_(t+1)，但G-only rollout下一步究竟直接把h^-当h、还是执行Q仍不确定。一致性项鼓励latent接近不能代替实际计算图。若真实部署actor还使用Q估计的执行记忆，而想象actor只用G prior，也需明确两个闭环策略的差异；有限历史/递归估计的初始h与优化参数依赖也必须固定。

修正：在方案写三条可执行递推：起点由过去记录Q得到h_0；imagined第k步command和actor各自读哪个latent；下一步明确令h_(k+1)=G(...)，或固定采用Q(predicted observation)。同时固定多步损失是否全程open-loop、是否detach各段、burn-in长度/梯度边界。两种方案可作开发对照，但确认前只能选定一套语义。

验证：相同初值的独立reference rollout逐步比对；改变未来真观测不改想象轨迹；autograd/FD在latent递推全链路一致。这个问题可在实现前用伪代码解决。

## P1-03：接触任务10秒成功目标与H8标签的时间跨度尚未连接

位置：04协议33、45、63行；03方法70、90行；当前 `sim_env.py:63,134–137`。

旧control_dt约1/30秒，H8只有约0.27秒，H16约0.53秒；新插入任务需10秒内完成并保持0.5秒。沿用旧局部28参数tracking actor和H8 reward gradient，不会自动优化整个插入成功率；接触task也未指定actor adaptation及trajectory reference来源。这个能力缺口比“是否开启contact”更大。

修正：明确T2/T3部署是单一冻结策略跑完整约300控制步，还是receding horizon；指定训练标签的目标长度、终端value/子目标或分段采样机制。若保留H8，只将它称局部代理目标，并把成功收益作为实证外部目标；开发gate必须显示该代理对已定义插入过程有可测更新作用。记录control_dt实际值、horizon和训练/评估目标差异。

验证：手工可达策略只验证环境；另做开发集局部gradient更新与完整episode成功/回报的一致性诊断。按实际H重新计FD预算，不能沿用1344 steps估计长任务。

## P1-04：接触场景snapshot维度变化不是必然，应改为实际模型派生能力

位置：04协议43行。

“修改MJCF会改变状态维度”过于绝对：增加静态fixture和固定工具geom未必改变nq/nv/nu或integration state width；但模型hash与接触/solver上下文确实变了。沿用固定197断言不合理，不代表必须凭空增加状态。

修正：写成“修改MJCF可能改变状态维度；始终由当前mj_stateSize及spec派生大小，严格检查新模型hash，禁止历史hash豁免”。完整保存controller状态、RNG与MuJoCo积分/求解器相关上下文；静态几何也应触发hash隔离。

验证：新task自己的snapshot replay/FD重复性；测试同维度不同MJCF也必须拒绝交叉restore。这是窄文档修正，不应阻塞开发。

## P1-05：固定数据消融与等交互预算优化尚有冲突

位置：03方法102行；04协议65、97–99行。

G=0与G=1共用数据、同训练steps时，G=1额外消耗FD；若把unused labelbank也计给G=0，得到的是固定信息/计算干预对比，不是有效利用同预算的竞争。Direct-FD/SPSA“同训练交互上限”也未说明是否先扣掉它们不需要的world-model数据，以及视觉共享预训练如何扣除。数字未冻结本身允许，但计费逻辑现在就应一致。

修正：区分两条结果轨道：(a)固定相同data/updates的机制消融，真实报告成本差；(b)严格总训练query预算下的性能比较，Base可将剩余label预算分配额外动态数据/训练（规则仅开发确定），FD/SPSA用完整可用优化余额。视觉共有预训练向每个方法同额计入，但不得把未用标签视作方法必须成本。标明主比较属于哪条。

验证：每方法ledger相加恰好等于总cap或注明实际未使用；编写小预算调度测试，无法负余额/超cap；成本表分别列共享实际成本与逻辑归属成本。

## P1-06：确认集初始化latent/cache的历史来源与费用还没固定

位置：03方法FIFO cache及70行；04协议35、37、97行。

“独立episode一个起点”可以是reset或warm-up后中途snapshot，两者执行别名/hidden lag分布差别很大。若所有评测都在reset，B内部状态常统一到qtarget，记忆能力几乎没有受测；若中途截取而不给因果burn-in，latent模型无法初始化。反过来从snapshot读取真实hidden state会越权。

修正：固定起点frame分布、warm-up行为策略/动作序列、提供给各方法的过去记录窗口、latent reset语义及缓存warm-up。所有方法相同episode与warm-up，不需使用过去记录的模型可以忽略；warm-up costs单列。训练/测试trajectory隔离应覆盖warm-up，不只覆盖被计分片段。

验证：test histories属于该episode且全部早于起点；full snapshot供物理环境restore，deployable init只读合法history；包括相同x/c_obs不同hidden-memory起点的开发诊断。

## P1-07：视觉主比较同时改变执行记忆的模型/策略角色，需固定归因

位置：03方法58、92、96行；04协议51–57行；03架构V-Mem。

RGB+recurrent+G中的recurrent可能是world-model execution latent、policy memory，或两者；当前actor adapter公式只有records而没有明确h。若参考policy不具记忆而候选policy读h，回报提高可以来自策略信息优势，不能全归因更准确world-model梯度。共享32维adapter不足以控制这一点。

修正：主比较保持相同部署policy输入/记忆结构，仅改变world-model execution representation；若要验证policy memory作用，另设因子或明确主比较是联合系统且收窄归因。写清当前图像/过去图像/历史动作分别进入policy与model的哪些模块，固定base encoder+adapter是否完全共享初始化和预训练目标。

验证：policy输入合同及模型输入合同分别测试；只替换world model时theta=0两方法在同episode产生同一初始动作/return；有意不同policy结构的结果单列。

## P2-08：secondary Holm4尚不能覆盖文档列出的多任务、多指标机制检验

位置：04协议88–89行。

4个contrast×return/cosine×T1/T2已经至少16项可能检验，若再涉及T3会更多。写Holm4但不指定每项endpoint/task，容易在多个指标里挑显著者形成机制故事。

修正：冻结四项确切task+metric+contrast；其它组合仅描述性，或把所有拟主张的机制p值放同一明示family。D×R交互应写具体均值差公式及对G平均还是条件于G=1。只观察回报与cosine都改善仍不能证明中介因果，当前警告应保留。

验证：统计配置枚举所有检验，family长度程序断言；主稿每个显著性说法能够回溯到唯一预注册键。

## 结论

无需为本次设计审查运行训练。建议先修正01/02/03/05/06/07的方案歧义，再提交作者批准；04是小范围文档准确性修正，08可在确认性冻结前锁定。已有理论推导可保留为条件性设计笔记，不能提前升级为新方法定理或有效性证据。
