# 冻结开发流水线独立只读复核

复核日期：2026-11-14。范围：当前冻结 `eagm2/{data,models,training,gradients,rollout,evaluation,contact,visual_training,budget_controls,extra_data,runtime}.py` 与开发 runner、batch runner、分析脚本。只读源码和工程测试定义；**未读取 development_v1 的效果结果，不据部分结果选择方法**。未修改任何冻结源码或测试。主审告知启动前全套 78 测试通过；通过测试不等于本报告覆盖了所有边界。

结论：本次代码复核未发现足以要求立即停止全部开发训练的 P0（如反向优化、测试输入用于训练、方法间不同评测起点、确认根种子启动）。这不是无缺陷认证。发现一项确定的后处理阻断及若干科学解释/计账边界，均在下列记录。已立即告知主审，主审决定保留冻结训练，待批次结束后单独版本化修复分析，并保留自动分析失败记录。

## 明确问题与解释限制

| 严重性/范围 | 代码证据与现状 | 影响和后续处理 |
|---|---|---|
| P1：自动分析必然报错 | runner 默认 `methods=None`，完整列表在 `config.resolved_methods`；分析脚本第124–125行读取 `methods` 后执行 `list(None)` | 九根训练结果不会因此失效，但末尾自动分析将 TypeError。批次后新版本读取 resolved_methods 并兼容旧配置；保留原失败日志，不在运行中修改冻结源码 |
| P1：contact 时间边界 | `eval_h=ceil(10/control_dt)-32`，dt=.034，恢复frame32再走263步，终态frame295即10.030s；`ContactEnv.step`仅当 `frame*dt<=10` 才首次记录success | 最后10.030s步不能首次赚取成功，但其reward和末态几何进入评测，contact_fraction/force列表也包含它。最后合法成功采样时刻9.996s。不能将return称精确10.000s截断回报。当前结果按既有定义保留；未来协议须固定floor/ceil及终末步奖励规则，单独测试9.996/10.030s边界 |
| P1：成功保持离散采样 | within按每个control-step末态累计dt，需15个合格样本=.510s；几何/penetration是当前步末态判据 | 这是控制频率采样的保持判据，不证明每个物理substep都满足条件。测试有脚本成功与snapshot replay，但未见逐阈值±epsilon、超时与穿模后恢复的完整边界矩阵。后续补测；本批次不可事后改变成功标签 |
| P1：失败时精确交互成本 | `physical`和普通`collect`在`env.step`成功返回后增加completed_steps；若内部已积分再异常，失败那次调用未必计入 | 正常完整调用计数正确；失败partial物理成本仍可能低估。extra_data有issued-call补账，direct optimizer有预留预算并另存actual，二者不能冒充精确已执行physics substeps。失败恢复应保留状态而不把预留全额称实耗 |
| P1：资源/归属报告不完整 | runner有training_simulator_budget、direct_controls_resource；分析当前筛选预算字段靠control_steps等字符串，collection/gradient_labels/extra_collection不会全部进入长表；未读取独立direct_controls_resource | 原始日志仍在，但06报告后处理需要补读。预算实际共享采集与每方法归属分别列；CUDA elapsed不是GPU活跃核时间。critic/视觉公共参数、训练wall与峰值显存不可省略或重复累加 |
| 科学范围：warm-up与memory | 数据warm-up是theta0 actor加独立随机噪声，不是纯随机动作；历史窗口32条含当前观测，在frame32仅覆盖frame1..32 | 配对且因果，但不等于协议文字所述从reset逐条更新的完整latent。burnin首条从零开始；最初recurrent transition的前一filter用窗口首条近似，该近似会影响delay=2的首个burnin transition（即使alpha=.5），之后使用正确窗口缓存；其影响经后续memory递推传播，大小尚未量化。必须称有限窗口估计，不能声称精确hidden重建 |
| 科学范围：固定执行条件 | `data.CFG`为gain1.05/extra_lag1/rate.045/smooth.25，nominal delay按0/1/2循环 | 该批次不是gain/smoothing/rate/extra-lag OOD覆盖。无隐藏状态负对照、真正随机shuffle、privileged完整实验、RGB-only/state信息对照不因类/API存在而视为完成 |
| 科学范围：模型指标 | model_metrics最多取256个等距sequence，仅评测到训练horizon（tracking H8，contact/vision H32） | 不是全test sequence汇总，也不自动给tracking H16预测误差；delay/episode覆盖清单、逐delay回报、policy path质量需要额外描述性后处理，不能把theta traces当已评估policy path |
| 科学范围：梯度诊断 | gradient_metrics会删除非finite坐标再算cosine；evaluate_gradients过滤undefined probe后求均值 | raw记录有finite覆盖、undefined，可追溯。但“full coordinate”必须同时报告nonfinite与覆盖；出现非finite时不能暗称完整向量cosine，不能仅展示幸存均值 |

## 已检查的关键路径

**梯度方向与目标。** imagined/physical均累加`.99**k`的奖励；tracking使用对应Torch/NumPy reward，contact使用几何Torch reward与真实物理reward。FD取实际奖励的导数；模型策略优化执行`(-J).backward()`后Adam最小化，Direct FD/SPSA写入`theta.grad=-g`，方向为增大奖励，没有观察到正负号反转。G监督对相同theta、同start、同局部horizon匹配；非对称/边界FD使用可行三点导数而非悄悄裁剪中心。Vision imagined reward是训练集拟合的latent reward，FD/evaluation用真实reward，所以其误差包括reward approximation，不可全归因执行表示。

**配对起点和数据。** 每个根种子一次采集并共享所有方法的训练/validation/test起点；test每episode一个frame32起点。contact/vision训练frame32/64/96与delay交叉；模型与direct的优化都取同一前6个train起点。FD每次physical先restore同snapshot，包括Controller B与contact计时/成功状态。共同初theta为零；vision共享冻结的CNN/base。Base-extra-data沿用同策略起点，仅增加train episode并重拟合自身train normalization，不能将其NMSE分母与其他方法不同的事实隐去。

**因果执行记忆。** 部署/imagined memory输入公开x/c/previous-action历史；recurrent逐步使用自身transition，window使用自身预测next_x/next_c。没有从snapshot读取hidden状态初始化模型的路径，也没有未来真实图像进入imagined rollout的路径。snapshot只用于真实模拟器与标签。训练burnin可反传，imagined默认detach初始burnin；这限制G对posterior模块的直接监督，不是未来信息泄漏。

**真实预算。** 模型固定数据/更新的机制比较允许G=0少用FD标签。Direct每种获collection_train+labels_train控制步上限，并按完整iteration执行；FD与SPSA更新数可不同且最多50，剩余额度保留，不等于必然耗尽。Direct仍需要共同train起点（以及vision共同encoder）作为公共资产；其采集/预训练成本须在共享公共成本栏披露，不能称无采集成本的独立系统。Base-extra-data按labels_train上限采集额外train-only数据，余数不足合法历史时未用额度明确记录。PAML/VaGraM共享训练数据critic、零额外模拟查询，但额外训练和参数成本真实存在。

**任务成功语义。** 成功要求深度20–36.5mm、横向≤2mm、角度≤5°、当前penetration≤1mm并累计保持；成功一旦达到即锁存，后续退出不会取消。contact_fraction/force从reset历史累计（包括共同warmup），不是仅成功保持段统计。CompletionTime是reset起绝对时间，非policy接管后时间。右臂名义保持，只有工具—fixture碰撞启用；不能写成双臂抓取装配或完整全身接触安全验证。

**批次授权。** batch明列9探索根种子，runtime再次check_seed；45确认根种子禁止。每根前比较源码hash，非零worker或manifest失败停止后续根；不允许同attempt覆盖。当前冻结检查可追溯源码但无法把全部外部mesh/系统库运行时内容都自动冻结；环境与外部资产hash覆盖仍需验收。

## 未验证与后续验收

本复核未启动额外训练、未访问partial效果、未改协议或标签。未独立重跑78测试；未实施完整contact成功边界穷举、每个物理substep穿透审计、实际9根checkpoint恢复继续训练、最终episode/hash配对或完整成本对账。须在实际结果齐备后按逐seed原始值、缺失/失败/undefined、容量、查询归属与任务门槛验收。n=3只作探索描述；效果弱、no-go、容量/预算不可比或机制不支持均可成为结论。确认草案继续 `not_authorized`。
