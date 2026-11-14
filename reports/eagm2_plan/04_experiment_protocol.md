# EAGM 2.0 实验协议 v0 — 待作者确认，尚未运行

基准 `412cf8c`（Stage8.3稿+PART0/1审计）。本文件是可审阅方案，不是已经完成的预注册实验或结果。作者批准后，在独立 `EAGM-2.0-full-validation` 分支实现；开发种子用于可行性/参数选择，所有确认性数据访问前冻结最终协议、代码、环境、模型容量与完整查询预算。正式确认开始后不得改主要指标、排除规则或选择赢家。

指定 Windows Final 与正确补充仍缺失；研发可从所见版本推进，最终论文认证须取得相应文件或由作者明确选择现有研究稿作为重构基准。旁目录诊断论文不作为 EAGM supplement。

## 1. 科学问题及可证伪假设

核心问题：可观测执行记录不足时，模型遗漏的执行历史如何改变闭环策略梯度；更准确的因果执行表示能否在同样查询/容量预算下改善真实模拟器策略更新？

- H-D：修正 nominal delay 的即时作用路径，会降低逐delay命令误差/Jacobian误差，而非仅由残差容量改善预测。
- H-R：在隐藏 rate-limit/smoothing/extra-lag 造成状态混叠时，正确对齐的执行记忆优于同容量无时间信息对照；在无隐藏状态的负对照中收益应缩小。有限窗口GRU已存在，不能把加入GRU作为新贡献。
- H-G：梯度监督在给定执行表示后改善未参与标签生成的策略/初态上的梯度与回报。回报提升但cosine无提升时，只报告性能，不归因为gradient fidelity。
- H-V：视觉观测与执行历史具有互补信息；同视觉编码器、同策略参数化下，RGB+execution memory对延迟/隐藏状态变化更稳健。此项只有真实RGB部署策略才可检验。

执行表示(D)、记忆(R)、梯度目标(G)采用2×2×2设计，G首先固定为旧normalized-Huber的开/关，辅以容量匹配及输入信息负对照。新的direction/norm/calibration目标先属开发候选；若拟作贡献，确认前须增补其独立消融格、预算和检验family，不能仅凭开发结果进入论文。所有“能提升”均待检验，不预定组合模型为赢家。不同任务使用不同根种子，不混池。

## 2. 先修复审计缺陷，并保留原版复现轨道

1. 保留冻结原版 Base/Dir 的实现、配置及原始日志；新种子重新运行，不将新代码输出称为旧版本复现。
2. 独立测试确认 HistC 的 x/a/c 时间约定；修复版历史编码采用明确的 `(o_t,a_t,o_{t+1})` transition约定，训练、teacher-forced测试与imagined更新一致。旧错位结果保留并标记。
3. 隐状态probe按episode划分train/validation/test，归一化仅fit train，重新拟合；probe不得作为主模型输入。主模型隔离与probe隔离分开验证。
4. 把普通sign accuracy与balanced sign accuracy分开；类别缺失/空mask返回未定义并报告覆盖率。历史派生重算另存，不覆盖JSON。
5. FD记录实际左右步长，区别symmetric central/asymmetric secant/one-sided；低梯度和不稳定mask不冒充可信标签。报告受影响路径，不自动否定旧Stage5。
6. 当前环境MuJoCo元数据为3.5.0，旧环境报告曾为4.0.1；先核对实际import版本、模型hash和原版数值复现，不能承诺bitwise一致。

这些修复不直接计为方法创新。修复测试失败时不得启动对应模块训练；不得靠删除失败测试放行。

## 3. 三类任务与开发门槛

### T1 原有双臂tracking

使用归档pilot effective MJCF和原 Controller B；14维动作、28参数actor。先复现冻结 Base/Dir：3000 world-model updates、50 policy updates、H_imag=8、H_eval=16、lambda_u=1、lambda_d=.1。原版复现采用原始protocol语义；EAGM2统一协议中所有对照共用新采样/输入定义。不得把两套协议数据直接池化。

新确认性评估固定每seed至少30个独立episode起点（每个nominal delay 0/1/2各10个），均独立于train/val；每个episode只贡献一个主评测起点。主指标为seed内30起点平均物理回报；报告原始回报及相对共同初始策略delta。delay=2-only历史结果另外列，不能当全delay泛化证据。

统一评测起点采用每个独立episode在reset后执行32步预先固定的限幅随机warm-up，再开始计分；warm-up行为RNG独立于方法，所有方法看到相同的合法过去记录。latent从reset零状态通过这32步观测更新，FIFO缓存只读这些历史filter；训练/test的warm-up也严格按episode隔离。warm-up交互单列，不能从完整snapshot偷取B隐状态初始化部署模型。主结果不混入reset-only起点；reset-only另作描述性负对照。开发：3个根种子；检查reset/边界/alpha/历史一致性；初态随机化依预设位置和速度范围，落在合法关节限位。每个seed train/validation/IID-test由不同episode生成，不能把单条trajectory切片分成训练和测试。具体采样分布在开发可行性检查后写入冻结JSON；任何看过的开发test仍属开发数据。

### T2 接触任务：OpenArm工具固定式 peg insertion 候选

优先复用现有OpenArm，左末端刚性固定圆柱工具，静态插孔fixture，右臂保持预设安全姿态；这是单臂工具插入验证，不包装为双臂协作抓取或完整抓取-插入。若开发无法实现可重复接触/成功，停止此候选并在新的开发协议中记录替代任务，不能使用确认种子寻找容易任务。

现有ArmEnv默认 `enable_contact=False`（vendor/legacy_sim/sim_env.py:66,110–112），新增任务必须显式开启contact。修改MJCF可能改变状态维度；由当前mj_stateSize及spec计算大小，不能沿用CompatibleArmEnv的197维snapshot断言或历史snapshot哈希豁免。即便维度相同，不同模型hash也禁止交叉restore。

拟定验收定义（开发阶段验证几何一致性后冻结）：在10 s内轴向插入深度≥20 mm，横向偏差≤2 mm、轴线夹角≤5°，持续0.5 s，且不是穿模成功。孔径/工具半径/公差与成功阈值一起冻结；深度沿fixture坐标轴计算，不能由奖励阈值反推成功。起点位于孔口上方、姿态/位置随机化；示教或控制器必须在开发验证存在可达成功路线，不能用其成功宣称学习成功。

时间/策略接口：请求control_dt=1/30s，实际周期为round(requested_dt/physics_dt)×physics_dt；以真实累计仿真时间10s截止，步数ceil(10/actual_dt)，约300步；保持0.5s也用实际时间计。下文300步为约数，日志和FD horizon用实际冻结整数。T2沿用28维增益/偏置actor，但reference改为固定的对准—下插—保持轨迹，由已知fixture几何和名义机器人模型离线构造，所有方法相同；不是从测试真实状态生成的oracle动作。开发必须先验证actor类可以达到成功。训练采用H32的密集物理奖励局部代理目标和其matched-theta FD，而不是把H8梯度叫10s成功率梯度；模型/FD都优化同一H32目标，部署冻结策略连续执行300步。起点覆盖插入各阶段的训练episode片段，独立确认仍按完整episode计分。开发需同时检查局部梯度更新、完整episode回报和成功率：T2/T3候选需success差均值>0且至少2/3种子非负，才能进入确认。若短目标无法改善完整任务，记no-go；不得事后把确认指标换成局部reward。T3同H32训练/300步评测，32维adapter，不把其网络与旧28维actor称同一架构。

指标：SuccessRate（主）、MeanReturn、末端横向/轴向误差、CompletionTime（全部样本按10s上限截尾，并另列成功条件均值）、ContactStability（保持段接触占比/力或冲量波动/穿透与反弹）、每次训练/标注交互、成功率随累计交互。成功/超时/数值错误分别计数。奖励由距离、姿态、动作平滑及接触异常代价组成，具体权重仅在开发固定；不得用每个方法自己的surrogate reward评价。

### T3 RGB策略优化（与T2任务共享定义，但不共享确认种子）

部署策略必须只接收图像、任务已知目标与声明可观测执行记录/历史；state对照和privileged参考明确分列。RGB-only不输入q/qdot/模拟器latent；RGB+record只允许实际日志中的target/filter/delay及历史动作。视觉输入须在每次动作计算图中生效，不是仅作分类/诊断。

拟用64×64双帧或短序列RGB、小型CNN（从头训练），固定encoder/trunk后以32维policy adapter做初期EAGM梯度优化。encoder仅使用训练数据学得，预训练数据/交互/耗时对所有视觉方法一致计费；确认集上不得更新encoder。这个分阶段方案可称“RGB策略adapter优化”，不能称完整端到端视觉网络的全参数梯度优化。若要端到端主张，须额外设计随机方向或低秩梯度监督，独立登记。

禁止world model偷读未来真实图像：imagination只能使用自身预测的latent/观测与可部署递推状态；完整模拟器状态仅oracle和训练标签通道可用。图像输入隔离测试必须检测q/qdot隐藏字段的置换不改变RGB前向、改变RGB确实改变可学习输出、未来帧不影响当前输出。

至少30独立测试episode/seed；相机偏移±5 cm、姿态±5°及光照变化作为预先定义的单因素压力测试（须开发检查场景仍可见），不能与执行参数变化混为一项。主要视觉比较为RGB+recurrent-world-model+G vs RGB+observable-record-world-model+G：两者部署policy固定为同一RGB+当前可观测record的CNN+32维adapter，不让actor读取world-model latent，初始theta=0时动作与回报应一致；只改变world-model执行记忆，因而能隔离模型表示。RGB-only/state/privileged是信息权限对照，不和这个主比较混为纯梯度效应。图像历史进入world model，主对照actor只用相同双帧RGB/records；另外用recurrent不加G区分执行记忆与梯度目标。

## 4. 对照与配对

T1/T2确认矩阵各16方法条件×15种子（240 method-seed runs）；T3为6条件×15种子（90），上限570个method-seed runs；FD/SPSA无需训练world model，不能把该数字叫570个神经网络训练。开发筛选/复现/失败重跑另记。

T1/T2条件：原Base、原Dir、Delay-Base、Delay-Dir、Recurrent-Base、Recurrent-Dir、Delay+Recurrent-Base、Delay+Recurrent-Dir、capacity-MLP、capacity-GRU、PAML-critic adaptation、VaGraM-inspired adaptation、Direct-FD、SPSA、Base-extra-data、privileged-state reference。确认矩阵中的“原”指原模型行为和损失，在统一新评测协议下；严格历史复现是单独开发轨道。

T1/T2的deployable model比较使用相同可观测信息权限、共同训练/测试episode、共同初始策略和数据顺序。不同架构不能伪称完全同权重初始化；共享形状模块复制同参数，不同模块采用同初始化分布与配对RNG。容量控制目标为候选±5%，报告精确参数/FLOPs；若未达标先调整宽度，不能静默称匹配。

PAML为最近目标基线：依据官方代码，用真实next-state与模型next-state下的critic驱动actor参数梯度差训练模型；不能把EAGM的FD损失换名。当前确定性actor/执行接口属于adaptation，不声称完整原论文复现。VaGraM采用价值状态梯度逐坐标平方权重，同一可观测表示/数据拟合critic。Base-extra-data将对应任务Dir的标签查询额度（T1 16,128；T2 H32为64,512，在相同12起点/3尺度下）用于额外动力学训练数据，与Base结构相同，作为预算公平性对照。critic额外参数、训练更新和收集查询计账；不同观测充分性假设单列。GAMPS需策略密度与重要性比，不能直接嫁接确定性actor冒充完整实现；先作为相关理论对照，不强行加入本矩阵。

Privileged参考给额外信息，不参与公平deployable排行榜，**不是数学保证的性能上界**。随机时间乱序对照在开发真正随机置换历史，不能把固定reverse称shuffle。

## 5. 开发选择与确认冻结

预留开发根种子：T1 200001–200003、T2 201001–201003、T3 202001–202003。确认根种子：T1 200101–200115、T2 201101–201115、T3 202101–202115。它们在已扫描698个跟踪文件/262个JSON中未发现历史碰撞；静态扫描不是完整RNG证明，见04_seed_scan.json。

开发阶段须先通过结构正确性、真实物理奖励与泄漏测试。候选选择只用开发validation：与Dir相比平均回报差>0且至少2/3种子为正，未见policy-probe梯度cosine平均差>0，且容量/预算约束满足。若多候选通过，优先模块更少者，平局按预定义 D、R、D+R 顺序；不看确认集。此筛选是探索规则，不是显著性检验，也不证明被选者有效。

确认前写入selection_record：固定selected candidate、各对照容量、超参数、采样分布、成功定义、episode ID/hash、环境与代码commit、primary comparison、所有查询上限和异常规则；全部文件hash。无候选通过则报告no-go，不为追求正结果扩大搜索空间；新的研究路线需另立开发方案，不能消耗确认种子。

所有15确认种子启动后完整执行，不因负面中期表现停某方法；不按中期test更换selected candidate。特定任务未过可行性门槛，不启动其确认阶段，也不声称已完成整个EAGM2。

## 6. 主指标、统计和终止规则

- T1 primary：selected deployable candidate−Dir 的seed级平均物理回报差。
- T2 primary：selected deployable candidate−Dir 的seed级成功率差；continuous return为预定义次要指标。
- T3 primary：RGB+recurrent+G−RGB+observable-record+G 的seed级成功率差；训练数据/视觉编码器相同。
- 三个primary组成固定family，配对双侧exact sign-flip检验（n=15枚举32768种符号），Holm校正；未启动任务作为“未检验”，为避免缩减家族带来优势，校正时占位p=1。检验需要配对差值在零假设下可交换/对称的假设；报告精确sign test敏感性（零差单列）及全部差值，不暗称无假设的随机试验。
- 每seed先汇总起点，再对15个seed差值bootstrap 20,000次给95%区间，CI为未校正描述性区间并标注；bootstrap种子登记且与训练分开。不将450个episode当450独立训练seed。种子间标准差、median、wins/ties、leave-one-seed-out一并报告。
- 梯度机制支持只固定T1 tracking 的未见probe H8全坐标gradient cosine为一个endpoint，4项secondary family：D作用、R作用、G作用、D×R交互；Holm4。令Y_drg为各seed先汇总probe的cosine，D=(1/4)sum_rg(Y_1rg−Y_0rg)，R/G同理，D×R=(1/2)sum_g(Y_11g−Y_10g−Y_01g+Y_00g)。T2/T3或return/norm上的同类对比仅描述性；若拟声称显著，确认前扩family并固定全部键。回报与cosine相关仅作相关分析，不代替中介因果证明；方向/模长独立消融在开发筛选后若进入主张，必须进入额外冻结确认family。
- 次要模型指标：one-step NMSE（分母用训练尺度）、H1/2/4/8/16 rollout误差、command MAE、状态误差、全坐标与stable-mask cosine/sign/balanced-sign/norm-ratio/per-coordinate error、mask覆盖、policy path(0/10/25/50更新)质量。零真梯度下cosine为undefined并报告比例；不能只保留高cosine样本。
- OOD：nominal delay按0/1/2分层；holdout执行参数用已有配置族范围外的预先固定集合，按gain、smoothing、rate-limit、extra-lag逐因素和联合组合分报。名义实现只支持0/1/2时，不能靠extra_lag=3称unseen nominal delay=3。一般d>2须先实现并验证FIFO，另作扩展实验。
- 不排除低回报、失败、超时或负面种子。可重跑只限已定位基础设施错误/日志损坏，保留失败attempt，并对所有受同bug影响条件统一修复。若bug影响科学含义且已看确认结果，原确认批次作废为探索，不能修完仍称同一未触碰确认。
- NaN/物理失稳作为失败结果保留，报告发生率；主要回报若无法定义，按预先冻结的任务failure floor计入并给敏感性，不能事后挑惩罚。failure floor必须在确认前明确，否则不得启动。
- 停止规则：开发未过科学/正确性gate则no-go；确认只按固定种子数结束，硬件/磁盘错误暂停可恢复，不以效果好坏停止；预算不足在启动前缩减方案并重新冻结，不能做到一半选择性结束。

## 7. 查询预算与执行记录

结果分两条轨道：(a)固定共同数据/更新次数的D/R/G机制消融，G=0真实少用FD预算，不能把未使用标签费用强塞给它来声称成本相等；primary candidate−Dir属于此轨道并透明报告成本。(b)固定总训练query上限的效率比较，Base-extra-data用剩余额度采集，FD/SPSA不扣其未使用的world-model训练数据，使用全部合法优化余额；视觉共同预训练对各方法同额计费。所有部署方法的cost-effectiveness结论须来自(b)，不能由(a)推出。

所有方法记录训练采样、标签生成、critic、优化查询、评测、diagnostic分别的control steps和physics substeps；共享数据同时报告一次性实际采集成本与每方法归属成本，不能靠共享缓存让一个方法看起来免费。真正相同snapshot+policy+reward+epsilon才可复用FD标签；有状态/奖励变化必须重算并计费。

Direct-FD和SPSA以同训练交互上限停止，优化步数可能少于50；另列equal-update-cost曲线，不能伪称同时等步数又等查询。模型方法的诊断FD不反馈训练/选择，预算单列。视觉encoder预训练与critic消耗必须包含。

日志目录使用 task/phase/root_seed/method/config_hash/attempt_id，禁止覆盖。每run保存完整config、git diff/commit、环境版本、GPU/峰值内存、随机流映射、模型参数、原始episode结果、checkpoint/optimizer/RNG恢复状态、异常和失败。所有方法共享同一测试起点清单但不能读取测试来选checkpoint；选择只用validation。

开发通过后才发布完整run_manifest.json、experiment_status.csv并启动确认；当前不创建伪results/checkpoints。每项测试保存日志：FIFO/time order、delay切换、bounds、autograd-FD、mask、history causality、输入隔离、snapshot恢复、RNG/checkpoint、参数量、multi-step vs模拟器。只能把真运行日志写入05_*报告。

## 8. 阶段批准点

本轮提交研究方案和预注册草案。请求作者确认：以当前Stage8.3为研发基准，先进入实现/测试和三个任务的开发门槛；确认性矩阵在门槛通过并提交冻结清单后，按本方案执行。若预算或任务定义发生实质变化，先更新可审阅方案。原始稿、历史原始实验和旧分支不改动。
