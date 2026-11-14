# EAGM 2.0 方法设计（待作者确认；未实现、未训练）

基准：`412cf8c`（研究代码继承 `9416b92`）。本文件是 PART 3 方案，不是有效性证明或论文贡献清单。指定 Final 与正确补充材料仍未取得。已知基线含有限窗口 GRU、H8 imagination 和50次策略更新；这些不能被包装为全新功能。

## 科学问题与可证伪假设

中心问题：在执行状态部分不可观测时，哪些可获得的因果信息足以减少**用于策略更新的闭环导数误差**，以及这种减少是否在固定计算预算下带来独立测试回报收益？

- H-D（结构）：修正名义延迟的瞬时动作通路，可减少 command Jacobian 误差；不保证整体回报提高。
- H-M（状态）：因果历史在存在执行别名的分布下减小命令预测误差及多步梯度误差，且收益不能由参数量解释。若容量对照相同有效，拒绝“记忆独立贡献”。
- H-G（监督）：方向与模长监督对更新效用作用不同；联合监督是否优于单独项是实证问题。若回报上升但梯度未改善，不作“通过梯度保真改善回报”的因果主张。
- H-V（外部验证）：在真正闭环RGB策略中，执行记录/历史可补充图像难以观测的信息。图像与记忆的互补性需交互消融，不能由状态任务的结果推断。

所有假设均需要独立种子与未用于选择的测试分布。标准链式法则、GRU、FD、CNN 和FIFO本身不作为创新；潜在贡献必须是执行信息的结构性限制、可部署建模和预算受限更新效用之间的证据链。

## 1. 时间索引和可获得信息

以调用 `step(a_t)` **之前**为 t。`r_t=qtarget`，`f_t=filtered`，`q_t=[f_(t-1),f_t]` 为名义FIFO（正常更新不变量下），`d_t∈{0,1,2}` 为本步已知声明延迟。代码依据：`controllers/partial_unknown.py:160–178`；reset初始化依据 `vendor/legacy_sim/sim_env.py:155–198`。

先限幅动作，再目标限幅，再滤波，最后读取旧FIFO：

\[
\bar a_t=\mathrm{clip}(a_t,-a_{\max},a_{\max}),\quad
r_{t+1}=\mathrm{clip}(r_t+\bar a_t,\ell,u),\quad
f_{t+1}=\alpha r_{t+1}+(1-\alpha)f_t.
\]

\[
v_t=U_{0,\mathrm{delay}}=
\begin{cases}f_{t+1}&d_t=0,\\f_t&d_t=1,\\q_t[0]=f_{t-1}&d_t=2.\end{cases}
\quad q_{t+1}=[f_t,f_{t+1}].
\]

`v_t` 叫 **nominal command**；真实 executed command `u_t` 还经过 hidden gain→extra FIFO→rate limit→smoothing→joint clip（`partial_unknown.py:199–226`）。特别注意 `smooth_beta>0` 使用 `beta*limited+(1-beta)*old`，`beta=0` 是关闭分支，不是该公式的连续极限。额外延迟 `extra_lag` 与声明延迟 `d_t` 不混为一个字段。

当前 `models/eagm.py:14–16` 始终使用 `f_(t+1)`，未读取 delay；它是 delay-free prior。在动作/目标非饱和且 d>0 时，真实名义瞬时 `∂v_t/∂a_t=0`，旧prior却为 `alpha I`。残差可能学会抵消，但这必须验证。

### FIFO的部署可用性与边界

1. 首选维护**可观测 filter 的历史缓存**，从真实reset开始逐步保存。它不读 simulator FIFO；每个缓存值来源必须可追溯到当时观测。恢复中途checkpoint时保存这份可部署缓存，否则 burn-in 至所需长度。
2. 固定 `alpha≠1` 且上步更新不变量成立时，`f_(t-1)=(f_t-alpha*r_t)/(1-alpha)` 可作为无历史短接口的等价构造；reset时需核实 `r=f=q=初值`。用单元测试比对缓存与逆式，不凭浮点相等推断任意状态有效。
3. `alpha→1` 时逆式放大观测误差；`alpha=1` 不可逆，必须有历史缓存，不能置零伪造。alpha时变时必须知道上一步实际alpha；未知alpha不适用逆式。
4. 当前控制器只支持d=0/1/2；不得宣称已支持任意delay。扩展到D需要另外实现D长因果缓存及控制器对照。未知或未来delay不能用测试调度泄漏；若本步delay不可得，应建立显式belief并另列实验，不能调用本精确分支。
5. 动态已知delay切换只改变读取索引，缓存每步持续更新。cache warm-up、reset、动作饱和、目标饱和和非法delay必须测试；切换处/clip边界导数不按光滑区域定理解释。

## 2. 方向 D：Delay-aware structured prior

四个命令头至少比较：Original U0（不学习残差的解析控制）；Delay U0（解析名义映射）；Free command head；Delay U0 + learned residual。并保留真实历史 EAGM-Base/Dir 的 U0+residual 作为可复现对照。无残差的名义映射不是B真实命令估计的合理oracle，其误差是预期边界。

`u_hat_t = U0_delay(c_obs_t,a_t,d_t,cache_t)+Delta_omega(c_obs_t,a_t[,h_t])`；物理预测 `x_hat_(t+1)=F_phi(x_t,u_hat_t)`。自由命令头要同输入、可训练参数和训练预算匹配。先单独测试D，不同时添加记忆/新损失。

指标：nominal逐步数值误差（实现正确性）；真实command MAE（模型能力）；interior Jacobian误差/余弦与零通路违规率；H8 policy-gradient cosine/norm ratio；独立起点真实物理回报。跨delay分别报，不能用平均抵消d=1/2失配。

## 3. 方向 M：因果执行latent，而非再加一个GRU

现有 `HistoryEncoder/HistoryEncoderC` 已在 `models/eagm.py:19–57`；每步有限窗口重编码。HistC数据路径追加旧c_obs而rollout追加新c_obs（`dataset.py:129–134` vs `rollout.py:128–132`），probe又重用训练样本（`exec_state_probe.py:67–95`）。因此首先在新分支建立**修复版既有GRU基线**；旧结果与checkpoint原样保留。

可部署记录统一为 `e_t=(o_t,c_obs_t,a_(t-1), optional_logged_u_(t-1), valid_mask)`。状态策略的o为x；RGB策略的o为编码器 `E(I_t)`。上一条真实命令只有部署接口确实记录时才可输入；默认实验关闭该通道，另作命令日志可用性消融，所有方法同接口。

采用显式预测/观测更新两步：

\[
h_t=Q_\psi(h^-_t,e_t),\qquad
\hat u_t=v_t+R_\omega(h_t,c_t^{obs},a_t),\qquad
h^-_{t+1}=G_\psi(h_t,c_t^{obs},a_t,\hat u_t).
\]

`Q`和`G`可先用GRUCell实现；内部状态不是“已识别物理隐状态”，只称执行预测latent。状态T1/T2以及视觉主对照的actor不读取该world-model latent；在真实执行时Q读当前可获得记录；在imagination时G递推，**不读取未来真观测**。若用Q进行想象校正，输入只能是模型预测o/c且必须把该路径写入计算图，训练/评测语义一致。主候选采用G-only想象：h_0由起点之前32步合法burn-in经Q得到并在开始优化theta时detach；第k步command读h_k，actor读取其规定的预测观测/record而不读h_k，计算x/z_(k+1)，再令h_(k+1)=G(h_k,c_k,a_k,u_hat_k)，下一步直接使用这个prior，不调用Q。latent递推和预测观测至actor反馈均不detach；训练固定窗口open-loop多步损失沿同一G路径，只有posterior target及burn-in边界detach。Q(predicted observation)是另列开发对照，不可在确认中混用。

训练包含单步动态/命令损失与任务指定horizon的多步预测损失（T1 H8、T2/T3 H32）；posterior/prior一致性用 `||G(h_t,...)-stopgrad(h_(t+1))||²` 作为可消融辅助项，不能据此声称belief正确。Teacher forcing训练与open-loop误差分别报告。训练切片包含burn-in；初始latent只由reset或过去记录建立，禁止用held-out future优化h0。模型不得输入`controller_b` snapshot、`exec_state_scalars`、真实physics状态（RGB条件）或未来真实u。

参照矩阵：feed-forward、flattened-history MLP、修复窗口GRU、persistent latent recurrence、D、D+M；配capacity-matched MLP/GRU；privileged hidden-state模型作为**信息参考**。有限模型的oracle不保证性能上界，更不保证优化器会利用额外信息；必须分别报告它的架构/查询权限/误差。

状态估计probe只在冻结latent上独立拟合，按episode划分，normalizer只fit probe-train；训练用和评估用完整隐状态均不进入部署前向。比较历史长度L∈{1,4,8,16}，在同预算下报告参数量、burn-in成本和命令/梯度/回报；不能只画相关系数证明可辨识性。

## 4. 方向 G：梯度方向、模长及置信校准

旧损失 `training/dir_loss.py:24–33` 是normalized Huber discrepancy，已监督方向和模长，不应称纯direction。固定同一标签bank，比较：无梯度损失；旧Huber；direction-only；norm-only；direction+norm。

设稳定mask矩阵M，`p=M g_hat, q=M g_FD`。当两者范数超过预设epsilon时，`L_dir=1-<p,q>/(||p||||q||)`；零/近零标签没有方向，应另报覆盖率而非填余弦0。`L_norm=Huber(log((||p||+epsilon)/(||q||+epsilon)))`。联合项为加权和；初始权重只在开发集候选网格选择后冻结。主实验固定参数probe匹配策略theta，禁止跨theta复用而不标识。

FD改进先明确 stencil：对称central、forward/backward、非对称三点各分开记录 `h_minus,h_plus`。旧feasible分支在近边界实际为secant而标central（`gradients.py:107–137`）。新三点公式见推导PDF，旧结果不覆盖。clip/接触切换处没有经典导数保证；记录接触模式、饱和、尺度一致性以及未解析标签比例，拒绝“mask通过即真导数”。

开发集实验：同theta标签 vs 人为stale标签；固定重标间隔K∈{0,5,10}（0为从不重标），同总查询预算下比较；H∈{4,8,16}多步一致性；epsilon在训练/开发标签上选择，独立测试仅评测。Path-aware已有失败，不能预定周期重标有效。

置信度只作为**待校准分数**：尺度spread、ensemble方差、policy离标签分布距离。用独立calibration episodes检查误差分位数覆盖；没有exchangeability/分布假设时不称95%梯度置信区间。种子级统计CI与逐样本梯度误差预测区间必须区分。先记录分数与实际误差的关系；不先据此过滤测试难例。

## 5. RGB策略与接触扩展的可实现边界

当前actor只有28参数（`rollout.py:47–65`）；3尺度坐标FD一个H8标签成本 `2×3×28×8=1344` simulator control steps。CNN全参数坐标FD不可直接照搬。

第一条可检验路线：训练集上学习视觉编码器与基础策略（训练监督来源、BC示范成本及共享预训练成本全部计费），冻结基础后用预先固定32维adapter `theta` 做真实RGB闭环策略优化。动作可写 `a=clip(pi_base(E(I),records)+B(E(I),records)theta)`，B的权重冻结，theta在模型梯度下优化。由真实渲染输入决定动作，物理state只可用于训练奖励/非部署标签；所有视觉比较共享同一视觉训练预算。此结论只能叫**视觉策略adapter优化**，不能称端到端全视觉网络优化。

第二条（单独开发候选）：使用K个随机单位方向的投影导数监督 `v_k^T grad_theta J`，使标签成本随K而非全部参数P增长。随机方向bank在训练前冻结，测试方向独立；有限K不能声称恢复全梯度。与SPSA/直接方向更新以相同真实查询预算比较；低维adapter与随机方向不可同时改变后把收益归因一个模块。

视觉world model须有 `z_(t+1)=F(z_t,u_hat,h_t)`、图像特征/奖励预测与open-loop训练；不能在想象中偷用未来真图像或真实x替代z。它与原状态F不完全相同，需作为明确的architecture adaptation。RGB、RGB+执行记录、RGB+执行记忆、state策略和privileged信息参考分列。接触任务必须先确定真实接触、成功条件、reset稳定性及预注册阈值；此处不宣称任何任务已稳定运行。

## 6. 公平选择与晋级

开发种子、确认种子和统计方案由 `04_*` 协议统一冻结；不使用历史0–20或已审视Stage8数据作为新确认。修复实现与数值测试是所有方法共有前置条件，不算候选性能优势。

D/M/G固定数据机制轨道采用同可用数据和labelbank，但G=0不使用标签、真实成本较低；独立等query效率轨道用Base-extra-data和FD/SPSA按合法余额竞争，不能靠计入未用标签伪造等成本。参数量计入encoders、latent、command、dynamics、reward及可训练adapter，配对容量偏差目标±5%，不能只报主MLP。训练交互、FD/诊断交互、GPU时间、wall time分别记账；共享数据也要向每个逻辑方法归属完整成本，避免因缓存少计成本。容量无法匹配时保留原型并标不公平，另补匹配对照后再解释。

候选晋级仅依赖开发集：通过接口/因果/梯度测试；在固定预算上相对参考的种子级平均回报差>0，且若主张gradient mechanism，开发集梯度余弦差也须>0；合格候选至少2/3开发种子回报差为正，平均回报差与cosine差均>0；多个通过时优先模块更少者，再按D、M、DM固定顺序。精确定义以04协议为准。组合不自动优先。没有候选通过时报告no-go，不挑测试集赢家，也不把开发集改名确认集。

确认阶段冻结一个主候选与完整2×2×2 D/M/G因子矩阵，并保留capacity MLP/GRU、VaGraM-inspired adaptation、PAML-critic adaptation、direct FD/SPSA、Base-extra-data预算对照、privileged reference（tracking与contact各16格×15种子；根种子和任务可行性门槛见04协议）。PAML适配以真实next-state与model next-state路径的critic/actor参数梯度作目标，不把既有FD目标重命名成PAML；其critic训练与额外数据预算单列。开发集单格负面不能删去预注册因子格。任何中途负面结果不得删种子或提前停止。若确认失败，论文保留失败并限制主张。理论推导文件中的条件界不证明当前latent满足假设，不能替代实验。

硬件预算约束：主审探测到单张RTX4060 8GB（约6GB空闲），不假定多GPU；MuJoCo现有版本3.5.0与历史报告4.0.1不一致，原版复现须先核查实际依赖和数值差异。此信息来自本轮环境探测，不是新实验性能。
