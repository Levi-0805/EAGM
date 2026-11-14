# 两份条件性推导的独立数学复核

复核日期：2026-11-14。只读检查 `gradient_error_bound.tex` 与 `causal_jacobian_derivation.tex` 的关键代数、假设与实现对应；没有修改作者文件，没有等待PDF排版，也没有读取训练结果。本文不是实证验收或文献新颖性认证。

本次检查版本的SHA256：

- gradient_error_bound.tex：`5f7b103ec9b5587d26a446c9df8452d1f33d34fe7fa7bd2305f9d6f68b6bd7ba`
- causal_jacobian_derivation.tex：`2e85814377f50dbc567e3ad3a4f47bc7b82b43de2cc7aa8cf2969c8868e16591`

结论：在文中给定的统一局部正则性、同坐标表示与固定执行配置假设下，未发现所检查递推、梯度误差界和FD误差常数的数学错误。不代表假设已在MuJoCo接触、实际训练GRU或视觉reward上得到验证，也不证明经验有效、策略改善或方法新颖。

## 1. Controller与因果Jacobian

核对 `controllers/partial_unknown.py` 的step和_apply_unknown、`eagm2/prior.py` 的observable_step/nominal_command：

- pre-call状态采用r_t/f_t、缓存b_t=f_(t−1)，FIFO为[b_t,f_t]；delay0/1/2分别选新filter、当前filter、缓存。这与代码在更新队列前选command的顺序一致。旧vendor注释采用不同下标不能混用。
- 固定旧状态时，d>0即时action偏导为零；参数总导数仍通过旧状态与缓存传播。R更新中的D_r(R_t+D_a A_t)、F更新及B_(t+1)=F_t均正确，不能以即时偏导零推断延迟策略梯度零。
- gain→extra lag→rate limit→smoothing→output clip的顺序对应实现。rate sensitivity为D_rho W+(I−D_rho)U_prev，输出clip后U与内部未clip的S区分正确。
- extra FIFO填充时使用previous executed command，填满后的常gain/常lag公式W_t=gV_(t−L)正确。文中没有把启动阶段错误地套用稳态公式。
- β=0走旁路，确实不是β趋近零正值时recurrence的极限。整数lag、delay及该分支不在可微参数范围内。
- filter代数逆只在合法历史、固定0<α<1下成立；α=1奇异、接近1病态，文中已限制。缓存正确性仍依赖真实前序观测，不自动覆盖实际有限burnin首步的近似缓存。

## 2. 闭环误差递推与reward项

链式法则给A=F_z+F_a π_z、B=F_a π_theta。使用联合输入距离||Δz||+||Δa||，由policy的P_z界得到1+P_z因子；文中K_A、K_B各项及同点epsilon_A、epsilon_B正确。比较不同轨迹上的Jacobian时，先加减真系统在model-state处的导数，所以只要求真导数的Lipschitz界即可；不必额外假设model导数Lipschitz来完成当前证明。

Sensitivity差按hat_A(hat_S−S)+(hat_A−A)S+(hat_B−B)分解，给出L E_t+(epsilon_A+K_A e_t)s_t+epsilon_B+K_B e_t。其求和展开指数正确；e_t与s_t的几何和在L=1时直接解释为项数，无需除以L−1。

Reward同样需要closed-loop显式theta导数D，文中未遗漏epsilon_D/K_D项。将next-state reward吸收到R中时，该R实际上包括transition复合；即使底层物理reward解析相同，hat_R与R的导数误差也未必为零，文中已明确。视觉learned reward更不能置零。

固定保存snapshot/history且模型权重固定时，s_0=E_0=0是条件rollout导数的正确陈述；如果初态/encoder/history本身随本次优化theta变化则不适用。e_0不因此归零。

适用域必须包含比较涉及的状态点、连接段以及相应(z,π_theta(z))输入；后续应用不能只验证离散训练点上的常数。所用向量/算子范数需一致且满足乘积次乘性。本文是有限维通常范数意义下的条件界，不是逐样本MSE的直接推论。

## 3. FD常数

独立检查对常数、线性、二次项的抵消和三次项：

| stencil | 截断界 | 有界objective误差项 | 核对结果 |
|---|---|---|---|
| −h,+h中心 | M3 h²/6 | η/h | 正确 |
| −a,0,b非均匀三点 | M3 ab/6 | η Σ|w_i| | 正确 |
| 0,h,2h前向二阶 | M3 h²/3 | 4η/h | 正确 |

非均匀三点的三次矩为ab，分别取两边Taylor remainder可直接给M3 ab/6。前向模板直接把两条Taylor remainder绝对值相加会得到较松常数，不能这样推出1/3；文中的积分核两段均非正，其绝对积分恰为h²/3，因此更紧的1/3常数成立。反射到上边界仅改符号，不改绝对界。

另外用Python Fraction完成三个非均匀模板的精确有理矩检查、前向绝对权重和4以及三次矩−2检查，均通过。此检查只验证代数，不是接触FD准确性实验。

η假设指每个objective值的确定性有界误差；实际浮点加权求和舍入若要认证，也需单独覆盖，不能因写了η就自动获得机器误差证明。多尺度标签一致性不提供M3或η界，单尺度同样不能证明稳定性；文中未将其当置信区间。

## 4. 非光滑、部分可观测与更新保证

- clip/rate active-set内部的导数是局部条件结论，切换面或tie上的autograd选择不等于物理经典导数。
- 固定contact active set本身不足以保证smooth solver；solver分支与事件变化同样影响可微性。文中已经补充solver-map光滑条件，范围合理。
- C³ FD界要求整个stencil区间满足条件，不是仅中心点光滑；跨接触/饱和切换时不能套用。
- 完整状态z与hat_z必须有可比较坐标。部署GRU latent及32维RGB latent并不天然与真实hidden state同构；可微lifting/representation需要另证。历史有限窗口或小预测误差不证明可辨识性/充分性。
- epsilon*sin(x/epsilon²)反例确实满足前向误差≤epsilon而零点导数为1/epsilon，足以否定“仅forward误差控制Jacobian”的无附加假设主张；不能据反例断言实际网络产生这种振荡。
- 实际step d的下界hat_g^T d−delta||d||−L_J||d||²/2来自平滑下界与Cauchy–Schwarz，符号正确。Adam和projection须代入实际d，不能替换成默认学习率乘raw gradient。高cosine不提供delta/步长界。

## 5. 未覆盖事项

未做形式化证明助手验证；未认证任何uniform常数、lifting、contact全域正则性或有限精度η；未审查最终论文的文献对照、所有MD改稿及实证表述；未将数学检查外推为learning improvement、收敛、全网络视觉RL或确认实验通过。作者若改变上述tex版本，尤其范数、假设、reward定义或FD模板，应重新针对变更复核。
