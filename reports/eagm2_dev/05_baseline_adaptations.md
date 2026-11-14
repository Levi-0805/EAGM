# 梯度标注与外部目标适配实现记录

基准：81ecd5f，开发分支EAGM-2.0-full-validation。仅新增eagm2命名空间，不修改历史训练文件或论文。以下为实现/解析单测结论，不是任务性能结果，也不是确认性证据。

## 1. 已实现API

- `finite_difference(objective, theta, epsilon=.01, scales=(.5,1,2), bounds=(-1,1))`：返回gradient、mask、queries、每坐标每尺度stencil/offset/weight/value、estimates和spread。输入theta越界直接拒绝，不改变标签位置；对称中心差分、非均匀三点、边界单侧三点分别标记。默认缓存完全相同query向量，queries为真实objective调用次数；环境必须每次恢复同snapshot/reference/RNG。`queries × horizon`只有在各objective固定horizon时才等于control steps，主runner负责记实际步数。
- `calibration_loss(pred,target,mask,mode=...)`：none、legacy normalized Huber、direction、norm、joint。标签detach，空mask返回连图零；近零目标无方向不冒称可信方向标签。主runner须记录被跳过数量及mask覆盖。
- `gradient_metrics`：cosine、norm ratio、普通sign accuracy、balanced sign accuracy、RMSE、relative L2、每坐标误差及有效覆盖。单类balanced accuracy、零范数cosine等返回null，不能填1或静默删除困难样本。
- `spsa_gradient`：Rademacher方向，固定seed，真实2×directions调用；边界不静默clip扰动点，需由调度器使用合法内点或明确另一个stencil适配。
- `QCritic`、`ValueCritic`、`critic_regression_loss`：小型SiLU网络和目标回归；训练数据、归一化、Bellman/return target以及费用由runner登记。

## 2. PAML-critic adaptation

依据已核查的[PAML官方models.py](https://raw.githubusercontent.com/rabachi/paml/master/models.py)中的one-step路径，在真实next state和预测next state分别计算`−Q(s',πθ(s'))`的actor参数梯度，再比较二者L2平方差。真实目标detach；模型支路create_graph=True，以便梯度回到world-model参数。critic参数在计算目标时冻结，但保留Q对state/action的导数。actor参数仅用于求导，更新model时不得调用actor optimizer。

接口：`paml_loss(critic, actor_fn, true_next, pred_next, actor_parameters)`。默认参数维mean squared error；`reduction='sum'`可显式切换；必须记录该缩放与lambda。真实/模型两支的可观测context必须相同。它没有读取FD标签，不是把EAGM梯度损失换名。

适配差异：EAGM deterministic 28维actor/32维adapter和部分可观测执行接口不同于原论文完整agent；当前只提供同架构目标对照，不声称完整PAML算法复现。critic失真可影响效果，因此需要独立validation critic误差、critic参数量/更新次数/数据来源和耗时；不得使用测试转移训练critic。

## 3. VaGraM-inspired adaptation

依据[VaGraM官方vaml_mlp.py](https://raw.githubusercontent.com/pairlab/vagram/main/mbrl/models/vaml_mlp.py)中的scaled-MSE形式，目标是逐坐标`mean[(∂V/∂s_j × (ŝ_j−s_j))²]`，不是先点积再平方的Taylor VAML。权重在真实next-state计算并detach；world model梯度只沿预测误差传播。

接口：`vagram_loss(value_fn, true_next, pred_next, state_scale=None, normalize_weights=False, weight_floor=0)`。state_scale同时变换导数和误差，因此默认物理目标保持坐标单位一致。weight normalization或floor默认关闭；启用必须标记为额外适配并做消融。`value_fn`可为独立V critic或在固定actor下的Q closure；必须记录选择与critic target。

同样不复现完整MBPO/VaGraM框架，不借用外部论文回报作对照。

## 4. 公平费用与隔离要求

PAML/VaGraM不消耗EAGM FD标签，但训练critic不是免费：额外交互、critic训练/GPU时间、参数与前向/导数计算逐项计费。固定数据机制轨道可共享train trajectories；等query效率轨道允许按已登记规则使用未消耗额度，不能把未用FD标签算成它们实际成本。所有值函数/actor/model优化器保存各自checkpoint及RNG。

成功标准须由9个探索性种子的真实命令/梯度/回报验证；单测通过只证明目标实现的数值性质。45确认种子未运行。

## 5. 本轮解析单测

`tests/eagm2/test_gradients.py`：11项。覆盖所有FD stencil的二次函数精确性、指数函数收敛、cache真实查询计数、越界不静默clip、非有限/单尺度/不稳定mask、SPSA复现、legacy公式一致、方向缩放不变性、模型二阶梯度可回传、空mask、类不平衡及未定义指标。

`tests/eagm2/test_baselines.py`：5项。PAML在bilinear Q和线性actor下与解析actor-gradient差及其model导数一致；同状态差为零；critic权重无更新梯度；神经模型可获得有限非零梯度；VaGraM验证逐坐标平方不发生正负抵消、坐标单位变换一致、权重detach。

日志：`tests_gradients.log`、`tests_baselines.log`。环境：eagm_gpu Python，CPU解析测试；没有用这些数值单测声称GPU训练收益。

## Integration correction: contextual fixed-policy critic

The initial truncated behavior-return regression critic was replaced before the corrected development run. `train_critic` now fits a continuing discounted FQE/TD proxy at theta=0, on training trajectories only. Inputs are `[x,c_obs,reference]` (71 coordinates for physical observations; 75 for 32D visual latent), together with action. The frozen Polyak target uses aligned true next observation, next observable record, next reference, and the actual fixed policy action. Discount 0.99, target interpolation 0.01, Adam 0.001. Context normalization and reward RMS derive from training only. Returned Q is in original reward units. Both adaptations query this same contextual critic, with the physical 28D policy adapter or visual 32D adapter as appropriate.

This remains an **adaptation**, not an original-method reproduction. Offline behavior coverage is not guaranteed. The reported behavior-to-theta0 action RMSE is descriptive, not proof of support. Observable context still aliases Controller B hidden state. The continuing critic is not an exact finite-horizon value or a solution to partial observability. Optional heldout TD RMSE is not true value error and is not used for selection. PAML/VaGraM objectives are fixed at theta0 while directional labels may cover several policy positions; this difference must remain visible in comparisons. Critic fitting consumes no new simulator calls but its training wall time and compute must be charged to these baselines. `critic.training_checkpoint` contains optimizer, target network and RNG tensors; the runner must persist it separately from JSON metadata.

Common model parameters are now initialized from parameter-name and shape-specific seeds, independent of architecture constructor order. Matrices use fan-in uniform bounds; vectors use their own length for a shape-consistent uniform bound. Prior residual output layers retain zero initialization. The model exposes `initialization_manifest` for exact rules/seeds/shapes. Equal parameter shapes have identical initialization; differently shaped capacity controls do not claim identical weights.

Synthetic CPU integration validation: `tests/eagm2/test_critic_integration.py` checks shared initialization, preserved zero residual, validation-data isolation, JSON-safe metadata, frozen target/main critic state, and one genuine backward/update through each adaptation for physical and visual contexts. Three tests pass; log `tests_critic_integration.log`. The environment lacks pytest, so the three plain test functions were invoked through runpy. This verifies plumbing, not robotics performance or critic accuracy.
