# Delay prior 与执行记忆实现说明

实现文件：`eagm2/prior.py`, `eagm2/models.py`。新增命名空间，原始controller、模型、训练和论文未修改。

## 接口

`nominal_command(c,a,f_prev,lower,upper,alpha,max_increment=.03)` 返回nominal而非executed；`observable_step` 返回 `(next_c,next_f_prev)`。c为29维pre-step target/filter/delay，f_prev来自上一时刻真实可观测filter缓存。alpha=1走缓存，不使用不可逆公式。`previous_filter_from_invariant`仅提供alpha∈(0,1)下的受限代数校验。

`ExecutionWorldModel(stats, obs_dim=28,width=160,memory_dim=32,memory='none',prior='original',residual=True,history_len=8)`：

- `step(x,c,a,f_prev,h=None)` → `(next_x,command,h_next)`；recurrent想象只G递推。
- `observe(h_prior,x,c,prev_a)` → posterior；部署输入不接受snapshot/controller_b/privileged_state。
- `burnin(xhist,chist,ahist,detach=True)`：x/c与前一action对齐，recurrent先用前一action作G prior再读当前Q；不得提供未来记录。第一条记录是声明的reset/burn-in窗口起点，若从中途开始，零latent只是有限窗口估计，不能称真实hidden状态。
- `memory='window'`：有限窗口记录重新经GRUCell编码；追加预测next x、解析next c及当前a，修复旧HistC时序差。
- `memory='none'`：feed-forward。
- `prior='delay'/'original'/'free'`：精确nominal FIFO / delay-free nominal / free learned head。
- GRUCell不使用fused cuDNN GRU路径，支持模型训练所需二阶自动微分。
- obs_dim可32以服务视觉latent，模型本身不读取原始物理状态标签。

**`prior='original'`仅是新架构下delay-free对照，不是冻结旧EAGMBase的逐参数复现。** 新command head有输入/输出归一化；严格旧版复现必须另走原始代码轨道。

`PrivilegedExecutionWorldModel`只有显式`step_privileged(...,privileged_state=...)`可用，常规step拒绝运行。这是更多输入的一步参考，尚不能当自动传播未来真实隐状态的可部署rollout，也不是数学保证性能上界。

`capacity_matched_width`只按trainable参数量搜索width，不读实验表现；使用fork_rng不扰动训练初始化。若不能±5%匹配则抛错，不静默称匹配。

## 验证边界

`test_prior.py`使用真实Controller B和MuJoCo step。通过patch仅把新materialized XML写到临时目录，绝不改历史assets。其独立B栈参考用于检验gain/extra FIFO/rate/smoothing顺序；名义prior与真实executed值分开比较，不能期待nominal等于executed。

`test_models.py`覆盖输入权限、历史因果API、window时序、checkpoint恢复、32维latent、容量控制及多步模型autograd/FD一致性。这是实现正确性测试，不是模型物理准确性、GRU可辨识性或训练收益证明。模型误差/gradient cosine/回报须由独立开发实验报告。

当前控制器只支持nominal d=0/1/2；不宣称任意delay。无新训练结果进入本文件。

## 冻结原模型适配器

`eagm2/legacy_adapter.py:LegacyWorldAdapter(stats)`直接持有未修改的`models.eagm.EAGMBase`作为注册子模块；只增加step三元返回值与runner元数据，不更改command/dynamics公式或参数化。所有统计buffer通过属性读取同一个子模块buffer，`.to(device/dtype)`与checkpoint由PyTorch递归管理，无CPU别名缓存。

适配器只允许原28维物理输入；`memory='none', prior='legacy', width=160`。在统一新数据/更新/多步损失协议训练legacy_base/legacy_dir，只能称“原架构在新共同协议下比较”；严格历史3000-update训练协议由单独复现轨道负责。

`tests_legacy_adapter.log`记录非零残差参数下，forward、输入梯度及所有参数梯度与冻结原实现逐元素严格相同（atol=rtol=0），以及buffer dtype/对象同步和checkpoint验证。没有修改原模型。
