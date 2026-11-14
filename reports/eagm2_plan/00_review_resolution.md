# 独立方案复核整改记录

原审查与复核分别保存在00_independent_plan_review.md及followup。它们针对草案当时的行号，修订后行号会变化；不删除历史评语。

| 问题 | 本轮处理 | 状态 |
|---|---|---|
| G矩阵/选择规则 | 8格G固定旧Huber开关，新损失属开发；主候选按模块数及D/R/DR顺序；新增Base-extra-data，总570格 | 方案已澄清，待实现/冻结 |
| latent递推 | 起点32步Q burn-in并detach；想象h_next=G，不读未来Q；actor不读model latent | 方案已澄清，待因果测试 |
| 接触时间/actor | H32局部FD目标、约10s完整评测、固定reference actor；必须通过局部更新与完整成功验证 | 待开发可行性 |
| snapshot维度 | 改为可能变化，实际mj_stateSize/spec派生，模型hash严格隔离 | 待实现测试 |
| 公平预算 | 固定数据机制轨道与等query效率轨道分开；Base-extra-data使用剩余FD额度 | 待预算日志校验 |
| latent初始化 | 32步合法warmup、每episode独立、缓存/隐状态权限隔离、warmup成本单列 | 待实现测试 |
| 视觉归因 | 主比较相同RGB+record actor及adapter，仅改变world-model记忆 | 待输入隔离测试 |
| Holm family | secondary仅T1 full cosine的D/R/G/D×R四项，公式明示；其它描述性 | 待冻结分析配置 |
| 复核小同步 | actual_dt决定10s/0.5s时间；T1 H8、T2/T3 H32；Base-extra-data任务化计费 | 文档已同步 |

独立复核未发现条件性数学推导代数错误；这不等于数值测试或新方法有效性验证。确认性预注册仍需开发后固定超参数、各任务奖励权重、错误惩罚、资产与代码hash、实测资源预算。当前可批准的是实现和开发阶段。
