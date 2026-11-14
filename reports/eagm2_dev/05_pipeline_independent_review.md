# 开发流水线独立复核（首次集成版本）

审查范围：eagm2/data.py、training.py、rollout.py、evaluation.py、runtime.py及scripts/eagm2/run_development.py。本文记录初次阅读时的问题；后续主审修复应追加处置记录，不能把本报告当最终状态。没有修改这些源文件。

## 实际独立测试

新增tests/eagm2/test_data_pipeline.py，使用显式合成public controller而不是研究任务性能结果。运行7项，6通过、1失败；初次日志保存tests_data_pipeline.log。

通过：确认/历史/错任务seed拒绝；episode及派生RNG隔离；x/c/previous-action/filter-cache时刻对齐；normalization不受test数值污染；采集计数等于实际steps；none/recurrent/window三个模型的短闭环imagined autograd与独立FD一致。

失败：start历史切片长度33，sequence训练长度32。具体data.collect starts使用`max(0,t-history):t+1`而sequences使用`t-history+1:t+1`；应统一公开history合同并保留此失败日志，再另存修复后日志。

## 必须在解释结果前处理的事项

| 问题 | 定位 | 建议处理 |
|---|---|---|
| 视觉PAML/VaGraM使用错误actor | training.train_model中的th=28和物理actor callback | 视觉须调用实际32维adapter的action_from_latent，或明确禁止该未适配组合；不能把latent坐标当前14维当真实joint |
| critic丢失可用context | train_critic输入只有x/a，target依赖c_obs和reference/time | 采用同可观测context输入，并在真实/预测next两支提供一致的next_c/next_ref；否则标为低信息受限适配 |
| behavior MC不等于actor Q | train_critic以带噪收集轨迹H-step回报为target；基线θ=0而EAGM标签含随机θ | 明确behavior-value critic近似及policy-region限制；更忠实目标需train-only FQE/TD、记录额外成本与验证误差，不称完整PAML复现 |
| 主梯度指标被mask改变 | evaluate_gradients只对mask算mean_cosine | full与stable分列，primary full-coordinate与协议一致；报告未定义/无稳定坐标覆盖 |
| model误差仅测试头128样本 | model_metrics按序取前128 sequences | 很可能只覆盖一个episode/delay；按episode/延迟分层采样，列覆盖，不称全部test误差 |
| 源码版本不可追溯 | environment只有proposal commit和未跟踪状态 | 每run保存当前源码hash+快照或执行前commit；训练后commit不能证明当时版本 |
| 失败partial交互漏计 | physical只结束后budget.add | finally计实际成功steps与异常耗时；失败query同样计费 |
| 参数初始化没有真正共享 | make_model同RNG，但memory模块先构造导致共同trunk抽样偏移 | 复制同形状named parameters或准确标记仅共根seed，不宣称完全paired initialization |
| 逻辑费用未逐method分摊 | Budget聚合collection/labels/evaluation；results无各method差额 | 共用实际采样与每方法使用成本分别列；未使用FD不可强归给Base；critic训练单列 |

这些问题不会自动否定所有Base/Dir结果；它们限制特定基线、公平比较和归因用途。开发数据仍为探索性，可修复后重跑并保留所有attempt；45确认种子必须继续锁定。

当前runner的执行配置是固定Controller B加nominal delay分层。若未另跑gain/rate/smoothing/extra-lag压力测试，不能宣称gradient calibration跨执行配置持续有效。Direct FD/SPSA和Base-extra-data完整预算比较尚需确认实际接入与结果，不能只凭helper存在认定已完成。
