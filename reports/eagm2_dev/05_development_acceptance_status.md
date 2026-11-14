# EAGM 2.0 开发验收状态与证据框架

启动更新：`development_v1` 已通过三任务 smoke 和完整 78 项测试门槛，在 RTX 4060 上顺序运行九个授权探索根种子。启动代码 commit 为 `5037c4904dcd3ee99d15ab95e31c15cac413de94`，已推送 GitHub。实时状态见 `runs/eagm2_batches/development_v1/experiment_status.csv`；完成整理器状态见本目录 `development_completion_worker.json`。这只是运行启动，不是九种子验收通过。冻结旧分析脚本的解析错误将保留，由独立测试过的 postrun 副本生成报告；不得把后处理错误误记成训练失败。以下是启动前的原始状态快照，保留其时间顺序。

状态快照：2026-11-14，本文件编写时 `development_v1` 尚未启动。当前是实现、工程测试和 smoke 阶段，未完成三个任务各三个根种子的完整探索批次。本文件不登记学习效果，不选赢家，也不把 smoke 数值用于验收。后续实际批次必须以其源码快照、配置、完整日志与结果重新核对本表；旧日志只证明当时版本。

## 授权边界

允许的探索根种子只有 tracking 200001–200003、contact 201001–201003、vision 202001–202003。45 个确认根种子（tracking 200101–200115、contact 201101–201115、vision 202101–202115）禁止启动，状态为 `not_authorized`。确认阶段必须另获作者批准；开发通过不自动构成批准。历史论文、原始数据和旧分支保持冻结，开发结果不得回填为原论文实验。

## 当前可核对的事实

| 范围 | 现有证据 | 当前结论与边界 |
|---|---|---|
| nominal prior/Controller B | `tests_prior.log`：3 tests OK，972 actual control steps；独立 executed-stack 对照 | 实现一致性证据；不是训练收益、任意 delay 或 OOD 泛化证据 |
| 模块集成 | `tests_integration_fixed.log`：43 tests OK；`tests_gradients.log`：11 tests OK | 特定修复版本工程测试通过；最新并行修改仍需对应新日志，不能将未结束的 `tests_pre_batch.log` 当通过 |
| 早期独立审查 | `05_pipeline_independent_review.md`、初始失败日志 | 曾发现 history 长度、critic/context、视觉 actor、成本和初始化等问题；报告保留，修复状态须由实际新测试逐项证明 |
| 旧模型轨道 | `tests_legacy_adapter.log`、`05_models_implementation_notes.md`；独立 `legacy_reproduction_v1` 目录 | 冻结模型适配器与严格历史协议复现是不同层次；新共同协议下的 legacy_base/dir 不等于完整旧实验复现，不能混池 |
| 接触实现 | `tasks_contact_tests_final.log`、`tasks_implementation.md`、`tasks_feasibility.json` | FK/reward/snapshot/contact 工程测试与脚本控制器可达性证据；identity_d0 失败保留。未证明学习策略成功或控制器变化鲁棒性 |
| 视觉实现 | `tasks_vision_tests_final.log`、`tasks_visual_training_tests.log`（1 test OK） | 当前单帧 RGB，CNN/base 冻结后优化外部 32D adapter；不是双帧协议，也不是整个视觉网络端到端 RL。输入隔离/梯度存在性不等于学习有效 |
| PAML/VaGraM | `05_baseline_adaptations.md`、`tests_critic_integration.log`（3 tests passed） | 确定性 actor/可观测接口适配；不得声称原论文完整复现。critic 参数、拟合误差与额外成本仍需实际批次核查 |
| 预算对照 | `budget_controls_tests.log`（7 tests OK）、`eagm2/budget_controls.py` | FD/SPSA helper 与预算防护已测试；runner 集成与完整学习结果仍待。Base-extra-data 同样须核对实际额外采样而非仅代码存在 |
| privileged/oracle | `PrivilegedExecutionWorldModel` 与权限分离测试 | 仅显式 privileged 一步接口实现/工程测试；没有完整 oracle 实验结果，不是可部署模型，也不是数学保证性能上界 |
| smoke | tracking/contact/vision 的显式 smoke attempt 目录及失败记录 | 仅用于管线连通与错误定位；失败与耗时保留。任何 smoke 回报、cosine、成功率均不进入开发效果结论 |
| 正式探索统计 | `scripts/eagm2/analyze_development.py` 已通过临时合成结构检查 | 尚未对 `development_v1` 生成结果报告；合成夹具不属于科学结果，已自动删除 |

## 完整开发批次待填验收表

| 验收项 | 当前状态 | 必须填入的实际证据 |
|---|---|---|
| 每任务 3 根种子完整性 | 未完成 | 所有预声明方法 × 根种子的 completed/failed/missing 清单、失败 attempt、checkpoint |
| 数据/部署权限 | 工程测试已有，批次待核 | train/validation/test 独立 episode IDs、hash、归一化 train-only、历史对齐、未来观测隔离 |
| D/R/G 与容量 | 未有完整结果 | 八格逐 seed 原始指标、D/R/G/D×R 配对差、精确参数量与 ±5% 容量核对；FLOPs 缺失不能称齐全 |
| 梯度机制 | 未有完整结果 | full-coordinate 与 stable-mask 分列，undefined 和覆盖率；正回报不能替代 cosine 改善证据 |
| T1 tracking | 未有完整结果 | 同起点/初策略的物理回报与 delta，delay 分层；validation 筛选证据与 test 描述严格分开 |
| T2 contact | 未有完整结果 | 完整 episode success、return、几何误差、失败/超时/数值错误；H32 局部优化不是 10s success 梯度 |
| T3 vision | 未有完整结果 | 真 RGB 策略部署、共同冻结 encoder/32D adapter、同 actor 不同 world-model 比较；披露单帧偏离 |
| OOD/负对照 | 未完成 | gain/smoothing/rate/extra-lag 单因素与联合记录、无隐藏状态及真实 shuffle；nominal 0/1/2 不冒称更大 nominal delay |
| 查询预算与失败成本 | 运行后核算 | train/label/critic/optimization/evaluation/diagnostic 分账、共享一次性与每方法归属、control/physics steps、失败成本 |
| 资源 | 运行后核算 | wall elapsed、CUDA-event elapsed、峰值 allocated/reserved；CUDA event 不是 GPU-active kernel time，嵌套时长不相加 |
| 冻结确认草案 | not_authorized | selection/no-go、配置/环境/源码/hash、主比较、异常/failure floor、预算与额外作者批准 |

## 统计与结论约束

仅显式指定 `development_v1` 读取授权根种子。每任务最多 n=3，episode/probe 不充种子，不跨任务合并，不声称显著性。原始失败、负面与空值均保留；smoke/invalid 不进入效应汇总但成本仍可审计。Direct FD/SPSA 没有 world model，其模型/梯度 fidelity 指标应为不适用，不能填零。不同更新数下的固定查询预算对照与共同数据机制消融分别解释。

分析程序输出 `06_*` CSV 与审计报告，但不执行候选选择。后续验收必须允许 no-go、效果不稳定、只改善性能不支持机制、容量或预算不可比等负面结论。任何新方法/指标、确认性 family 或任务定义变更，需明确记录，不能由探索 test 结果倒推冻结规则。
