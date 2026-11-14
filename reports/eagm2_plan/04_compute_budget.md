# 算力与交互预算（估算，不是实测新训练结果）

## 已读取的机器状态

2026-11-14：宿主机只读nvidia-smi返回1×RTX4060，8188 MiB显存、6187 MiB空闲、利用率3%、驱动580.178.04；CPU可见8逻辑核，磁盘约111 GiB空闲。GPU在沙箱内不可见而宿主探测成功，不应把沙箱失败写成硬件损坏。没有多GPU资源证据。

当前eagm_gpu环境包元数据：Python环境路径 `/home/aaa/miniconda3/envs/eagm_gpu`；torch2.11.0+cu128、mujoco3.5.0、numpy2.1.3、matplotlib3.11.2、Pillow13.0.0。pytest/scipy/gymnasium/robosuite/mani_skill未安装。测试可先使用标准unittest；科学统计可用NumPy实现并解析案例校验，不为了方便改动旧环境。若需新依赖，独立eagm2环境锁版本。源模型依赖 `/home/aaa/HIL-OpenArm/third_party/openarm_mujoco/v2/assets`，新worktree不自动拥有原来的未跟踪symlink/大checkpoint，启动前须记录外部资产hash与恢复步骤。

历史环境报告曾写MuJoCo4.0.1，当前元数据不同；先确认实际import与原版回放数值。`reports/gpu_bench_raw.json`记录约2520 control steps/s、简单MLP约429GPU updates/s，但来自不同环境且不含二阶Dir或渲染，**不能直接当本轮训练吞吐**。Stage5A日志约369s/seed涵盖其当时方法组；HistC seed10约2356s，均不是本方案每格耗时。

## 查询成本的可核算部分

坐标FD：`2 × scales × parameters × horizon × starts`，额外center、重标与诊断另计。

- 原28维、3尺度、H8、12标签起点：16,128 control steps。
- 32维视觉adapter、3尺度、H8、12起点：18,432 steps；若H16则36,864。渲染wall-time额外，不免费。
- T2/T3方案已明确H32局部目标：28维3尺度12起点为64,512 steps；32维为73,728 steps。10s/300步成功评估是独立任务效用，不能叫H32标签直接求其成功率梯度。后续预算必须用这两个数，不能沿用T1的16,128。
- 原数据train采样4,800 steps仅供量级参考；验证/测试采样不算train但计入总成本。
- Direct FD每次更新若12起点、28维、H8、单尺度：5,376 steps；50次需268,800。故等查询预算下不能同时要求它跑50步；按预固定预算可执行步数，报告性能/step曲线。
- SPSA每次2条扰动、H8、12起点：192 steps；多方向或多尺度成倍计费。它估计与EAGM相同目标与初始化，不能给单起点SPSA却给多起点EAGM并称完全公平。

Base-extra-data将对应任务的有限差分查询额度（T1 16,128；T2 64,512）改为额外动力学采集，保持Base架构及统一评测。不同模型训练、critic、encoder更新仍需单列算力，等交互不等于等FLOPs/wall-time。

共享标签只在相同reward、snapshot、theta和stencil下复用；实际唯一交互总量与每方法逻辑成本同时报。测试diagnostics不可用于选择/relabel。初始化、验证、训练、确认、诊断的查询分桶保存。

## 分阶段资源封顶草案

以下是作者可批准的资源上限建议，不是保证能在其内完成。确认前必须用开发smoke实测更新估算；预算不足先改协议，不能运行后丢条件/种子。

| 阶段 | 工作量 | 初始资源封顶建议 | 超限处理 |
|---|---|---|---|
| 实现与正确性 | 单元/集成、原版回放、任务可达/渲染检查 | 8 CPU wall-hours + 2 GPU wall-hours | 记录阻碍，先定位错误 |
| T1开发 | 3根种子、单模块/组合与标签消融 | 24 CPU wall-hours + 8 GPU wall-hours | 未过gate不得扩确认 |
| T2开发 | 3根种子、接触成功条件与对照 | 48 CPU wall-hours + 16 GPU wall-hours | 未稳定时保留失败并重设计方案 |
| T3开发 | 3根种子、小CNN/32维adapter | 48 CPU wall-hours + 24 GPU wall-hours | 不能改成仅视觉诊断冒充完成 |
| 确认 | T1 240、T2 240、T3 90 method-seed runs | 尚不报实测承诺；启动前由smoke×矩阵+25%余量估算并冻结 | 若预计超批准资源，先提交具体调整 |

CPU wall-hours是所用worker累计占用时间，GPU wall-hours是占卡时间；总日历时间不可简单相加，单GPU也不可假定8路训练。开发建议最多2个CPU仿真worker、1个GPU训练worker，逐步实测；不终止用户现有进程腾资源。固定线程数及采样顺序避免并行改变配对。

资源估算公式：`sum(cell measured runtime × seed_count) × 1.25`，分采集/FD/模型训练/critic/策略更新/渲染/评测。最大矩阵570个method-seed runs并非570个world-model训练（FD/SPSA不训练model）。论文主张需要额外direction/norm或信息负对照确认格时，预算一并增加，不能把探索曲线当确认。

## 存储

64×64 RGB uint8每帧12,288 bytes；每100,000帧原始约1.23 GB，float32则约4.92 GB；不要逐方法重复存公共数据，也不能假设压缩比。各任务独立manifest引用不可变共享分片。开发存储上限先定40GB，保持至少30GB系统余量；确认前按实测帧数、checkpoint数、optimizer/RNG恢复文件估算完整空间，若不足扩存储后再启动。

每个训练保存best-val和final checkpoint、optimizer/RNG恢复点、完整配置与逐seed原始结果；可按预先规则限制中间checkpoint数量，不删除失败结果。外部上传大型数据须先核查文件大小与仓库策略；新报告/代码可正常Git管理。GitHub连接用已验证的github-direct，不关闭TLS。

## 当前实际消耗

本轮仅只读环境、文献与源码核查、报告和数学PDF编译；没有新增仿真性能实验、训练标签、训练checkpoint或确认结果。上列预算是未来执行限制，未假定作者已批准。
