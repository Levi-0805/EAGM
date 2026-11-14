# 架构对比与输入权限（设计稿，尚未实现）

| ID | 模块 | 可部署输入 | 状态传播 | 用途与限制 |
|---|---|---|---|---|
| O-Base/O-Dir | 历史U0+residual+F；无/旧Huber | x,c_obs,a | 解析r/f；无B隐状态 | 原版复现，不悄悄修成新方法 |
| D0 | delay-free名义U0 | r,f,a | 无 | 解析映射对照；不是B executed oracle |
| D1 | delay-aware名义U0 | r,f,d,a,合法filter缓存 | 精确名义FIFO | 验证名义索引；不处理隐藏B链路 |
| F | Free command head+F | 与D+residual同输入 | 无额外隐状态 | 测结构约束是否有独立价值 |
| D | delay-aware U0+residual+F | x,c_obs,a,filter缓存 | 名义FIFO | 单模块候选 |
| H-MLP | flattened history MLP | x/c_obs历史、过去a | 固定窗口 | 参数量匹配history对照 |
| H-GRU | 修复后的已有窗口GRU | 同上 | 每步重编码窗口 | 必须修正HistC错位，不能当成新发明 |
| M | posterior+prior execution latent | 当前可观测记录、过去a、h | Q观测更新，G想象递推 | 新候选；无完整隐状态保证 |
| DM | D+M | 同M及合法filter缓存 | 名义FIFO+latent | 组合只有独立胜出才能晋级 |
| G | 梯度目标变化 | 与选定固定架构完全相同 | 架构不变 | 标签与query预算匹配，隔离损失作用 |
| DMG | D+M+G | 同DM | 同DM | 不预设为最终方法 |
| P-Ref | privileged模型 | B内部队列、u_prev、u_smooth、参数 | 显式或learned | 信息参考，不保证可达性能上界；不能部署比较混报 |
| V-Base | RGB基础encoder+32维adapter | 当前RGB（其他输入按条件明确） | 视觉model latent | 真图像闭环；冻结基础后的adapter优化 |
| V-Exec | RGB+observable execution record | RGB,r,f,d,过去a | 明确名义缓存 | 状态q/qdot禁止部署输入 |
| V-Mem | RGB+causal execution estimator | RGB与允许的可观测历史 | Q/G latent | 必须检验相机/初态/执行配置泛化 |

容量控制：报告所有可训练与冻结参数；尽量让总可训练参数在±5%内，同时列FLOPs/显存/推理耗时。GRU对照须匹配隐维/窗口/输入；MLP可调宽度但只能用开发配置冻结，不按测试表现搜索。视觉encoder的冻结参数和预训练数据成本不能隐去。

输入权限测试：分别扰动未来观测、B内部状态、snapshot元数据、测试标签；可部署输出在其合法过去输入固定时必须不变。允许训练命令标签/奖励state标签不等于允许模型部署读取它们。随机改变RGB必须能改变动作且计算图连通，不通过图像统计诊断替代。

Oracle概念：数据生成器提供更多信息只扩大可用信息集合；受限架构、有限训练和优化器并不保证性能单调，因此P-Ref最多是经验信息参考。只有在同目标上对策略类全局最优化且信息类包含关系成立时，最优值才有形式上的上界关系；当前实验不满足该保证。

代码差异预期：新模块未来放独立eagm2命名空间，原文件保留；修复版数据/指标有版本号与迁移报告。当前阶段没有执行这些修改。
