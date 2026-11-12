# 本机OpenArm仿真、自训练与预实验

本目录已搭建Windows CPU运行入口，使用现有OpenArm v2 MJCF与MuJoCo、PyTorch。它用于RSS方案中的执行链路诊断，不连接CAN、不向机器人发送指令。

已完成主运行：21,600个采集转移、3种delay、3个模型种子、12个世界模型，另进行20次模型策略梯度更新。主流程实测173.1秒，原始方向诊断记录6,912行（含4种模型对同一批探针的预测，不是6,912个独立实验）。环境队列、全状态重放和模型解析梯度的数值检查已通过。结果见下方报告；不据此声称完整论文实验已结束。

## 快速使用

1. 双击`启动仿真.cmd`打开MuJoCo可视化窗口，关闭窗口退出。
2. 阅读`runs/pilot_20261101/预实验报告.md`查看主预实验实际结果。
3. 在本目录运行以下命令，生成新的训练与诊断记录（名称必须未使用）：

```powershell
& '.\.venv\Scripts\python.exe' experiment.py --run-name my_new_run --updates 1000 --seeds 3
& '.\.venv\Scripts\python.exe' report_results.py --run-name my_new_run
```

快速验证可使用：

```powershell
& '.\.venv\Scripts\python.exe' sim_env.py --smoke
& '.\.venv\Scripts\python.exe' experiment.py --run-name my_smoke --trajectories 6 --train-count 2 --val-count 2 --steps 90 --updates 100 --seeds 1
```

播放主实验训练后的策略参数（效果以报告为准，不保证改善）：

```powershell
& '.\.venv\Scripts\python.exe' view_sim.py --interactive --delay 1 --actor '.\runs\pilot_20261101\trained_actor_theta.npy'
```

## 环境与依赖

主机约28GiB内存、AMD Radeon 780M、无CUDA。采用CPU PyTorch 2.10.0、MuJoCo 3.5.0、Python3.13；原MJCF已实测能由3.5加载。无需在这台机器上安装完整Isaac Sim训练栈。

`.venv`具有独立解释器入口，通过`--system-site-packages`复用本机已有依赖，**不是依赖完全隔离的环境**。本次没有替换或升级系统包。精确观测版本见`requirements-observed.txt`和运行目录的environment.json。迁移到其他计算机时需安装匹配的CPU依赖，并提供项目内`third_party/openarm_mujoco/v2`及其mesh资源；不能只复制本目录期待模型自动存在。

## 模型与证据边界

- 使用现有OpenArm双臂质量、惯量和position actuator；固定base、将手指焊接在零位、关闭接触，保留手指质量几何。不是自由托箱或完整抓取仿真。
- 14维rad增量，目标累积+0.5低通滤波+0/1/2帧FIFO；物理2ms，控制17个子步即34ms，约29.4118Hz。
- 自训练包含4种世界模型的监督学习和一次20步模型策略梯度优化；不是已完成的大规模PPO/SuperMimic复现。
- 主预实验使用3个模型种子、24条轨迹和共同方向库。主报告保留所有种子、有效探针比例及反例，无统计显著性结论。
- 未完成完整R5门控算法验证、同状态控制记忆干预、滤波关闭对照、完整R0–R5比较和实机迁移。不要据此删除RSS方案全部待实验标记。

## 文件入口

`sim_env.py`：模型加载、控制执行链、完整状态快照和队列/重放检查。

`experiment.py`：数据采集、3步世界模型拟合、验证集方向库、测试集有限差分、自训练试跑与成本记录。

`report_results.py`：从真实日志生成报告、CSV摘要和科学绘图。

`view_sim.py`：仿真画面或交互查看器；默认生成PNG，`--interactive`才打开窗口。

`smoke_check.json`及`runs/smoke_pipeline`：环境检查和小规模排错记录，与主预实验分开。

现有IsaacLab/机器人控制代码及RSS稿件未被替换。新方法有效性由测量决定，若训练使真实仿真回报降低，报告也会保留。
