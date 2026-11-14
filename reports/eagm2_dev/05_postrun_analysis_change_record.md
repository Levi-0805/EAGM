# 冻结批次后的独立分析修正记录

2026-11-14。新增 `reports/eagm2_dev/analyze_development_postrun.py`，复制冻结分析程序后仅在报告目录修正。未修改 `scripts/eagm2`、`tests/eagm2` 或训练代码；不改变任何原始结果。`parents[2]`仍解析为仓库根目录。未读取实际 `development_v1` 的部分效果。

变更：

- 方法列表优先用 `resolved_methods`，兼容旧版逗号字符串；默认 `methods=None` 不再触发 `list(None)`。校验method目录名，完整预声明列表中的缺失结果仍保留。
- `direct_controls_resource.json`作为两种direct controls共同资源范围读取；不是两份独立费用，不得重复分摊相加。
- `training_simulator_budget`所有数值字段进入每方法归属表，显式区分实际额外成本；布尔标记不转换成步数。
- 仍支持direct结果无status/seed，明确合法stop_reason且physical存在才判完成；`optimization.json`的failed即使无results也保留为失败，实际/预留步数分开保留。
- 原始NMSE增加分母标识。Base-extra-data用common+extra train尺度，其他模型用common train尺度；涉及extra-data的跨方法NMSE/validation NMSE差值为null并标为不可同尺度比较，原值仍显示。不会重新定义或用test重新拟合归一化。
- 继续只读9授权探索根种子，不合并任务、不选赢家、不将episode当种子；确认草案依然 `not_authorized`。

## 验证

`test_analyze_development_postrun.py`在自动删除的临时目录构造三任务×三根种子、15种方法的合成结构，使用显式attempt `development_v1`。覆盖默认methods=None/full resolved名单、direct异构结果/checkpoint、failed与missing、validation NMSE、预算归属与direct资源、失败actual成本、禁止覆盖、确认锁。该夹具只验证程序，数值不是实验结果。

运行 `python reports/eagm2_dev/test_analyze_development_postrun.py`：1项完整集成测试通过，日志 `tests_analysis_postrun.log`；测试前后冻结分析脚本SHA256相同。未产生实际实验 `06_*` 报告。

## 批次全部训练结束后的命令

独立completion worker先确认九根状态与保留的原自动分析失败日志，再在仓库根运行：

```bash
/home/aaa/miniconda3/envs/eagm_gpu/bin/python reports/eagm2_dev/analyze_development_postrun.py --attempt development_v1 --output-dir reports/eagm2_dev/development_v1_postrun_analysis --report-tag development_v1_postrun
```

若目标tag已存在，程序拒绝覆盖，需明确新tag。原批次可能因冻结旧分析的None解析错误标为failed；那是后处理失败，不应据此掩盖训练完成状态或删除原日志。新分析输出也不回写原batch manifest。实际失败/缺失不能因新分析通过被改成completed。
