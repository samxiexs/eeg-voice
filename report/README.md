# Perceived EEG-to-Speech 报告

| 文件 | 用途 |
|---|---|
| [最终执行方案](06_execution_plan_2026-10-13.md) | 固定单一研究路线、数据划分、损失与科学放行门槛 |
| [项目运行说明](../README.md) | 已实现的完整入口、运行顺序与 EEG-only 输入合同 |
| [清理记录](cleanup_2026-10-13.json) | 用户授权删除的旧流水线、数据缓存与失败输出目录 |
| [真实文件与模型检查](implementation_checks.json) | 实际 EEG / WavLM / Vocos / ASR 接口及完整梯度链路；科研效果仍未验证 |
| [数据单位审计](dataset_unit_audit.json) | 固定快照的原作者数值尺度核对、缩放依据及绝对单位解释限制 |

完成代码测试后，implementation_checks.json 记录真实 EEG / 音频 / 冻结模型的接口与梯度核对。正式训练与评估完成后，report 命令生成结果表；计算图核对不会被表述为语音重建性能提升。
