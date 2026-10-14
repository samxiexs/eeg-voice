# Perceived EEG-to-Speech 报告

当前研究方案为 NeuroSpeech v2.1；现有代码与接口检查仍属于 v1。

| 文件 | 用途 |
|---|---|
| [NeuroSpeech 研究方案 v2.1](neurospeech_research_plan_2026-10-13.md) | 当前方案：四个英语Perceived来源（Kymata仅英文）与KaraOne迁移；REVE适配、生理对齐、CFM、Simulation和跨被试；逐批处理只留processed |
| [数据大小与英文目录核查](dataset_storage_audit_2026-10-13.json) | Kymata英文所选文件117.767 GB、KaraOne分包24.824 GB及来源证据；processed大小为预算估算 |
| [历史执行方案 v1](06_execution_plan_2026-10-13.md) | 现有单数据集 TCN 实现的协议；研究范围已由 v2 取代 |
| [项目运行说明](../README.md) | 当前 v1 代码入口、运行顺序与 EEG-only 输入合同 |
| [清理记录](cleanup_2026-10-13.json) | 用户授权删除的旧流水线、数据缓存与失败输出目录 |
| [真实文件与模型检查](implementation_checks.json) | 实际 EEG / WavLM / Vocos / ASR 接口及完整梯度链路；科研效果仍未验证 |
| [数据单位审计](dataset_unit_audit.json) | 固定快照的原作者数值尺度核对、缩放依据及绝对单位解释限制 |

完成代码测试后，implementation_checks.json 记录真实 EEG / 音频 / 冻结模型的接口与梯度核对。正式训练与评估完成后，report 命令生成结果表；计算图核对不会被表述为语音重建性能提升。
