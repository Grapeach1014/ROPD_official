# ROPD 4×H100 25-step pilot：验证报告

**验证完成时间**：2026-07-13 11:36–12:22 UTC  
**实验目录**：`outputs/ropd-repro-90m-4gpu/validation/`  
**结论摘要**：六个候选（基座模型及 step 5/10/15/20/25）均成功完成验证。step 15 在本次验证集上表现最佳；相较基座模型，单次 rollout 准确率从 6.0% 提升至 16.5%，官方 `best@4` 指标从 13.21% 提升至 33.77%。但该结果来自 50 道题和随机采样，曲线并不单调，当前应将 step 15 作为后续独立测试的候选，而不能将其 val 成绩作为最终泛化结论。

## 1. 验证目的与对象

本次验证用于比较一次 25-step ROPD pilot 训练前后的模型状态，并从保存的 checkpoint 中选择后续测试候选。

已验证的候选为：

| 候选 | 权重来源 |
| --- | --- |
| `base` | 本地 `/home/work/migoo_ai_public/ruiyu_lu/models/Qwen3-4B`，未加载训练 checkpoint |
| `step_5` | `checkpoints/ropd-repro-90m-4gpu/global_step_5` |
| `step_10` | `checkpoints/ropd-repro-90m-4gpu/global_step_10` |
| `step_15` | `checkpoints/ropd-repro-90m-4gpu/global_step_15` |
| `step_20` | `checkpoints/ropd-repro-90m-4gpu/global_step_20` |
| `step_25` | `checkpoints/ropd-repro-90m-4gpu/global_step_25` |

训练本身使用 400 个 train prompt、batch size 16、每题 4 个 rollout，共 25 个更新 step。验证独立读取 `data/math/val.parquet`，不读取训练 Teacher Index。

## 2. 验证协议

- 数据：`data/math/val.parquet`，50 道 held-out 数学题。
- 推理：4 卡 H100、每题 `n=4` 个随机 rollout，`temperature=1.0`、`top_p=0.95`。
- 计分：项目的 `naive` reward manager，即 math ground-truth scorer；验证期间不初始化 Teacher、Rubricator 或 Verifier，因此该成绩不受在线 Judge API 波动影响。
- 规模：每个候选 50 × 4 = 200 条生成，共 1,200 条生成。
- 入口：[run_ropd_repro_90m_val_run.sh](/home/work/migoo_ai_public/ruiyu_lu/opd/ROPD_official/run_ropd_repro_90m_val_run.sh)；配置：[ropd_repro_90m_val.yaml](/home/work/migoo_ai_public/ruiyu_lu/opd/ROPD_official/verl/trainer/config/ropd_repro_90m_val.yaml)。

六个目录均具有 `metrics.json`、生成 JSONL 及 `main_ppo.log`。step 5 日志中存在一次 Torch Inductor 临时文件警告，但该进程仍写出了完整的 200 条生成和 metrics，故不影响本次结果使用。

## 3. 官方 metrics

下表直接转录各候选 `metrics.json` 中的核心字段。`mean@4` 表示 4 个 rollout 的平均得分；`best@4` 是框架输出的候选聚合指标。

| 候选 | `acc/mean@4` | `acc/best@4` | `reward/mean@4` | 相对 base：mean acc |
| --- | ---: | ---: | ---: | ---: |
| base | 6.00% | 13.21% | -0.88 | — |
| step 5 | 11.00% | 25.37% | -0.78 | +5.00 pp |
| step 10 | 8.00% | 17.01% | -0.84 | +2.00 pp |
| **step 15** | **16.50%** | **33.77%** | **-0.67** | **+10.50 pp** |
| step 20 | 8.00% | 15.79% | -0.84 | +2.00 pp |
| step 25 | 15.00% | 25.72% | -0.70 | +9.00 pp |

按官方 metrics 选择，**step 15 是当前最佳 checkpoint**。它相对 base 的 `acc/best@4` 增加约 20.55 个百分点，平均 reward 也从 -0.88 改善到 -0.67。

## 4. 对生成文件的独立复核

对每个 `generations/*.jsonl` 逐条汇总，可得到以下直观统计。此处的“直接 pass@4”定义为 50 道题中“至少一个 rollout 的 `acc=true`”的题目占比；它是对生成文件的独立统计，不应与框架的 `acc/best@4` 字段混为同一指标。

| 候选 | 正确 rollout / 200 | 直接 pass@4（题数 / 50） | 无法提取 `pred` 的 rollout |
| --- | ---: | ---: | ---: |
| base | 12 | 18%（9） | 70 |
| step 5 | 22 | 34%（17） | 63 |
| step 10 | 16 | 22%（11） | 49 |
| **step 15** | **33** | **42%（21）** | 45 |
| step 20 | 16 | 20%（10） | 49 |
| step 25 | 30 | 30%（15） | 38 |

该复核与官方排序一致：step 15 在 50 题中有 21 题至少产生过一个正确解，明显高于 base 的 9 题；step 25 位居第二。

## 5. 结果解读

1. **训练产生了可见的正向信号。** step 5 已优于 base，step 15 达到本次最高点，说明数据流、reward、更新及 checkpoint 加载并非“无效训练”。
2. **曲线不单调。** step 10 和 step 20 都明显回落，step 25 恢复但未超过 step 15。这与仅 25 个更新、400 条训练题、随机 rollout 和 GRPO/在线 reward 噪声的 pilot 设定相符。
3. **当前最合理的模型选择是 step 15。** 不建议仅依据最后一个 checkpoint（step 25）作为候选；本次 checkpoint sweep 的价值正是在于发现中间 checkpoint 更好。
4. **答案格式仍是明显的测量因素。** 六个候选均存在大量空 `pred`；例如 base 为 70/200，step 15 仍为 45/200。生成样本中也可见数学推理正确但结尾写成 `**Answer: 6**`，导致提取结果为 `6**` 而被严格 ground-truth scorer 判错的情况。因此当前指标同时衡量解题能力与遵循最终答案格式的能力。

## 6. 有效性边界

- 验证集仅 50 题，step 15 的直接 pass@4 为 21/50；样本量不足以精确区分相近 checkpoint。
- 推理开启随机采样，且本轮未记录多个随机种子下的重复评估，因此 step 15 与 step 25 的差异还可能部分包含采样方差。
- 训练只覆盖 25 step / 400 prompt，远低于论文 Math track 的规模，不能用于与论文的最终数值直接对齐。
- 本次是 ground-truth math 验证，不评估 ROPD 的 Teacher/Rubricator/Verifier Judge 质量；它刻意回答的是“训练后模型能否在 held-out 数学题上提高”。

## 7. 建议的下一步

1. 先固定 step 15 为候选，保留 step 25 作为次候选。
2. 使用尚未用于模型选择的 `data/math/test.parquet` 对 step 15 做一次独立评估；不要用该 test 结果继续挑 checkpoint。
3. 为减小方差，可在 val 上对 step 15 与 step 25 使用多个固定随机种子重复，或扩大验证题数；若希望接近论文协议，可逐步将 `n` 从 4 增至 8/16。
4. 单独检查 answer extractor 与提示词约束，确保最终行严格输出 `Answer: <答案>`、不要加粗或附加 Markdown。该修复可能直接减少格式性误判。
5. 若独立 test 仍显示 step 15 的提升，再增加训练数据与 step 数，并继续每隔固定 step 保存/验证 checkpoint。

## 8. 结果文件索引

- [base metrics](/home/work/migoo_ai_public/ruiyu_lu/opd/ROPD_official/outputs/ropd-repro-90m-4gpu/validation/base/metrics.json)
- [step 15 metrics](/home/work/migoo_ai_public/ruiyu_lu/opd/ROPD_official/outputs/ropd-repro-90m-4gpu/validation/step_15/metrics.json)
- [全部验证目录](/home/work/migoo_ai_public/ruiyu_lu/opd/ROPD_official/outputs/ropd-repro-90m-4gpu/validation)
- [step 15 生成记录](/home/work/migoo_ai_public/ruiyu_lu/opd/ROPD_official/outputs/ropd-repro-90m-4gpu/validation/step_15/generations/15.jsonl)
