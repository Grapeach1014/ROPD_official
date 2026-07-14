# ROPD 4×H100 GPU10 实验报告

## 结论

本次实验完成了 10/10 个训练 step，4 卡 FSDP、vLLM rollout、离线 Teacher、Rubricator、Verifier、GRPO 更新与 checkpoint 保存均已实际执行。`global_step_5` 和 `global_step_10` 的四个 FSDP model shard、optimizer shard 和训练状态均完整写入。

但这不是可用于判断模型效果提升的干净实验：第 9、10 步的 Judge provider 出现 circuit breaker，第 10 步只有 1/16 个 prompt group 获得有效评分。训练基础设施成功，前 8 步的 reward 信号总体可用；末两步，尤其第 10 步的学习信号不可靠。

## 实验配置

入口为官方路径：

```bash
bash training/train.sh --config-name ropd_gpu10
```

主要参数：

| 项目 | 值 |
|---|---:|
| GPU | 4 × H100 80GB |
| Student | 本地 `Qwen3-4B` |
| 训练 step | 10 |
| `train_batch_size` | 16 prompts / step |
| `rollout.n` | 4 |
| Student rollout | 64 / step，640 总计 |
| 最大 prompt / response | 2048 / 4096 tokens |
| FSDP actor/ref micro batch | 2 / GPU |
| vLLM memory utilization | 0.50 |
| vLLM batched tokens | 65536 |
| Judge 并发 / transport in-flight | 8 / 16 |
| checkpoint | step 5、10 |

运行期间 `nvidia-smi dmon` 显示四张卡在 GPU 密集阶段可同时达到 `SM=100%`，后续稳定计算阶段约为 `SM=77–79%`，显存约为 50GB/卡。因此未观察到多卡计算倾斜；GPU 空闲区间主要来自同步 Judge 调用与阶段切换。

## 训练与 reward 汇总

| Step | 有效 Student / 64 | 有效 UID 组 | 有差异 reward 组 | Student score 均值 | Reward 均值 | Fallback 组率 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 15 | 13 / 16 | 5.344 | 0.304 | 6.25% |
| 2 | 60 | 15 | 14 / 16 | 7.625 | 0.409 | 6.25% |
| 3 | 64 | 15 | 12 / 15 | 6.766 | 0.380 | 0% |
| 4 | 64 | 16 | 15 / 16 | 5.469 | 0.290 | 0% |
| 5 | 60 | 15 | 11 / 16 | 7.297 | 0.412 | 6.25% |
| 6 | 64 | 16 | 14 / 16 | 9.328 | 0.543 | 0% |
| 7 | 64 | 16 | 14 / 16 | 5.484 | 0.305 | 0% |
| 8 | 56 | 14 | 13 / 16 | 6.219 | 0.325 | 12.50% |
| 9 | 40 | 10 | 8 / 16 | 5.266 | 0.281 | 37.50% |
| 10 | 4 | 1 | 1 / 16 | 0.391 | 0.022 | 93.75% |

全程共 640 条 Student rollout，其中 536 条有有效 Judge 结果（83.75%）。按所有 step 平均，Student score 为 5.919，Reward 为 0.327；72.38% 的 group 有组内 reward 差异。前 8 步的有效率为 492/512（96.09%），因此这些 step 能提供正常的 GRPO 相对偏好信号。

GRPO 只使用同一 prompt 的 rollout 间相对 reward。大多数 group 的四个 reward 存在明显差异，例如 `[0.11, 0.89, 0.94, 0.17]`，会产生非零 advantage。少数四条 reward 相同的 group 会得到零 advantage，这是 GRPO 的正常行为，不是训练故障。

## Checkpoint 与参数更新证据

- `checkpoints/ropd-gpu10-4gpu/latest_checkpointed_iteration.txt` 为 `10`。
- `global_step_5` 与 `global_step_10` 均保存了四个 model shard 和四个 optimizer shard。
- step-5 与 step-10 的 `model_world_size_4_rank_0.pt` 内容不同，表明模型参数在两次 checkpoint 间发生了变化。

因此，本次实验确认了真实的多卡训练与参数更新；不是仅完成 rollout 或 reward 的 dry run。

## 异常与风险

### 1. Judge provider circuit breaker

第 9 步开始出现较高失败率，第 10 步 15/16 个 group 失败。失败记录显示 Rubricator 或 Verifier provider 的 `circuit_open`：请求在 cooldown 内被拒绝。此前还出现 Rubricator `parse_error`，即模型输出了 JSON 后继续输出解释性文本，导致 JSON 解析失败。该错误累积后触发了 provider circuit breaker。

这使得末段 group 的 reward 回退为零，因而：

- 第 9 步只有 10 个有效 group；
- 第 10 步只有 1 个有效 group；
- step 10 的 reward 均值 0.022 不代表 Student 性能突然下降，而主要反映 Judge 不可用；
- 不应把 `global_step_10` 作为可信的效果 checkpoint。

尽管 fallback 已达 93.75%，debug 记录中的 `quality_gate_stop` 仍为 `false`。这表明当前质量门没有阻止该低质量 step 继续更新，需在正式长训前单独验证这一行为。

### 2. 重复 UID 分组

第 3 步有 64 个 rollout，却只有 15 个 UID group；其中一个 UID 聚合为 8 个 rollout。Reward manager 按 uid 聚合所有同题 rollout，因此重复 UID 会使该题在 GRPO 中获得高于普通题目的权重。正式训练集应按 uid 去重，或确保每个 batch 的 uid 唯一。

### 3. Teacher 与 Student 的相对评分

多个 step 中 `teacher_below_student_group_rate` 较高（约 0.60–0.93）。这不阻断当前 reward 计算，但提示离线 Teacher answer、Rubricator 和 Verifier 的一致性需要抽样审计；不能仅依赖 Student 的绝对 reward 均值宣称效果优于 Teacher。

## 对模型效果的解读

不能从这 10 个训练 step 得出泛化效果提升：每一步题目和动态 rubric 不同，reward 均值在 0.29–0.54 间波动，且最后两步被 Judge 故障污染。

可以得出的正面结论是：前 8 步中，Student 的 4 个候选回答通常可被稳定地区分好坏，奖励不为零且具备组内差异；这满足 GRPO 学习所需的基本条件。

## 建议的后续动作

1. 修复或缓解 Judge 稳定性后再做长训：先将 Judge 并发从 8 降到 4，确认 provider 无 circuit breaker，再逐步增加。
2. 对 Rubricator 增强结构化输出处理：拒绝 JSON 后附加自由文本的响应，或在解析前可靠提取唯一 JSON 对象。
3. 在每个 step 后强制检查 fallback rate；当其超过质量门阈值时阻止 PPO update，而不是写入零 reward。
4. 对训练 parquet 按 uid 去重，保证每个 prompt 的 rollout group 大小恒为 4。
5. 使用固定且与训练集不重叠的评测集，在 step 0、5、10 或更长间隔比较同一批题目的 reward、正确率和答案文本。正式续训时优先从 `global_step_5` 重启并在 Judge 稳定后重跑后续 step；不要将 step-10 checkpoint 直接视作质量最优版本。

## 产物位置

- Reward 调试记录：`outputs/ropd-gpu10-4gpu/reward_debug/ropd_reward_debug.jsonl`
- 中间 checkpoint：`checkpoints/ropd-gpu10-4gpu/global_step_5`
- 末 checkpoint：`checkpoints/ropd-gpu10-4gpu/global_step_10`
