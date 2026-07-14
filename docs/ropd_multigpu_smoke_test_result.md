# ROPD 四卡 Smoke Test：运行结果（2026-07-13）

## 结论

本次运行已经确认 **多卡 Ray、模型加载、vLLM rollout、离线 Teacher Index、Anthropic
Rubricator、Anthropic Verifier 和 ROPD reward** 可以连通并完成一次真实的 reward
闭环。

持久化的 reward 证据位于：

`outputs/ropd_smoke/reward_debug/ropd_reward_debug.jsonl`

该文件含一个 reward 调用记录，覆盖 4 个 prompt、每 prompt 2 个 rollout（共 8 个
student response）。所有 group 的状态为 success，且没有 fallback。

## 实测结果

| 项目 | 结果 |
| --- | --- |
| Ray / 四卡资源 | 已在此前 probe 中确认 4 个 GPU worker 分别分配 GPU 0--3。 |
| 本次 trainer 配置 | `nnodes=1`，`n_gpus_per_node=4`，FSDP actor，vLLM TP=1。 |
| 训练数据 | 4 个与 Teacher Index 精确匹配的 prompt。 |
| rollout | 8 / 8 responses；每条生成长度均为 256（触及 smoke 上限）。 |
| Offline Teacher | 4 / 4 group 成功读取 index；Teacher scores 为 23、6、20、16。 |
| Rubricator | 4 / 4 成功；产生 4 个非空 rubric hash。 |
| Verifier | 4 / 4 成功；无 transport / parsing / schema error。 |
| Reward quality gate | `effective_group_rate=1.0`，`fallback_rate=0.0`，`excluded_student_count=0`。 |
| 更新样本掩码 | 8 个 response 均为 `true`，没有被 reward pipeline 排除。 |

关键 reward control：

```text
total_group_count: 4
effective_uid_count: 4
effective_student_count: 8
fallback_rate_initial: 0.0
fallback_rate_repaired: 0.0
quality_gate_stop: false
update_mask: [true, true, true, true, true, true, true, true]
```

## 重要限制：本步没有有效学习信号

四个 group 的两条 student response 均被 Verifier 评为 0 分：

```text
student_scores: [0.0, 0.0]
reward_scores:  [0.0, 0.0]
```

ROPD 使用 GRPO group mean baseline。两条 rollout 的 reward 同为零时，二者的优势
均为零；因此即便 PPO/FSDP update 调用已进入，梯度也会是零或等价于零。这不是
Teacher、Rubricator 或 Verifier 失败，而是初始 4B policy 在 256-token 截断下没有
获得 rubric 分数且组内没有 reward 差异。

因此本次结果应表述为：

> **训练基础设施和真实 ROPD reward 闭环通过；非零 reward / 非零 actor gradient
> 尚未由本次最小样本证明。**

此外，当前 smoke 配置刻意使用 `save_freq=-1`，未留下 `global_step_1` checkpoint；
Hydra 的 `main_ppo.log` 也为空。因此仅凭当前持久化文件，不能独立审计进程的最终
exit code、actor grad norm 或 checkpoint 写入。若终端命令确实以 0 退出，则可确认
一次 1-step trainer loop 正常结束；否则应将终端末尾日志保留为最终判据。

## 建议的下一次证明性运行

为了把 smoke test 提升为“确认一次非零反向更新”，保持正式路径不变，仅做以下
Hydra 覆盖：

```bash
trainer.save_freq=1 \
actor_rollout_ref.rollout.response_length=512 \
data.max_response_length=512
```

并保持 `rollout.n=2`。验收时检查：

1. `checkpoints/.../global_step_1/` 存在；
2. trainer 日志中出现 step-1 actor metrics（特别是 `actor/grad_norm`）；
3. reward debug 至少有一个 group 的两个 `student_scores` 不完全相同，或至少存在
   一个非零 reward。

若仍全部为零分，可增加 response length、换取更容易得到部分分的 prompt，或保留
更大的 rollout 数；不应以 mock reward 替代真实 Judge。
