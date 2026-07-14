# 日报｜2026-07-13

## 今日完成

- 完成 ROPD 项目训练架构与官方启动路径梳理：训练由 `training/train.sh` 进入 `verl.trainer.main_ppo`，通过 Ray 调度 FSDP Actor/Reference 与 vLLM rollout，使用 ROPD Reward Manager 串联离线 Teacher、Rubricator 与 Verifier。
- 完成最小 Smoke Test：验证了 Ray 初始化、FSDP/vLLM Worker 创建、离线 Teacher Index、Rubricator/Verifier、rollout、reward 与训练更新等主链路可以启动并执行。
- 完成一次 4 卡、25 step 的 ROPD 数学训练试验，成功保存 step 5、10、15、20、25 的完整 FSDP checkpoint（模型分片和优化器分片均齐全）。


## 训练观察

- 25 step 训练完成，4 卡 FSDP、vLLM rollout、Ray、Reward 与 checkpoint 链路均正常工作。
- 本轮训练中 Judge 有效题组平均覆盖率约为 70%，部分 step 存在 Verifier 输出格式异常导致的 reward fallback；训练未因该问题中止，但后续需要继续关注 Judge 输出稳定性与 reward 信号质量。
- 当前已具备使用不同 checkpoint 比较基模与训练后模型的条件。

## 进行中

- 正在搭建并运行基于项目官方 `val_only` 路径的离线验证流程：对 Base Model 及 step 5/10/15/20/25 checkpoint，在固定 val 集上使用 ground truth 计算统一指标。
- 验证完成后，将根据 val 结果选择最佳 checkpoint；最终仅对最佳 checkpoint 执行一次 test 集评估。

## 后续计划

- 汇总 Base Model 与各 checkpoint 的 val 指标，判断训练是否带来稳定的正确率提升。
- 对最佳 checkpoint 进行 test 集一次性评估，形成初步实验结论。
- 根据验证结果调整 Judge 并发与结构化输出稳定性设置，开展下一轮更稳定的训练实验。
