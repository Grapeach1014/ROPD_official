# ROPD 单机四卡 Smoke Test 报告

## 状态

**实现和静态预检：通过。**

**完整一轮训练：待在目标四 H100 主机执行。** 当前自动化工作区只暴露了一个
MIG 设备，不能代替目标主机完成四卡 FSDP/vLLM 进程组的真实启动。本文不把静态
检查或 Ray 探针误报为训练成功。

## 训练链路（按源码）

```text
training/train.sh
  -> verl.trainer.main_ppo.run_ppo()
  -> Ray TaskRunner.run()
  -> ResourcePoolManager([n_gpus_per_node] * nnodes)
  -> RayWorkerGroup（同一 global_pool，共置）
       -> AsyncActorRolloutRefWorker（FSDP actor）
       -> vLLM rollout engine
  -> rollout responses
  -> RopdRewardManager
       -> OfflineTeacherIndex(uid + canonical raw_prompt hash)
       -> Anthropic Rubricator
       -> Anthropic Verifier
       -> token-level reward
  -> GRPO advantage
  -> FSDP actor update
  -> checkpoint / clean exit
```

关键实现在：

| 阶段 | 源码位置 | Smoke 覆盖方式 |
| --- | --- | --- |
| Ray 初始化与 runtime environment | `verl/trainer/main_ppo.py`、`verl/trainer/constants_ppo.py` | 四 GPU Ray probe 已通过；正式入口保留 `working_dir=None` 防止 Ray/uv 创建空 `.venv`。 |
| 资源池与共卡 worker | `verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py` | `trainer.nnodes=1`、`n_gpus_per_node=4`。 |
| FSDP actor / GRPO | `verl/workers/fsdp_workers.py`、`verl/trainer/ppo/core_algos.py` | batch=4、rollout n=2、每 GPU micro batch=1。 |
| vLLM rollout | `verl/workers/rollout/vllm_rollout/` | TP=1，四卡数据并行 rollout，256 response tokens。 |
| 数据集 | `verl/utils/dataset/rl_dataset.py` | 真实 parquet、四样本；保留 `prompt` 与 `extra_info.index`。 |
| Teacher | `algo/ropd_teacher_index.py` | 指定离线 JSONL，不进行在线 teacher 生成。 |
| Rubricator / Verifier | `algo/ropd/client.py`、`algo/ropd_clients.py` | Anthropic profile 和模型名全部由环境变量选择。 |
| reward / retry / fallback | `algo/ropd/reward_manager.py` | 写出 reward debug JSONL，要求无 fallback。 |

## Smoke Test 设计

新配置 `verl/trainer/config/ropd_smoke.yaml` 继承正式 `ropd.yaml`，没有替换任何
trainer、worker、reward 或 rollout 代码。它仅缩小规模：

- 4 个训练 prompt；
- 每 prompt 2 个 rollout（保留 GRPO 分组而非退化为 n=1）；
- 512 prompt tokens + 256 response tokens；
- 1 个 training step；
- 4 个 GPU、FSDP、vLLM TP=1；
- teacher answer count=1（index 本身仍为每 prompt 的多答案 index）；
- rubricator/verifier 并发=1，降低外部 API 压力。

所用数据是 `tmp/dapo-math-17k.train.teacher_index_aligned.smoke1.parquet`。已做
离线校验：800/800 条 parquet row 的 `extra_info.index` 加 canonical prompt hash 都能
命中 `tmp/opus_teacher_index_math100_n4.jsonl`。因此本测试不会以 mock 或 fallback
替代 Teacher。

## 所做修改及原因

1. 新增 `verl/trainer/config/ropd_smoke.yaml`：正式 ROPD recipe 的最小 Hydra 配置。
2. 新增 `tests/ray_startup_probe.py`：独立验证 Ray 及四个单 GPU worker；它不替代
   训练，只用于把 Ray 启动问题与训练问题分离。
3. 新增本报告与 `docs/ropd_multigpu_smoke_test.md`：记录命令、证据和验收条件。

没有修改 ROPD、FSDP、vLLM、Ray 或 Judge 的核心训练逻辑。

## 已执行的验证与关键日志

### 1. 目标机运行环境

目标机已确认：4×H100 80GB、Torch 2.9.0+cu126、vLLM 0.12.0、Ray 2.53.0、
Transformers 4.57.1，且 `torch.cuda.is_available()=True`、GPU count=4。

### 2. Ray 四卡 worker probe

执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  timeout 180s uv run --no-sync python tests/ray_startup_probe.py --expected-gpus 4
```

关键结果：

```text
Ray cluster resources: ... 'GPU': 4.0 ...
GPU probe 0: ... 'ray_visible_gpus': '0' ... H100 ...
GPU probe 1: ... 'ray_visible_gpus': '1' ... H100 ...
GPU probe 2: ... 'ray_visible_gpus': '2' ... H100 ...
GPU probe 3: ... 'ray_visible_gpus': '3' ... H100 ...
PASS: Ray started and scheduled all requested GPU tasks.
```

首次 probe 暴露 Ray 自动打包项目后在临时 cwd 中被 uv 创建空 `.venv` 的问题；probe
已改为 `runtime_env={"working_dir": None}`。正式 `main_ppo` 也使用同类 safeguard。

### 3. Teacher/index 与 judge 配置预检

在不发起网络调用的情况下，已验证：

```text
teacher= offline_index anthropic_messages claude-opus-4-6 480.0 8192
rubricator= anthropic anthropic_messages claude-opus-4-6
verifier= anthropic anthropic_messages claude-opus-4-6
PASS: offline teacher lookup succeeded for first smoke row
```

这同时证明 teacher fingerprint 与用户提供的 index 相符。Teacher 没有被改成在线调用。

### 4. Hydra/verl 静态配置预检

`ropd_smoke` 已通过正式 `validate_config`：

```text
[validate_config] All configuration checks passed successfully!
PASS: ropd_smoke satisfies the same verl static configuration checks as the Trainer.
```

## 目标机实际执行命令

以下变量必须在目标机中以安全方式预先导出：`ANTHROPIC_AUTH_TOKEN`、
`ANTHROPIC_BASE_URL`。不要将 token 写入本仓库或日志。

```bash
export ROPD_MODEL_PATH=/home/work/migoo_ai_public/ruiyu_lu/models/Qwen3-4B
export ROPD_TEACHER_INDEX_PATH="$PWD/tmp/opus_teacher_index_math100_n4.jsonl"
export ROPD_TEACHER_PROFILE=claude_compass
export ROPD_TEACHER_MODEL=claude-opus-4-6
export ROPD_RUBRICATOR_PROVIDER=anthropic
export ROPD_RUBRICATOR_PROFILE=claude_compass
export ROPD_RUBRICATOR_MODEL=claude-opus-4-6
export ROPD_VERIFIER_PROVIDER=anthropic
export ROPD_VERIFIER_PROFILE=claude_compass
export ROPD_VERIFIER_MODEL=claude-opus-4-6
export EXPERIMENT=ropd-smoke-$(date +%Y%m%d-%H%M%S)

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  timeout 45m bash training/train.sh --config-name ropd_smoke
```

`training/train.sh` retains its normal official entrypoint; Hydra accepts the
second `--config-name` and uses `ropd_smoke` (verified with `--cfg job`).

## Completion criteria

Mark the smoke test **successful** only when the command exits with status 0
and all conditions below hold:

1. one trainer step is logged;
2. `outputs/ropd_smoke/reward_debug/ropd_reward_debug.jsonl` exists;
3. every last-record group has `final_status == "success"`, `fallback_used ==
   false`, and a non-empty `rubric_hash`;
4. process exits without a Ray worker, FSDP, vLLM, Teacher-index, Rubricator,
   Verifier, or backward/update exception.

Use:

```bash
uv run --no-sync python - <<'PY'
import json
from pathlib import Path

path = Path('outputs/ropd_smoke/reward_debug/ropd_reward_debug.jsonl')
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert records, f'No reward records in {path}'
groups = records[-1]['groups']
assert groups, 'Reward record has no groups'
assert all(item['final_status'] == 'success' for item in groups), groups
assert all(not item['fallback_used'] for item in groups), groups
assert all(item['rubric_hash'] for item in groups), groups
print(f'PASS: {len(groups)} ROPD groups completed without fallback.')
PY
```

## Failure triage

| First failure point | Likely cause | Next action |
| --- | --- | --- |
| before `Started a local Ray instance` | stale Ray session or environment | run the Ray probe; inspect `/tmp/ray/session_latest/logs`. |
| worker has no `ray` / creates empty `.venv` | Ray runtime env packages project cwd | ensure official `main_ppo` runtime env is used; do not launch workers through a separately packaged cwd. |
| teacher fingerprint/index error | profile/model/timeout/index mismatch | retain teacher `offline_index`; use `claude_compass`, the specified model, 480 seconds and specified index. |
| rubricator/verifier HTTP/auth error | missing token, endpoint, profile or gateway response | validate secure environment export and inspect the redacted provider error. |
| vLLM/FSDP OOM | insufficient hybrid allocation | lower `gpu_memory_utilization`, response length, or batch only after retaining four GPUs and n=2. |
| GRPO/update error | invalid batch partition | preserve batch=4, n=2, mini batch=4, micro batch per GPU=1. |
