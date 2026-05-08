# ROPD: Rubric-based On-policy Distillation

This repository contains the official implementation of **ROPD**, a rubric-based on-policy distillation framework for scalable black-box distillation of large language models.

> **Paper:** Rubric-based On-policy Distillation  
> **Method:** Rubric-based On-policy Distillation (ROPD)  
> **Codebase:** ROPD trainer built on a customized [verl](https://github.com/volcengine/verl) backend

## Overview

On-policy distillation (OPD) is an effective paradigm for transferring capabilities from a teacher model to a student policy. Conventional OPD methods, however, typically depend on access to the teacher's token-level logits. This white-box requirement limits their applicability to proprietary frontier models and complicates distillation across heterogeneous architectures.

ROPD studies a complementary direction:

> **Can we preserve the on-policy nature of OPD without accessing teacher logits?**

We answer this question with **rubric-based on-policy distillation**. Instead of supervising student trajectories with token-level probability distributions, ROPD converts teacher-student behavioral differences into prompt-specific semantic rubrics. These rubrics are then used to score student rollouts and provide rewards for on-policy optimization.

## Key Idea

ROPD replaces token-level imitation with structured semantic supervision.

For each input prompt:

1. The student policy generates multiple on-policy rollouts.
2. The teacher provides one or more reference responses.
3. A **Rubricator** induces prompt-specific rubrics by contrasting teacher and student responses.
4. A **Verifier** evaluates each rollout against the induced rubric.
5. The weighted rubric score is used as the reward for GRPO-style policy optimization.

This design allows ROPD to operate in black-box teacher settings, requiring only textual teacher responses rather than logits, hidden states, or tokenizer alignment.

## Method

### Problem Setting

Let $x$ denote an input prompt, $\pi_T$ a teacher model, and $\pi_\theta$ a trainable student policy. Traditional OPD supervises student-generated trajectories using the teacher's next-token distribution. In contrast, ROPD assumes that only teacher-generated text is available.

The goal is to construct an effective reward signal from black-box teacher interactions while retaining the on-policy training dynamics of OPD.

### Rubric Induction

Given a prompt $x$, ROPD collects:

- **Teacher responses** $\mathcal{Y}^T_x$
- **Student rollouts** $\mathcal{Y}^S_x$

A Rubricator then produces a prompt-specific rubric set:

$$
\mathcal{C}_x = \{c_k\}_{k=1}^{K}
$$

where each rubric item contains a textual criterion and an importance weight. These rubrics are shared across all student rollouts for the same prompt, providing a consistent group-level reward signal.

### Rubric-based Verification

For each student rollout, the Verifier determines whether the response satisfies each rubric criterion. The final rollout score is computed as a weighted pass rate:

$$
s_i =
\frac{
\sum_{k=1}^{K} w_k v_{i,k}
}{
\sum_{k=1}^{K} w_k + \epsilon
}
$$

where $v_{i,k} \in \{0,1\}$ indicates whether rollout $i$ satisfies criterion $k$. This score is used as the reward for on-policy optimization.

## Why Rubrics?

ROPD reframes distillation from token-level imitation to semantic principle transfer.

Compared with logit-based OPD, rubric-based OPD offers several practical and conceptual advantages:

- **Black-box compatibility**  
  ROPD only requires teacher text outputs, making it applicable to proprietary API-based teachers.

- **Cross-architecture flexibility**  
  ROPD does not require aligned tokenizers, shared vocabularies, or comparable output distributions.

- **Improved sample efficiency**  
  By filtering out surface-form token-level noise, rubrics emphasize task-level reasoning principles and can substantially improve data utilization.

- **Interpretable supervision**  
  The reward signal is decomposed into explicit criteria, making the training signal easier to inspect, debug, and analyze.

## Main Results

ROPD is evaluated across mathematical reasoning, scientific reasoning, medical reasoning, and instruction-following benchmarks.

The evaluation suite includes:

- **Mathematics:** AIME 2024, AIME 2025, HMMT 2025
- **Science:** GPQA-Diamond
- **Medical reasoning:** HealthBench
- **Instruction following:** IFEval

The experiments study both black-box and white-box teacher settings, including teacher-student pairs based on Qwen, Gemma, and GPT-series models.

### Black-box Distillation

In black-box scenarios, ROPD consistently outperforms representative black-box distillation baselines, including static teacher-output SFT, Teacher-as-Judge rewards, OVD, and GAD.

### White-box Comparison

Even when teacher logits are available, ROPD remains competitive with, and often surpasses, advanced logit-based OPD methods while still using only textual teacher responses.

### Sample Efficiency

ROPD achieves up to a **10x improvement in sample efficiency**, suggesting that high-level semantic rubrics can provide more useful supervision than dense token-level logits for complex reasoning tasks.

## Repository Structure

```text
.
├── algo/
│   ├── ropd/                    # Core ROPD implementation
│   │   ├── client.py            # Rubricator and verifier clients
│   │   ├── prompts.py           # Prompt rendering utilities
│   │   └── reward_manager.py    # ROPD reward manager
│   ├── ropd_pipeline.py         # Rollout and group bookkeeping
│   ├── ropd_scheduler.py        # Bounded judge request scheduler
│   ├── ropd_teacher_index.py    # Offline teacher response index
│   └── openai_env.py            # OpenAI-compatible environment resolution
├── prompts/
│   ├── rubricator.txt           # Rubric induction template
│   └── verifier.txt             # Rubric verification template
├── verl/                        # Vendored/customized verl trainer
├── training/
│   ├── train.sh                 # Training entry point
│   └── launch_judge_vllm.sh     # Optional local vLLM judge server
├── pyproject.toml               # Dependency and package configuration
└── README.md
```

## Installation

ROPD uses `uv` and `pyproject.toml` for dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra math --extra gpu-generic
```

## Configuration

Before training, configure the environment variables:

```bash
cp .env.example .env
```

Important variables include:

- `ROPD_MODEL_PATH`
- `ROPD_TEACHER_INDEX_PATH`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `ROPD_RUBRICATOR_MODEL`
- `ROPD_VERIFIER_MODEL`
- `ROPD_TEACHER_MODEL`

## Training

To launch ROPD training:

```bash
bash training/train.sh
```

This wraps:

```bash
uv run --no-sync python -m verl.trainer.main_ppo --config-name ropd
```

Hydra overrides can be passed directly:

```bash
bash training/train.sh \
    trainer.total_training_steps=100 \
    data.train_batch_size=16 \
    actor_rollout_ref.rollout.n=4
```

## Teacher Response Index

ROPD supports an offline teacher-response index to decouple teacher generation from policy optimization. The default training configuration uses an offline teacher provider, where teacher answers are keyed by prompt fingerprints and replayed deterministically during training.

This design makes it possible to:

- run teacher inference outside the training loop;
- reduce GPU memory pressure during policy optimization;
- reproduce training rewards from a fixed teacher-response cache.

## Prompt Templates

ROPD uses two core prompt templates:

- **`prompts/rubricator.txt`**  
  Generates prompt-specific rubrics from teacher-student contrasts.

- **`prompts/verifier.txt`**  
  Scores each response against the generated rubric.

These templates implement the Rubricator-Verifier pipeline described in the paper and are central to the method.

## Reproducing Paper Experiments

The main experiments use:

- GRPO optimization
- learning rate $1 \times 10^{-6}$
- batch size 32
- 8 student rollouts per prompt
- 4 teacher references
- 4 to 12 rubric items

Training data:

- DAPO-Math-17K
- RaR-Science-20K
- RaR-Medical-20K

Evaluation settings:

- temperature 1.0
- top-p 0.95
- 16 samples per problem
- maximum generation length 32,768 tokens

## Citation

If you find ROPD useful, please cite:

```bibtex
@article{ropd2026,
  title   = {Rubric-based On-policy Distillation},
  author  = {Fang, Junfeng and Hong, Zhepei and Zheng, Mao and Song, Mingyang and Li, Gengsheng and Jiang, Houcheng and Zhang, Dan and Guo, Haiyun and Wang, Xiang and Chua, Tat-Seng},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

This repository builds upon [verl](https://github.com/volcengine/verl). We thank the open-source community for making scalable reinforcement learning training infrastructure available.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
