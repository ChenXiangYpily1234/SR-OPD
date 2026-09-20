# SR-OPD: Success-Referenced On-Policy Distillation

This repository contains the implementation of **SR-OPD**, introduced in *SR-OPD: Success-Referenced Pre-Query Rollout Routing for On-Policy Distillation*.

SR-OPD selects failed Student rollouts for Teacher supervision using successful same-prompt siblings as references. It ranks candidates by the magnitude and persistence of hidden-state departure, adjusted for Teacher input cost. Each mixed-outcome prompt receives one Teacher query. Routing reuses the Student log-probability forward and preserves the sampled-token OPD objective.

The implementation is built on [verl](https://github.com/volcengine/verl). The paper studies Qwen3-4B → Qwen3-1.7B, Skywork-OR1-Math-7B → DeepSeek-R1-Distill-Qwen-1.5B, and Granite-3.3-8B-Instruct → Granite-3.3-2B-Instruct.

## Repository layout

- `verl/`: modified verl training framework and routing implementation.
- `bash/sr_opd/`: SR-OPD launchers, calibration, and routing baselines.
- `bash/train/`: shared trainer and Full-OPD, TA-OPD, and TLR-OPD launchers.
- `bash/eval/`: checkpoint merging and downstream evaluation.
- `bash/produce/`: evaluation and cost summaries.
- `datasets/`: DAPO-Math-17k and six mathematical reasoning benchmarks.

## Key implementation files

- `verl/verl/utils/boundary_opd.py`: centered hidden states and persistent-departure scoring.
- `verl/verl/utils/ff_opd.py`: mixed-prompt gating and rollout selection.
- `verl/verl/utils/boundary_calibration.py`: frozen centering statistics.
- `verl/verl/workers/actor/dp_actor.py`: hidden-state capture from the Student forward.
- `verl/verl/trainer/ppo/ray_trainer.py`: routing, Teacher scoring, and training integration.

## Quick start

Use a CUDA environment with PyTorch, vLLM, Ray, Transformers, and FlashAttention. Install the included verl package and evaluation dependencies:

```bash
pip install -e verl/
pip install pandas sympy pylatexenc
```

Run from the repository root on a dedicated eight-GPU node. Set the model paths before launching:

```bash
export TEACHER_MODEL_PATH=/path/to/Qwen3-4B
export STUDENT_MODEL_PATH=/path/to/Qwen3-1.7B
export ACTOR_MODEL_PATH="$STUDENT_MODEL_PATH"
export REWARD_MODEL_PATH="$TEACHER_MODEL_PATH"

bash bash/sr_opd/main/run_sr_opd_m16.sh 42
```

The main setting uses four rollouts per prompt and 16 hidden-state positions. Centering statistics are prepared automatically on first use. For the resolution variants, use `run_sr_opd_m8.sh` or `run_sr_opd_m32.sh`. Set the corresponding model paths for the Skywork or Granite pair.

## Baselines

```bash
# Matched routing baselines
bash bash/sr_opd/main/run_frontier_tlr.sh 42
bash bash/sr_opd/strategy/run_random_wrong.sh 42
bash bash/sr_opd/strategy/run_shortest_wrong.sh 42

# Complete-system baselines
bash bash/train/full-opd.sh --epochs 1 --seed 42
bash bash/train/ta-opd.sh --token-retain-ratio 0.10 --seed 42
bash bash/train/tlr-opd.sh --seed 42
```

## Evaluation

```bash
bash bash/eval/eval.sh \
  --run-dir runs/Qwen3-4B__to__Qwen3-1.7B/sr_opd/sr-opd-m16-seed42 \
  --step 279

python3 bash/produce/build_method_table.py runs --output method_table.md
```

Evaluation reports Avg@16 and Pass@16 on AIME24, AIME25, AMC23, HMMT24, HMMT25, and MATH-500.

## Citation

```bibtex
@misc{sropd2027,
  title  = {SR-OPD: Success-Referenced Pre-Query Rollout Routing for On-Policy Distillation},
  author = {Anonymous Authors},
  year   = {2027},
  note   = {Manuscript under review}
}
```

## License

Apache-2.0. Upstream license and attribution notices are preserved in the vendored verl tree.
