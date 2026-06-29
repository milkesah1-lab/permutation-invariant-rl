# Permutation-Invariant Reinforcement Learning for Autonomous Driving

Deep reinforcement learning agents for autonomous highway driving where the observation is an
**unordered, variable-length set of surrounding vehicles**. Standard MLP policies treat this set as a
fixed, ordered vector and break when vehicle order or count changes. This project implements
**permutation-invariant policy/value networks** and trains them with a from-scratch **PPO**
implementation under a **4-stage curriculum** of increasing traffic difficulty.

Built in PyTorch on [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) (Gymnasium).

## Why permutation invariance

The agent observes a matrix of nearby vehicles (per-vehicle features). The *identity ordering* of those
rows is arbitrary — a good policy should produce the same action regardless of row order. This project
compares architectures that enforce that property against a standard MLP baseline.

## Architectures compared

| Model | Backbone | Idea |
|-------|----------|------|
| `basic_mlp` | MLP baseline | Flattened observation — **not** permutation invariant |
| `deep_sets_mean` | Deep Sets | Per-vehicle encoder + **mean** pooling |
| `deep_sets_mean_max` | Deep Sets | Per-vehicle encoder + **mean + max** pooling |
| `self_attention` | Self-attention | Attention over the vehicle set |
| `uncertainty_penalty_PPO` | Uncertainty-aware | PPO with an uncertainty penalty (lambda sweeps) |

## Method

- **PPO** implemented from scratch (actor/critic, GAE, clipped objective) — see `ppo.py`.
- **Permutation-invariant networks** — see `PI_network.py` in each model directory.
- **Curriculum learning** across 4 stages: open lane → easy overtake → mixed traffic → dense traffic.
- **Controlled comparison**: same curriculum, multi-seed studies, evaluated on episodic return and
  collision rate. See `run_*.py` study drivers.

## Results

Final-policy evaluation across curriculum stages (10 episodes/stage, full-length eval).
Higher return is better; lower collision rate is better.

| Model | Stage 3 (mixed) return | Stage 3 collision | Stage 4 (dense) return | Stage 4 collision |
|-------|----:|----:|----:|----:|
| `basic_mlp` (baseline) | 205 | 0.70 | **108** | **1.00** |
| `deep_sets_mean` | 155 | 0.80 | 88 | 0.90 |
| `deep_sets_mean_max` | **295** | **0.50** | **214** | **0.80** |
| `self_attention` | 122 | 0.80 | 60 | 1.00 |

**Headline:** under dense traffic the MLP baseline collapses (collides every episode), while the
permutation-invariant **mean+max** model retains the highest return and the lowest collision rate —
evidence that respecting the set structure of the observation improves robustness as traffic scales.

> Caveat: evaluation uses 10 episodes per stage; numbers above are from one fair multi-model run.
> Collision rates are high across the board — dense highway driving is hard — so results are best read
> as *relative* model comparisons, not absolute safety claims.

## Repository structure

```
Deep_sets_mean_model/              # Deep Sets (mean) — PI_network.py, ppo.py, curriculum_train.py, eval_policy.py
mean+max/                          # Deep Sets (mean+max)
self_attention_model/              # Self-attention backbone
uncertainty_penalty_PPO/           # Uncertainty-aware PPO
uncertainty_aware_*_backbone/      # Uncertainty-aware variants per backbone
baseline1/                         # MLP / shared baseline code
run_*.py                           # Experiment study drivers (multi-seed, multi-stage)
```

Each model directory is self-contained: `PI_network.py` (network), `ppo.py` (algorithm),
`curriculum_train.py` (training), `eval_policy.py` (evaluation), `highway_configs.py` (env configs).

## Running

Requires Python 3.10+, PyTorch, Gymnasium, and `highway-env`:

```bash
pip install torch gymnasium highway-env numpy
```

Train one model through the curriculum (run from inside its directory):

```bash
cd Deep_sets_mean_model
python curriculum_train.py
```

Reproduce the full multi-model comparison study:

```bash
python run_fair_curriculum_experiments.py
```

Training artifacts (model weights, CSV logs, videos) are written locally and are gitignored.

## Tech

PyTorch · Gymnasium · highway-env · NumPy · custom PPO · Deep Sets · self-attention
