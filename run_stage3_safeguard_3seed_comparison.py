from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(
    os.environ.get(
        "HIGHWAY_PYTHON",
        r"C:\Users\milke\miniconda3\envs\highway\python.exe",
    )
)

RUN_ID = os.environ.get(
    "RUN_ID",
    f"stage3_safeguard_3seed_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
OUTPUT_ROOT = ROOT / "experiment_runs" / RUN_ID
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

EVAL_SEEDS = list(range(1000, 1050))
EVAL_SEEDS_ENV = ",".join(str(seed) for seed in EVAL_SEEDS)
CONFIG_NAME = "curriculum_stage_3_mixed_traffic"
SEEDS = [12345, 23456, 34567]
SAFE_THRESHOLD = float(os.environ.get("SAFE_THRESHOLD", "0.0005"))
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "5"))


def make_child_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path_value = os.environ.get("Path") or os.environ.get("PATH")
    seen: set[str] = set()

    for key, value in os.environ.items():
        lower = key.lower()
        if lower == "path":
            continue
        if lower in seen:
            continue
        env[key] = value
        seen.add(lower)

    if path_value is not None:
        env["Path"] = path_value

    return env


RUNS: list[dict[str, object]] = []
for seed in SEEDS:
    base_dir = ROOT / "experiment_runs" / f"meanmax_vs_uncertainty_3seeds_20260430_173848_seed{seed}"
    RUNS.extend(
        [
            {
                "variant": "meanmax_baseline",
                "seed": seed,
                "script": ROOT / "mean+max" / "run_checkpoint_eval.py",
                "cwd": ROOT / "mean+max",
                "env": {
                    "HIGHWAY_CONFIG": CONFIG_NAME,
                    "ACTOR_CHECKPOINT": str(
                        base_dir
                        / "deep_sets_mean_max_baseline"
                        / "deep_sets_mean_max_baseline_actor.pth"
                    ),
                    "OUTPUT_DIR": str(OUTPUT_ROOT / f"meanmax_baseline_seed{seed}"),
                    "EVAL_SEEDS": EVAL_SEEDS_ENV,
                },
            },
            {
                "variant": "uncertainty_no_safeguard",
                "seed": seed,
                "script": ROOT / "uncertainty_aware2_PPO" / "run_threshold_sweep.py",
                "cwd": ROOT / "uncertainty_aware2_PPO",
                "env": {
                    "HIGHWAY_CONFIG": CONFIG_NAME,
                    "ACTOR_CHECKPOINT": str(
                        base_dir
                        / "uncertainty_aware_lambda_0.01"
                        / "uncertainty_aware_lambda_0.01_actor.pth"
                    ),
                    "CRITIC_CHECKPOINT": str(
                        base_dir
                        / "uncertainty_aware_lambda_0.01"
                        / "uncertainty_aware_lambda_0.01_critic.pth"
                    ),
                    "THRESHOLDS": "1.0",
                    "MC_SAMPLES": str(MC_SAMPLES),
                    "OUTPUT_DIR": str(OUTPUT_ROOT / f"uncertainty_no_safeguard_seed{seed}"),
                    "EVAL_SEEDS": EVAL_SEEDS_ENV,
                },
            },
            {
                "variant": "uncertainty_safeguard",
                "seed": seed,
                "script": ROOT / "uncertainty_aware2_PPO" / "run_threshold_sweep.py",
                "cwd": ROOT / "uncertainty_aware2_PPO",
                "env": {
                    "HIGHWAY_CONFIG": CONFIG_NAME,
                    "ACTOR_CHECKPOINT": str(
                        base_dir
                        / "uncertainty_aware_lambda_0.01"
                        / "uncertainty_aware_lambda_0.01_actor.pth"
                    ),
                    "CRITIC_CHECKPOINT": str(
                        base_dir
                        / "uncertainty_aware_lambda_0.01"
                        / "uncertainty_aware_lambda_0.01_critic.pth"
                    ),
                    "THRESHOLDS": str(SAFE_THRESHOLD),
                    "MC_SAMPLES": str(MC_SAMPLES),
                    "OUTPUT_DIR": str(OUTPUT_ROOT / f"uncertainty_safeguard_seed{seed}"),
                    "EVAL_SEEDS": EVAL_SEEDS_ENV,
                },
            },
        ]
    )


def run_job(job: dict[str, object]) -> Path:
    variant = str(job["variant"])
    seed = int(job["seed"])
    script = Path(job["script"])
    cwd = Path(job["cwd"])
    env = make_child_env()
    env.update({str(k): str(v) for k, v in dict(job["env"]).items()})

    log_path = OUTPUT_ROOT / f"{variant}_seed{seed}.log"
    print(f"Running {variant} seed={seed}")
    print(f"  script: {script}")
    print(f"  log:    {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [str(PYTHON), str(script)],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{variant}:{seed}] {line}", end="")
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"{variant} seed={seed} failed with exit code {return_code}")

    return Path(str(dict(job["env"])["OUTPUT_DIR"]))


def load_summary_row(variant: str, output_dir: Path) -> dict[str, str]:
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {summary_csv}")
    row = rows[0]
    if variant.startswith("uncertainty_"):
        row = {**row, "threshold": row.get("threshold", "")}
    return row


def main() -> int:
    run_outputs: list[dict[str, object]] = []
    for job in RUNS:
        output_dir = run_job(job)
        run_outputs.append(
            {
                "variant": job["variant"],
                "seed": job["seed"],
                "output_dir": str(output_dir),
                "summary": load_summary_row(str(job["variant"]), output_dir),
            }
        )

    per_run_csv = OUTPUT_ROOT / "per_run_summary.csv"
    per_run_fields = [
        "variant",
        "seed",
        "output_dir",
        "avg_episodic_return",
        "avg_episodic_length",
        "collision_rate",
        "avg_uncertainty_per_episode",
        "avg_episode_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate_per_episode",
        "total_activations",
        "threshold",
    ]
    with per_run_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_run_fields)
        writer.writeheader()
        for item in run_outputs:
            summary = dict(item["summary"])
            writer.writerow(
                {
                    "variant": item["variant"],
                    "seed": item["seed"],
                    "output_dir": item["output_dir"],
                    "avg_episodic_return": summary.get("avg_episodic_return", ""),
                    "avg_episodic_length": summary.get("avg_episodic_length", ""),
                    "collision_rate": summary.get("collision_rate", ""),
                    "avg_uncertainty_per_episode": summary.get("avg_uncertainty_per_episode", ""),
                    "avg_episode_max_uncertainty": summary.get("avg_episode_max_uncertainty", ""),
                    "avg_activations_per_episode": summary.get("avg_activations_per_episode", ""),
                    "avg_activation_rate_per_episode": summary.get("avg_activation_rate_per_episode", ""),
                    "total_activations": summary.get("total_activations", ""),
                    "threshold": summary.get("threshold", ""),
                }
            )

    grouped: dict[str, list[dict[str, str]]] = {}
    for item in run_outputs:
        grouped.setdefault(str(item["variant"]), []).append(dict(item["summary"]))

    aggregate_csv = OUTPUT_ROOT / "aggregate_summary.csv"
    aggregate_fields = [
        "variant",
        "runs",
        "train_seeds",
        "avg_eval_return",
        "avg_eval_length",
        "avg_collision_rate",
        "avg_uncertainty_per_episode",
        "avg_episode_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate_per_episode",
        "threshold",
    ]
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        for variant, rows in grouped.items():
            numeric = lambda key: [float(row[key]) for row in rows if row.get(key, "") not in {"", None}]
            seed_list = [str(seed) for seed in SEEDS]
            writer.writerow(
                {
                    "variant": variant,
                    "runs": len(rows),
                    "train_seeds": ";".join(seed_list),
                    "avg_eval_return": sum(numeric("avg_episodic_return")) / len(rows),
                    "avg_eval_length": sum(numeric("avg_episodic_length")) / len(rows),
                    "avg_collision_rate": sum(numeric("collision_rate")) / len(rows),
                    "avg_uncertainty_per_episode": (
                        sum(numeric("avg_uncertainty_per_episode")) / len(rows)
                        if numeric("avg_uncertainty_per_episode")
                        else ""
                    ),
                    "avg_episode_max_uncertainty": (
                        sum(numeric("avg_episode_max_uncertainty")) / len(rows)
                        if numeric("avg_episode_max_uncertainty")
                        else ""
                    ),
                    "avg_activations_per_episode": (
                        sum(numeric("avg_activations_per_episode")) / len(rows)
                        if numeric("avg_activations_per_episode")
                        else ""
                    ),
                    "avg_activation_rate_per_episode": (
                        sum(numeric("avg_activation_rate_per_episode")) / len(rows)
                        if numeric("avg_activation_rate_per_episode")
                        else ""
                    ),
                    "threshold": rows[0].get("threshold", ""),
                }
            )

    run_config = {
        "python": str(PYTHON),
        "config_name": CONFIG_NAME,
        "eval_seeds": EVAL_SEEDS,
        "train_seeds": SEEDS,
        "safe_threshold": SAFE_THRESHOLD,
        "mc_samples": MC_SAMPLES,
        "jobs": [
            {
                "variant": str(job["variant"]),
                "seed": int(job["seed"]),
                "script": str(job["script"]),
                "cwd": str(job["cwd"]),
                "env": {str(k): str(v) for k, v in dict(job["env"]).items()},
            }
            for job in RUNS
        ],
    }
    config_json = OUTPUT_ROOT / "run_config.json"
    config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print()
    print(f"Per-run summary: {per_run_csv}")
    print(f"Aggregate summary: {aggregate_csv}")
    print(f"Run config: {config_json}")
    print("Stage-3 safeguard comparison complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
