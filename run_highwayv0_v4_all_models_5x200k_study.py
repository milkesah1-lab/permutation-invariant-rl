from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SINGLE_RUNNER = ROOT / "run_meanmax_highwayv0_stage2_study.py"

DEFAULT_PYTHON = Path(r"C:\Users\milke\miniconda3\envs\highway\python.exe")
PYTHON = Path(
    os.environ.get(
        "PYTHON_EXE",
        str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else sys.executable),
    )
)

DEFAULT_RUN_ID = f"highwayv0_v4_all_models_5x200k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
OUT_DIR = ROOT / "experiment_runs" / RUN_ID

CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "highway_v0_stage_2_benchmark_v4")
TARGET_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "200000"))
TRAIN_SEEDS = json.loads(
    os.environ.get("TRAIN_SEEDS", "[12345, 23456, 34567, 45678, 56789]")
)
EVAL_SEEDS = json.loads(
    os.environ.get("EVAL_SEEDS", json.dumps(list(range(1000, 1050))))
)
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
SAFEGUARD_THRESHOLD = float(os.environ.get("SAFEGUARD_THRESHOLD", "0.001"))
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "5"))
RESUME = os.environ.get("RESUME", "").strip().lower() in {"1", "true", "yes", "y", "on"}

MODELS = [
    {
        "model_label": "mlp_baseline",
        "model_subdir": "baseline1",
        "display_name": "MLP baseline",
        "eval_action_mode": "actor",
    },
    {
        "model_label": "deep_sets_mean",
        "model_subdir": "Deep_sets_mean_model",
        "display_name": "Deep Sets mean",
        "eval_action_mode": "actor",
    },
    {
        "model_label": "deep_sets_mean_max",
        "model_subdir": "mean+max",
        "display_name": "Deep Sets mean+max",
        "eval_action_mode": "actor",
    },
    {
        "model_label": "self_attention",
        "model_subdir": "self_attention_model",
        "display_name": "Self-attention",
        "eval_action_mode": "actor",
    },
    {
        "model_label": "uncertainty_aware2_threshold",
        "model_subdir": "uncertainty_aware2_PPO",
        "display_name": "Uncertainty-aware2 threshold",
        "eval_action_mode": "safeguarded",
        "safeguard_threshold": SAFEGUARD_THRESHOLD,
        "mc_samples": MC_SAMPLES,
    },
    {
        "model_label": "uncertainty2_self_attention_threshold",
        "model_subdir": "uncertainty_aware2_PPO_self_attention_backbone",
        "display_name": "Uncertainty2 self-attention threshold",
        "eval_action_mode": "safeguarded",
        "safeguard_threshold": SAFEGUARD_THRESHOLD,
        "mc_samples": MC_SAMPLES,
    },
]

MODEL_LABELS_RAW = os.environ.get("MODEL_LABELS", "").strip()
if MODEL_LABELS_RAW:
    try:
        parsed_model_labels = json.loads(MODEL_LABELS_RAW)
    except json.JSONDecodeError:
        parsed_model_labels = [
            part.strip()
            for part in MODEL_LABELS_RAW.split(",")
            if part.strip()
        ]
    requested_model_labels = set(parsed_model_labels)
    MODELS = [model for model in MODELS if str(model["model_label"]) in requested_model_labels]
    missing_model_labels = requested_model_labels - {str(model["model_label"]) for model in MODELS}
    if missing_model_labels:
        raise ValueError(f"Unknown model labels requested: {sorted(missing_model_labels)}")


def write_study_config() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=RESUME)
    (OUT_DIR / "study_config.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "env_id": "highway-v0",
                "config_name": CONFIG_NAME,
                "target_timesteps": TARGET_TIMESTEPS,
                "train_seeds": TRAIN_SEEDS,
                "eval_seeds": EVAL_SEEDS,
                "eval_type": "full",
                "torch_num_threads": TORCH_NUM_THREADS,
                "python": str(PYTHON),
                "single_runner": str(SINGLE_RUNNER),
                "safeguard_threshold": SAFEGUARD_THRESHOLD,
                "mc_samples": MC_SAMPLES,
                "resume": RESUME,
                "models": MODELS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_single_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def run_artifacts_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "training_summary.csv",
        run_dir / "evaluation_summary.csv",
        run_dir / "evaluation_per_episode.csv",
        run_dir / "checkpoint_actor.pth",
        run_dir / "checkpoint_critic.pth",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def build_result_from_run_dir(
    model: dict[str, object],
    train_seed: int,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "model_label": str(model["model_label"]),
        "model_subdir": str(model["model_subdir"]),
        "display_name": str(model["display_name"]),
        "train_seed": train_seed,
        "run_dir": str(run_dir),
        "eval_action_mode": str(model["eval_action_mode"]),
        "training_summary": load_single_row(run_dir / "training_summary.csv"),
        "evaluation_summary": load_single_row(run_dir / "evaluation_summary.csv"),
        "evaluation_per_episode_csv": str(run_dir / "evaluation_per_episode.csv"),
    }


def archive_incomplete_run(run_dir: Path, log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir.exists():
        archive_dir = run_dir.with_name(f"{run_dir.name}_interrupted_{timestamp}")
        shutil.move(str(run_dir), str(archive_dir))
        print(f"Archived incomplete run directory: {archive_dir}", flush=True)
    if log_path.exists():
        archive_log = log_path.with_name(f"{log_path.stem}_interrupted_{timestamp}{log_path.suffix}")
        shutil.move(str(log_path), str(archive_log))
        print(f"Archived incomplete run log: {archive_log}", flush=True)


def run_one(model: dict[str, object], train_seed: int) -> dict[str, object]:
    model_label = str(model["model_label"])
    model_subdir = str(model["model_subdir"])
    display_name = str(model["display_name"])
    eval_action_mode = str(model["eval_action_mode"])
    child_run_id = f"{RUN_ID}\\{model_label}\\seed{train_seed}"
    run_dir = ROOT / "experiment_runs" / child_run_id
    log_path = OUT_DIR / f"{model_label}_seed{train_seed}.log"

    if run_artifacts_complete(run_dir):
        print()
        print(
            f"Skipping completed {display_name} ({model_subdir}) | seed={train_seed}",
            flush=True,
        )
        return build_result_from_run_dir(model, train_seed, run_dir)

    if RESUME and run_dir.exists():
        archive_incomplete_run(run_dir, log_path)

    print()
    print(
        f"Running {display_name} ({model_subdir}) | seed={train_seed} | "
        f"timesteps={TARGET_TIMESTEPS} | eval={eval_action_mode}",
        flush=True,
    )
    print(f"Log: {log_path}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHON_EXE": str(PYTHON),
            "MODEL_SUBDIR": model_subdir,
            "MODEL_LABEL": model_label,
            "HIGHWAY_CONFIG": CONFIG_NAME,
            "TOTAL_TIMESTEPS": str(TARGET_TIMESTEPS),
            "TRAIN_SEED": str(train_seed),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "TORCH_NUM_THREADS": str(TORCH_NUM_THREADS),
            "RUN_ID": child_run_id,
            "EVAL_ACTION_MODE": eval_action_mode,
            "SAFEGUARD_THRESHOLD": str(model.get("safeguard_threshold", SAFEGUARD_THRESHOLD)),
            "MC_SAMPLES": str(model.get("mc_samples", MC_SAMPLES)),
        }
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [str(PYTHON), str(SINGLE_RUNNER)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{model_label}:{train_seed}] {line}", end="")
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"{model_label} seed={train_seed} failed with exit code {return_code}"
        )

    train_summary_path = run_dir / "training_summary.csv"
    eval_summary_path = run_dir / "evaluation_summary.csv"
    eval_per_episode_path = run_dir / "evaluation_per_episode.csv"

    return {
        "model_label": model_label,
        "model_subdir": model_subdir,
        "display_name": display_name,
        "train_seed": train_seed,
        "run_dir": str(run_dir),
        "eval_action_mode": eval_action_mode,
        "training_summary": load_single_row(train_summary_path),
        "evaluation_summary": load_single_row(eval_summary_path),
        "evaluation_per_episode_csv": str(eval_per_episode_path),
    }


def write_per_run_summary(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "per_run_summary.csv"
    fieldnames = [
        "model_label",
        "display_name",
        "model_subdir",
        "train_seed",
        "run_dir",
        "target_timesteps",
        "eval_action_mode",
        "final_train_return",
        "final_train_length",
        "avg_eval_return",
        "avg_eval_length",
        "collision_rate",
        "safeguard_threshold",
        "mc_samples",
        "avg_uncertainty",
        "avg_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate",
        "total_activations",
        "training_summary_csv",
        "evaluation_summary_csv",
        "evaluation_per_episode_csv",
        "actor_checkpoint",
        "critic_checkpoint",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            train = dict(item["training_summary"])
            eval_ = dict(item["evaluation_summary"])
            writer.writerow(
                {
                    "model_label": item["model_label"],
                    "display_name": item["display_name"],
                    "model_subdir": item["model_subdir"],
                    "train_seed": item["train_seed"],
                    "run_dir": item["run_dir"],
                    "target_timesteps": train.get("target_timesteps", ""),
                    "eval_action_mode": eval_.get("eval_action_mode", item["eval_action_mode"]),
                    "final_train_return": train.get("avg_episodic_return", ""),
                    "final_train_length": train.get("avg_episodic_length", ""),
                    "avg_eval_return": eval_.get("avg_episodic_return", ""),
                    "avg_eval_length": eval_.get("avg_episodic_length", ""),
                    "collision_rate": eval_.get("collision_rate", ""),
                    "safeguard_threshold": eval_.get("safeguard_threshold", ""),
                    "mc_samples": eval_.get("mc_samples", ""),
                    "avg_uncertainty": eval_.get("avg_uncertainty", ""),
                    "avg_max_uncertainty": eval_.get("avg_max_uncertainty", ""),
                    "avg_activations_per_episode": eval_.get("avg_activations_per_episode", ""),
                    "avg_activation_rate": eval_.get("avg_activation_rate", ""),
                    "total_activations": eval_.get("total_activations", ""),
                    "training_summary_csv": str(Path(item["run_dir"]) / "training_summary.csv"),
                    "evaluation_summary_csv": str(Path(item["run_dir"]) / "evaluation_summary.csv"),
                    "evaluation_per_episode_csv": item["evaluation_per_episode_csv"],
                    "actor_checkpoint": train.get("actor_checkpoint", ""),
                    "critic_checkpoint": train.get("critic_checkpoint", ""),
                }
            )

    return out_path


def average_nonblank(values: list[str]) -> str:
    filtered = [float(value) for value in values if str(value).strip() != ""]
    if not filtered:
        return ""
    return str(sum(filtered) / len(filtered))


def write_aggregate_summary(results: list[dict[str, object]]) -> Path:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in results:
        grouped.setdefault(str(item["model_label"]), []).append(item)

    out_path = OUT_DIR / "aggregate_summary.csv"
    fieldnames = [
        "model_label",
        "display_name",
        "runs",
        "train_seeds",
        "target_timesteps",
        "eval_action_mode",
        "avg_train_return",
        "avg_train_length",
        "avg_eval_return",
        "avg_eval_length",
        "collision_rate",
        "safeguard_threshold",
        "mc_samples",
        "avg_uncertainty",
        "avg_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate",
        "total_activations",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODELS:
            label = str(model["model_label"])
            rows = grouped.get(label, [])
            if not rows:
                continue
            eval_rows = [dict(item["evaluation_summary"]) for item in rows]
            train_rows = [dict(item["training_summary"]) for item in rows]

            writer.writerow(
                {
                    "model_label": label,
                    "display_name": model["display_name"],
                    "runs": len(rows),
                    "train_seeds": ";".join(
                        str(item["train_seed"])
                        for item in sorted(rows, key=lambda row: int(row["train_seed"]))
                    ),
                    "target_timesteps": TARGET_TIMESTEPS,
                    "eval_action_mode": model["eval_action_mode"],
                    "avg_train_return": average_nonblank(
                        [row.get("avg_episodic_return", "") for row in train_rows]
                    ),
                    "avg_train_length": average_nonblank(
                        [row.get("avg_episodic_length", "") for row in train_rows]
                    ),
                    "avg_eval_return": average_nonblank(
                        [row.get("avg_episodic_return", "") for row in eval_rows]
                    ),
                    "avg_eval_length": average_nonblank(
                        [row.get("avg_episodic_length", "") for row in eval_rows]
                    ),
                    "collision_rate": average_nonblank(
                        [row.get("collision_rate", "") for row in eval_rows]
                    ),
                    "safeguard_threshold": model.get("safeguard_threshold", ""),
                    "mc_samples": model.get("mc_samples", ""),
                    "avg_uncertainty": average_nonblank(
                        [row.get("avg_uncertainty", "") for row in eval_rows]
                    ),
                    "avg_max_uncertainty": average_nonblank(
                        [row.get("avg_max_uncertainty", "") for row in eval_rows]
                    ),
                    "avg_activations_per_episode": average_nonblank(
                        [row.get("avg_activations_per_episode", "") for row in eval_rows]
                    ),
                    "avg_activation_rate": average_nonblank(
                        [row.get("avg_activation_rate", "") for row in eval_rows]
                    ),
                    "total_activations": sum(
                        int(float(row.get("total_activations", "0") or 0))
                        for row in eval_rows
                    ),
                }
            )

    return out_path


def write_combined_eval_csv(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "combined_evaluation_per_episode.csv"
    fieldnames = [
        "model_label",
        "display_name",
        "train_seed",
        "episode_index",
        "eval_seed",
        "start_lane",
        "episodic_return",
        "episodic_length",
        "collided",
        "capped",
        "terminated",
        "truncated",
        "eval_action_mode",
        "safeguard_threshold",
        "mc_samples",
        "avg_uncertainty",
        "max_uncertainty",
        "activation_count",
        "activation_rate",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            per_episode_path = Path(item["evaluation_per_episode_csv"])
            with per_episode_path.open("r", newline="", encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    writer.writerow(
                        {
                            "model_label": item["model_label"],
                            "display_name": item["display_name"],
                            "train_seed": item["train_seed"],
                            "episode_index": row.get("episode_index", ""),
                            "eval_seed": row.get("eval_seed", ""),
                            "start_lane": row.get("start_lane", ""),
                            "episodic_return": row.get("episodic_return", ""),
                            "episodic_length": row.get("episodic_length", ""),
                            "collided": row.get("collided", ""),
                            "capped": row.get("capped", ""),
                            "terminated": row.get("terminated", ""),
                            "truncated": row.get("truncated", ""),
                            "eval_action_mode": row.get("eval_action_mode", item["eval_action_mode"]),
                            "safeguard_threshold": row.get("safeguard_threshold", ""),
                            "mc_samples": row.get("mc_samples", ""),
                            "avg_uncertainty": row.get("avg_uncertainty", ""),
                            "max_uncertainty": row.get("max_uncertainty", ""),
                            "activation_count": row.get("activation_count", ""),
                            "activation_rate": row.get("activation_rate", ""),
                        }
                    )

    return out_path


def main() -> int:
    if not SINGLE_RUNNER.exists():
        raise FileNotFoundError(f"Missing runner: {SINGLE_RUNNER}")
    if len(EVAL_SEEDS) != 50:
        raise ValueError(f"Expected 50 evaluation seeds, got {len(EVAL_SEEDS)}")
    if not TRAIN_SEEDS:
        raise ValueError("At least one training seed is required")
    if not MODELS:
        raise ValueError("No models selected for the v4 study")

    write_study_config()

    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Output directory: {OUT_DIR}", flush=True)
    print(f"Python: {PYTHON}", flush=True)
    print(f"Config: {CONFIG_NAME}", flush=True)
    print(f"Target timesteps: {TARGET_TIMESTEPS}", flush=True)
    print(f"Train seeds: {TRAIN_SEEDS}", flush=True)
    print(f"Eval seeds: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} ({len(EVAL_SEEDS)} episodes)", flush=True)
    print(f"Models: {[model['model_label'] for model in MODELS]}", flush=True)
    print(f"Safeguard threshold: {SAFEGUARD_THRESHOLD}", flush=True)
    print(f"MC samples: {MC_SAMPLES}", flush=True)
    print(f"Resume: {RESUME}", flush=True)

    results: list[dict[str, object]] = []
    for model in MODELS:
        for train_seed in TRAIN_SEEDS:
            results.append(run_one(model, int(train_seed)))

    per_run_csv = write_per_run_summary(results)
    aggregate_csv = write_aggregate_summary(results)
    combined_eval_csv = write_combined_eval_csv(results)

    print()
    print("Highway-v0 v4 all-model study complete.", flush=True)
    print(f"Per-run summary: {per_run_csv}", flush=True)
    print(f"Aggregate summary: {aggregate_csv}", flush=True)
    print(f"Combined per-episode evaluation: {combined_eval_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
