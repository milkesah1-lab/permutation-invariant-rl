from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SINGLE_STAGE_RUNNER = ROOT / "run_meanmax_single_stage_study.py"

DEFAULT_PYTHON = Path(r"C:\Users\milke\miniconda3\envs\highway\python.exe")
PYTHON = Path(
    os.environ.get(
        "PYTHON_EXE",
        str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else sys.executable),
    )
)

DEFAULT_RUN_ID = f"stage3_backbones_repeat2_100k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
OUT_DIR = ROOT / "experiment_runs" / RUN_ID

CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_3_mixed_traffic")
TARGET_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
TRAIN_SEEDS = json.loads(os.environ.get("TRAIN_SEEDS", "[12345, 23456]"))
EVAL_SEEDS = json.loads(os.environ.get("EVAL_SEEDS", json.dumps(list(range(1000, 1050)))))
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
EGO_START_LANE_POLICY = os.environ.get("EGO_START_LANE_POLICY", "random")
EVAL_MAX_STEPS = os.environ.get("EVAL_MAX_STEPS", "full")
EVAL_ACTION_MODE = os.environ.get("EVAL_ACTION_MODE", "actor")

ORIGINAL_EVAL_SUMMARY = (
    ROOT
    / "experiment_runs"
    / "stage3_only_all_models_100k_20260424_101952_consistent_full_eval_50"
    / "summary.csv"
)
ORIGINAL_TRAIN_SUMMARY = (
    ROOT
    / "experiment_runs"
    / "stage3_only_all_models_100k_20260424_101952"
    / "stage_training_final_summary.csv"
)

MODELS = [
    {
        "model_label": "basic_mlp",
        "model_subdir": "baseline1",
        "display_name": "Basic MLP",
    },
    {
        "model_label": "deep_sets_mean",
        "model_subdir": "Deep_sets_mean_model",
        "display_name": "Deep Sets mean",
    },
    {
        "model_label": "deep_sets_mean_max",
        "model_subdir": "mean+max",
        "display_name": "Deep Sets mean+max",
    },
    {
        "model_label": "self_attention",
        "model_subdir": "self_attention_model",
        "display_name": "Self-attention",
    },
]


def ps_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def make_command_header(model: dict[str, str], train_seed: int, child_run_id: str) -> str:
    assignments = {
        "MODEL_SUBDIR": model["model_subdir"],
        "MODEL_LABEL": model["model_label"],
        "HIGHWAY_CONFIG": CONFIG_NAME,
        "TOTAL_TIMESTEPS": TARGET_TIMESTEPS,
        "TRAIN_SEED": train_seed,
        "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
        "EVAL_MAX_STEPS": EVAL_MAX_STEPS,
        "EVAL_ACTION_MODE": EVAL_ACTION_MODE,
        "TORCH_NUM_THREADS": TORCH_NUM_THREADS,
        "RUN_ID": child_run_id,
        "EGO_START_LANE_POLICY": EGO_START_LANE_POLICY,
    }
    prefix = "; ".join(f"$env:{key}={ps_quote(value)}" for key, value in assignments.items())
    return f"{prefix}; & {ps_quote(PYTHON)} {ps_quote(SINGLE_STAGE_RUNNER)}"


def write_study_config() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "study_config.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "config_name": CONFIG_NAME,
                "target_timesteps": TARGET_TIMESTEPS,
                "train_seeds": TRAIN_SEEDS,
                "eval_seeds": EVAL_SEEDS,
                "eval_type": "full" if str(EVAL_MAX_STEPS).lower() == "full" else "capped",
                "eval_max_steps": None if str(EVAL_MAX_STEPS).lower() == "full" else EVAL_MAX_STEPS,
                "eval_action_mode": EVAL_ACTION_MODE,
                "torch_num_threads": TORCH_NUM_THREADS,
                "ego_start_lane_policy": EGO_START_LANE_POLICY,
                "python": str(PYTHON),
                "runner": str(SINGLE_STAGE_RUNNER),
                "models": MODELS,
                "original_eval_summary": str(ORIGINAL_EVAL_SUMMARY),
                "original_train_summary": str(ORIGINAL_TRAIN_SUMMARY),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_one(model: dict[str, str], train_seed: int) -> dict[str, object]:
    model_label = model["model_label"]
    model_subdir = model["model_subdir"]
    display_name = model["display_name"]
    child_run_id = f"{RUN_ID}\\{model_label}\\seed{train_seed}"
    command_header = make_command_header(model, train_seed, child_run_id)
    log_path = OUT_DIR / f"{model_label}_seed{train_seed}.log"

    print()
    print(
        f"Running {display_name} ({model_subdir}) | "
        f"seed={train_seed} | timesteps={TARGET_TIMESTEPS}",
        flush=True,
    )
    print("Command:", flush=True)
    print(command_header, flush=True)
    print(f"Log: {log_path}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODEL_SUBDIR": model_subdir,
            "MODEL_LABEL": model_label,
            "HIGHWAY_CONFIG": CONFIG_NAME,
            "TOTAL_TIMESTEPS": str(TARGET_TIMESTEPS),
            "TRAIN_SEED": str(train_seed),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "EVAL_MAX_STEPS": EVAL_MAX_STEPS,
            "EVAL_ACTION_MODE": EVAL_ACTION_MODE,
            "TORCH_NUM_THREADS": str(TORCH_NUM_THREADS),
            "RUN_ID": child_run_id,
            "EGO_START_LANE_POLICY": EGO_START_LANE_POLICY,
            "COMMAND_HEADER": command_header,
        }
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(command_header)
        log_file.write("\n\n")
        log_file.flush()
        process = subprocess.Popen(
            [str(PYTHON), str(SINGLE_STAGE_RUNNER)],
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
        raise RuntimeError(f"{model_label} seed={train_seed} failed with exit code {return_code}")

    run_dir = ROOT / "experiment_runs" / child_run_id
    train_summary_path = run_dir / "training_summary.csv"
    eval_summary_path = run_dir / "evaluation_summary.csv"
    eval_per_episode_path = run_dir / "evaluation_per_episode.csv"

    train_summary = load_single_row(train_summary_path)
    eval_summary = load_single_row(eval_summary_path)

    return {
        "model_label": model_label,
        "model_subdir": model_subdir,
        "display_name": display_name,
        "train_seed": train_seed,
        "run_dir": str(run_dir),
        "command": command_header,
        "training_summary": train_summary,
        "evaluation_summary": eval_summary,
        "evaluation_per_episode_csv": str(eval_per_episode_path),
    }


def load_single_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def write_per_run_summary(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "per_run_summary.csv"
    fieldnames = [
        "model_label",
        "display_name",
        "model_subdir",
        "train_seed",
        "run_dir",
        "target_timesteps",
        "final_train_return",
        "final_train_length",
        "avg_eval_return",
        "avg_eval_length",
        "collision_rate",
        "training_summary_csv",
        "evaluation_summary_csv",
        "evaluation_per_episode_csv",
        "actor_checkpoint",
        "critic_checkpoint",
        "command",
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
                    "final_train_return": train.get("avg_episodic_return", ""),
                    "final_train_length": train.get("avg_episodic_length", ""),
                    "avg_eval_return": eval_.get("avg_episodic_return", ""),
                    "avg_eval_length": eval_.get("avg_episodic_length", ""),
                    "collision_rate": eval_.get("collision_rate", ""),
                    "training_summary_csv": str(Path(item["run_dir"]) / "training_summary.csv"),
                    "evaluation_summary_csv": str(Path(item["run_dir"]) / "evaluation_summary.csv"),
                    "evaluation_per_episode_csv": item["evaluation_per_episode_csv"],
                    "actor_checkpoint": train.get("actor_checkpoint", ""),
                    "critic_checkpoint": train.get("critic_checkpoint", ""),
                    "command": item["command"],
                }
            )

    return out_path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


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
        "avg_train_return",
        "avg_train_length",
        "avg_eval_return",
        "avg_eval_length",
        "collision_rate",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODELS:
            label = model["model_label"]
            rows = grouped.get(label, [])
            if not rows:
                continue

            writer.writerow(
                {
                    "model_label": label,
                    "display_name": model["display_name"],
                    "runs": len(rows),
                    "train_seeds": ";".join(str(item["train_seed"]) for item in rows),
                    "target_timesteps": TARGET_TIMESTEPS,
                    "avg_train_return": mean(
                        [float(item["training_summary"]["avg_episodic_return"]) for item in rows]
                    ),
                    "avg_train_length": mean(
                        [float(item["training_summary"]["avg_episodic_length"]) for item in rows]
                    ),
                    "avg_eval_return": mean(
                        [float(item["evaluation_summary"]["avg_episodic_return"]) for item in rows]
                    ),
                    "avg_eval_length": mean(
                        [float(item["evaluation_summary"]["avg_episodic_length"]) for item in rows]
                    ),
                    "collision_rate": mean(
                        [float(item["evaluation_summary"]["collision_rate"]) for item in rows]
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
        "episodic_return",
        "episodic_length",
        "collided",
        "capped",
        "terminated",
        "truncated",
        "scenario",
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
                            "episodic_return": row.get("episodic_return", ""),
                            "episodic_length": row.get("episodic_length", ""),
                            "collided": row.get("collided", ""),
                            "capped": row.get("capped", ""),
                            "terminated": row.get("terminated", ""),
                            "truncated": row.get("truncated", ""),
                            "scenario": row.get("scenario", ""),
                        }
                    )

    return out_path


def load_original_training_rows() -> dict[str, dict[str, str]]:
    if not ORIGINAL_TRAIN_SUMMARY.exists():
        return {}
    with ORIGINAL_TRAIN_SUMMARY.open("r", newline="", encoding="utf-8") as handle:
        return {row["model"]: row for row in csv.DictReader(handle)}


def write_eval_average_with_original(results: list[dict[str, object]]) -> tuple[Path, Path] | None:
    if not ORIGINAL_EVAL_SUMMARY.exists():
        return None

    original_train = load_original_training_rows()
    per_run_path = OUT_DIR / "per_run_summary_with_original.csv"
    aggregate_path = OUT_DIR / "aggregate_summary_with_original.csv"
    fieldnames = [
        "source",
        "model_label",
        "display_name",
        "train_seed",
        "final_train_return",
        "final_train_length",
        "avg_eval_return",
        "avg_eval_length",
        "collision_rate",
    ]

    rows: list[dict[str, str]] = []
    with ORIGINAL_EVAL_SUMMARY.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            display_name = next((item["display_name"] for item in MODELS if item["model_label"] == model), model)
            train = original_train.get(model, {})
            rows.append(
                {
                    "source": "original_20260424",
                    "model_label": model,
                    "display_name": display_name,
                    "train_seed": "",
                    "final_train_return": train.get("avg_episodic_return", ""),
                    "final_train_length": train.get("avg_episodic_length", ""),
                    "avg_eval_return": row["avg_episodic_return"],
                    "avg_eval_length": row["avg_episodic_length"],
                    "collision_rate": row["collision_rate"],
                }
            )

    for item in results:
        train = dict(item["training_summary"])
        eval_ = dict(item["evaluation_summary"])
        rows.append(
            {
                "source": "repeat_20260512",
                "model_label": str(item["model_label"]),
                "display_name": str(item["display_name"]),
                "train_seed": str(item["train_seed"]),
                "final_train_return": train.get("avg_episodic_return", ""),
                "final_train_length": train.get("avg_episodic_length", ""),
                "avg_eval_return": eval_.get("avg_episodic_return", ""),
                "avg_eval_length": eval_.get("avg_episodic_length", ""),
                "collision_rate": eval_.get("collision_rate", ""),
            }
        )

    with per_run_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_label",
                "display_name",
                "runs",
                "sources",
                "avg_eval_return",
                "avg_eval_length",
                "collision_rate",
            ],
        )
        writer.writeheader()
        for model in MODELS:
            model_rows = [row for row in rows if row["model_label"] == model["model_label"]]
            writer.writerow(
                {
                    "model_label": model["model_label"],
                    "display_name": model["display_name"],
                    "runs": len(model_rows),
                    "sources": ";".join(row["source"] for row in model_rows),
                    "avg_eval_return": mean([float(row["avg_eval_return"]) for row in model_rows]),
                    "avg_eval_length": mean([float(row["avg_eval_length"]) for row in model_rows]),
                    "collision_rate": mean([float(row["collision_rate"]) for row in model_rows]),
                }
            )

    return per_run_path, aggregate_path


def main() -> int:
    if not SINGLE_STAGE_RUNNER.exists():
        raise FileNotFoundError(f"Missing runner: {SINGLE_STAGE_RUNNER}")
    if len(EVAL_SEEDS) != 50:
        raise ValueError(f"Expected 50 evaluation seeds, got {len(EVAL_SEEDS)}")
    if len(TRAIN_SEEDS) != 2:
        raise ValueError(f"Expected exactly 2 repeat training seeds, got {len(TRAIN_SEEDS)}")

    write_study_config()

    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Output directory: {OUT_DIR}", flush=True)
    print(f"Python: {PYTHON}", flush=True)
    print(f"Config: {CONFIG_NAME}", flush=True)
    print(f"Target timesteps: {TARGET_TIMESTEPS}", flush=True)
    print(f"Train seeds: {TRAIN_SEEDS}", flush=True)
    print(f"Eval seeds: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} ({len(EVAL_SEEDS)} episodes)", flush=True)
    print("Every per-run log starts with the equivalent PowerShell command.", flush=True)

    results: list[dict[str, object]] = []
    for model in MODELS:
        for train_seed in TRAIN_SEEDS:
            results.append(run_one(model, int(train_seed)))

    per_run_csv = write_per_run_summary(results)
    aggregate_csv = write_aggregate_summary(results)
    combined_eval_csv = write_combined_eval_csv(results)
    combined_with_original = write_eval_average_with_original(results)

    print()
    print("Stage 3 backbone repeat study complete.", flush=True)
    print(f"Per-run summary: {per_run_csv}", flush=True)
    print(f"Aggregate summary for the two new repeats: {aggregate_csv}", flush=True)
    print(f"Combined per-episode evaluation for repeats: {combined_eval_csv}", flush=True)
    if combined_with_original:
        print(f"Per-run summary including original table: {combined_with_original[0]}", flush=True)
        print(f"Aggregate summary including original table: {combined_with_original[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
