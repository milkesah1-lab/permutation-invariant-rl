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

CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_2_easy_overtake")
TARGET_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "200000"))
TRAIN_SEEDS = json.loads(
    os.environ.get("TRAIN_SEEDS", "[12345, 23456, 34567, 45678, 56789]")
)
EVAL_SEEDS = json.loads(
    os.environ.get("EVAL_SEEDS", json.dumps(list(range(1000, 1050))))
)
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
EGO_START_LANE_POLICY = os.environ.get("EGO_START_LANE_POLICY", "random")

SELF_ATTENTION_RUN_ID = os.environ.get(
    "SELF_ATTENTION_RUN_ID",
    "self_attention_stage2_5x200k_foreground_20260508_162148",
)
MEANMAX_RUN_ID = os.environ.get(
    "MEANMAX_RUN_ID",
    f"meanmax_stage2_5x200k_foreground_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

FAMILIES = [
    {
        "display_name": "Self-attention",
        "model_label": "self_attention",
        "model_subdir": "self_attention_model",
        "run_id": SELF_ATTENTION_RUN_ID,
    },
    {
        "display_name": "Deep Sets mean+max",
        "model_label": "deep_sets_mean_max",
        "model_subdir": "mean+max",
        "run_id": MEANMAX_RUN_ID,
    },
]


def ensure_family_dir(run_id: str) -> Path:
    family_dir = ROOT / "experiment_runs" / run_id
    family_dir.mkdir(parents=True, exist_ok=True)
    return family_dir


def seed_run_dir(run_id: str, train_seed: int) -> Path:
    return ROOT / "experiment_runs" / run_id / f"seed{train_seed}"


def evaluation_summary_path(run_id: str, train_seed: int) -> Path:
    return seed_run_dir(run_id, train_seed) / "evaluation_summary.csv"


def write_family_config(family: dict[str, str]) -> None:
    family_dir = ensure_family_dir(family["run_id"])
    (family_dir / "study_config.json").write_text(
        json.dumps(
            {
                "run_id": family["run_id"],
                "model_label": family["model_label"],
                "model_subdir": family["model_subdir"],
                "display_name": family["display_name"],
                "config_name": CONFIG_NAME,
                "target_timesteps": TARGET_TIMESTEPS,
                "train_seeds": TRAIN_SEEDS,
                "eval_seeds": EVAL_SEEDS,
                "eval_type": "full",
                "torch_num_threads": TORCH_NUM_THREADS,
                "ego_start_lane_policy": EGO_START_LANE_POLICY,
                "python": str(PYTHON),
                "runner": str(SINGLE_STAGE_RUNNER),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_seed(family: dict[str, str], train_seed: int) -> None:
    model_label = family["model_label"]
    display_name = family["display_name"]
    run_id = family["run_id"]
    child_run_id = f"{run_id}\\seed{train_seed}"
    log_path = ROOT / "experiment_runs" / run_id / f"{model_label}_seed{train_seed}.log"

    print(
        f"Running {display_name} | seed={train_seed} | timesteps={TARGET_TIMESTEPS}",
        flush=True,
    )
    print(f"Log: {log_path}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODEL_SUBDIR": family["model_subdir"],
            "MODEL_LABEL": model_label,
            "HIGHWAY_CONFIG": CONFIG_NAME,
            "TOTAL_TIMESTEPS": str(TARGET_TIMESTEPS),
            "TRAIN_SEED": str(train_seed),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "TORCH_NUM_THREADS": str(TORCH_NUM_THREADS),
            "RUN_ID": child_run_id,
            "EGO_START_LANE_POLICY": EGO_START_LANE_POLICY,
        }
    )

    with log_path.open("w", encoding="utf-8") as log_file:
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
        raise RuntimeError(
            f"{family['display_name']} seed={train_seed} failed with exit code {return_code}"
        )


def load_single_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def write_family_summaries(family: dict[str, str]) -> None:
    run_id = family["run_id"]
    family_dir = ROOT / "experiment_runs" / run_id
    rows: list[dict[str, object]] = []

    for train_seed in TRAIN_SEEDS:
        run_dir = seed_run_dir(run_id, int(train_seed))
        train_summary_path = run_dir / "training_summary.csv"
        eval_summary_path = run_dir / "evaluation_summary.csv"
        if not train_summary_path.exists() or not eval_summary_path.exists():
            continue

        train = load_single_row(train_summary_path)
        eval_ = load_single_row(eval_summary_path)
        rows.append(
            {
                "model_label": family["model_label"],
                "display_name": family["display_name"],
                "train_seed": int(train_seed),
                "run_dir": str(run_dir),
                "final_train_return": float(train["avg_episodic_return"]),
                "final_train_length": float(train["avg_episodic_length"]),
                "avg_eval_return": float(eval_["avg_episodic_return"]),
                "avg_eval_length": float(eval_["avg_episodic_length"]),
                "collision_rate": float(eval_["collision_rate"]),
                "training_summary_csv": str(train_summary_path),
                "evaluation_summary_csv": str(eval_summary_path),
                "evaluation_per_episode_csv": str(run_dir / "evaluation_per_episode.csv"),
            }
        )

    if not rows:
        return

    per_run_csv = family_dir / "combined_5run_per_run_summary.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_csv = family_dir / "combined_5run_aggregate_summary.csv"
    avg = lambda key: sum(float(row[key]) for row in rows) / len(rows)
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model_label",
            "display_name",
            "runs",
            "train_seeds",
            "avg_train_return",
            "avg_train_length",
            "avg_eval_return",
            "avg_eval_length",
            "collision_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model_label": family["model_label"],
                "display_name": family["display_name"],
                "runs": len(rows),
                "train_seeds": ";".join(str(row["train_seed"]) for row in rows),
                "avg_train_return": avg("final_train_return"),
                "avg_train_length": avg("final_train_length"),
                "avg_eval_return": avg("avg_eval_return"),
                "avg_eval_length": avg("avg_eval_length"),
                "collision_rate": avg("collision_rate"),
            }
        )


def main() -> int:
    if not SINGLE_STAGE_RUNNER.exists():
        raise FileNotFoundError(f"Missing runner: {SINGLE_STAGE_RUNNER}")
    if len(EVAL_SEEDS) != 50:
        raise ValueError(f"Expected 50 evaluation seeds, got {len(EVAL_SEEDS)}")

    print(f"Config: {CONFIG_NAME}", flush=True)
    print(f"Target timesteps: {TARGET_TIMESTEPS}", flush=True)
    print(f"Train seeds: {TRAIN_SEEDS}", flush=True)
    print(f"Eval seeds: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} ({len(EVAL_SEEDS)} episodes)", flush=True)
    print(f"Python: {PYTHON}", flush=True)
    print()

    for family in FAMILIES:
        write_family_config(family)
        family_dir = ensure_family_dir(family["run_id"])
        print(
            f"=== {family['display_name']} | output: {family_dir} ===",
            flush=True,
        )
        for train_seed in TRAIN_SEEDS:
            summary_path = evaluation_summary_path(family["run_id"], int(train_seed))
            if summary_path.exists():
                print(
                    f"Skipping {family['display_name']} seed={train_seed} "
                    f"(already completed)",
                    flush=True,
                )
                continue
            run_seed(family, int(train_seed))
        write_family_summaries(family)
        print(f"Completed {family['display_name']} family.", flush=True)
        print()

    print("Continuation queue complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
