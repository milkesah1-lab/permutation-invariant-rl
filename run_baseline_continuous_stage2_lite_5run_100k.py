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
SINGLE_STAGE_RUNNER = ROOT / "run_meanmax_single_stage_study.py"

DEFAULT_PYTHON = Path(r"C:\Users\milke\miniconda3\envs\highway\python.exe")
PYTHON = Path(
    os.environ.get(
        "PYTHON_EXE",
        str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else sys.executable),
    )
)

DEFAULT_RUN_ID = f"baseline_continuous_stage2_lite_5run_100k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
OUT_DIR = ROOT / "experiment_runs" / RUN_ID

MODEL_LABEL = os.environ.get("MODEL_LABEL", "mlp_baseline")
MODEL_SUBDIR = os.environ.get("MODEL_SUBDIR", "baseline1")
DISPLAY_NAME = os.environ.get("DISPLAY_NAME", "MLP baseline")
ENV_ID = "continuous-spawn-highway-v0"
BASE_CONFIG_NAME = "curriculum_stage_2_easy_overtake"
CONFIG_VARIANT = "curriculum_stage_2_easy_overtake_lite"
TARGET_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
TRAIN_SEEDS = json.loads(
    os.environ.get("TRAIN_SEEDS", "[12345, 23456, 34567, 45678, 56789]")
)
EVAL_SEEDS = json.loads(
    os.environ.get("EVAL_SEEDS", json.dumps(list(range(1000, 1050))))
)
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
EGO_START_LANE_POLICY = os.environ.get("EGO_START_LANE_POLICY", "random")
RESUME = os.environ.get("RESUME", "").strip().lower() in {"1", "true", "yes", "y", "on"}
EVAL_ACTION_MODE = os.environ.get("EVAL_ACTION_MODE", "actor").strip().lower()
SAFEGUARD_THRESHOLD = float(os.environ.get("SAFEGUARD_THRESHOLD", "0.0005"))
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "5"))

LITE_CONFIG_OVERRIDES = {
    "spawn_probability": 0.07,
    "spawn_interval": 4,
    "max_vehicles": 10,
    "spawn_min_gap": 24,
    "lead_vehicle_distance_range": [30, 44],
    "lead_vehicle_speed_range": [14, 18],
    "same_lane_blocker_probability": 0.15,
    "same_lane_blocker_distance_range": [30, 42],
    "same_lane_blocker_speed_range": [14, 18],
    "scenario_probabilities": {
        "overtake_easy": 0.8,
        "overtake_blocked": 0.1,
        "stay_best": 0.1,
    },
    "blocked_adjacent_front_distance_range": [22, 34],
    "blocked_adjacent_rear_distance_range": [18, 30],
    "semi_open_adjacent_front_distance_range": [38, 55],
}


def write_study_config() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=RESUME)
    (OUT_DIR / "study_config.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "model": MODEL_LABEL,
                "display_name": DISPLAY_NAME,
                "model_subdir": MODEL_SUBDIR,
                "env_id": ENV_ID,
                "base_config_name": BASE_CONFIG_NAME,
                "config_variant": CONFIG_VARIANT,
                "config_overrides": LITE_CONFIG_OVERRIDES,
                "ego_start_lane_policy": EGO_START_LANE_POLICY,
                "target_timesteps": TARGET_TIMESTEPS,
                "train_seeds": TRAIN_SEEDS,
                "eval_seeds": EVAL_SEEDS,
                "eval_type": "full",
                "eval_action_mode": EVAL_ACTION_MODE,
                "safeguard_threshold": SAFEGUARD_THRESHOLD if EVAL_ACTION_MODE == "safeguarded" else None,
                "mc_samples": MC_SAMPLES if EVAL_ACTION_MODE == "safeguarded" else None,
                "torch_num_threads": TORCH_NUM_THREADS,
                "python": str(PYTHON),
                "single_stage_runner": str(SINGLE_STAGE_RUNNER),
                "resume": RESUME,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_dir_for_seed(train_seed: int) -> Path:
    return OUT_DIR / f"seed{train_seed}"


def log_path_for_seed(train_seed: int) -> Path:
    return OUT_DIR / f"{MODEL_LABEL}_seed{train_seed}.log"


def run_artifacts_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "training_summary.csv",
        run_dir / "evaluation_summary.csv",
        run_dir / "evaluation_per_episode.csv",
        run_dir / "checkpoint_actor.pth",
        run_dir / "checkpoint_critic.pth",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


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


def load_single_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def run_seed(train_seed: int) -> dict[str, object]:
    run_dir = run_dir_for_seed(train_seed)
    log_path = log_path_for_seed(train_seed)

    if run_artifacts_complete(run_dir):
        print()
        print(f"Skipping completed {DISPLAY_NAME} | seed={train_seed}", flush=True)
        return collect_seed_result(train_seed)

    if RESUME and run_dir.exists():
        archive_incomplete_run(run_dir, log_path)

    print()
    print(
        f"Running {DISPLAY_NAME} ({MODEL_SUBDIR}) | seed={train_seed} | "
        f"timesteps={TARGET_TIMESTEPS}",
        flush=True,
    )
    print(f"Log: {log_path}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODEL_SUBDIR": MODEL_SUBDIR,
            "MODEL_LABEL": MODEL_LABEL,
            "HIGHWAY_CONFIG": BASE_CONFIG_NAME,
            "TOTAL_TIMESTEPS": str(TARGET_TIMESTEPS),
            "TRAIN_SEED": str(train_seed),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "TORCH_NUM_THREADS": str(TORCH_NUM_THREADS),
            "RUN_ID": f"{RUN_ID}\\seed{train_seed}",
            "EGO_START_LANE_POLICY": EGO_START_LANE_POLICY,
            "CONFIG_OVERRIDES": json.dumps(LITE_CONFIG_OVERRIDES),
            "EVAL_ACTION_MODE": EVAL_ACTION_MODE,
            "SAFEGUARD_THRESHOLD": str(SAFEGUARD_THRESHOLD),
            "MC_SAMPLES": str(MC_SAMPLES),
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
            print(f"[{MODEL_LABEL}:{train_seed}] {line}", end="")
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"{MODEL_LABEL} seed={train_seed} failed with exit code {return_code}")

    return collect_seed_result(train_seed)


def collect_seed_result(train_seed: int) -> dict[str, object]:
    run_dir = run_dir_for_seed(train_seed)
    train_summary_path = run_dir / "training_summary.csv"
    eval_summary_path = run_dir / "evaluation_summary.csv"
    train = load_single_row(train_summary_path)
    eval_ = load_single_row(eval_summary_path)
    return {
        "model": MODEL_LABEL,
        "display_name": DISPLAY_NAME,
        "train_seed": train_seed,
        "run_id": f"{RUN_ID}\\seed{train_seed}",
        "run_folder": str(run_dir),
        "training_summary": train,
        "evaluation_summary": eval_,
        "training_log": str(run_dir / "training_log.csv"),
        "training_summary_csv": str(train_summary_path),
        "evaluation_per_episode": str(run_dir / "evaluation_per_episode.csv"),
        "evaluation_summary_csv": str(eval_summary_path),
        "actor_checkpoint": str(run_dir / "checkpoint_actor.pth"),
        "critic_checkpoint": str(run_dir / "checkpoint_critic.pth"),
    }


def write_per_run_summary(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "per_run_summary.csv"
    fieldnames = [
        "run_index",
        "model",
        "train_seed",
        "env_id",
        "base_config_name",
        "config_variant",
        "ego_start_lane_policy",
        "target_timesteps",
        "final_training_return",
        "final_training_length",
        "eval_episodes",
        "eval_type",
        "avg_episodic_return",
        "avg_episodic_length",
        "collision_rate",
        "eval_action_mode",
        "safeguard_threshold",
        "mc_samples",
        "avg_uncertainty",
        "avg_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate",
        "total_activations",
        "run_id",
        "run_folder",
        "training_log",
        "training_summary",
        "evaluation_per_episode",
        "evaluation_summary",
        "actor_checkpoint",
        "critic_checkpoint",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, result in enumerate(results, start=1):
            train = dict(result["training_summary"])
            eval_ = dict(result["evaluation_summary"])
            writer.writerow(
                {
                    "run_index": index,
                    "model": MODEL_LABEL,
                    "train_seed": result["train_seed"],
                    "env_id": ENV_ID,
                    "base_config_name": BASE_CONFIG_NAME,
                    "config_variant": CONFIG_VARIANT,
                    "ego_start_lane_policy": EGO_START_LANE_POLICY,
                    "target_timesteps": TARGET_TIMESTEPS,
                    "final_training_return": train.get("avg_episodic_return", ""),
                    "final_training_length": train.get("avg_episodic_length", ""),
                    "eval_episodes": eval_.get("eval_episodes", ""),
                    "eval_type": eval_.get("eval_type", ""),
                    "avg_episodic_return": eval_.get("avg_episodic_return", ""),
                    "avg_episodic_length": eval_.get("avg_episodic_length", ""),
                    "collision_rate": eval_.get("collision_rate", ""),
                    "eval_action_mode": eval_.get("eval_action_mode", EVAL_ACTION_MODE),
                    "safeguard_threshold": eval_.get("safeguard_threshold", ""),
                    "mc_samples": eval_.get("mc_samples", ""),
                    "avg_uncertainty": eval_.get("avg_uncertainty", ""),
                    "avg_max_uncertainty": eval_.get("avg_max_uncertainty", ""),
                    "avg_activations_per_episode": eval_.get("avg_activations_per_episode", ""),
                    "avg_activation_rate": eval_.get("avg_activation_rate", ""),
                    "total_activations": eval_.get("total_activations", ""),
                    "run_id": result["run_id"],
                    "run_folder": result["run_folder"],
                    "training_log": result["training_log"],
                    "training_summary": result["training_summary_csv"],
                    "evaluation_per_episode": result["evaluation_per_episode"],
                    "evaluation_summary": result["evaluation_summary_csv"],
                    "actor_checkpoint": result["actor_checkpoint"],
                    "critic_checkpoint": result["critic_checkpoint"],
                }
            )
    return out_path


def write_aggregate_summary(results: list[dict[str, object]], per_run_summary: Path) -> Path:
    out_path = OUT_DIR / "aggregate_summary.csv"

    def avg_eval(key: str) -> float:
        return sum(float(dict(item["evaluation_summary"])[key]) for item in results) / len(results)

    def avg_train(key: str) -> float:
        return sum(float(dict(item["training_summary"])[key]) for item in results) / len(results)

    def avg_eval_optional(key: str) -> float | str:
        values = []
        for item in results:
            value = dict(item["evaluation_summary"]).get(key, "")
            if value not in {"", None}:
                values.append(float(value))
        return round(sum(values) / len(values), 6) if values else ""

    def sum_eval_optional(key: str) -> int | str:
        values = []
        for item in results:
            value = dict(item["evaluation_summary"]).get(key, "")
            if value not in {"", None}:
                values.append(float(value))
        return int(sum(values)) if values else ""

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "env_id",
            "base_config_name",
            "config_variant",
            "ego_start_lane_policy",
            "runs",
            "train_seeds",
            "eval_episodes_per_run",
            "total_eval_episodes",
            "avg_final_training_return",
            "avg_final_training_length",
            "avg_of_eval_avg_episodic_return",
            "avg_of_eval_avg_episodic_length",
            "avg_collision_rate",
            "eval_action_mode",
            "safeguard_threshold",
            "mc_samples",
            "avg_uncertainty",
            "avg_max_uncertainty",
            "avg_activations_per_episode",
            "avg_activation_rate",
            "total_activations",
            "per_run_summary",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model": MODEL_LABEL,
                "env_id": ENV_ID,
                "base_config_name": BASE_CONFIG_NAME,
                "config_variant": CONFIG_VARIANT,
                "ego_start_lane_policy": EGO_START_LANE_POLICY,
                "runs": len(results),
                "train_seeds": ";".join(str(item["train_seed"]) for item in results),
                "eval_episodes_per_run": len(EVAL_SEEDS),
                "total_eval_episodes": len(results) * len(EVAL_SEEDS),
                "avg_final_training_return": round(avg_train("avg_episodic_return"), 6),
                "avg_final_training_length": round(avg_train("avg_episodic_length"), 6),
                "avg_of_eval_avg_episodic_return": round(avg_eval("avg_episodic_return"), 6),
                "avg_of_eval_avg_episodic_length": round(avg_eval("avg_episodic_length"), 6),
                "avg_collision_rate": round(avg_eval("collision_rate"), 6),
                "eval_action_mode": EVAL_ACTION_MODE,
                "safeguard_threshold": SAFEGUARD_THRESHOLD if EVAL_ACTION_MODE == "safeguarded" else "",
                "mc_samples": MC_SAMPLES if EVAL_ACTION_MODE == "safeguarded" else "",
                "avg_uncertainty": avg_eval_optional("avg_uncertainty"),
                "avg_max_uncertainty": avg_eval_optional("avg_max_uncertainty"),
                "avg_activations_per_episode": avg_eval_optional("avg_activations_per_episode"),
                "avg_activation_rate": avg_eval_optional("avg_activation_rate"),
                "total_activations": sum_eval_optional("total_activations"),
                "per_run_summary": str(per_run_summary),
            }
        )
    return out_path


def main() -> int:
    if len(EVAL_SEEDS) != 50:
        raise ValueError(f"Expected 50 evaluation seeds, got {len(EVAL_SEEDS)}")
    if not (ROOT / MODEL_SUBDIR).exists():
        raise FileNotFoundError(f"Model directory not found: {ROOT / MODEL_SUBDIR}")
    if EVAL_ACTION_MODE not in {"actor", "safeguarded"}:
        raise ValueError(f"Unsupported EVAL_ACTION_MODE={EVAL_ACTION_MODE!r}")

    write_study_config()

    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Output directory: {OUT_DIR}", flush=True)
    print(f"Python: {PYTHON}", flush=True)
    print(f"Model: {DISPLAY_NAME} ({MODEL_SUBDIR})", flush=True)
    print(f"Env: {ENV_ID}", flush=True)
    print(f"Base config: {BASE_CONFIG_NAME}", flush=True)
    print(f"Config variant: {CONFIG_VARIANT}", flush=True)
    print(f"Target timesteps: {TARGET_TIMESTEPS}", flush=True)
    print(f"Train seeds: {TRAIN_SEEDS}", flush=True)
    print(f"Eval seeds: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} ({len(EVAL_SEEDS)} episodes)", flush=True)
    print(f"Eval action mode: {EVAL_ACTION_MODE}", flush=True)
    if EVAL_ACTION_MODE == "safeguarded":
        print(f"Safeguard threshold: {SAFEGUARD_THRESHOLD}", flush=True)
        print(f"MC samples: {MC_SAMPLES}", flush=True)
    print(f"Resume: {RESUME}", flush=True)

    results = [run_seed(int(seed)) for seed in TRAIN_SEEDS]
    per_run_summary = write_per_run_summary(results)
    aggregate_summary = write_aggregate_summary(results, per_run_summary)

    print()
    print(f"Completed {DISPLAY_NAME} continuous stage2 lite 5-run study.", flush=True)
    print(f"Per-run summary: {per_run_summary}", flush=True)
    print(f"Aggregate summary: {aggregate_summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
