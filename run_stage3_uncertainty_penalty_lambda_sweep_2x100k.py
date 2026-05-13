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
PYTHON = Path(os.environ.get("PYTHON_EXE", str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else sys.executable)))

DEFAULT_RUN_ID = f"stage3_uncertainty_penalty_lambda_sweep_2x100k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
OUT_DIR = ROOT / "experiment_runs" / RUN_ID

MODEL_SUBDIR = "uncertainty_penalty_PPO"
DISPLAY_NAME = "Uncertainty penalty PPO"
CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_3_mixed_traffic")
TARGET_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
TRAIN_SEEDS = json.loads(os.environ.get("TRAIN_SEEDS", "[12345, 23456]"))
LAMBDA_VALUES = json.loads(os.environ.get("LAMBDA_VALUES", "[0.01, 0.05]"))
EVAL_SEEDS = json.loads(os.environ.get("EVAL_SEEDS", json.dumps(list(range(1000, 1050)))))
EVAL_MAX_STEPS = os.environ.get("EVAL_MAX_STEPS", "full")
EGO_START_LANE_POLICY = os.environ.get("EGO_START_LANE_POLICY", "random")
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
DROPOUT_P = float(os.environ.get("DROPOUT_P", "0.1"))
PPO_MC_SAMPLES = int(os.environ.get("PPO_MC_SAMPLES", "5"))


def lambda_label(lambda_u: float) -> str:
    return str(lambda_u).replace(".", "_")


def ps_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def make_command_header(lambda_u: float, train_seed: int, child_run_id: str) -> str:
    model_label = f"uncertainty_penalty_lambda_{lambda_label(lambda_u)}"
    assignments = {
        "MODEL_SUBDIR": MODEL_SUBDIR,
        "MODEL_LABEL": model_label,
        "HIGHWAY_CONFIG": CONFIG_NAME,
        "TOTAL_TIMESTEPS": TARGET_TIMESTEPS,
        "TRAIN_SEED": train_seed,
        "LAMBDA_U": lambda_u,
        "DROPOUT_P": DROPOUT_P,
        "PPO_MC_SAMPLES": PPO_MC_SAMPLES,
        "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
        "EVAL_MAX_STEPS": EVAL_MAX_STEPS,
        "EVAL_ACTION_MODE": "actor",
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
                "display_name": DISPLAY_NAME,
                "model_subdir": MODEL_SUBDIR,
                "config_name": CONFIG_NAME,
                "target_timesteps": TARGET_TIMESTEPS,
                "train_seeds": TRAIN_SEEDS,
                "lambda_values": LAMBDA_VALUES,
                "eval_seeds": EVAL_SEEDS,
                "eval_type": "full" if str(EVAL_MAX_STEPS).lower() == "full" else "capped",
                "eval_max_steps": None if str(EVAL_MAX_STEPS).lower() == "full" else EVAL_MAX_STEPS,
                "eval_action_mode": "actor",
                "dropout_p": DROPOUT_P,
                "ppo_mc_samples": PPO_MC_SAMPLES,
                "torch_num_threads": TORCH_NUM_THREADS,
                "ego_start_lane_policy": EGO_START_LANE_POLICY,
                "python": str(PYTHON),
                "runner": str(SINGLE_STAGE_RUNNER),
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


def run_one(lambda_u: float, train_seed: int) -> dict[str, object]:
    model_label = f"uncertainty_penalty_lambda_{lambda_label(lambda_u)}"
    child_run_id = f"{RUN_ID}\\lambda_{lambda_label(lambda_u)}\\seed{train_seed}"
    command_header = make_command_header(lambda_u, train_seed, child_run_id)
    log_path = OUT_DIR / f"lambda_{lambda_label(lambda_u)}_seed{train_seed}.log"

    print()
    print(
        f"Running {DISPLAY_NAME} | lambda_u={lambda_u} | seed={train_seed} | timesteps={TARGET_TIMESTEPS}",
        flush=True,
    )
    print("Command:", flush=True)
    print(command_header, flush=True)
    print(f"Log: {log_path}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODEL_SUBDIR": MODEL_SUBDIR,
            "MODEL_LABEL": model_label,
            "HIGHWAY_CONFIG": CONFIG_NAME,
            "TOTAL_TIMESTEPS": str(TARGET_TIMESTEPS),
            "TRAIN_SEED": str(train_seed),
            "LAMBDA_U": str(lambda_u),
            "DROPOUT_P": str(DROPOUT_P),
            "PPO_MC_SAMPLES": str(PPO_MC_SAMPLES),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "EVAL_MAX_STEPS": EVAL_MAX_STEPS,
            "EVAL_ACTION_MODE": "actor",
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
    return {
        "lambda_u": lambda_u,
        "model_label": model_label,
        "display_name": f"{DISPLAY_NAME} lambda_u={lambda_u}",
        "train_seed": train_seed,
        "run_dir": str(run_dir),
        "command": command_header,
        "training_summary": load_single_row(run_dir / "training_summary.csv"),
        "evaluation_summary": load_single_row(run_dir / "evaluation_summary.csv"),
        "evaluation_per_episode_csv": str(run_dir / "evaluation_per_episode.csv"),
    }


def write_per_run_summary(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "per_run_summary.csv"
    fieldnames = [
        "lambda_u",
        "model_label",
        "display_name",
        "train_seed",
        "run_dir",
        "target_timesteps",
        "final_train_return",
        "final_train_length",
        "final_raw_train_return",
        "final_avg_critic_uncertainty",
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
            run_dir = Path(str(item["run_dir"]))
            writer.writerow(
                {
                    "lambda_u": item["lambda_u"],
                    "model_label": item["model_label"],
                    "display_name": item["display_name"],
                    "train_seed": item["train_seed"],
                    "run_dir": item["run_dir"],
                    "target_timesteps": train.get("target_timesteps", ""),
                    "final_train_return": train.get("avg_episodic_return", ""),
                    "final_train_length": train.get("avg_episodic_length", ""),
                    "final_raw_train_return": train.get("avg_raw_episodic_return", ""),
                    "final_avg_critic_uncertainty": train.get("avg_critic_uncertainty", ""),
                    "avg_eval_return": eval_.get("avg_episodic_return", ""),
                    "avg_eval_length": eval_.get("avg_episodic_length", ""),
                    "collision_rate": eval_.get("collision_rate", ""),
                    "training_summary_csv": str(run_dir / "training_summary.csv"),
                    "evaluation_summary_csv": str(run_dir / "evaluation_summary.csv"),
                    "evaluation_per_episode_csv": item["evaluation_per_episode_csv"],
                    "actor_checkpoint": train.get("actor_checkpoint", ""),
                    "critic_checkpoint": train.get("critic_checkpoint", ""),
                    "command": item["command"],
                }
            )
    return out_path


def avg(values: list[float]) -> float:
    return sum(values) / len(values)


def to_float(value: object) -> float | None:
    text = str(value)
    if text == "":
        return None
    return float(text)


def write_aggregate_summary(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "aggregate_summary.csv"
    fields = [
        "lambda_u",
        "runs",
        "train_seeds",
        "target_timesteps",
        "avg_train_return",
        "avg_train_length",
        "avg_raw_train_return",
        "avg_critic_uncertainty",
        "avg_eval_return",
        "avg_eval_length",
        "collision_rate",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for lambda_u in LAMBDA_VALUES:
            group = [item for item in results if float(item["lambda_u"]) == float(lambda_u)]
            if not group:
                continue
            train_rows = [dict(item["training_summary"]) for item in group]
            eval_rows = [dict(item["evaluation_summary"]) for item in group]

            def average_rows(rows: list[dict[str, str]], key: str) -> str:
                values = [to_float(row.get(key, "")) for row in rows]
                present = [value for value in values if value is not None]
                return "" if not present else avg(present)

            writer.writerow(
                {
                    "lambda_u": lambda_u,
                    "runs": len(group),
                    "train_seeds": ";".join(str(item["train_seed"]) for item in group),
                    "target_timesteps": TARGET_TIMESTEPS,
                    "avg_train_return": average_rows(train_rows, "avg_episodic_return"),
                    "avg_train_length": average_rows(train_rows, "avg_episodic_length"),
                    "avg_raw_train_return": average_rows(train_rows, "avg_raw_episodic_return"),
                    "avg_critic_uncertainty": average_rows(train_rows, "avg_critic_uncertainty"),
                    "avg_eval_return": average_rows(eval_rows, "avg_episodic_return"),
                    "avg_eval_length": average_rows(eval_rows, "avg_episodic_length"),
                    "collision_rate": average_rows(eval_rows, "collision_rate"),
                }
            )
    return out_path


def write_combined_eval_csv(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "combined_evaluation_per_episode.csv"
    fields = [
        "lambda_u",
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            with Path(str(item["evaluation_per_episode_csv"])).open("r", newline="", encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    writer.writerow(
                        {
                            "lambda_u": item["lambda_u"],
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


def main() -> int:
    if not SINGLE_STAGE_RUNNER.exists():
        raise FileNotFoundError(f"Missing runner: {SINGLE_STAGE_RUNNER}")
    if len(EVAL_SEEDS) != 50:
        raise ValueError(f"Expected 50 evaluation seeds, got {len(EVAL_SEEDS)}")
    if len(TRAIN_SEEDS) != 2:
        raise ValueError(f"Expected 2 training seeds, got {len(TRAIN_SEEDS)}")

    write_study_config()
    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Output directory: {OUT_DIR}", flush=True)
    print(f"Model: {DISPLAY_NAME} ({MODEL_SUBDIR})", flush=True)
    print(f"Config: {CONFIG_NAME}", flush=True)
    print(f"Target timesteps: {TARGET_TIMESTEPS}", flush=True)
    print(f"Lambda values: {LAMBDA_VALUES}", flush=True)
    print(f"Train seeds per lambda: {TRAIN_SEEDS}", flush=True)
    print(f"Eval seeds: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} ({len(EVAL_SEEDS)} episodes)", flush=True)

    results = []
    for lambda_u in LAMBDA_VALUES:
        for seed in TRAIN_SEEDS:
            results.append(run_one(float(lambda_u), int(seed)))

    per_run = write_per_run_summary(results)
    aggregate = write_aggregate_summary(results)
    combined_eval = write_combined_eval_csv(results)

    print()
    print("Stage 3 uncertainty penalty lambda sweep complete.", flush=True)
    print(f"Per-run summary: {per_run}", flush=True)
    print(f"Aggregate summary: {aggregate}", flush=True)
    print(f"Combined per-episode evaluation: {combined_eval}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
