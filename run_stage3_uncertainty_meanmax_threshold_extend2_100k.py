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

DEFAULT_RUN_ID = f"stage3_uncertainty_meanmax_threshold_extend2_100k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
OUT_DIR = ROOT / "experiment_runs" / RUN_ID
PRIOR_PER_RUN_SUMMARY = Path(
    os.environ.get(
        "PRIOR_PER_RUN_SUMMARY",
        str(
            ROOT
            / "experiment_runs"
            / "stage3_uncertainty_meanmax_threshold_3run_100k_20260512_142042"
            / "per_run_summary.csv"
        ),
    )
)

MODEL_LABEL = "uncertainty_meanmax_threshold"
MODEL_SUBDIR = "uncertainty_aware_mean+max_backbone"
DISPLAY_NAME = "Uncertainty mean+max threshold"
CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_3_mixed_traffic")
TARGET_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
TRAIN_SEEDS = json.loads(os.environ.get("TRAIN_SEEDS", "[45678, 56789]"))
EVAL_SEEDS = json.loads(os.environ.get("EVAL_SEEDS", json.dumps(list(range(1000, 1050)))))
EVAL_MAX_STEPS = os.environ.get("EVAL_MAX_STEPS", "full")
EGO_START_LANE_POLICY = os.environ.get("EGO_START_LANE_POLICY", "random")
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "1"))
SAFEGUARD_THRESHOLD = float(os.environ.get("SAFEGUARD_THRESHOLD", "0.0005"))
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "5"))


def ps_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def make_command_header(train_seed: int, child_run_id: str) -> str:
    assignments = {
        "MODEL_SUBDIR": MODEL_SUBDIR,
        "MODEL_LABEL": MODEL_LABEL,
        "HIGHWAY_CONFIG": CONFIG_NAME,
        "TOTAL_TIMESTEPS": TARGET_TIMESTEPS,
        "TRAIN_SEED": train_seed,
        "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
        "EVAL_MAX_STEPS": EVAL_MAX_STEPS,
        "EVAL_ACTION_MODE": "safeguarded",
        "SAFEGUARD_THRESHOLD": SAFEGUARD_THRESHOLD,
        "MC_SAMPLES": MC_SAMPLES,
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
                "model_label": MODEL_LABEL,
                "display_name": DISPLAY_NAME,
                "model_subdir": MODEL_SUBDIR,
                "config_name": CONFIG_NAME,
                "target_timesteps": TARGET_TIMESTEPS,
                "train_seeds": TRAIN_SEEDS,
                "eval_seeds": EVAL_SEEDS,
                "eval_type": "full" if str(EVAL_MAX_STEPS).lower() == "full" else "capped",
                "eval_max_steps": None if str(EVAL_MAX_STEPS).lower() == "full" else EVAL_MAX_STEPS,
                "eval_action_mode": "safeguarded",
                "safeguard_threshold": SAFEGUARD_THRESHOLD,
                "mc_samples": MC_SAMPLES,
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


def run_one(train_seed: int) -> dict[str, object]:
    child_run_id = f"{RUN_ID}\\seed{train_seed}"
    command_header = make_command_header(train_seed, child_run_id)
    log_path = OUT_DIR / f"seed{train_seed}.log"

    print()
    print(f"Running {DISPLAY_NAME} | seed={train_seed} | timesteps={TARGET_TIMESTEPS}", flush=True)
    print("Command:", flush=True)
    print(command_header, flush=True)
    print(f"Log: {log_path}", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MODEL_SUBDIR": MODEL_SUBDIR,
            "MODEL_LABEL": MODEL_LABEL,
            "HIGHWAY_CONFIG": CONFIG_NAME,
            "TOTAL_TIMESTEPS": str(TARGET_TIMESTEPS),
            "TRAIN_SEED": str(train_seed),
            "EVAL_SEEDS": json.dumps(EVAL_SEEDS),
            "EVAL_MAX_STEPS": EVAL_MAX_STEPS,
            "EVAL_ACTION_MODE": "safeguarded",
            "SAFEGUARD_THRESHOLD": str(SAFEGUARD_THRESHOLD),
            "MC_SAMPLES": str(MC_SAMPLES),
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
            print(f"[{MODEL_LABEL}:{train_seed}] {line}", end="")
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"{MODEL_LABEL} seed={train_seed} failed with exit code {return_code}")

    run_dir = ROOT / "experiment_runs" / child_run_id
    return {
        "model_label": MODEL_LABEL,
        "display_name": DISPLAY_NAME,
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
        "model_label",
        "display_name",
        "train_seed",
        "run_dir",
        "target_timesteps",
        "final_train_return",
        "final_train_length",
        "avg_eval_return",
        "avg_eval_length",
        "collision_rate",
        "avg_uncertainty",
        "avg_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate",
        "total_activations",
        "safeguard_threshold",
        "mc_samples",
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
                    "model_label": item["model_label"],
                    "display_name": item["display_name"],
                    "train_seed": item["train_seed"],
                    "run_dir": item["run_dir"],
                    "target_timesteps": train.get("target_timesteps", ""),
                    "final_train_return": train.get("avg_episodic_return", ""),
                    "final_train_length": train.get("avg_episodic_length", ""),
                    "avg_eval_return": eval_.get("avg_episodic_return", ""),
                    "avg_eval_length": eval_.get("avg_episodic_length", ""),
                    "collision_rate": eval_.get("collision_rate", ""),
                    "avg_uncertainty": eval_.get("avg_uncertainty", ""),
                    "avg_max_uncertainty": eval_.get("avg_max_uncertainty", ""),
                    "avg_activations_per_episode": eval_.get("avg_activations_per_episode", ""),
                    "avg_activation_rate": eval_.get("avg_activation_rate", ""),
                    "total_activations": eval_.get("total_activations", ""),
                    "safeguard_threshold": eval_.get("safeguard_threshold", ""),
                    "mc_samples": eval_.get("mc_samples", ""),
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


def numeric(rows: list[dict[str, object]], summary_key: str, csv_key: str) -> list[float]:
    out = []
    for item in rows:
        value = dict(item[summary_key]).get(csv_key, "")
        if value != "":
            out.append(float(value))
    return out


def write_aggregate_summary(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "aggregate_summary.csv"
    fields = [
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
        "avg_uncertainty",
        "avg_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate",
        "total_activations",
        "safeguard_threshold",
        "mc_samples",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        total_activations = numeric(results, "evaluation_summary", "total_activations")
        writer.writerow(
            {
                "model_label": MODEL_LABEL,
                "display_name": DISPLAY_NAME,
                "runs": len(results),
                "train_seeds": ";".join(str(item["train_seed"]) for item in results),
                "target_timesteps": TARGET_TIMESTEPS,
                "avg_train_return": avg(numeric(results, "training_summary", "avg_episodic_return")),
                "avg_train_length": avg(numeric(results, "training_summary", "avg_episodic_length")),
                "avg_eval_return": avg(numeric(results, "evaluation_summary", "avg_episodic_return")),
                "avg_eval_length": avg(numeric(results, "evaluation_summary", "avg_episodic_length")),
                "collision_rate": avg(numeric(results, "evaluation_summary", "collision_rate")),
                "avg_uncertainty": avg(numeric(results, "evaluation_summary", "avg_uncertainty")),
                "avg_max_uncertainty": avg(numeric(results, "evaluation_summary", "avg_max_uncertainty")),
                "avg_activations_per_episode": avg(
                    numeric(results, "evaluation_summary", "avg_activations_per_episode")
                ),
                "avg_activation_rate": avg(numeric(results, "evaluation_summary", "avg_activation_rate")),
                "total_activations": int(sum(total_activations)),
                "safeguard_threshold": SAFEGUARD_THRESHOLD,
                "mc_samples": MC_SAMPLES,
            }
        )
    return out_path


def write_combined_eval_csv(results: list[dict[str, object]]) -> Path:
    out_path = OUT_DIR / "combined_evaluation_per_episode.csv"
    fields = [
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
        "eval_action_mode",
        "safeguard_threshold",
        "mc_samples",
        "avg_uncertainty",
        "max_uncertainty",
        "activation_count",
        "activation_rate",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            with Path(str(item["evaluation_per_episode_csv"])).open("r", newline="", encoding="utf-8") as csv_file:
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
                            "eval_action_mode": row.get("eval_action_mode", ""),
                            "safeguard_threshold": row.get("safeguard_threshold", ""),
                            "mc_samples": row.get("mc_samples", ""),
                            "avg_uncertainty": row.get("avg_uncertainty", ""),
                            "max_uncertainty": row.get("max_uncertainty", ""),
                            "activation_count": row.get("activation_count", ""),
                            "activation_rate": row.get("activation_rate", ""),
                        }
                    )
    return out_path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def write_combined_five_seed_summaries(current_per_run: Path):
    if not PRIOR_PER_RUN_SUMMARY.exists():
        print(f"Prior per-run summary not found, skipping 5-seed aggregate: {PRIOR_PER_RUN_SUMMARY}", flush=True)
        return None, None

    rows = read_csv_rows(PRIOR_PER_RUN_SUMMARY) + read_csv_rows(current_per_run)
    combined_per_run = OUT_DIR / "combined_5seed_per_run_summary.csv"
    combined_aggregate = OUT_DIR / "combined_5seed_aggregate_summary.csv"

    with combined_per_run.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fields = [
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
        "avg_uncertainty",
        "avg_max_uncertainty",
        "avg_activations_per_episode",
        "avg_activation_rate",
        "total_activations",
        "safeguard_threshold",
        "mc_samples",
        "sources",
    ]
    with combined_aggregate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "model_label": MODEL_LABEL,
                "display_name": DISPLAY_NAME,
                "runs": len(rows),
                "train_seeds": ";".join(row["train_seed"] for row in rows),
                "target_timesteps": TARGET_TIMESTEPS,
                "avg_train_return": avg([csv_float(row, "final_train_return") for row in rows]),
                "avg_train_length": avg([csv_float(row, "final_train_length") for row in rows]),
                "avg_eval_return": avg([csv_float(row, "avg_eval_return") for row in rows]),
                "avg_eval_length": avg([csv_float(row, "avg_eval_length") for row in rows]),
                "collision_rate": avg([csv_float(row, "collision_rate") for row in rows]),
                "avg_uncertainty": avg([csv_float(row, "avg_uncertainty") for row in rows]),
                "avg_max_uncertainty": avg([csv_float(row, "avg_max_uncertainty") for row in rows]),
                "avg_activations_per_episode": avg(
                    [csv_float(row, "avg_activations_per_episode") for row in rows]
                ),
                "avg_activation_rate": avg([csv_float(row, "avg_activation_rate") for row in rows]),
                "total_activations": int(sum(csv_float(row, "total_activations") for row in rows)),
                "safeguard_threshold": SAFEGUARD_THRESHOLD,
                "mc_samples": MC_SAMPLES,
                "sources": f"{PRIOR_PER_RUN_SUMMARY};{current_per_run}",
            }
        )

    return combined_per_run, combined_aggregate


def main() -> int:
    if not SINGLE_STAGE_RUNNER.exists():
        raise FileNotFoundError(f"Missing runner: {SINGLE_STAGE_RUNNER}")
    if len(EVAL_SEEDS) != 50:
        raise ValueError(f"Expected 50 evaluation seeds, got {len(EVAL_SEEDS)}")
    if len(TRAIN_SEEDS) != 2:
        raise ValueError(f"Expected 2 extension training seeds, got {len(TRAIN_SEEDS)}")

    write_study_config()
    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Output directory: {OUT_DIR}", flush=True)
    print(f"Model: {DISPLAY_NAME} ({MODEL_SUBDIR})", flush=True)
    print(f"Config: {CONFIG_NAME}", flush=True)
    print(f"Target timesteps: {TARGET_TIMESTEPS}", flush=True)
    print(f"Train seeds: {TRAIN_SEEDS}", flush=True)
    print(f"Eval seeds: {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} ({len(EVAL_SEEDS)} episodes)", flush=True)
    print(f"Safeguard threshold: {SAFEGUARD_THRESHOLD}", flush=True)
    print(f"MC samples: {MC_SAMPLES}", flush=True)

    results = [run_one(int(seed)) for seed in TRAIN_SEEDS]
    per_run = write_per_run_summary(results)
    aggregate = write_aggregate_summary(results)
    combined_eval = write_combined_eval_csv(results)
    combined_per_run, combined_aggregate = write_combined_five_seed_summaries(per_run)

    print()
    print("Stage 3 uncertainty mean+max threshold study complete.", flush=True)
    print(f"Per-run summary: {per_run}", flush=True)
    print(f"Aggregate summary: {aggregate}", flush=True)
    print(f"Combined per-episode evaluation: {combined_eval}", flush=True)
    if combined_per_run and combined_aggregate:
        print(f"Combined 5-seed per-run summary: {combined_per_run}", flush=True)
        print(f"Combined 5-seed aggregate summary: {combined_aggregate}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
