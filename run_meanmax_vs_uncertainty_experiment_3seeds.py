from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\milke\miniconda3\envs\highway\python.exe")
BASE_RUNNER = ROOT / "run_meanmax_vs_uncertainty_experiment.py"

DEFAULT_RUN_ID = f"meanmax_vs_uncertainty_3seeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_ID = os.environ.get("RUN_ID", DEFAULT_RUN_ID)
CONFIG_NAME = os.environ.get("HIGHWAY_CONFIG", "curriculum_stage_3_mixed_traffic")
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
TRAIN_SEEDS = json.loads(os.environ.get("TRAIN_SEEDS", "[12345, 23456, 34567]"))

OUT_DIR = ROOT / "experiment_runs" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def run_one_seed(train_seed: int, run_index: int) -> dict:
    child_run_id = f"{RUN_ID}_seed{train_seed}"
    log_path = OUT_DIR / f"seed_{run_index:02d}_{train_seed}.log"

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": child_run_id,
            "HIGHWAY_CONFIG": CONFIG_NAME,
            "TOTAL_TIMESTEPS": str(TOTAL_TIMESTEPS),
            "TRAIN_SEED": str(train_seed),
        }
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [str(PYTHON), str(BASE_RUNNER)],
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    child_out_dir = ROOT / "experiment_runs" / child_run_id
    summary_csv = child_out_dir / "summary.csv"
    training_csv = child_out_dir / "training_final_summary.csv"
    per_episode_csv = child_out_dir / "per_episode.csv"

    summary_rows = load_csv_rows(summary_csv)
    training_rows = load_csv_rows(training_csv)
    per_episode_rows = load_csv_rows(per_episode_csv)

    return {
        "train_seed": train_seed,
        "run_index": run_index,
        "run_id": child_run_id,
        "out_dir": str(child_out_dir),
        "log_path": str(log_path),
        "summary_rows": summary_rows,
        "training_rows": training_rows,
        "per_episode_rows": per_episode_rows,
    }


def write_combined_files(run_results: list[dict]) -> None:
    runs_csv = OUT_DIR / "runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_index", "train_seed", "run_id", "out_dir", "log_path"])
        for result in run_results:
            writer.writerow(
                [
                    result["run_index"],
                    result["train_seed"],
                    result["run_id"],
                    result["out_dir"],
                    result["log_path"],
                ]
            )

    summary_all_csv = OUT_DIR / "summary_all_runs.csv"
    with summary_all_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "run_index",
                "train_seed",
                "model",
                "config_name",
                "total_timesteps",
                "eval_episodes",
                "avg_episodic_return",
                "avg_episodic_length",
                "collision_rate",
            ]
        )
        for result in run_results:
            for row in result["summary_rows"]:
                writer.writerow(
                    [
                        result["run_index"],
                        result["train_seed"],
                        row["model"],
                        row["config_name"],
                        row["total_timesteps"],
                        row["eval_episodes"],
                        row["avg_episodic_return"],
                        row["avg_episodic_length"],
                        row["collision_rate"],
                    ]
                )

    training_all_csv = OUT_DIR / "training_final_summary_all_runs.csv"
    with training_all_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "run_index",
                "train_seed",
                "model",
                "iteration",
                "timesteps_so_far",
                "avg_episodic_length",
                "avg_episodic_return",
                "avg_loss",
                "avg_raw_episodic_return",
                "avg_critic_uncertainty",
                "iteration_seconds",
            ]
        )
        for result in run_results:
            for row in result["training_rows"]:
                writer.writerow(
                    [
                        result["run_index"],
                        result["train_seed"],
                        row["model"],
                        row["iteration"],
                        row["timesteps_so_far"],
                        row["avg_episodic_length"],
                        row["avg_episodic_return"],
                        row["avg_loss"],
                        row["avg_raw_episodic_return"],
                        row["avg_critic_uncertainty"],
                        row["iteration_seconds"],
                    ]
                )

    per_episode_all_csv = OUT_DIR / "per_episode_all_runs.csv"
    with per_episode_all_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["run_index", "train_seed", "model", "seed", "episodic_return", "episodic_length", "collided"])
        for result in run_results:
            for row in result["per_episode_rows"]:
                writer.writerow(
                    [
                        result["run_index"],
                        result["train_seed"],
                        row["model"],
                        row["seed"],
                        row["episodic_return"],
                        row["episodic_length"],
                        row["collided"],
                    ]
                )


def aggregate_model_metrics(run_results: list[dict]) -> tuple[list[dict], list[dict]]:
    summary_by_model: dict[str, list[dict[str, str]]] = {}
    training_by_model: dict[str, list[dict[str, str]]] = {}

    for result in run_results:
        for row in result["summary_rows"]:
            summary_by_model.setdefault(row["model"], []).append(row)
        for row in result["training_rows"]:
            training_by_model.setdefault(row["model"], []).append(row)

    aggregated_eval = []
    for model, rows in summary_by_model.items():
        aggregated_eval.append(
            {
                "model": model,
                "config_name": rows[0]["config_name"],
                "total_timesteps": int(rows[0]["total_timesteps"]),
                "num_training_runs": len(rows),
                "eval_episodes_per_run": int(rows[0]["eval_episodes"]),
                "avg_episodic_return_mean": sum(float(row["avg_episodic_return"]) for row in rows) / len(rows),
                "avg_episodic_length_mean": sum(float(row["avg_episodic_length"]) for row in rows) / len(rows),
                "collision_rate_mean": sum(float(row["collision_rate"]) for row in rows) / len(rows),
            }
        )

    aggregated_training = []
    for model, rows in training_by_model.items():
        aggregated_training.append(
            {
                "model": model,
                "num_training_runs": len(rows),
                "final_training_return_mean": sum(float(row["avg_episodic_return"]) for row in rows) / len(rows),
                "final_training_length_mean": sum(float(row["avg_episodic_length"]) for row in rows) / len(rows),
                "final_training_loss_mean": sum(float(row["avg_loss"]) for row in rows) / len(rows),
                "final_raw_training_return_mean": (
                    sum(float(row["avg_raw_episodic_return"]) for row in rows if row["avg_raw_episodic_return"]) / len(rows)
                    if any(row["avg_raw_episodic_return"] for row in rows)
                    else None
                ),
                "final_avg_critic_uncertainty_mean": (
                    sum(float(row["avg_critic_uncertainty"]) for row in rows if row["avg_critic_uncertainty"]) / len(rows)
                    if any(row["avg_critic_uncertainty"] for row in rows)
                    else None
                ),
            }
        )

    aggregated_eval.sort(key=lambda row: row["model"])
    aggregated_training.sort(key=lambda row: row["model"])
    return aggregated_eval, aggregated_training


def write_aggregates(aggregated_eval: list[dict], aggregated_training: list[dict]) -> None:
    summary_avg_csv = OUT_DIR / "summary_average_over_3_runs.csv"
    with summary_avg_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "config_name",
                "total_timesteps",
                "num_training_runs",
                "eval_episodes_per_run",
                "avg_episodic_return_mean",
                "avg_episodic_length_mean",
                "collision_rate_mean",
            ]
        )
        for row in aggregated_eval:
            writer.writerow(
                [
                    row["model"],
                    row["config_name"],
                    row["total_timesteps"],
                    row["num_training_runs"],
                    row["eval_episodes_per_run"],
                    round(row["avg_episodic_return_mean"], 6),
                    round(row["avg_episodic_length_mean"], 6),
                    round(row["collision_rate_mean"], 6),
                ]
            )

    training_avg_csv = OUT_DIR / "training_average_over_3_runs.csv"
    with training_avg_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "num_training_runs",
                "final_training_return_mean",
                "final_training_length_mean",
                "final_training_loss_mean",
                "final_raw_training_return_mean",
                "final_avg_critic_uncertainty_mean",
            ]
        )
        for row in aggregated_training:
            writer.writerow(
                [
                    row["model"],
                    row["num_training_runs"],
                    round(row["final_training_return_mean"], 6),
                    round(row["final_training_length_mean"], 6),
                    round(row["final_training_loss_mean"], 6),
                    "" if row["final_raw_training_return_mean"] is None else round(row["final_raw_training_return_mean"], 6),
                    "" if row["final_avg_critic_uncertainty_mean"] is None else round(row["final_avg_critic_uncertainty_mean"], 6),
                ]
            )


def main() -> int:
    print(f"Run ID: {RUN_ID}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Training seeds: {TRAIN_SEEDS}")

    run_results = []
    for run_index, train_seed in enumerate(TRAIN_SEEDS, start=1):
        print(f"Running seed {train_seed} ({run_index}/{len(TRAIN_SEEDS)})...")
        run_results.append(run_one_seed(train_seed, run_index))

    write_combined_files(run_results)
    aggregated_eval, aggregated_training = aggregate_model_metrics(run_results)
    write_aggregates(aggregated_eval, aggregated_training)

    print("Completed 3-seed experiment.")
    print(f"Per-run summary: {OUT_DIR / 'summary_all_runs.csv'}")
    print(f"Averaged eval summary: {OUT_DIR / 'summary_average_over_3_runs.csv'}")
    print(f"Averaged training summary: {OUT_DIR / 'training_average_over_3_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
