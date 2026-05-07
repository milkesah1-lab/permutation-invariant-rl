from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get(
    "HIGHWAY_PYTHON",
    r"C:\Users\milke\miniconda3\envs\highway\python.exe",
)
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("fair_curriculum_%Y%m%d_%H%M%S"))
EVAL_EPISODES = os.environ.get("EVAL_EPISODES", "10")
CURRICULUM_STAGE_LIMIT = os.environ.get("CURRICULUM_STAGE_LIMIT", "4")
CAPPED_EVAL_STEPS = os.environ.get("CAPPED_EVAL_STEPS", "300")

MODELS = [
    ("basic_mlp", ROOT / "baseline", {"PPO_NETWORK_TYPE": "mlp"}),
    ("deep_sets_mean", ROOT / "Deep_sets_mean_model", {}),
    ("deep_sets_mean_max", ROOT / "baseline", {"PPO_NETWORK_TYPE": "mean_max"}),
    ("self_attention", ROOT / "self_attention_model", {}),
]


def make_child_env() -> dict[str, str]:
    """Build a Windows-safe environment for subprocesses.

    Some shells expose both Path and PATH. Windows treats those as the same
    variable, and process launch can fail if both are forwarded.
    """
    env: dict[str, str] = {}
    path_value = os.environ.get("Path") or os.environ.get("PATH")

    for key, value in os.environ.items():
        if key.lower() == "path":
            continue
        if key.lower() in {existing.lower() for existing in env}:
            continue
        env[key] = value

    if path_value is not None:
        env["Path"] = path_value

    return env


def run_model(
    model_label: str,
    model_dir: Path,
    logs_dir: Path,
    env_overrides: dict[str, str] | None = None,
) -> int:
    log_path = logs_dir / f"{model_label}.log"
    env = make_child_env()
    env.update(
        {
            "RUN_ID": RUN_ID,
            "MODEL_LABEL": model_label,
            "EVAL_EPISODES": EVAL_EPISODES,
            "CURRICULUM_STAGE_LIMIT": CURRICULUM_STAGE_LIMIT,
            "CAPPED_EVAL_STEPS": CAPPED_EVAL_STEPS,
        }
    )
    if env_overrides:
        env.update(env_overrides)

    print()
    print(f"==================== Running {model_label} ====================")
    print(f"Folder: {model_dir}")
    print(f"Log: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [PYTHON, "curriculum_train.py"],
            cwd=model_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{model_label}] {line}", end="")
            log_file.write(line)

        return process.wait()


def combine_evaluation_csvs(logs_dir: Path) -> Path:
    combined_path = logs_dir / "combined_final_evaluation.csv"
    header_written = False

    with combined_path.open("w", newline="", encoding="utf-8") as out_file:
        writer = None
        for model_label, model_dir, _ in MODELS:
            eval_csv = (
                model_dir
                / "curriculum_artifacts"
                / RUN_ID
                / "evaluation"
                / f"{model_label}_final_evaluation.csv"
            )
            if not eval_csv.exists():
                print(f"Missing evaluation CSV for {model_label}: {eval_csv}")
                continue

            with eval_csv.open("r", newline="", encoding="utf-8") as in_file:
                reader = csv.reader(in_file)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(out_file)
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)

    return combined_path


def main() -> int:
    logs_dir = ROOT / "experiment_runs" / RUN_ID
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {RUN_ID}")
    print(f"Python: {PYTHON}")
    print(f"Evaluation episodes per eval: {EVAL_EPISODES}")
    print(f"Curriculum stages used: {CURRICULUM_STAGE_LIMIT}")
    print(f"Capped eval steps: {CAPPED_EVAL_STEPS}")
    print(f"Experiment logs: {logs_dir}")
    print("All models train from scratch; no existing weights are loaded.")

    failures = []
    for model_label, model_dir, env_overrides in MODELS:
        return_code = run_model(model_label, model_dir, logs_dir, env_overrides)
        if return_code != 0:
            failures.append((model_label, return_code))
            print(f"{model_label} failed with exit code {return_code}. Stopping.")
            break

    combined_eval = combine_evaluation_csvs(logs_dir)
    print()
    print(f"Combined evaluation CSV: {combined_eval}")

    if failures:
        print(f"Failures: {failures}")
        return 1

    print("All fair curriculum experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
