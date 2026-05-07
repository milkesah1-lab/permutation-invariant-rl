@echo off
setlocal

set "ROOT=%~dp0"
set "RUN_ID=%~1"
if "%RUN_ID%"=="" set "RUN_ID=fair_curriculum_manual"

if "%EVAL_EPISODES%"=="" set "EVAL_EPISODES=10"
if "%CURRICULUM_STAGE_LIMIT%"=="" set "CURRICULUM_STAGE_LIMIT=4"
if "%CAPPED_EVAL_STEPS%"=="" set "CAPPED_EVAL_STEPS=300"
set "PYTHONUNBUFFERED=1"
set "HIGHWAY_PYTHON=C:\Users\milke\miniconda3\envs\highway\python.exe"

set "LOG_DIR=%ROOT%experiment_runs\%RUN_ID%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

"%HIGHWAY_PYTHON%" "%ROOT%run_fair_curriculum_experiments.py" > "%LOG_DIR%\scheduled_runner.stdout.log" 2> "%LOG_DIR%\scheduled_runner.stderr.log"

endlocal
