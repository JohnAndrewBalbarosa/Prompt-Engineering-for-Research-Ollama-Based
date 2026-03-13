# Prompt Engineering for Research Ollama Based

This repository runs a prompt-engineering experiment on GSM8K in strict local-only mode.

- Generation provider: Ollama only
- Judge provider: Ollama only
- Dataset source: local JSONL snapshot only

## Current Scope

- Compare prompt strategies on a fixed local model setup
- Parse final answers and evaluate exact-match correctness
- Optionally run a local judge model for rubric-based scoring
- Export structured outputs for analysis

## Requirements

- Python virtual environment in this workspace
- Ollama running locally
- Models pulled in Ollama for generation and judge (current config uses llama3.1:8b)
- Local snapshot file present at data/raw/gsm8k_snapshot.jsonl

## Configuration

Primary config file:

- config/experiment.json

Quick local profile:

- config/experiment-gemini.json

Despite its filename, config/experiment-gemini.json is now a local-only quick profile.

## Environment Variables

Supported local env variables are defined in .env.example:

- EXPERIMENT_CONFIG_PATH
- OLLAMA_BASE_URL
- OLLAMA_REQUEST_INTERVAL_SECONDS
- OLLAMA_JUDGE_REQUEST_INTERVAL_SECONDS
- OLLAMA_RETRY_BACKOFF_SECONDS
- OLLAMA_MAX_RETRY_BACKOFF_SECONDS
- OLLAMA_JUDGE_RETRY_BACKOFF_SECONDS
- OLLAMA_JUDGE_MAX_RETRY_BACKOFF_SECONDS
- OLLAMA_NUM_GPU
- OLLAMA_DEVICES
- OLLAMA_DEBUG
- OLLAMA_GENERATION_BUCKET
- OLLAMA_JUDGE_BUCKET

Precedence order:

1. CLI arguments
2. Config file values
3. Environment fallback values
4. Code defaults

## Run

From repository root:

```powershell
& "c:/Users/Drew/Desktop/Prompt Engineering for Research Ollama Based/.venv/bin/python.exe" -m src.main --config config/experiment.json
```

## One-Click Scripts

These scripts automate local setup and execution:

- Install Ollama if missing
- Start Ollama server
- Pull the required model
- Create virtual environment
- Install Python dependencies if present
- Create .env from .env.example if missing
- Run the experiment

Windows PowerShell:

```powershell
pwsh -ExecutionPolicy Bypass -File scripts/one_click_windows.ps1
```

Linux:

```bash
bash scripts/one_click_linux.sh
```

Optional arguments:

- Windows: `-ConfigPath config/experiment.json -ModelNames llama3.1:8b,deepseek-coder-v2 -SkipRun`
- Linux: `--config config/experiment.json --model llama3.1:8b --skip-run`

Windows behavior notes:

- Pulls all Ollama models listed under `models[]` where `provider` is `ollama`
- Also pulls Ollama judge model when `judge.enabled=true` and `judge.provider=ollama`
- You can add extra pulls via `-ModelName` (single) or `-ModelNames` (multiple)
- If `OLLAMA_NUM_GPU` is not set, the script defaults it to `999` before starting `ollama serve`

GPU tips:

- Set `OLLAMA_NUM_GPU=999` to prefer maximum layer offload on GPU
- Set `OLLAMA_DEVICES` to pin specific GPU devices on multi-GPU systems
- Set `OLLAMA_DEBUG=1` for device/backend diagnostics while troubleshooting

Run a single model:

```powershell
& "c:/Users/Drew/Desktop/Prompt Engineering for Research Ollama Based/.venv/bin/python.exe" -m src.main --config config/experiment.json --model ollama_main
```

Run a single strategy:

```powershell
& "c:/Users/Drew/Desktop/Prompt Engineering for Research Ollama Based/.venv/bin/python.exe" -m src.main --config config/experiment.json --strategy zero_shot
```

## Outputs

Default outputs are configured in config/experiment.json:

- results/runs/local_raw_generations.jsonl
- results/runs/local_parsed_answers.csv
- results/runs/local_metrics_summary.csv
- results/runs/local_confusion_matrices.json
- results/runs/local_run_metadata.json

## Notes

- Non-ollama providers fail fast by design in local-only mode.
- Dataset download is not performed; snapshot file is required.
- Successful rows are skipped on reruns based on parsed output state.

