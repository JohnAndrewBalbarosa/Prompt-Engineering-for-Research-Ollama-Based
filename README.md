# Prompt Engineering for Research Ollama Based

This repository runs a prompt-engineering experiment on GSM8K with local-first and optional automatic remote retrieval.

- Generation provider: Ollama only
- Judge provider: Ollama only
- Dataset source: local snapshot, or automatic Hugging Face retrieval

## Current Scope

- Compare prompt strategies on a fixed local model setup
- Parse final answers and evaluate exact-match correctness
- Optionally run a local judge model for rubric-based scoring
- Run a dedicated llama3.1 confusion checker for TP/TN/FP/FN analysis
- Export structured outputs for analysis

## Requirements

- Python virtual environment in this workspace
- Ollama running locally
- Models pulled in Ollama for generation and judge (current config uses llama3.1:8b)
- Local snapshot file at data/raw/gsm8k_snapshot.jsonl (auto-fetch can populate or expand it)

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

Useful runtime flags:

- `--dataset-source local|auto|remote`
- `--dataset-split test|train`
- `--sample-size <int>`
- `--interactive-dataset-source` (prompts at startup when source not provided)
- `--storage sql|parallel|file` (default is `sql`)

Examples:

```powershell
# Auto-fetch enough GSM8K rows for a larger run and persist in SQLite only
& ".venv/Scripts/python.exe" -m src.main --config config/experiment.json --dataset-source auto --sample-size 500 --storage sql

# Force remote retrieval and prompt for mode if omitted
& ".venv/Scripts/python.exe" -m src.main --config config/experiment.json --interactive-dataset-source
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
- Windows pass-through to main CLI: `-MainArgs @("--dataset-source","auto","--sample-size","500","--storage","sql")`

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

Primary output is SQLite (default storage mode `sql`):

- results/runs/local_experiment_results.sqlite

Quick SQL examples:

```sql
-- Latest parsed rows for one strategy
SELECT run_id, model_id, item_id, exact_match_label, judge_correctness
FROM v_results_by_strategy
WHERE prompt_strategy = 'zero_shot'
ORDER BY run_id DESC, model_id, item_id;

-- Confusion metrics by strategy and model
SELECT run_id, prompt_strategy, model_id, tp, fp, fn, tn, precision, recall, f1, no_data_reason
FROM v_confusion_by_strategy
ORDER BY run_id DESC, prompt_strategy, model_id;

-- Quantitative normalized confusion metrics
SELECT run_id, prompt_strategy, model_id, aggregate_type, precision, recall, f1, specificity, npv, balanced_accuracy
FROM v_quantitative_by_strategy
ORDER BY run_id DESC, prompt_strategy, model_id, aggregate_type;
```

When `--storage parallel` or `--storage file` is used, additional files are written:

- results/runs/local_raw_generations.jsonl
- results/runs/local_parsed_answers.csv
- results/runs/local_metrics_summary.csv
- results/runs/local_confusion_matrices.json
- results/runs/local_quantitative_summary.csv
- results/runs/local_quantitative_details.json
- results/runs/local_run_metadata.json
- results/runs/by_strategy/<strategy>/raw_generations.jsonl
- results/runs/by_strategy/<strategy>/parsed_answers.csv
- results/runs/by_strategy/<strategy>/metrics_summary.csv
- results/runs/by_strategy/<strategy>/confusion_matrices.json
- results/runs/by_strategy/<strategy>/quantitative_summary.csv
- results/runs/by_strategy/<strategy>/quantitative_details.json

## Notes

- Non-ollama providers fail fast by design in local-only mode.
- Dataset retrieval supports local snapshot and automatic Hugging Face download (via `datasets`).
- Successful rows are skipped on reruns based on parsed output state.

