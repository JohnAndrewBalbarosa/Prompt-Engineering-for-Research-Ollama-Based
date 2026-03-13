# Local-Only Ollama Prompt Engineering Experiment Plan

## Objective

Build a reproducible local experiment pipeline that measures how prompt engineering affects answer correctness on GSM8K using Ollama for both response generation and optional judge evaluation.

This phase is intentionally local-only and single-provider. It is designed to be operationally stable and easy to rerun without cloud quota dependencies.

## Research Questions

1. Does prompt engineering improve final-answer correctness when the model is held constant?
2. How do zero-shot and chain-of-thought prompting compare under the same local model?
3. How reliable is local model-as-judge output compared with exact GSM8K ground truth?
4. How do local runtime settings (timeouts, intervals, retries) affect reproducibility and completion rate?

## Scope

### What This Study Is

- A local-only Ollama prompt-engineering experiment
- A controlled within-model comparison of prompt strategies
- Exact-match correctness evaluation on GSM8K
- Optional judge reliability analysis using the same local provider

### What This Study Is Not

- A cloud-provider comparison study
- A cross-family benchmark across multiple APIs
- A quota-constrained free-tier benchmark design

## Active Implementation Constraints

- Generation provider is restricted to ollama
- Judge provider is restricted to ollama
- Dataset loading is snapshot-only from local JSONL
- Non-ollama providers fail fast by design

## Current Execution Workflow

1. Load experiment config.
2. Load records from local snapshot file.
3. Build prompt text for each strategy.
4. Generate answers via Ollama.
5. Save raw generations immediately.
6. Parse final answers.
7. Compute exact-match correctness.
8. Run optional Ollama judge on successful generations.
9. Export metrics and confusion matrices.
10. Write run metadata.

## Dataset Plan

### Dataset Source

- Local snapshot file only: data/raw/gsm8k_snapshot.jsonl

### Required Record Fields

Normalized rows must provide:

- item_id
- question
- gold_final_answer
- split

Loader also accepts raw rows with:

- question
- answer

When raw rows are used, gold_final_answer is derived automatically during load.

## Prompt Strategy Plan

### Current Required Strategies

- zero_shot
- chain_of_thought

### Optional Future Strategies

- question_only
- ablation_no_reasoning
- ablation_no_format

## Evaluation Plan

### Primary Metric

- Exact-match correctness against GSM8K gold final answer

### Judge Metrics

- Judge correctness flag
- Reasoning score
- Arithmetic score
- Format-following score
- Judge explanation

### Reliability Outputs

- TP, FP, TN, FN
- Precision, recall, F1

## Configuration Plan

Primary config:

- config/experiment.json

Quick profile:

- config/experiment-gemini.json (local-only quick run profile)

Current baseline in config/experiment.json:

- Generation model: llama3.1:8b
- Judge model: llama3.1:8b
- Sample size: 25

## Environment Plan

Use .env.example for supported local runtime variables:

- EXPERIMENT_CONFIG_PATH
- OLLAMA_BASE_URL
- OLLAMA_REQUEST_INTERVAL_SECONDS
- OLLAMA_JUDGE_REQUEST_INTERVAL_SECONDS
- OLLAMA_RETRY_BACKOFF_SECONDS
- OLLAMA_MAX_RETRY_BACKOFF_SECONDS
- OLLAMA_JUDGE_RETRY_BACKOFF_SECONDS
- OLLAMA_JUDGE_MAX_RETRY_BACKOFF_SECONDS
- OLLAMA_GENERATION_BUCKET
- OLLAMA_JUDGE_BUCKET

Precedence order:

1. CLI arguments
2. Config file values
3. Environment fallbacks
4. Code defaults

## Output Artifacts

Default output paths:

- results/runs/local_raw_generations.jsonl
- results/runs/local_parsed_answers.csv
- results/runs/local_metrics_summary.csv
- results/runs/local_confusion_matrices.json
- results/runs/local_run_metadata.json

## Interpretation Guidance

- Separate model reasoning outcomes from runtime errors.
- Treat judge results as reliability analysis, not replacement for exact-match ground truth.
- Report completion status honestly when rows fail due to local runtime issues.

## Immediate Execution Recommendation

1. Start with zero_shot only on a small sample.
2. Run chain_of_thought next.
3. Keep judge enabled only if runtime remains stable.
4. Expand strategy count or sample size only after a stable baseline run.

## Future Expansion Path

Once local baseline is stable:

1. Add more prompt ablations.
2. Evaluate multiple local Ollama models.
3. Reintroduce optional multi-provider support only if needed for a later comparative phase.

## Final Positioning Statement

This project is currently a local-only pilot evaluation of prompt engineering on GSM8K, built for reproducibility and iterative expansion without cloud dependency.
