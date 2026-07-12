# Prompt-Engineering-for-Research-Ollama-Based

## Overview

A reproducible, local-only experiment pipeline using Ollama to measure how prompt engineering strategies affect final-answer correctness on the GSM8K dataset.

Repository: [JohnAndrewBalbarosa/Prompt-Engineering-for-Research-Ollama-Based](https://github.com/JohnAndrewBalbarosa/Prompt-Engineering-for-Research-Ollama-Based)

## Problem and Goal

**Problem.** Prompt-strategy claims are often based on cloud models or anecdotal examples, making local model comparisons difficult to reproduce.

**Goal.** Run a local-only GSM8K experiment that compares zero-shot and chain-of-thought prompting across Ollama models using repeatable exact-match evaluation.

## System Design

- `src/`: experiment loading, Ollama inference, answer extraction, and evaluation logic.
- `prompts/`: versioned prompt strategies.
- `config/` + `scripts/`: model/run configuration and experiment entry points.
- `results/`: saved outputs and aggregate comparisons; `data/`: dataset inputs/cache.

## Setup and Usage

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Start Ollama and pull the configured local models, then inspect scripts/
python scripts/run_experiment.py --help
```

## Evaluation Method

- Evaluated zero-shot and chain-of-thought prompting on GSM8K using local Ollama models.
- Used exact-match final-answer accuracy as the primary metric.
- Compared five local models with 20 questions per cell.

## Results

- Mean accuracy improved from 50% zero-shot to 67% chain-of-thought.
- Mean lift: +17 percentage points.
- 4 of 5 tested models improved with chain-of-thought prompting.
- qwen2.5:7b improved from 60% to 95%.
- gemma2:9b improved from 55% to 90%.
- llama2:13b-chat regressed from 20% to 10%, showing that chain-of-thought was not universal.

## Interpretation

- Chain-of-thought prompting improved final-answer accuracy for most capable small and mid-sized local models in this run.
- The regression on llama2:13b-chat suggests the strategy must be evaluated per model instead of assumed globally.
- The sample size is directional rather than final because each cell used 20 questions.

## Limitations

- Results should only be treated as validated when this README includes the dataset, sample size, metric definition, and reproduction steps.
- Any AI-generated, OCR-based, scraped, or heuristic output requires manual review before being used as ground truth.
- Environment-dependent measurements such as latency, memory use, browser behavior, and API reliability should be re-measured on the target machine.

## Recommendations and Future Work

- Re-run with a larger sample size such as 500 questions per model/strategy cell.
- Add confidence intervals and repeated runs to separate prompt effects from sampling variance.
- Track latency and accuracy together so a model is not selected on accuracy alone.

## Documentation Standard

This README follows a technical-project structure: overview, goal, system design, setup, evaluation method, results, interpretation, limitations, and recommendations. Update the Results section whenever new measurements are available so project claims stay evidence-backed.
