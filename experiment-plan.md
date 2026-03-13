# Gemini-Focused Prompt Engineering Experiment Plan

## Objective

Build a reproducible Python experiment pipeline that measures how prompt engineering affects answer correctness on GSM8K using Gemini as the only accessible model family for now.

This study is no longer framed as a cross-model comparison. It is now framed as a within-model prompt-engineering evaluation using Gemini for both response generation and optional judge-based evaluation.

## Practical Research Framing

Because only Gemini API access is currently available, the research should focus on these questions:

1. Does prompt engineering improve Gemini's final-answer correctness on GSM8K?
2. How do zero-shot and chain-of-thought prompting compare within the same Gemini model?
3. How reliable is Gemini as a judge when compared with exact GSM8K ground truth?
4. What quota, rate-limit, and prompt-length constraints affect reproducible evaluation with free Gemini access?

## What This Study Is And Is Not

### What It Is

- A Gemini-only prompt-engineering experiment
- A controlled comparison of prompt strategies under one model family
- A correctness and judge-reliability evaluation on GSM8K
- A practical, quota-aware benchmark workflow

### What It Is Not

- A broad comparison across OpenAI, Anthropic, Google, Meta, and DeepSeek
- A claim about which frontier model is best overall
- A final multi-model benchmark study

That broader study can still be added later once access and quota improve.

## Recommended Thesis Or Paper Positioning

Use wording like this in your methodology and scope sections:

> This phase of the study evaluates prompt-engineering effects within a single accessible model family, Gemini, due to current API and quota constraints. The design is intended as a reproducible pilot benchmark that can later be extended to additional LLM providers.

That keeps the research defensible instead of pretending the study is already multi-model when it is not.

## Current Execution Scope

The current build should support:

- GSM8K loading from Hugging Face dataset infrastructure
- Prompt variants such as zero-shot and chain-of-thought
- Gemini generation through Google AI Studio API access
- Exact-answer extraction and correctness scoring
- Gemini-as-a-judge evaluation
- Confusion matrix analysis for judge reliability
- CSV and JSON export for reporting

## Recommended Project Structure

```text
Sir Alex/
  experiment-plan.md
  README.md
  .env.example
  config/
    experiment-gemini.json
  data/
    raw/
  prompts/
    zero_shot.txt
    cot.txt
    judge_prompt.txt
  results/
    runs/
      gemini_raw_generations.jsonl
      gemini_parsed_answers.csv
      gemini_metrics_summary.csv
      gemini_confusion_matrices.json
      gemini_run_metadata.json
  src/
    main.py
    config.py
    dataset_loader.py
    prompt_builder.py
    answer_parser.py
    evaluator.py
    judge.py
    metrics.py
    io_utils.py
    env_utils.py
    models/
      base.py
      google_client.py
      openai_client.py
```

Note: the OpenAI client may remain in the codebase as optional future support, but it is not part of the active Gemini-focused study design.

## Gemini-Focused Experimental Design

## Independent Variable

Prompt strategy:

- zero-shot
- chain-of-thought
- optional future ablations if quota allows

## Controlled Factors

- Same dataset subset
- Same Gemini model for all prompt conditions
- Same temperature
- Same max token settings
- Same answer extraction rules
- Same judging rubric

## Dependent Variables

- final-answer accuracy
- parse success rate
- judge agreement with ground truth
- precision, recall, and F1 for judge reliability
- average latency
- failure rate caused by quota or rate limiting

## Revised Research Logic

Since model family is fixed, the key causal question becomes:

> When the model is held constant, does changing the prompt improve performance?

That is a valid and much cleaner experimental design than forcing a weak multi-model setup without access.

## End-To-End Workflow

1. Load Gemini experiment configuration.
2. Download or load cached GSM8K data.
3. Normalize gold answers.
4. Build prompt variants.
5. Run each question through Gemini for each prompt strategy.
6. Save raw responses immediately.
7. Parse final answers.
8. Compare parsed answers with GSM8K gold answers.
9. Run Gemini judge on successful generations.
10. Compute metrics and confusion matrices.
11. Export results for writing and plotting.

## Configuration Plan

### Primary Config File

- `config/experiment-gemini.json`

### What It Should Define

- dataset split and sample size
- prompt strategies
- Gemini generation model name
- Gemini judge model name
- temperature and token limits
- retry limits
- output file paths

### Current Recommended Gemini Setup

- Generation model: `gemini-flash-lite-latest`
- Judge model: `gemini-flash-lite-latest`
- Sample size: start small, then scale up only if quota allows

### Why This Matters

Gemini free-tier limits are the central operational constraint in this project. Config must make it easy to reduce sample size, disable judging, or run only one strategy.

## Dataset Plan

### Dataset

- GSM8K test split

### Required Record Fields

- `item_id`
- `question`
- `raw_answer`
- `gold_final_answer`
- `split`

### Why GSM8K Still Works Well Here

Even with one model family, GSM8K remains a valid benchmark because the study is about prompt effects on mathematical reasoning accuracy.

## Prompt Strategy Plan

### Minimum Required Strategies

- `zero_shot`
- `chain_of_thought`

### Optional Future Strategies

- `question_only`
- `ablation_no_reasoning`
- `ablation_no_format`

### Recommendation Under Free-Tier Quota

Do not add more prompt variants until the first two can be run consistently.

The correct expansion order is:

1. zero-shot
2. chain-of-thought
3. judge-on-successful-rows only
4. ablations later

## Answer Extraction Plan

The parser should continue to extract answers in this order:

1. explicit `Final Answer:` line
2. fallback formatted answer patterns
3. last numeric expression

This is necessary because Gemini sometimes follows formatting well and sometimes does not, especially under quota or short output budgets.

## Evaluation Plan

## Primary Metric

Exact-match correctness against GSM8K gold answers.

This remains the primary metric. Judge output should never replace the ground truth answer check.

## Judge Metric

Use Gemini as a judge only for rows where generation succeeded.

Judge output should measure:

- whether the response is correct
- quality of reasoning
- arithmetic quality
- format-following quality

## Confusion Matrix Interpretation

The confusion matrix compares:

- exact-match correctness
- Gemini judge correctness

This measures judge reliability, not raw task performance.

If there are no successful rows, the confusion matrix should still show zero-valued rows rather than remaining empty.

## Quota-Aware Execution Strategy

This is now a core part of the plan, not a side note.

### Known Constraint

Free Gemini access can fail mid-run with `429 RESOURCE_EXHAUSTED`.

### Required Design Responses

- keep sample sizes small
- save every generation immediately
- skip only successful rows on rerun
- allow rerunning one strategy at a time
- allow judging only after generation succeeds

### Recommended Run Order

1. Run `zero_shot` only.
2. Check quota health.
3. Run `chain_of_thought` only.
4. Judge successful rows.
5. Export metrics.

This is better than running everything in one batch and exhausting quota halfway through.

## Revised Metrics Plan

For each prompt strategy, export:

- total rows
- success rows
- error rows
- judged rows
- correct rows
- incorrect rows
- unparseable rows
- accuracy on successful rows
- parse failure rate on successful rows
- average latency

For judge reliability, export:

- TP
- FP
- TN
- FN
- precision
- recall
- F1

## Revised Interpretation Guidance

When writing the results section:

- Separate model-performance outcomes from quota failures.
- Do not mix API failure with reasoning failure.
- Report incomplete conditions honestly.

For example:

> Zero-shot prompting completed on 8 of 10 sampled items before quota exhaustion and achieved 100% correctness on completed rows. Chain-of-thought prompting could not be fully evaluated in the same run due to Gemini free-tier request limits.

That is methodologically stronger than pretending the missing rows are model failures.

## Recommended Immediate Study Design

For the current phase, the most defensible design is:

- one Gemini model
- two prompt strategies
- one small GSM8K subset
- Gemini judge on successful rows only
- explicit reporting of quota failures

This is a valid pilot study.

## Future Expansion Path

Once quota or additional provider access becomes available, extend in this order:

1. Increase Gemini sample size
2. Add more prompt ablations
3. Add a second Gemini model tier
4. Add a second provider family
5. Convert the study into a true cross-model comparison

## Recommended Final Position

The research plan should now be described as:

> A Gemini-centered pilot evaluation of prompt engineering on GSM8K, designed for reproducibility under limited-cost API access and structured to expand into a broader multi-model benchmark later.

That matches what you can actually execute today.