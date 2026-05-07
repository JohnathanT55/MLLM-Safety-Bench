# Quick Demo

## Recommended Live Demo

This demo runs one real sample end-to-end and asks LM Studio to load a fresh model first.

### Command

```bash
python experiments\final_benchmark.py --models glm4_6v_flash --attacks bvs --configs Physical_Harm --samples 1 --llm-judge --audit-rate 1.0 --defense --prompt-style strong --output-file demo_full_pipeline_1sample.json
```

### Why this command

- `glm4_6v_flash`: forces LM Studio to load a different model
- `bvs + Physical_Harm`: one of the strongest completed full-pipeline conditions
- `--samples 1`: keeps the demo short
- `--llm-judge`: enables the LLM judge stage
- `--audit-rate 1.0`: forces judge audit on the single sample
- `--defense`: runs the CoCA defense stage too

### What to say

"This one command loads a model, generates the visual jailbreak attack, evaluates the response with the LLM judge, runs audit, and then applies CoCA defense on the same sample."

### What to open after it finishes

- `results/final_benchmark/demo_full_pipeline_1sample.json`

## Safer Fallback Demo

If live inference fails, use the aggregation-only demo:

```bash
python scripts\aggregate_paper_results.py --results-root . --table-dir results/demo_tables --figure-dir results/demo_figures
```
