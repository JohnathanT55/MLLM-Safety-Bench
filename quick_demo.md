# Quick Demo

## Recommended Live Demo

This demo runs one real sample end-to-end and asks LM Studio to load a fresh model first.

If `python` is not available as Python 3 on macOS, use `python3` instead.

## Read This First

If you have not read the full `README.md`, complete these sections before running the demo:

- `README.md` -> `External Requirements`
- `README.md` -> `Data Preparation`
- `README.md` -> `Full Setup` steps `1` through `4`

In practice, that means:

1. Install dependencies in a virtual environment.
2. Copy `example.env` to `.env` and fill in `API_KEY`.
3. Start LM Studio on port `1234`.
4. Make sure these exact models are available in LM Studio:
   - `qwen/qwen3.5-9b`
   - `google/gemma-3-12b`
   - `zai-org/glm-4.6v-flash`
5. Run `python tests/test_lmstudio.py` to confirm the local setup works.

### Command

```bash
# Cross-platform
python experiments/final_benchmark.py --models glm4_6v_flash --attacks bvs --configs Physical_Harm --samples 1 --llm-judge --audit-rate 1.0 --defense --prompt-style strong --output-file demo_full_pipeline_1sample.json
```

### Why this command

- `glm4_6v_flash`: forces LM Studio to load a different model
- `bvs + Physical_Harm`: one of the strongest completed full-pipeline conditions
- `--samples 1`: keeps the demo short
- `--llm-judge`: enables the LLM judge stage
- `--audit-rate 1.0`: forces judge audit on the single sample
- `--defense`: runs the CoCA defense stage too

This one command loads a model, generates the visual jailbreak attack, evaluates the response with the LLM judge, runs audit, and then applies CoCA defense on the same sample.

### What to open after it finishes

- `results/final_benchmark/demo_full_pipeline_1sample.json`
- This file contains attack metrics, defense metrics, and the single-sample raw outputs.

## Safer Fallback Demo

If live inference fails, use the included plotting helper on the bundled formal results:

```bash
python scripts/plot_final_results.py --benchmark results/final_benchmark/figstep_full_3x13x10.json --benign results/benign_utility/benign_utility_full_3x13x10_typo.json --output-dir results/demo_figures
```

There is no database preparation step for this demo. The only data dependency is the MM-SafetyBench dataset, which is downloaded automatically and cached locally by Hugging Face on first use.

For full setup, full benchmark commands, and important file descriptions, see `README.md`.
