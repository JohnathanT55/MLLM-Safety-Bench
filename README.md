# CSE 674 Project

Multimodal safety benchmarking pipeline for local MLLMs using MM-SafetyBench.

## Scope

This repository evaluates three local vision-language models against three image-based jailbreak attacks and an optional CoCA defense.

- Models: `qwen3_5_9b`, `gemma3_12b`, `glm4_6v_flash`
- Attacks: `figstep`, `scenetap`, `bvs`
- Defense: `CoCA`
- Judge: `gemini-3-flash-preview` with `gemini-3.1-pro-preview` audit sampling

## Included Results

Formal result files included in `results/`:

- `results/final_benchmark/bvs_expansion_3x3x10.json`
- `results/final_benchmark/bvs_expansion_highrisk_3x3x10.json`
- `results/final_benchmark/bvs_remaining_3x7x10.json`
- `results/final_benchmark/scenetap_full_3x13x10.json`
- `results/final_benchmark/figstep_full_3x13x10.json`
- `results/final_benchmark/bvs_glm_coca_followup_2x10.json`
- `results/final_benchmark/bvs_glm_physicalharm_coca_10.json`
- `results/final_benchmark/bvs_coca_expansion_5conds.json`
- `results/benign_utility/benign_utility_full_3x13x10_typo.json`

Judge benchmarking artifacts are included under:

- `results/judge_comparison/`
- `results/llm_judge_benchmark_bundle/`

## Current Status

- Completed: full baseline sweep for `BVS`, `SceneTAP`, and `FigStep` across all 13 scenarios with 3 models and 10 samples per condition
- Completed: formal benign utility evaluation on the `TYPO` split
- Completed: BVS CoCA expansion for all BVS conditions with baseline `ASR > 0`
- Pending: CoCA expansion for `SceneTAP` and `FigStep`

## Quick Start

### Install

```bash
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### Configure LM Studio

1. Start LM Studio local server on port `1234`.
2. Ensure these exact model IDs are available:
   - `qwen/qwen3.5-9b`
   - `google/gemma-3-12b`
   - `zai-org/glm-4.6v-flash`

### Verify Connectivity

```bash
python tests\test_lmstudio.py
```

### Run Final Benchmark

```bash
python experiments\final_benchmark.py --attacks figstep --samples 10 --llm-judge --audit-rate 0.1 --prompt-style strong
```

### Run Benign Utility Evaluation

```bash
python experiments\benign_utility_eval.py --splits TYPO --samples 10 --llm-judge --audit-rate 0.1
```

## Repository Layout

```text
configs/
docs/
experiments/
results/
scripts/
src/
tests/
```

Key entrypoints:

- `experiments/final_benchmark.py`: main formal experiment runner
- `experiments/benign_utility_eval.py`: benign over-refusal evaluation
- `scripts/plot_final_results.py`: plotting helper
- `docs/LLM_JUDGE_SETUP.md`: judge setup notes

Legacy scripts kept for reference:

- `experiments/milestone_mvp.py`
- `experiments/full_comparison.py`

## Metrics

- `ASR`: Attack Success Rate
- `RR`: Refusal Rate
- `SUI`: Safety-Usability Index

## Notes

- `final_benchmark.py` supports `--resume` and checkpoints both attack and CoCA sample progress.
- Because LM Studio serves one model at a time, experiments are intended to run sequentially.
- This package excludes smoke tests, pilot runs, logs, and cache artifacts.
