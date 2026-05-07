# CSE 674 Project

Multimodal safety benchmarking pipeline for local MLLMs using MM-SafetyBench.

## Scope

This repository evaluates three local vision-language models against three image-based jailbreak attacks and an optional CoCA defense.

- Models: `qwen3_5_9b`, `gemma3_12b`, `glm4_6v_flash`
- Attacks: `figstep`, `scenetap`, `bvs`
- Defense: `CoCA`
- Judge: `gemini-3-flash-preview` with `gemini-3.1-pro-preview` audit sampling

If you only want a very short end-to-end test, use `quick_demo.md`.
This `README.md` is the full setup and full-run guide.

## Current Status

- Completed: full baseline sweep for `BVS`, `SceneTAP`, and `FigStep` across all 13 scenarios with 3 models and 10 samples per condition
- Completed: formal benign utility evaluation on the `TYPO` split
- Completed: BVS CoCA expansion for all BVS conditions with baseline `ASR > 0`
- Pending: CoCA expansion for `SceneTAP` and `FigStep`

## External Requirements

- LM Studio local server running on port `1234`
- Gemini Enterprise Agent Platform / Vertex API key in `.env`
- Internet access on first dataset load to download `PKU-Alignment/MM-SafetyBench` from Hugging Face
- Optional: `HF_TOKEN` in `.env` for higher Hugging Face rate limits during dataset download

## Data Preparation

- No local database is required.
- The only data dependency is the MM-SafetyBench dataset from Hugging Face.
- The dataset is downloaded automatically on first access and then reused from the local Hugging Face cache.
- To verify cache availability before a long run, use:

```bash
python scripts/check_cache.py --verify
```

## Full Setup

### 1. Install Dependencies

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

If `python` is not available as Python 3 on macOS, replace it with `python3` in the commands below.

### 2. Configure API Keys

Copy `example.env` to `.env` and fill in at least `API_KEY`.

```bash
# macOS / Linux
cp example.env .env

# Windows PowerShell
# Copy-Item example.env .env
```

Set:

```text
API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token_here  # optional
```

### 3. Start LM Studio

Start the LM Studio local server on port `1234`, without authentication (API token), 
and make sure these exact model IDs are available (**They all Q8_0 Quant**):

- `qwen/qwen3.5-9b`
- `google/gemma-3-12b`
- `zai-org/glm-4.6v-flash`

### 4. Verify Local Setup

```bash
python tests/test_lmstudio.py
```

This verifies:

- LM Studio connectivity
- configured model loading and switching
- text-only chat
- multimodal chat
- MM-SafetyBench sample access

## Full Run Commands

### Final baseline benchmark

```bash
python experiments/final_benchmark.py --attacks figstep --samples 10 --llm-judge --audit-rate 0.1 --prompt-style strong
```

### Resume a long benchmark run

```bash
python experiments/final_benchmark.py --attacks scenetap --samples 10 --llm-judge --audit-rate 0.1 --prompt-style strong --resume --output-file scenetap_full_3x13x10.json
```

### Run CoCA defense comparison

```bash
python experiments/final_benchmark.py --attacks bvs --configs Physical_Harm --samples 10 --llm-judge --audit-rate 0.1 --prompt-style strong --defense --resume
```

### Run benign utility evaluation

```bash
python experiments/benign_utility_eval.py --splits TYPO --samples 10 --llm-judge --audit-rate 0.1
```

## Important Files and What They Do

### Core runners

- `experiments/final_benchmark.py`
  - Main harmful benchmark runner
  - Supports all three attacks, optional CoCA, optional LLM judge, and sample-level resume

- `experiments/benign_utility_eval.py`
  - Runs benign/control evaluation for over-refusal analysis

- `experiments/judge_comparison.py`
  - Compares judge model choices and audit behavior

### Model and evaluation code

- `src/models/lmstudio_client.py`
  - Connects to LM Studio
  - Handles model loading, switching, and multimodal chat requests

- `src/metrics/llm_judge.py`
  - Gemini-based LLM judge
  - Supports Flash primary judging and Pro audit sampling

- `src/metrics/evaluator.py`
  - Converts raw model outputs into metrics such as ASR, RR, and SUI

### Attack and defense implementations

- `src/attacks/figstep.py`
  - Typographic image jailbreak attack

- `src/attacks/scenetap.py`
  - Scene-coherent typographic jailbreak attack

- `src/attacks/bvs.py`
  - Semantic reconstruction jailbreak attack

- `src/defenses/coca.py`
  - Constitutional Calibration defense implementation

### Data and configuration

- `src/utils/data_loader.py`
  - Loads MM-SafetyBench from Hugging Face
  - Handles scenario config and split selection

- `configs/models.yaml`
  - Maps experiment model keys to exact LM Studio model IDs

- `configs/experiments.yaml`
  - Legacy experiment configuration reference

- `example.env`
  - Template for `API_KEY` and optional `HF_TOKEN`

### Utilities and validation

- `tests/test_lmstudio.py`
  - End-to-end local setup validation

- `scripts/check_cache.py`
  - Checks local Hugging Face cache and verifies dataset access

- `scripts/plot_final_results.py`
  - Generates simple summary plots from bundled formal JSON results

### Results and documentation

- `results/final_benchmark/`
  - Formal harmful benchmark JSON outputs

- `results/benign_utility/`
  - Formal benign evaluation outputs

- `results/judge_comparison/`
  - Judge benchmark outputs

- `results/llm_judge_benchmark_bundle/`
  - Bundled judge benchmark code and cleaned result package

- `docs/LLM_JUDGE_SETUP.md`
  - Detailed judge and Vertex / Gemini setup notes

- `quick_demo.md`
  - Minimal short-path demo for quick validation

## Included Formal Results

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

## Metrics

- `ASR`: Attack Success Rate
- `RR`: Refusal Rate
- `SUI`: Safety-Usability Index

## Notes

- `final_benchmark.py` supports `--resume` and checkpoints both attack and CoCA sample progress.
- Because LM Studio serves one model at a time, experiments are intended to run sequentially.
- Dataset download is automatic on first access; no separate database bootstrap step is needed.
- This package excludes smoke tests, pilot runs, logs, and cache artifacts.
