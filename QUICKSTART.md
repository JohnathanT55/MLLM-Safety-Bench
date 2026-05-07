# Quick Start

## 1. Install Dependencies

```bash
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## 2. Start LM Studio

Use the local server on port `1234` and make sure these models are available with the exact IDs used by `configs/models.yaml`:

- `qwen/qwen3.5-9b`
- `google/gemma-3-12b`
- `zai-org/glm-4.6v-flash`

## 3. Verify Local Setup

```bash
python tests\test_lmstudio.py
```

Optional cache check:

```bash
python scripts\check_cache.py --verify
```

## 4. Main Experiment Commands

### Final baseline benchmark

```bash
python experiments\final_benchmark.py --attacks figstep --samples 10 --llm-judge --audit-rate 0.1 --prompt-style strong
```

### Resume a long benchmark run

```bash
python experiments\final_benchmark.py --attacks scenetap --samples 10 --llm-judge --audit-rate 0.1 --prompt-style strong --resume --output-file scenetap_full_3x13x10.json
```

### Run CoCA defense comparison

```bash
python experiments\final_benchmark.py --attacks bvs --configs Physical_Harm --samples 10 --llm-judge --audit-rate 0.1 --prompt-style strong --defense --resume
```

### Run benign utility evaluation

```bash
python experiments\benign_utility_eval.py --splits TYPO --samples 10 --llm-judge --audit-rate 0.1
```

## 5. Included Formal Results

Current formal JSON outputs in this package:

- `results/final_benchmark/bvs_expansion_3x3x10.json`
- `results/final_benchmark/bvs_expansion_highrisk_3x3x10.json`
- `results/final_benchmark/bvs_remaining_3x7x10.json`
- `results/final_benchmark/scenetap_full_3x13x10.json`
- `results/final_benchmark/figstep_full_3x13x10.json`
- `results/final_benchmark/bvs_glm_coca_followup_2x10.json`
- `results/final_benchmark/bvs_glm_physicalharm_coca_10.json`
- `results/final_benchmark/bvs_coca_expansion_5conds.json`
- `results/benign_utility/benign_utility_full_3x13x10_typo.json`

## 6. Current Scope Boundaries

- Included: all three full baseline attacks
- Included: benign utility formal run
- Included: BVS CoCA expansion
- Not yet included: SceneTAP and FigStep CoCA expansions
- Not yet included: final publication-grade plots and significance tables
