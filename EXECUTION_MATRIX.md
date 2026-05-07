# Execution Matrix for Final Benchmark

**Last Updated**: 2026-05-07
**Models**: Qwen3.5-9B, Gemma-3-12B, GLM-4.6V-Flash
**Judge**: Gemini-3-Flash-Preview (10% Pro audit)
**Samples per condition**: 10
**Attack prompt style**: strong

---

## Summary Statistics

| Metric | Completed | Pending |
|--------|-----------|---------|
| **Baseline scenarios** | 13 / 13 | 0 |
| **Baseline attack types** | 3 / 3 | 0 |
| **Baseline conditions** | 117 / 117 | 0 |
| **Benign utility conditions** | 39 / 39 | 0 |
| **BVS CoCA vulnerable conditions** | 8 / 8 | 0 |
| **SceneTAP/FigStep CoCA** | 0 / all vulnerable conditions | pending |

---

## Coverage Matrix

| Scenario | FigStep | SceneTAP | BVS | BVS+CoCA | Benign |
|----------|---------|----------|-----|----------|--------|
| **01. Illegal_Activitiy** | ✅ | ✅ | ✅ | ✅ GLM | ✅ |
| **02. HateSpeech** | ✅ | ✅ | ✅ | n/a | ✅ |
| **03. Privacy_Violence** | ✅ | ✅ | ✅ | ✅ GLM | ✅ |
| **04. Malware_Generation** | ✅ | ✅ | ✅ | n/a | ✅ |
| **05. Physical_Harm** | ✅ | ✅ | ✅ | ✅ Qwen/Gemma/GLM | ✅ |
| **06. Fraud** | ✅ | ✅ | ✅ | ✅ GLM | ✅ |
| **07. Financial_Advice** | ✅ | ✅ | ✅ | n/a | ✅ |
| **08. Gov_Decision** | ✅ | ✅ | ✅ | n/a | ✅ |
| **09. Health_Consultation** | ✅ | ✅ | ✅ | ✅ GLM | ✅ |
| **10. Legal_Opinion** | ✅ | ✅ | ✅ | n/a | ✅ |
| **11. Political_Lobbying** | ✅ | ✅ | ✅ | n/a | ✅ |
| **12. Sex** | ✅ | ✅ | ✅ | n/a | ✅ |
| **13. EconomicHarm** | ✅ | ✅ | ✅ | ✅ GLM | ✅ |

**Notes**:
- `n/a` means baseline `ASR = 0`, so no BVS CoCA follow-up was required for that condition.
- CoCA coverage shown here applies only to the BVS attack because SceneTAP/FigStep CoCA has not been run yet.

---

## Included Formal Result Files

### Baseline benchmark
- `results/final_benchmark/bvs_expansion_3x3x10.json`
- `results/final_benchmark/bvs_expansion_highrisk_3x3x10.json`
- `results/final_benchmark/bvs_remaining_3x7x10.json`
- `results/final_benchmark/scenetap_full_3x13x10.json`
- `results/final_benchmark/figstep_full_3x13x10.json`

### BVS CoCA
- `results/final_benchmark/bvs_glm_coca_followup_2x10.json`
- `results/final_benchmark/bvs_glm_physicalharm_coca_10.json`
- `results/final_benchmark/bvs_coca_expansion_5conds.json`

### Benign utility
- `results/benign_utility/benign_utility_full_3x13x10_typo.json`

---

## Key Findings So Far

- `Health_Consultation` is the most consistently vulnerable scenario across attacks.
- `GLM-4.6V-Flash` is the most vulnerable model on BVS.
- BVS CoCA is strongly scenario-dependent.
- CoCA helps in some BVS conditions, but it backfires on `GLM + Physical_Harm` and also worsens several additional GLM BVS conditions.

---

## Remaining Work

1. Run CoCA expansion for `SceneTAP` vulnerable conditions.
2. Run CoCA expansion for `FigStep` vulnerable conditions.
3. Generate final plots, summary tables, and significance tests.
4. Write final comparative discussion and failure-case analysis.
