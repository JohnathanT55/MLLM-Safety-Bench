# LLM Judge Benchmark Bundle

This folder consolidates the recent Gemini judge benchmarking work.

## Included Code
- `code/judge_comparison.py`: comparison experiment script (Flash-Lite vs Flash, Pro arbitration).

## Included Results
- `results/final_report.json`: cleaned final summary used for decision making.
- `results/judge_comparison_final_1775364901.json`: full merged report.
- `results/batches/batch_0.json` ... `results/batches/batch_4.json`: intermediate batch outputs.

## Key Metrics (150 samples)
- Flash-Lite accuracy vs Pro arbitration: 74.0%
- Flash accuracy vs Pro arbitration: 99.3%
- Arbitration rate (Flash-Lite vs Flash disagreement): 26.7%
- Flash-Lite avg latency: 1.55s
- Flash avg latency: 3.86s
- Pro avg latency (arbitration only): 7.15s
