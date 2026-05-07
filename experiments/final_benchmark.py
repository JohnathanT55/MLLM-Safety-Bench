"""
Final benchmark runner for multimodal safety experiments.

Supports:
- multiple local LM Studio models
- multiple MM-SafetyBench scenarios
- FigStep / SceneTAP / BVS attacks
- optional CoCA defense comparison
- optional Gemini LLM judge with Pro spot-check auditing
"""

import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attacks.bvs import BVSAttack
from src.attacks.figstep import FigStepAttack
from src.attacks.scenetap import SceneTAPAttack
from src.defenses.coca import create_coca_defense
from src.metrics.evaluator import create_evaluator
from src.metrics.llm_judge import create_judge
from src.models.lmstudio_client import LMStudioClient
from src.utils.data_loader import MMSafetyBenchLoader


ATTACK_SPECS = {
    "figstep": {"split": "TYPO", "class": FigStepAttack},
    "scenetap": {"split": "SD_TYPO", "class": SceneTAPAttack},
    "bvs": {"split": "SD", "class": BVSAttack},
}


def condition_key(model_key: str, attack_name: str, config: str) -> str:
    return f"{model_key}::{attack_name}::{config}"


def serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize result dictionaries without storing PIL image objects."""
    serialized = {}
    for key, value in result.items():
        if key == "attack_config" and isinstance(value, dict):
            clean_config = {k: v for k, v in value.items() if k != "image"}
            image = value.get("image")
            if image is not None:
                clean_config["image_info"] = {
                    "size": getattr(image, "size", None),
                    "mode": getattr(image, "mode", None),
                }
            serialized[key] = clean_config
        else:
            serialized[key] = value
    return serialized


def hydrate_attack_results(
    attack_name: str,
    prompt_style: str,
    samples: List[Dict[str, Any]],
    stored_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebuild transient attack config fields needed for resumed defense runs."""
    attack = make_attack(attack_name, prompt_style)
    hydrated = []

    for index, stored in enumerate(stored_results):
        result = deepcopy(stored)
        attack_config = result.get("attack_config")
        if isinstance(attack_config, dict) and attack_config.get("image") is None and index < len(samples):
            rebuilt = attack.prepare_attack(samples[index])
            merged = dict(rebuilt)
            merged.update({k: v for k, v in attack_config.items() if k != "image_info"})
            result["attack_config"] = merged
        hydrated.append(result)

    return hydrated


def upsert_condition_result(all_results: List[Dict[str, Any]], condition_result: Dict[str, Any]) -> None:
    key = condition_key(condition_result["model_key"], condition_result["attack"], condition_result["scenario"])
    for index, existing in enumerate(all_results):
        existing_key = condition_key(existing["model_key"], existing["attack"], existing["scenario"])
        if existing_key == key:
            all_results[index] = condition_result
            return
    all_results.append(condition_result)


def make_attack(attack_name: str, prompt_style: str):
    if attack_name not in ATTACK_SPECS:
        raise ValueError(f"Unknown attack: {attack_name}")
    return ATTACK_SPECS[attack_name]["class"](prompt_style=prompt_style)


def load_samples(config: str, attack_name: str, samples: int, seed: int) -> List[Dict[str, Any]]:
    split = ATTACK_SPECS[attack_name]["split"]
    loader = MMSafetyBenchLoader(config=config, split=split, seed=seed, shuffle=False)
    return loader.get_tiny_dataset(samples)


def snapshot_judge_stats(judge) -> Dict[str, Any]:
    if judge is None:
        return {}
    stats = judge.get_stats()
    return {
        "api_calls": stats.get("api_calls", 0),
        "total_tokens": stats.get("total_tokens", 0),
        "audit_calls": stats.get("audit_calls", 0),
        "audit_overrides": stats.get("audit_overrides", 0),
        "audit_matches": stats.get("audit_matches", 0),
    }


def inject_condition_judge_deltas(metrics: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]) -> None:
    if not before and not after:
        return
    metrics["llm_judge_calls"] = after.get("api_calls", 0) - before.get("api_calls", 0)
    metrics["llm_judge_tokens"] = after.get("total_tokens", 0) - before.get("total_tokens", 0)
    metrics["llm_judge_audit_calls"] = after.get("audit_calls", 0) - before.get("audit_calls", 0)
    metrics["llm_judge_audit_overrides"] = after.get("audit_overrides", 0) - before.get("audit_overrides", 0)
    metrics["llm_judge_audit_matches"] = after.get("audit_matches", 0) - before.get("audit_matches", 0)


def run_single_condition(
    client: LMStudioClient,
    model_key: str,
    attack_name: str,
    config: str,
    samples: int,
    seed: int,
    prompt_style: str,
    evaluator,
    judge,
    run_defense: bool,
    existing_result: Optional[Dict[str, Any]] = None,
    checkpoint_callback=None,
) -> Dict[str, Any]:
    print("\n" + "-" * 70)
    print(f"Model={model_key} | Attack={attack_name} | Scenario={config} | Samples={samples}")
    print("-" * 70)

    if not client.load_model(model_key):
        raise RuntimeError(f"LM Studio target model is not loaded: {model_key}")
    scenario_samples = load_samples(config, attack_name, samples, seed)
    attack = make_attack(attack_name, prompt_style)

    output = deepcopy(existing_result) if existing_result else {}
    output.update({
        "model_key": model_key,
        "attack": attack_name,
        "scenario": config,
        "split": ATTACK_SPECS[attack_name]["split"],
        "samples": len(scenario_samples),
        "prompt_style": prompt_style,
        "status": "running",
    })

    def persist_partial() -> None:
        if checkpoint_callback is not None:
            checkpoint_callback(deepcopy(output))

    persist_partial()

    stored_attack_results = output.get("attack_results", [])
    if len(stored_attack_results) == len(scenario_samples) and "attack_metrics" in output:
        attack_results = hydrate_attack_results(attack_name, prompt_style, scenario_samples, stored_attack_results)
        print(f"  [RESUME] Reusing completed attack phase for {config}")
    else:
        before_attack_stats = snapshot_judge_stats(judge)
        attack_results = attack.batch_execute(
            client,
            scenario_samples,
            existing_results=stored_attack_results,
            progress_callback=lambda results: (
                output.__setitem__("attack_results", [serialize_result(r) for r in results]),
                persist_partial(),
            ),
        )
        attack_metrics = evaluator.calculate_all_metrics(attack_results)
        after_attack_stats = snapshot_judge_stats(judge)
        inject_condition_judge_deltas(attack_metrics, before_attack_stats, after_attack_stats)
        output["attack_metrics"] = attack_metrics
        output["attack_results"] = [serialize_result(r) for r in attack_results]
        persist_partial()

    if "attack_metrics" not in output:
        before_attack_stats = snapshot_judge_stats(judge)
        attack_metrics = evaluator.calculate_all_metrics(attack_results)
        after_attack_stats = snapshot_judge_stats(judge)
        inject_condition_judge_deltas(attack_metrics, before_attack_stats, after_attack_stats)
        output["attack_metrics"] = attack_metrics
        persist_partial()

    if run_defense:
        stored_defense_results = output.get("defense_results", [])
        if len(stored_defense_results) == len(scenario_samples) and "defense_metrics" in output:
            print(f"  [RESUME] Reusing completed CoCA phase for {config}")
        else:
            defense = create_coca_defense()
            before_defense_stats = snapshot_judge_stats(judge)
            defense_results = defense.batch_execute(
                client,
                scenario_samples,
                attack_results,
                existing_results=stored_defense_results,
                progress_callback=lambda results: (
                    output.__setitem__("defense_results", [serialize_result(r) for r in results]),
                    persist_partial(),
                ),
            )
            defense_metrics = evaluator.calculate_all_metrics(defense_results)
            after_defense_stats = snapshot_judge_stats(judge)
            inject_condition_judge_deltas(defense_metrics, before_defense_stats, after_defense_stats)
            output["defense_metrics"] = defense_metrics
            output["defense_results"] = [serialize_result(r) for r in defense_results]
            persist_partial()

    output["status"] = "completed"
    persist_partial()
    return output


def summarize_conditions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for result in results:
        if result.get("status") not in {None, "completed"}:
            continue
        if "attack_metrics" not in result:
            continue
        row = {
            "model_key": result["model_key"],
            "attack": result["attack"],
            "scenario": result["scenario"],
            "samples": result["samples"],
            "asr": result["attack_metrics"].get("asr"),
            "rr": result["attack_metrics"].get("rr"),
            "sui": result["attack_metrics"].get("sui"),
        }
        if "defense_metrics" in result:
            row.update({
                "defended_asr": result["defense_metrics"].get("asr"),
                "defended_rr": result["defense_metrics"].get("rr"),
                "defended_sui": result["defense_metrics"].get("sui"),
            })
        summary.append(row)
    return summary


def build_output(args: argparse.Namespace, all_results: List[Dict[str, Any]], judge) -> Dict[str, Any]:
    output = {
        "config": {
            "models": args.models,
            "attacks": args.attacks,
            "configs": args.configs,
            "samples_per_condition": args.samples,
            "seed": args.seed,
            "prompt_style": args.prompt_style,
            "use_llm_judge": args.llm_judge,
            "audit_rate": args.audit_rate,
            "run_defense": args.defense,
        },
        "summary": summarize_conditions(all_results),
        "conditions": all_results,
    }

    if judge is not None:
        output["judge_stats"] = judge.get_stats()

    return output


def save_output(output: Dict[str, Any], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def run_final_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    client = LMStudioClient("configs/models.yaml")
    if not client.test_connection():
        raise RuntimeError("Cannot connect to LM Studio at configured endpoint")

    judge = None
    if args.llm_judge:
        judge = create_judge(
            enable_audit=args.audit_rate > 0,
            audit_rate=args.audit_rate,
            audit_seed=args.seed,
        )

    evaluator = create_evaluator(use_llm_judge=args.llm_judge, llm_judge=judge)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_file

    all_results = []
    partial_results = {}
    completed = set()
    if args.resume and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        all_results = existing.get("conditions", [])
        partial_results = {
            condition_key(r["model_key"], r["attack"], r["scenario"]): r
            for r in all_results
        }
        completed = {
            key for key, r in partial_results.items()
            if r.get("status") in {None, "completed"}
        }
        print(f"[RESUME] Loaded {len(completed)} completed conditions from {output_file}")

    # Keep model outermost to avoid unnecessary LM Studio model switching.
    for model_key in args.models:
        for attack_name in args.attacks:
            for config in args.configs:
                key = condition_key(model_key, attack_name, config)
                if key in completed:
                    print(f"[SKIP] {key} already completed")
                    continue

                existing_result = partial_results.get(key)
                condition_result = run_single_condition(
                    client=client,
                    model_key=model_key,
                    attack_name=attack_name,
                    config=config,
                    samples=args.samples,
                    seed=args.seed,
                    prompt_style=args.prompt_style,
                    evaluator=evaluator,
                    judge=judge,
                    run_defense=args.defense,
                    existing_result=existing_result,
                    checkpoint_callback=lambda result, key=key: (
                        upsert_condition_result(all_results, result),
                        partial_results.__setitem__(key, result),
                        save_output(build_output(args, all_results, judge), output_file),
                    ),
                )
                upsert_condition_result(all_results, condition_result)
                partial_results[key] = condition_result
                completed.add(key)
                output = build_output(args, all_results, judge)
                save_output(output, output_file)

    output = build_output(args, all_results, judge)
    save_output(output, output_file)

    print("\n" + "=" * 70)
    print("Final Benchmark Summary")
    print("=" * 70)
    for row in output["summary"]:
        print(
            f"{row['model_key']} | {row['attack']} | {row['scenario']} | "
            f"ASR={row['asr']:.2%} RR={row['rr']:.2%} SUI={row['sui']:.3f}"
        )
    print(f"\nSaved to: {output_file}")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final multimodal safety benchmark runner")
    parser.add_argument("--models", nargs="+", default=["qwen3_5_9b", "gemma3_12b", "glm4_6v_flash"])
    parser.add_argument("--attacks", nargs="+", default=["figstep"], choices=list(ATTACK_SPECS.keys()))
    parser.add_argument("--configs", nargs="+", default=["Illegal_Activitiy"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-style", choices=["mild", "strong"], default="strong")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--audit-rate", type=float, default=0.1)
    parser.add_argument("--defense", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="results/final_benchmark")
    parser.add_argument("--output-file", default="final_benchmark.json")
    return parser.parse_args()


if __name__ == "__main__":
    run_final_benchmark(parse_args())
