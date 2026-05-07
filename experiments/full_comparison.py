"""
Multi-Model Comparison Experiment
Runs FigStep attack on multiple models with paired samples

Key guarantees:
1. All models tested on EXACTLY the same samples
2. Sample order preserved for McNemar's test
3. Unified system prompts for fairness
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lmstudio_client import LMStudioClient
from src.utils.data_loader import MMSafetyBenchLoader
from src.attacks.figstep import FigStepAttack
from src.metrics.evaluator import SafetyEvaluator, McNemarTest, create_evaluator
from src.metrics.llm_judge import GeminiJudge, create_judge


def run_multi_model_comparison(
    model_keys: List[str] = None,
    num_samples: int = 50,
    use_llm_judge: bool = False,
    seed: int = 42,
    config: str = "Illegal_Activitiy",
    split: str = "TYPO",
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Multi-model comparison experiment with paired samples
    
    Args:
        model_keys: List of model keys to compare
        num_samples: Number of paired samples
        use_llm_judge: Whether to use LLM-as-a-Judge
        seed: Random seed for reproducibility
        config: Dataset config name
        split: Data split (TYPO for FigStep)
        save_results: Whether to save results
        
    Returns:
        Dictionary with metrics, mcnemar results, and raw results
    """
    if model_keys is None:
        model_keys = ["qwen3_5_9b", "gemma3_12b", "glm4_6v_flash"]
    
    print("=" * 60)
    print("Multi-Model Comparison Experiment")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Models: {model_keys}")
    print(f"  Samples: {num_samples} (paired)")
    print(f"  Seed: {seed}")
    print(f"  Config: {config}/{split}")
    print(f"  LLM Judge: {use_llm_judge}")
    
    # Step 1: Load fixed samples (shared by all models)
    print(f"\n{'='*60}")
    print("Step 1: Loading Paired Samples")
    print(f"{'='*60}")
    
    loader = MMSafetyBenchLoader(
        config=config,
        split=split,
        seed=seed,
        shuffle=False  # Critical: No shuffle for perfect pairing
    )
    samples = loader.get_tiny_dataset(num_samples)
    print(f"Loaded {len(samples)} paired samples")
    
    # Step 2: Execute attack on each model
    print(f"\n{'='*60}")
    print("Step 2: Running Attacks on All Models")
    print(f"{'='*60}")
    
    all_results = {}
    for model_key in model_keys:
        print(f"\n{'-'*60}")
        print(f"Testing model: {model_key}")
        print(f"{'-'*60}")
        
        client = LMStudioClient()
        client.load_model(model_key)
        
        attack = FigStepAttack()
        results = attack.batch_execute(client, samples)
        
        all_results[model_key] = results
        print(f"Completed: {model_key}")
    
    # Step 3: Calculate metrics
    print(f"\n{'='*60}")
    print("Step 3: Calculating Metrics")
    print(f"{'='*60}")
    
    # Initialize evaluator with optional LLM judge
    llm_judge = create_judge() if use_llm_judge else None
    evaluator = create_evaluator(
        use_llm_judge=use_llm_judge,
        llm_judge=llm_judge
    )
    
    metrics = {}
    for model_key, results in all_results.items():
        print(f"\nCalculating metrics for {model_key}...")
        metrics[model_key] = evaluator.calculate_all_metrics(results)
        
        # Print summary
        m = metrics[model_key]
        print(f"  ASR: {m['asr']:.2%}")
        print(f"  RR:  {m['rr']:.2%}")
        print(f"  SUI: {m['sui']:.3f}")
        
        if use_llm_judge:
            print(f"  LLM Judge Calls: {m.get('llm_judge_calls', 0)}")
    
    # Step 4: McNemar's Test (paired comparison)
    print(f"\n{'='*60}")
    print("Step 4: McNemar's Test (Paired Comparison)")
    print(f"{'='*60}")
    
    mcnemar_results = {}
    for i, model_a in enumerate(model_keys):
        for model_b in model_keys[i+1:]:
            key = f"{model_a}_vs_{model_b}"
            print(f"\n{model_a} vs {model_b}:")
            
            test_result = McNemarTest.test(
                all_results[model_a],
                all_results[model_b],
                evaluator
            )
            
            mcnemar_results[key] = test_result
            
            print(f"  Chi-square: {test_result['chi2']:.4f}")
            print(f"  p-value:    {test_result['p_value']:.4f}")
            print(f"  Significant: {test_result['significant']}")
            
            # Show contingency table
            table = test_result['contingency_table']
            print(f"  Contingency Table:")
            print(f"    A=Compliance, B=Not: {table['a']}")
            print(f"    A=Not, B=Compliance: {table['b']}")
            print(f"    Both Compliance:     {table['c']}")
            print(f"    Both Not:            {table['d']}")
    
    # Step 5: Save results
    if save_results:
        print(f"\n{'='*60}")
        print("Step 5: Saving Results")
        print(f"{'='*60}")
        
        results_dir = Path("results/raw_responses")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_dir = Path("results/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = metrics_dir / f"comparison_metrics_seed{seed}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'models': model_keys,
                    'num_samples': num_samples,
                    'seed': seed,
                    'dataset_config': config,
                    'split': split,
                    'use_llm_judge': use_llm_judge
                },
                'metrics': metrics,
                'mcnemar': mcnemar_results
            }, f, indent=2)
        
        print(f"Metrics saved to: {metrics_file}")
        
        # Save raw results (serialized)
        serialized_results = {}
        for model_key, results in all_results.items():
            serialized_results[model_key] = [
                serialize_result(r) for r in results
            ]
        
        raw_file = results_dir / f"comparison_raw_seed{seed}.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'models': model_keys,
                    'num_samples': num_samples,
                    'seed': seed,
                },
                'results': serialized_results
            }, f, indent=2)
        
        print(f"Raw results saved to: {raw_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print("\nSummary:")
    for model_key in model_keys:
        m = metrics[model_key]
        print(f"  {model_key}: ASR={m['asr']:.2%}, RR={m['rr']:.2%}, SUI={m['sui']:.3f}")
    
    return {
        'metrics': metrics,
        'mcnemar': mcnemar_results,
        'raw_results': all_results
    }


def serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize result for JSON storage (remove PIL Images)"""
    serialized = {}
    for key, value in result.items():
        if key == 'attack_config':
            serialized[key] = {
                k: v for k, v in value.items() 
                if k != 'image'
            }
            if 'image' in value and value['image'] is not None:
                serialized[key]['image_info'] = {
                    'size': value['image'].size,
                    'mode': value['image'].mode
                }
        else:
            serialized[key] = value
    return serialized


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Model Comparison Experiment")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["qwen3_5_9b", "gemma3_12b", "glm4_6v_flash"],
        help="Model keys to compare"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of paired samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use LLM-as-a-Judge"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Illegal_Activitiy",
        help="Dataset config"
    )
    
    args = parser.parse_args()
    
    run_multi_model_comparison(
        model_keys=args.models,
        num_samples=args.samples,
        use_llm_judge=args.llm_judge,
        seed=args.seed,
        config=args.config
    )
