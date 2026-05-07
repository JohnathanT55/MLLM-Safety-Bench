"""
Gemini 3.1 Judge Comparison Experiment
Compares Flash-Lite vs Flash with Pro arbitration

Workflow:
1. Load existing experiment data (150 samples)
2. Dual judgment: Flash-Lite + Flash
3. Arbitration by Pro if disagreement
4. Calculate accuracy, efficiency, agreement metrics
5. Analyze by scenario
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.llm_judge import GeminiJudge


class DualJudgeComparison:
    """
    Dual judge comparison with arbitration
    
    Judges: Flash-Lite, Flash
    Arbitrator: Pro
    """
    
    LITE_MODEL = "gemini-3.1-flash-lite-preview"
    FLASH_MODEL = "gemini-3-flash-preview"
    PRO_MODEL = "gemini-3.1-pro-preview"
    
    def __init__(self):
        print("Initializing judges...")
        self.lite = GeminiJudge(self.LITE_MODEL)
        print(f"  Lite: {self.LITE_MODEL}")
        
        self.flash = GeminiJudge(self.FLASH_MODEL)
        print(f"  Flash: {self.FLASH_MODEL}")
        
        self.pro = GeminiJudge(self.PRO_MODEL)
        print(f"  Pro: {self.PRO_MODEL} (arbitrator)")
        
        self.results: List[Dict[str, Any]] = []
        
        self.arbitration_count = 0
        self.lite_matches_pro = 0
        self.flash_matches_pro = 0
        self.agreement_count = 0
    
    def load_existing_data(self, data_file: str = "results/raw_responses/comparison_raw_seed42.json") -> List[Dict[str, Any]]:
        """Load existing experiment data"""
        print(f"\nLoading data from: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        models = data['config']['models']
        
        for model_key in models:
            model_results = data['results'][model_key]
            for i, result in enumerate(model_results):
                sample = {
                    'sample_id': i,
                    'model_key': model_key,
                    'question': result['attack_config']['original_question'],
                    'response': result['content'],
                    'scenario': result['attack_config']['scenario']
                }
                samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from {len(models)} models")
        return samples
    
    def judge_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Judge a single sample with dual judges
        
        Returns judgment with arbitration info
        """
        question = sample['question']
        response = sample['response']
        scenario = sample['scenario']
        
        lite_result = self.lite.judge_response(question, response, scenario)
        flash_result = self.flash.judge_response(question, response, scenario)
        
        lite_cat = lite_result.get('category', '[Irrelevant]')
        flash_cat = flash_result.get('category', '[Irrelevant]')
        
        need_arbitration = lite_cat != flash_cat
        
        if need_arbitration:
            self.arbitration_count += 1
            
            pro_result = self.pro.judge_response(question, response, scenario)
            pro_cat = pro_result.get('category', '[Irrelevant]')
            final_cat = pro_cat
            
            if lite_cat == pro_cat:
                self.lite_matches_pro += 1
            if flash_cat == pro_cat:
                self.flash_matches_pro += 1
        else:
            self.agreement_count += 1
            pro_result = None
            pro_cat = None
            final_cat = lite_cat
            
            self.lite_matches_pro += 1
            self.flash_matches_pro += 1
        
        return {
            'sample_id': sample['sample_id'],
            'model_key': sample['model_key'],
            'scenario': scenario,
            'lite': {
                'category': lite_cat,
                'latency': lite_result.get('latency', 0),
                'tokens': lite_result.get('tokens', 0),
                'confidence': lite_result.get('confidence', 0)
            },
            'flash': {
                'category': flash_cat,
                'latency': flash_result.get('latency', 0),
                'tokens': flash_result.get('tokens', 0),
                'confidence': flash_result.get('confidence', 0)
            },
            'arbitration_needed': need_arbitration,
            'final_category': final_cat,
            'pro': {
                'category': pro_cat,
                'latency': pro_result.get('latency', 0) if pro_result else 0,
                'tokens': pro_result.get('tokens', 0) if pro_result else 0,
                'confidence': pro_result.get('confidence', 0) if pro_result else 0
            } if need_arbitration else None
        }
    
    def run_comparison(self, samples: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Run comparison on all samples"""
        total = len(samples)
        print(f"\nRunning dual judgment on {total} samples...")
        
        results = []
        start_time = time.time()
        
        for i, sample in enumerate(samples):
            if show_progress:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{total}] Arbitration: {self.arbitration_count} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r', flush=True)
            
            result = self.judge_sample(sample)
            results.append(result)
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"\n  Completed in {total_time:.1f}s")
        
        self.results = results
        return results
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate overall metrics"""
        total = len(self.results)
        
        lite_accuracy = self.lite_matches_pro / total if total > 0 else 0
        flash_accuracy = self.flash_matches_pro / total if total > 0 else 0
        agreement_rate = self.agreement_count / total if total > 0 else 0
        arbitration_rate = self.arbitration_count / total if total > 0 else 0
        
        lite_stats = self.lite.get_stats()
        flash_stats = self.flash.get_stats()
        pro_stats = self.pro.get_stats()
        
        category_dist = defaultdict(int)
        for r in self.results:
            category_dist[r['final_category']] += 1
        
        return {
            'total_samples': total,
            'lite_accuracy': lite_accuracy,
            'flash_accuracy': flash_accuracy,
            'agreement_rate': agreement_rate,
            'arbitration_rate': arbitration_rate,
            'arbitration_count': self.arbitration_count,
            'lite_stats': lite_stats,
            'flash_stats': flash_stats,
            'pro_stats': pro_stats,
            'category_distribution': dict(category_dist)
        }
    
    def analyze_by_scenario(self) -> Dict[str, Dict[str, Any]]:
        """Analyze metrics by scenario"""
        scenario_data = defaultdict(list)
        
        for r in self.results:
            scenario_data[r['scenario']].append(r)
        
        scenario_metrics = {}
        
        for scenario, results in scenario_data.items():
            total = len(results)
            
            lite_match = sum(1 for r in results if not r['arbitration_needed'] or r['lite']['category'] == r['final_category'])
            flash_match = sum(1 for r in results if not r['arbitration_needed'] or r['flash']['category'] == r['final_category'])
            
            arbitration = sum(1 for r in results if r['arbitration_needed'])
            agreement = total - arbitration
            
            lite_latencies = [r['lite']['latency'] for r in results]
            flash_latencies = [r['flash']['latency'] for r in results]
            
            category_dist = defaultdict(int)
            for r in results:
                category_dist[r['final_category']] += 1
            
            scenario_metrics[scenario] = {
                'total': total,
                'lite_accuracy': lite_match / total,
                'flash_accuracy': flash_match / total,
                'agreement_count': agreement,
                'arbitration_count': arbitration,
                'lite_avg_latency': sum(lite_latencies) / len(lite_latencies) if lite_latencies else 0,
                'flash_avg_latency': sum(flash_latencies) / len(flash_latencies) if flash_latencies else 0,
                'category_distribution': dict(category_dist)
            }
        
        return scenario_metrics
    
    def get_arbitration_details(self) -> List[Dict[str, Any]]:
        """Get details of cases requiring arbitration"""
        details = []
        
        for r in self.results:
            if r['arbitration_needed']:
                winner = 'lite' if r['lite']['category'] == r['final_category'] else 'flash' if r['flash']['category'] == r['final_category'] else 'neither'
                details.append({
                    'sample_id': r['sample_id'],
                    'model_key': r['model_key'],
                    'scenario': r['scenario'],
                    'lite_category': r['lite']['category'],
                    'flash_category': r['flash']['category'],
                    'pro_category': r['final_category'],
                    'winner': winner
                })
        
        return details
    
    def save_results(self, output_dir: str = "results/judge_comparison"):
        """Save all results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = self.calculate_metrics()
        scenario_metrics = self.analyze_by_scenario()
        arbitration_details = self.get_arbitration_details()
        
        output = {
            'config': {
                'lite_model': self.LITE_MODEL,
                'flash_model': self.FLASH_MODEL,
                'pro_model': self.PRO_MODEL,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'overall_metrics': metrics,
            'scenario_analysis': scenario_metrics,
            'arbitration_details': arbitration_details
        }
        
        output_file = output_path / f"judge_comparison_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print summary report"""
        metrics = self.calculate_metrics()
        scenario_metrics = self.analyze_by_scenario()
        
        print("\n" + "=" * 70)
        print("Gemini 3.1 Judge Comparison Results")
        print("=" * 70)
        
        print(f"\nTotal Samples: {metrics['total_samples']}")
        print(f"Arbitration Needed: {metrics['arbitration_count']} ({metrics['arbitration_rate']:.1%})")
        
        print("\nOverall Accuracy:")
        print(f"  Flash-Lite: {metrics['lite_accuracy']:.1%}")
        print(f"  Flash:      {metrics['flash_accuracy']:.1%}")
        print(f"  Agreement:  {metrics['agreement_rate']:.1%}")
        
        print("\nLatency Comparison:")
        print(f"  Flash-Lite: {metrics['lite_stats']['avg_latency']:.2f}s avg")
        print(f"  Flash:      {metrics['flash_stats']['avg_latency']:.2f}s avg")
        print(f"  Pro (arb):  {metrics['pro_stats']['avg_latency']:.2f}s avg")
        
        print("\nToken Usage:")
        print(f"  Flash-Lite: {metrics['lite_stats']['avg_tokens']:.0f} avg")
        print(f"  Flash:      {metrics['flash_stats']['avg_tokens']:.0f} avg")
        print(f"  Pro (arb):  {metrics['pro_stats']['avg_tokens']:.0f} avg")
        
        print("\nCategory Distribution:")
        for cat, count in sorted(metrics['category_distribution'].items()):
            pct = count / metrics['total_samples'] * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")
        
        print("\nScenario Analysis:")
        for scenario, sm in sorted(scenario_metrics.items()):
            print(f"\n  {scenario} ({sm['total']} samples):")
            print(f"    Lite:    {sm['lite_accuracy']:.1%}")
            print(f"    Flash:   {sm['flash_accuracy']:.1%}")
            print(f"    Arb:     {sm['arbitration_count']}")
            print(f"    Latency: Lite={sm['lite_avg_latency']:.2f}s, Flash={sm['flash_avg_latency']:.2f}s")


def run_experiment(data_file: str = None, save: bool = True):
    """Run the full comparison experiment"""
    comparison = DualJudgeComparison()
    
    if data_file is None:
        data_file = "results/raw_responses/comparison_raw_seed42.json"
    
    samples = comparison.load_existing_data(data_file)
    
    results = comparison.run_comparison(samples)
    
    comparison.print_summary()
    
    if save:
        comparison.save_results()
    
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemini 3.1 Judge Comparison")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to existing experiment data JSON"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to JSON"
    )
    
    args = parser.parse_args()
    
    run_experiment(args.data, args.save)