#!/usr/bin/env python3
"""
Script to analyze benchmark results from the JSONL log file.
"""

import json
import sys
from collections import defaultdict
from typing import Dict, List, Any

def load_results(filename: str) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def analyze_results(results: List[Dict]):
    """Analyze the benchmark results."""
    
    print("📊 Benchmark Results Analysis")
    print("=" * 60)
    
    # Group by model
    model_results = defaultdict(list)
    for result in results:
        model_results[result['model']].append(result)
    
    print(f"Total tests: {len(results)}")
    print(f"Models tested: {len(model_results)}")
    print()
    
    # Model performance summary
    print("🏆 Model Performance Summary:")
    print("-" * 40)
    
    model_stats = {}
    for model, model_data in model_results.items():
        scores = [r['score'] for r in model_data]
        avg_score = sum(scores) / len(scores)
        total_cost = sum(r['cost_usd'] for r in model_data)
        total_tokens = sum(r['tokens'] for r in model_data)
        avg_latency = sum(r['latency_seconds'] for r in model_data) / len(model_data)
        
        model_stats[model] = {
            'avg_score': avg_score,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'avg_latency': avg_latency,
            'test_count': len(model_data)
        }
        
        print(f"{model}:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Total Cost: ${total_cost:.4f}")
        print(f"  Total Tokens: {total_tokens:,}")
        print(f"  Average Latency: {avg_latency:.2f}s")
        print(f"  Tests: {len(model_data)}")
        print()
    
    # Find best performing model
    best_model = max(model_stats.items(), key=lambda x: x[1]['avg_score'])
    print(f"🥇 Best Average Score: {best_model[0]} ({best_model[1]['avg_score']:.3f})")
    
    # Cost efficiency analysis
    cost_efficient = min(model_stats.items(), key=lambda x: x[1]['total_cost'])
    print(f"💰 Most Cost Efficient: {cost_efficient[0]} (${cost_efficient[1]['total_cost']:.4f})")
    
    # Fastest model
    fastest = min(model_stats.items(), key=lambda x: x[1]['avg_latency'])
    print(f"⚡ Fastest: {fastest[0]} ({fastest[1]['avg_latency']:.2f}s avg)")
    
    print()
    
    # Category analysis
    print("📈 Performance by Category:")
    print("-" * 40)
    
    category_results = defaultdict(list)
    for result in results:
        category_results[result['category']].append(result)
    
    for category, cat_data in category_results.items():
        scores = [r['score'] for r in cat_data]
        avg_score = sum(scores) / len(scores)
        print(f"{category}: {avg_score:.3f} avg ({len(cat_data)} tests)")
    
    print()
    
    # Sample responses
    print("🔍 Sample Model Responses:")
    print("-" * 40)
    
    # Show a sample response for each model
    for model in list(model_results.keys())[:3]:  # Show first 3 models
        sample = model_results[model][0]
        print(f"\n{model} - {sample['category']}:")
        print(f"Question: {sample['input_question'][:100]}...")
        print(f"Response: {sample['model_response'][:200]}...")
        print(f"Expected: {sample['expected_answer'][:100]}...")
        print(f"Score: {sample['score']} ({sample['evaluation_choice']})")

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <log_file.jsonl>")
        print("Example: python analyze_results.py benchmark_results_20250828_104515.jsonl")
        sys.exit(1)
    
    filename = sys.argv[1]
    try:
        results = load_results(filename)
        analyze_results(results)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
