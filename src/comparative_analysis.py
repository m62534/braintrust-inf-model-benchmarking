#!/usr/bin/env python3
"""
Comparative Analysis for Braintrust Benchmarking

This script provides comprehensive comparative analysis of model performance,
including latency vs accuracy trade-offs, cost analysis, and user experience insights.
"""

import os
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from dotenv import load_dotenv
import braintrust as bt
from benchmark import evaluate_factuality_with_metrics, models
from dataset import benchmark_dataset, to_benchmark_format

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

class ComparativeAnalyzer:
    """Analyzes and compares model performance across multiple dimensions."""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = []
    
    async def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across all models."""
        print("🔍 Running comprehensive comparative analysis...")
        
        # Use full dataset for comprehensive analysis
        full_dataset = to_benchmark_format(benchmark_dataset)
        
        for model_config in models:
            print(f"\n📊 Testing {model_config.name} ({model_config.provider})...")
            
            model_results = {
                "model": model_config.name,
                "provider": model_config.provider,
                "cost_per_1k": model_config.cost_per_1k_tokens,
                "results": []
            }
            
            total_metrics = {
                "latency": 0,
                "tokens": 0,
                "cost": 0,
                "accuracy": 0,
                "successful_calls": 0
            }
            
            for i, data_point in enumerate(full_dataset):
                try:
                    result = await evaluate_factuality_with_metrics(
                        model=model_config.model,
                        input_text=data_point["input"]["input"],
                        output_text=data_point["input"]["output"],
                        expected_text=data_point["input"]["expected"]
                    )
                    
                    latency = result["metadata"].get("latency_seconds", 0)
                    tokens = result["metadata"].get("total_tokens", 0)
                    cost = (tokens / 1000) * (model_config.cost_per_1k_tokens or 0)
                    accuracy = result["score"]
                    
                    total_metrics["latency"] += latency
                    total_metrics["tokens"] += tokens
                    total_metrics["cost"] += cost
                    total_metrics["accuracy"] += accuracy
                    if accuracy > 0:
                        total_metrics["successful_calls"] += 1
                    
                    model_results["results"].append({
                        "test_id": i,
                        "latency": latency,
                        "tokens": tokens,
                        "cost": cost,
                        "accuracy": accuracy,
                        "error": result["metadata"].get("error")
                    })
                    
                    if i % 5 == 0:  # Progress indicator
                        print(f"    Completed {i+1}/{len(full_dataset)} tests...")
                        
                except Exception as e:
                    print(f"    Error on test {i}: {e}")
                    model_results["results"].append({
                        "test_id": i,
                        "latency": 0,
                        "tokens": 0,
                        "cost": 0,
                        "accuracy": 0,
                        "error": str(e)
                    })
            
            # Calculate averages
            num_tests = len(full_dataset)
            model_results["summary"] = {
                "avg_latency": total_metrics["latency"] / num_tests,
                "avg_tokens": total_metrics["tokens"] / num_tests,
                "total_cost": total_metrics["cost"],
                "avg_accuracy": total_metrics["accuracy"] / num_tests,
                "success_rate": total_metrics["successful_calls"] / num_tests,
                "tokens_per_second": total_metrics["tokens"] / total_metrics["latency"] if total_metrics["latency"] > 0 else 0
            }
            
            self.results[model_config.name] = model_results
            self.comparison_data.append({
                "Model": model_config.name,
                "Provider": model_config.provider,
                "Avg Latency (s)": model_results["summary"]["avg_latency"],
                "Avg Tokens": model_results["summary"]["avg_tokens"],
                "Total Cost ($)": model_results["summary"]["total_cost"],
                "Avg Accuracy": model_results["summary"]["avg_accuracy"],
                "Success Rate": model_results["summary"]["success_rate"],
                "Tokens/Second": model_results["summary"]["tokens_per_second"],
                "Cost per 1K Tokens": model_config.cost_per_1k_tokens or 0
            })
            
            print(f"  ✅ Completed: Avg Latency: {model_results['summary']['avg_latency']:.3f}s, "
                  f"Avg Accuracy: {model_results['summary']['avg_accuracy']:.2f}, "
                  f"Total Cost: ${model_results['summary']['total_cost']:.4f}")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("\n📊 Generating Comparative Analysis Report...")
        
        # Create DataFrame for analysis
        df = pd.DataFrame(self.comparison_data)
        
        # 1. Performance Rankings
        print("\n🏆 PERFORMANCE RANKINGS:")
        print("=" * 50)
        
        # Accuracy ranking
        accuracy_ranking = df.sort_values("Avg Accuracy", ascending=False)
        print("\n📈 Accuracy Ranking:")
        for i, row in accuracy_ranking.iterrows():
            print(f"  {i+1}. {row['Model']}: {row['Avg Accuracy']:.3f}")
        
        # Latency ranking (lower is better)
        latency_ranking = df.sort_values("Avg Latency (s)")
        print("\n⚡ Latency Ranking (fastest first):")
        for i, row in latency_ranking.iterrows():
            print(f"  {i+1}. {row['Model']}: {row['Avg Latency (s)']:.3f}s")
        
        # Cost efficiency ranking
        df["Cost per Accuracy"] = df["Total Cost ($)"] / df["Avg Accuracy"]
        cost_efficiency = df.sort_values("Cost per Accuracy")
        print("\n💰 Cost Efficiency Ranking (cost per accuracy point):")
        for i, row in cost_efficiency.iterrows():
            print(f"  {i+1}. {row['Model']}: ${row['Cost per Accuracy']:.4f}")
        
        # 2. Trade-off Analysis
        print("\n⚖️  LATENCY vs ACCURACY TRADE-OFFS:")
        print("=" * 50)
        
        # Find Pareto frontier
        pareto_models = []
        for _, row in df.iterrows():
            is_pareto = True
            for _, other_row in df.iterrows():
                if (other_row["Avg Latency (s)"] < row["Avg Latency (s)"] and 
                    other_row["Avg Accuracy"] >= row["Avg Accuracy"]):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_models.append(row["Model"])
        
        print(f"🎯 Pareto Optimal Models (best latency-accuracy combinations):")
        for model in pareto_models:
            model_data = df[df["Model"] == model].iloc[0]
            print(f"  • {model}: {model_data['Avg Latency (s)']:.3f}s latency, {model_data['Avg Accuracy']:.3f} accuracy")
        
        # 3. Cost Analysis
        print("\n💵 COST ANALYSIS:")
        print("=" * 50)
        
        total_cost = df["Total Cost ($)"].sum()
        print(f"Total cost across all models: ${total_cost:.4f}")
        
        cost_by_provider = df.groupby("Provider")["Total Cost ($)"].sum()
        print("\nCost by provider:")
        for provider, cost in cost_by_provider.items():
            print(f"  {provider}: ${cost:.4f}")
        
        # 4. User Experience Insights
        print("\n👥 USER EXPERIENCE INSIGHTS:")
        print("=" * 50)
        
        # Reliability analysis
        reliability_ranking = df.sort_values("Success Rate", ascending=False)
        print("\n🛡️  Reliability Ranking (success rate):")
        for i, row in reliability_ranking.iterrows():
            print(f"  {i+1}. {row['Model']}: {row['Success Rate']:.1%}")
        
        # Speed vs Quality analysis
        print("\n🚀 Speed vs Quality Analysis:")
        for _, row in df.iterrows():
            if row["Avg Latency (s)"] < 1.0 and row["Avg Accuracy"] > 0.7:
                print(f"  ⭐ {row['Model']}: Fast (<1s) AND Accurate (>70%)")
            elif row["Avg Latency (s)"] < 1.0:
                print(f"  ⚡ {row['Model']}: Fast (<1s) but lower accuracy")
            elif row["Avg Accuracy"] > 0.7:
                print(f"  🎯 {row['Model']}: Accurate (>70%) but slower")
            else:
                print(f"  ⚠️  {row['Model']}: Neither fast nor accurate")
        
        # 5. Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("=" * 50)
        
        # Best overall
        best_overall = df.loc[df["Avg Accuracy"].idxmax()]
        print(f"🎯 Best Overall Accuracy: {best_overall['Model']} ({best_overall['Avg Accuracy']:.3f})")
        
        # Best value
        best_value = df.loc[df["Cost per Accuracy"].idxmin()]
        print(f"💰 Best Value: {best_value['Model']} (${best_value['Cost per Accuracy']:.4f} per accuracy point)")
        
        # Fastest reliable
        reliable_models = df[df["Success Rate"] > 0.8]
        if not reliable_models.empty:
            fastest_reliable = reliable_models.loc[reliable_models["Avg Latency (s)"].idxmin()]
            print(f"⚡ Fastest Reliable: {fastest_reliable['Model']} ({fastest_reliable['Avg Latency (s)']:.3f}s, {fastest_reliable['Success Rate']:.1%} success)")
        
        # 6. Save detailed results
        self.save_detailed_results(df)
        
        return df
    
    def save_detailed_results(self, df: pd.DataFrame):
        """Save detailed results to files."""
        # Save comparison data
        df.to_csv("comparison_results.csv", index=False)
        print(f"\n💾 Detailed results saved to 'comparison_results.csv'")
        
        # Save raw results
        import json
        with open("raw_benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"💾 Raw results saved to 'raw_benchmark_results.json'")
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualizations for the comparison."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('LLM Performance Comparison', fontsize=16, fontweight='bold')
            
            # 1. Latency vs Accuracy scatter plot
            ax1 = axes[0, 0]
            scatter = ax1.scatter(df["Avg Latency (s)"], df["Avg Accuracy"], 
                                s=df["Total Cost ($)"]*1000, alpha=0.7, c=range(len(df)))
            ax1.set_xlabel("Average Latency (seconds)")
            ax1.set_ylabel("Average Accuracy")
            ax1.set_title("Latency vs Accuracy (bubble size = cost)")
            
            # Add model labels
            for i, row in df.iterrows():
                ax1.annotate(row["Model"], (row["Avg Latency (s)"], row["Avg Accuracy"]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 2. Cost comparison bar chart
            ax2 = axes[0, 1]
            bars = ax2.bar(df["Model"], df["Total Cost ($)"])
            ax2.set_xlabel("Model")
            ax2.set_ylabel("Total Cost ($)")
            ax2.set_title("Total Cost Comparison")
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Success rate comparison
            ax3 = axes[1, 0]
            bars = ax3.bar(df["Model"], df["Success Rate"])
            ax3.set_xlabel("Model")
            ax3.set_ylabel("Success Rate")
            ax3.set_title("Success Rate Comparison")
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Tokens per second (throughput)
            ax4 = axes[1, 1]
            bars = ax4.bar(df["Model"], df["Tokens/Second"])
            ax4.set_xlabel("Model")
            ax4.set_ylabel("Tokens per Second")
            ax4.set_title("Throughput Comparison")
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig("llm_comparison_analysis.png", dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to 'llm_comparison_analysis.png'")
            
        except ImportError:
            print("⚠️  matplotlib/seaborn not available. Skipping visualizations.")
        except Exception as e:
            print(f"⚠️  Error creating visualizations: {e}")

async def main():
    """Run the complete comparative analysis."""
    print("🎯 LLM Comparative Analysis")
    print("=" * 60)
    
    analyzer = ComparativeAnalyzer()
    
    # Run comprehensive benchmark
    await analyzer.run_comprehensive_benchmark()
    
    # Generate comparison report
    df = analyzer.generate_comparison_report()
    
    # Create visualizations
    analyzer.create_visualizations(df)
    
    print("\n" + "=" * 60)
    print("✅ Comparative analysis completed!")
    print("📊 Check the generated files for detailed insights:")
    print("  • comparison_results.csv - Tabular comparison data")
    print("  • raw_benchmark_results.json - Detailed raw results")
    print("  • llm_comparison_analysis.png - Performance visualizations")

if __name__ == "__main__":
    asyncio.run(main())
