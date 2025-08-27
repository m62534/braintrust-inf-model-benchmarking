#!/usr/bin/env python3
"""
Enhanced Braintrust Inference Model Benchmarking

This script provides detailed analysis across different categories and difficulty levels.
"""

import os
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv
import braintrust as bt
from benchmark import evaluate_factuality, models
from dataset import benchmark_dataset, filter_by_category, filter_by_difficulty, to_benchmark_format

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

async def run_category_benchmark(category: str, model_config: Any):
    """Run benchmark for a specific category."""
    print(f"  Running {category} benchmark...")
    
    # Filter dataset by category
    category_data = filter_by_category(benchmark_dataset, category)
    if not category_data:
        print(f"    No data found for category: {category}")
        return
    
    # Convert to benchmark format
    benchmark_data = to_benchmark_format(category_data)
    
    # Create Braintrust experiment
    experiment = bt.init(
        project="Inference-Model-Benchmark",
        experiment=f"{model_config.provider} - {model_config.name} - {category}",
        metadata={
            "model": model_config.model,
            "provider": model_config.provider,
            "modelName": model_config.name,
            "category": category,
        }
    )
    
    for data_point in benchmark_data:
        try:
            # Run factuality evaluation
            result = await evaluate_factuality(
                model=model_config.model,
                input_text=data_point["input"]["input"],
                output_text=data_point["input"]["output"],
                expected_text=data_point["input"]["expected"]
            )
            
            # Log results to Braintrust
            experiment.log(
                input=data_point["input"],
                output=result["score"],
                expected=data_point["expected"],
                scores={"factuality": result["score"]},
                metadata={
                    "factuality_score": result["score"],
                    "choice": result["metadata"].get("choice"),
                    "rationale": result["metadata"].get("rationale"),
                    "error": result["metadata"].get("error"),
                    "category": category,
                }
            )
            
        except Exception as e:
            print(f"    Error: {e}")
            experiment.log(
                input=data_point["input"],
                output=0,
                expected=data_point["expected"],
                scores={"factuality": 0},
                metadata={"error": str(e), "category": category}
            )
    
    # Don't call end() - let Braintrust handle experiment completion
    print(f"    Completed {category} benchmark with {len(benchmark_data)} tests")

async def run_difficulty_benchmark(difficulty: str, model_config: Any):
    """Run benchmark for a specific difficulty level."""
    print(f"  Running {difficulty} difficulty benchmark...")
    
    # Filter dataset by difficulty
    difficulty_data = filter_by_difficulty(benchmark_dataset, difficulty)
    if not difficulty_data:
        print(f"    No data found for difficulty: {difficulty}")
        return
    
    # Convert to benchmark format
    benchmark_data = to_benchmark_format(difficulty_data)
    
    # Create Braintrust experiment
    experiment = bt.init(
        project="Inference-Model-Benchmark",
        experiment=f"{model_config.provider} - {model_config.name} - {difficulty}",
        metadata={
            "model": model_config.model,
            "provider": model_config.provider,
            "modelName": model_config.name,
            "difficulty": difficulty,
        }
    )
    
    for data_point in benchmark_data:
        try:
            # Run factuality evaluation
            result = await evaluate_factuality(
                model=model_config.model,
                input_text=data_point["input"]["input"],
                output_text=data_point["input"]["output"],
                expected_text=data_point["input"]["expected"]
            )
            
            # Log results to Braintrust
            experiment.log(
                input=data_point["input"],
                output=result["score"],
                expected=data_point["expected"],
                scores={"factuality": result["score"]},
                metadata={
                    "factuality_score": result["score"],
                    "choice": result["metadata"].get("choice"),
                    "rationale": result["metadata"].get("rationale"),
                    "error": result["metadata"].get("error"),
                    "difficulty": difficulty,
                }
            )
            
        except Exception as e:
            print(f"    Error: {e}")
            experiment.log(
                input=data_point["input"],
                output=0,
                expected=data_point["expected"],
                scores={"factuality": 0},
                metadata={"error": str(e), "difficulty": difficulty}
            )
    
    # Don't call end() - let Braintrust handle experiment completion
    print(f"    Completed {difficulty} difficulty benchmark with {len(benchmark_data)} tests")

async def run_enhanced_benchmark():
    """Run enhanced benchmarking with category and difficulty analysis."""
    print("Starting enhanced inference model benchmarking...")
    print("Models to test:", ", ".join([f"{m.name} ({m.provider})" for m in models]))
    
    # Define categories and difficulties
    categories = [
        "factual_knowledge", "mathematics", "science", "history", 
        "geography", "reasoning", "creative", "programming"
    ]
    
    difficulties = ["easy", "medium", "hard"]
    
    for model_config in models:
        print(f"\n{'='*60}")
        print(f"Running enhanced evaluation for {model_config.name} ({model_config.provider})")
        print(f"{'='*60}")
        
        # 1. Run category-specific benchmarks
        print("\n📊 Category-specific benchmarks:")
        category_tasks = []
        for category in categories:
            task = run_category_benchmark(category, model_config)
            category_tasks.append(task)
        
        # Run category benchmarks concurrently
        await asyncio.gather(*category_tasks)
        
        # 2. Run difficulty-specific benchmarks
        print("\n📈 Difficulty-specific benchmarks:")
        difficulty_tasks = []
        for difficulty in difficulties:
            task = run_difficulty_benchmark(difficulty, model_config)
            difficulty_tasks.append(task)
        
        # Run difficulty benchmarks concurrently
        await asyncio.gather(*difficulty_tasks)
        
        # 3. Run comprehensive benchmark (all data)
        print("\n🎯 Comprehensive benchmark (all data):")
        all_data = to_benchmark_format(benchmark_dataset)
        
        experiment = bt.init(
            project="Inference-Model-Benchmark",
            experiment=f"{model_config.provider} - {model_config.name} - Comprehensive",
            metadata={
                "model": model_config.model,
                "provider": model_config.provider,
                "modelName": model_config.name,
                "type": "comprehensive",
            }
        )
        
        for data_point in all_data:
            try:
                result = await evaluate_factuality(
                    model=model_config.model,
                    input_text=data_point["input"]["input"],
                    output_text=data_point["input"]["output"],
                    expected_text=data_point["input"]["expected"]
                )
                
                experiment.log(
                    input=data_point["input"],
                    output=result["score"],
                    expected=data_point["expected"],
                    scores={"factuality": result["score"]},
                    metadata={
                        "factuality_score": result["score"],
                        "choice": result["metadata"].get("choice"),
                        "rationale": result["metadata"].get("rationale"),
                        "error": result["metadata"].get("error"),
                    }
                )
                
            except Exception as e:
                print(f"    Error: {e}")
                experiment.log(
                    input=data_point["input"],
                    output=0,
                    expected=data_point["expected"],
                    scores={"factuality": 0},
                    metadata={"error": str(e)}
                )
        
        # Don't call end() - let Braintrust handle experiment completion
        print(f"    Completed comprehensive benchmark with {len(all_data)} tests")
    
    print("\n" + "="*60)
    print("Enhanced benchmarking completed!")
    print("Check your Braintrust dashboard for detailed results across:")
    print("  - Categories: factual_knowledge, mathematics, science, history, geography, reasoning, creative, programming")
    print("  - Difficulties: easy, medium, hard")
    print("  - Comprehensive analysis")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(run_enhanced_benchmark())
