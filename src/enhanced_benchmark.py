#!/usr/bin/env python3
"""
Enhanced Braintrust Inference Model Benchmarking

This script provides detailed analysis across different categories and difficulty levels.
"""

import os
import asyncio
import json
import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
import braintrust as bt
from benchmark import evaluate_factuality_with_metrics, models
from dataset import benchmark_dataset, filter_by_category, filter_by_difficulty, to_benchmark_format

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

# Suppress gRPC warnings from Google API
import warnings
import logging
import os

# Set environment variables to reduce Google API noise
os.environ["GRPC_PYTHON_LOG_LEVEL"] = "error"
os.environ["ABSL_LOGGING_MIN_LEVEL"] = "1"  # Only show ERROR and above
os.environ["GRPC_TRACE"] = "all"
os.environ["GRPC_VERBOSITY"] = "ERROR"

warnings.filterwarnings("ignore", message=".*gRPC.*")
warnings.filterwarnings("ignore", message=".*absl.*")
warnings.filterwarnings("ignore", message=".*fork.*")
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

# Redirect stderr to suppress gRPC warnings
import sys
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

# Setup logging
def setup_logging():
    """Setup logging to file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"benchmark_results_{timestamp}.jsonl"
    return log_filename

def log_result(log_filename: str, model_name: str, category: str, test_data: Dict, model_response: str, expected: str, score: float, latency: float, tokens: int, cost: float, choice: str, rationale: str):
    """Log a single test result to the JSONL file."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name,
        "category": category,
        "input_question": test_data["input"]["input"],
        "model_response": model_response,
        "expected_answer": expected,
        "score": score,
        "latency_seconds": latency,
        "tokens": tokens,
        "cost_usd": cost,
        "evaluation_choice": choice,
        "evaluation_rationale": rationale
    }
    
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

async def generate_model_response(model: str, input_text: str) -> str:
    """Generate a response from the specified model for the given input."""
    import time
    import re
    from openai import OpenAI
    import anthropic
    import google.generativeai as genai
    
    # Initialize clients
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    start_time = time.time()
    
    def clean_response(text: str) -> str:
        """Clean response text to remove invalid control characters."""
        if not text:
            return ""
        
        # Remove invalid control characters (keep newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single
        text = re.sub(r' +', ' ', text)    # Replace multiple spaces with single
        text = text.strip()                # Remove leading/trailing whitespace
        
        return text
    
    try:
        # Make API call based on model provider
        if model.startswith("openai:"):
            actual_model = model.replace("openai:", "")
            
            # Use max_completion_tokens for GPT-5 models, max_tokens for others
            if "gpt-5" in actual_model.lower():
                response = openai_client.chat.completions.create(
                    model=actual_model,
                    messages=[
                        {"role": "user", "content": input_text}
                    ],
                    max_completion_tokens=2048
                )
            else:
                response = openai_client.chat.completions.create(
                    model=actual_model,
                    messages=[
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
            return clean_response(response.choices[0].message.content)
            
        elif model.startswith("claude-") or model.startswith("anthropic:"):
            # Handle both claude-* and anthropic:claude-* formats
            actual_model = model.replace("anthropic:", "") if model.startswith("anthropic:") else model
            
            response = anthropic_client.messages.create(
                model=actual_model,
                max_tokens=2048,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": input_text}
                ]
            )
            return clean_response(response.content[0].text)
            
        elif model.startswith("gemini-"):
            # Google Gemini models
            actual_model = model.replace("gemini-", "")
            
            # Use the correct method for the current Google Generative AI library
            # Map model names to correct identifiers
            model_mapping = {
                "1.5-pro": "gemini-1.5-pro",
                "2.5-pro": "gemini-2.5-pro"
            }
            model_id = model_mapping.get(actual_model, f"gemini-{actual_model}")
            
            # Suppress gRPC warnings during Google API calls
            with SuppressStderr():
                model_instance = genai.GenerativeModel(model_id)
                response = model_instance.generate_content(
                    input_text,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=2048,
                        temperature=0.1
                    )
                )
            return clean_response(response.text)
            
        else:
            raise ValueError(f"Unsupported model: {model}")
            
    except Exception as e:
        end_time = time.time()
        print(f"Error generating response for {model}: {e}")
        return f"Error: {str(e)}"

async def run_category_benchmark(category: str, model_config: Any, log_filename: str):
    """Run benchmark for a specific category."""
    print(f"  Running {category} benchmark for {model_config.name}...")
    
    try:
        # Filter dataset by category
        category_data = filter_by_category(benchmark_dataset, category)
        if not category_data:
            print(f"    No data found for category: {category}")
            return
        
        # Convert to benchmark format
        benchmark_data = to_benchmark_format(category_data)
        print(f"    Found {len(benchmark_data)} test cases for {category}")
        
        # Create Braintrust experiment
        experiment = bt.init(
            project="Inference-Model-Benchmark",
            experiment=f"{model_config.provider} - {model_config.name} - {category}",
            metadata={
                "model": model_config.model,
                "provider": model_config.provider,
                "modelName": model_config.name,
                "input_cost_per_1k": model_config.input_cost_per_1k,
                "output_cost_per_1k": model_config.output_cost_per_1k,
                "cost_per_1k_tokens": model_config.cost_per_1k_tokens,
                "category": category,
            }
        )
        
        successful_tests = 0
        failed_tests = 0
        
        for i, data_point in enumerate(benchmark_data, 1):
            try:
                print(f"    Test {i}/{len(benchmark_data)}: Generating response for {model_config.name}...")
                
                # Generate actual response from the model being tested
                model_response = await generate_model_response(
                    model=model_config.model,
                    input_text=data_point["input"]["input"]
                )
                
                if not model_response:
                    print(f"    ⚠️  No response generated for test {i}")
                    failed_tests += 1
                    continue
                
                # Run factuality evaluation comparing model's generated response to expected answer
                # Find the closest expected answer to determine the score
                best_match_score = 0.0
                best_match_answer = ""
                
                for quality, expected_data in data_point["expected_answers"].items():
                    expected_answer = expected_data["answer"]
                    expected_score = expected_data["score"]
                    
                    # Compare model response to this expected answer
                    result = await evaluate_factuality_with_metrics(
                        model=model_config.model,
                        input_text=data_point["input"]["input"],
                        output_text=model_response,
                        expected_text=expected_answer
                    )
                    
                    # If this expected answer matches well, use its score
                    if result["score"] >= 0.7:  # Good match threshold
                        best_match_score = expected_score
                        best_match_answer = expected_answer
                        break
                    elif result["score"] > 0.0:  # Some match
                        if expected_score > best_match_score:
                            best_match_score = expected_score
                            best_match_answer = expected_answer
                
                # If no good match found, use the evaluation score directly
                if best_match_score == 0.0:
                    result = await evaluate_factuality_with_metrics(
                        model=model_config.model,
                        input_text=data_point["input"]["input"],
                        output_text=model_response,
                        expected_text=data_point["expected_answers"]["excellent"]["answer"]
                    )
                    best_match_score = result["score"]
                    best_match_answer = data_point["expected_answers"]["excellent"]["answer"]
                
                # Calculate metrics with accurate input/output token costs
                latency = result["metadata"].get("latency_seconds", 0)
                input_tokens = int(result["metadata"].get("input_tokens", 0))
                output_tokens = int(result["metadata"].get("output_tokens", 0))
                total_tokens = int(result["metadata"].get("total_tokens", 0))
                
                # Calculate costs using input/output specific pricing
                input_cost = (input_tokens / 1000) * (model_config.input_cost_per_1k or model_config.cost_per_1k_tokens or 0)
                output_cost = (output_tokens / 1000) * (model_config.output_cost_per_1k or model_config.cost_per_1k_tokens or 0)
                total_cost = input_cost + output_cost
                
                # Calculate tokens per second
                tokens_per_second = total_tokens / latency if latency > 0 else 0
                
                # Ensure factuality score is within 0-1 range
                factuality_score = max(0, min(1, best_match_score))
                
                # Normalize values to 0-1 range for Braintrust scores
                max_cost = 0.1  # $0.10 max cost
                max_tokens = 2000  # 2000 tokens max
                max_latency = 20  # 20 seconds max
                
                normalized_input_cost = min(1.0, input_cost / max_cost)
                normalized_output_cost = min(1.0, output_cost / max_cost)
                normalized_total_cost = min(1.0, total_cost / max_cost)
                normalized_input_tokens = min(1.0, input_tokens / max_tokens)
                normalized_output_tokens = min(1.0, output_tokens / max_tokens)
                normalized_total_tokens = min(1.0, total_tokens / max_tokens)
                normalized_latency = min(1.0, latency / max_latency)
                
                # Log to file
                log_result(
                    log_filename=log_filename,
                    model_name=f"{model_config.provider} - {model_config.name}",
                    category=category,
                    test_data=data_point,
                    model_response=model_response,
                    expected=best_match_answer,
                    score=factuality_score,
                    latency=latency,
                    tokens=total_tokens,
                    cost=total_cost,
                    choice=result["metadata"].get("choice", ""),
                    rationale=result["metadata"].get("rationale", "")
                )
                
                # Log results to Braintrust with comprehensive metrics
                experiment.log(
                    input=data_point["input"],
                    output=factuality_score,
                    expected=best_match_answer,
                    scores={
                        "factuality": factuality_score,
                        "input_cost": normalized_input_cost,
                        "output_cost": normalized_output_cost,
                        "total_cost": normalized_total_cost,
                        "input_tokens": normalized_input_tokens,
                        "output_tokens": normalized_output_tokens,
                        "total_tokens": normalized_total_tokens,
                        "latency": normalized_latency,
                    },
                    metadata={
                        "input_cost_usd": input_cost,
                        "output_cost_usd": output_cost,
                        "total_cost_usd": total_cost,
                        "input_tokens_count": input_tokens,
                        "output_tokens_count": output_tokens,
                        "total_tokens_count": total_tokens,
                        "latency_seconds": latency,
                        "tokens_per_second": tokens_per_second,
                        "expected_output": best_match_answer,
                        "actual_output": model_response,
                        "choice": result["metadata"].get("choice"),
                        "rationale": result["metadata"].get("rationale"),
                        "error": result["metadata"].get("error"),
                        "category": category,
                    }
                )
                
                print(f"    ✅ Test {i} completed - Score: {factuality_score:.3f}, Latency: {latency:.3f}s, Tokens: {total_tokens}, Cost: ${total_cost:.4f}")
                successful_tests += 1
                
            except Exception as e:
                print(f"    ❌ Error in test {i} for {model_config.name}: {e}")
                failed_tests += 1
                
                # Log error to Braintrust
                experiment.log(
                    input=data_point["input"],
                    output=0,
                    expected=data_point["expected_answers"]["excellent"]["answer"],
                    scores={
                        "factuality": 0,
                        "input_cost": 0,
                        "output_cost": 0,
                        "total_cost": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "latency": 0,
                    },
                    metadata={
                        "error": str(e),
                        "category": category,
                        "input_cost_usd": 0,
                        "output_cost_usd": 0,
                        "total_cost_usd": 0,
                        "input_tokens_count": 0,
                        "output_tokens_count": 0,
                        "total_tokens_count": 0,
                        "latency_seconds": 0,
                        "tokens_per_second": 0,
                    }
                )
        
        print(f"    📊 {category} benchmark completed for {model_config.name}: {successful_tests} successful, {failed_tests} failed")
        
    except Exception as e:
        print(f"    ❌ Critical error in {category} benchmark for {model_config.name}: {e}")
        import traceback
        print(f"    Traceback: {traceback.format_exc()}")
        raise  # Re-raise the exception so the main loop can handle it

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
            "input_cost_per_1k": model_config.input_cost_per_1k,
            "output_cost_per_1k": model_config.output_cost_per_1k,
            "cost_per_1k_tokens": model_config.cost_per_1k_tokens,
            "difficulty": difficulty,
        }
    )
    
    for data_point in benchmark_data:
        try:
            # Generate actual response from the model being tested
            model_response = await generate_model_response(
                model=model_config.model,
                input_text=data_point["input"]["input"]
            )
            
            # Run factuality evaluation comparing model's generated response to expected answer
            # Find the closest expected answer to determine the score
            best_match_score = 0.0
            best_match_answer = ""
            
            for quality, expected_data in data_point["expected_answers"].items():
                expected_answer = expected_data["answer"]
                expected_score = expected_data["score"]
                
                # Compare model response to this expected answer
                result = await evaluate_factuality_with_metrics(
                    model=model_config.model,
                    input_text=data_point["input"]["input"],
                    output_text=model_response,
                    expected_text=expected_answer
                )
                
                # If this expected answer matches well, use its score
                if result["score"] >= 0.7:  # Good match threshold
                    best_match_score = expected_score
                    best_match_answer = expected_answer
                    break
                elif result["score"] > 0.0:  # Some match
                    if expected_score > best_match_score:
                        best_match_score = expected_score
                        best_match_answer = expected_answer
            
            # If no good match found, use the evaluation score directly
            if best_match_score == 0.0:
                result = await evaluate_factuality_with_metrics(
                    model=model_config.model,
                    input_text=data_point["input"]["input"],
                    output_text=model_response,
                    expected_text=data_point["expected_answers"]["excellent"]["answer"]
                )
                best_match_score = result["score"]
                best_match_answer = data_point["expected_answers"]["excellent"]["answer"]
            
            # Calculate metrics with accurate input/output token costs
            latency = result["metadata"].get("latency_seconds", 0)
            input_tokens = int(result["metadata"].get("input_tokens", 0))
            output_tokens = int(result["metadata"].get("output_tokens", 0))
            total_tokens = int(result["metadata"].get("total_tokens", 0))
            
            # Calculate costs using input/output specific pricing
            input_cost = (input_tokens / 1000) * (model_config.input_cost_per_1k or model_config.cost_per_1k_tokens or 0)
            output_cost = (output_tokens / 1000) * (model_config.output_cost_per_1k or model_config.cost_per_1k_tokens or 0)
            total_cost = input_cost + output_cost
            
            # Calculate tokens per second
            tokens_per_second = total_tokens / latency if latency > 0 else 0
            
            # Ensure factuality score is within 0-1 range
            factuality_score = max(0, min(1, best_match_score))
            
            # Normalize values to 0-1 range for Braintrust scores
            max_cost = 0.1  # $0.10 max cost
            max_tokens = 2000  # 2000 tokens max
            max_latency = 20  # 20 seconds max
            
            normalized_input_cost = min(1.0, input_cost / max_cost)
            normalized_output_cost = min(1.0, output_cost / max_cost)
            normalized_total_cost = min(1.0, total_cost / max_cost)
            normalized_input_tokens = min(1.0, input_tokens / max_tokens)
            normalized_output_tokens = min(1.0, output_tokens / max_tokens)
            normalized_total_tokens = min(1.0, total_tokens / max_tokens)
            normalized_latency = min(1.0, latency / max_latency)
            
            # Log results to Braintrust with comprehensive metrics
            experiment.log(
                input=data_point["input"],
                output=factuality_score,
                expected=best_match_answer,
                scores={
                    "factuality": factuality_score,
                    "input_cost": normalized_input_cost,
                    "output_cost": normalized_output_cost,
                    "total_cost": normalized_total_cost,
                    "input_tokens": normalized_input_tokens,
                    "output_tokens": normalized_output_tokens,
                    "total_tokens": normalized_total_tokens,
                    "latency": normalized_latency,
                },
                metadata={
                    "input_cost_usd": input_cost,
                    "output_cost_usd": output_cost,
                    "total_cost_usd": total_cost,
                    "input_tokens_count": input_tokens,
                    "output_tokens_count": output_tokens,
                    "total_tokens_count": total_tokens,
                    "latency_seconds": latency,
                    "tokens_per_second": tokens_per_second,
                    "expected_output": best_match_answer,
                    "actual_output": model_response,
                    "choice": result["metadata"].get("choice"),
                    "rationale": result["metadata"].get("rationale"),
                    "error": result["metadata"].get("error"),
                    "difficulty": difficulty,
                }
            )
            
            print(f"    - Score: {factuality_score}, Latency: {latency:.3f}s, Tokens: {int(total_tokens)}, Cost: ${total_cost:.4f}")
            
        except Exception as e:
            print(f"    Error: {e}")
            experiment.log(
                input=data_point["input"],
                output=0,
                expected=data_point["expected_answers"]["excellent"]["answer"],
                scores={
                    "factuality": 0,
                    "input_cost": 0,
                    "output_cost": 0,
                    "total_cost": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency": 0,
                },
                metadata={
                    "error": str(e),
                    "difficulty": difficulty,
                    "input_cost_usd": 0,
                    "output_cost_usd": 0,
                    "total_cost_usd": 0,
                    "input_tokens_count": 0,
                    "output_tokens_count": 0,
                    "total_tokens_count": 0,
                    "latency_seconds": 0,
                    "tokens_per_second": 0,
                }
            )
    
    # Don't call end() - let Braintrust handle experiment completion
    print(f"    Completed {difficulty} difficulty benchmark with {len(benchmark_data)} tests")

async def run_enhanced_benchmark():
    """Run enhanced benchmarking with category and difficulty analysis."""
    print("Starting enhanced inference model benchmarking...")
    print("Models to test:", ", ".join([f"{m.name} ({m.provider})" for m in models]))
    
    # Setup logging
    log_filename = setup_logging()
    print(f"📝 Logging results to: {log_filename}")
    
    # Define categories and difficulties
    categories = [
        "context_understanding", "inference", "problem_context", 
        "multi_step_reasoning"
    ]
    
    difficulties = ["easy", "medium", "hard"]
    
    for model_config in models:
        print(f"\n{'='*60}")
        print(f"Running enhanced evaluation for {model_config.name} ({model_config.provider})")
        print(f"{'='*60}")
        
        try:
            # 1. Run category-specific benchmarks
            print("\n📊 Category-specific benchmarks:")
            category_tasks = []
            for category in categories:
                task = run_category_benchmark(category, model_config, log_filename)
                category_tasks.append(task)
            
            # Run category benchmarks concurrently with proper error handling
            print(f"Starting {len(category_tasks)} category benchmarks for {model_config.name}...")
            results = await asyncio.gather(*category_tasks, return_exceptions=True)
            
            # Check for any exceptions in the results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"❌ Error in category {categories[i]} for {model_config.name}: {result}")
                else:
                    print(f"✅ Category {categories[i]} completed successfully for {model_config.name}")
            
            print(f"✅ Completed all category benchmarks for {model_config.name}")
            
        except Exception as e:
            print(f"❌ Critical error processing {model_config.name}: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Skipping to next model...")
            continue
        
        # 2. Run difficulty-specific benchmarks (COMMENTED OUT)
        # print("\n📈 Difficulty-specific benchmarks:")
        # difficulty_tasks = []
        # for difficulty in difficulties:
        #     task = run_difficulty_benchmark(difficulty, model_config)
        #     difficulty_tasks.append(task)
        # 
        # # Run difficulty benchmarks concurrently
        # await asyncio.gather(*difficulty_tasks)
        
        # 3. Run comprehensive benchmark (all data) (COMMENTED OUT)
        # print("\n🎯 Comprehensive benchmark (all data):")
        # all_data = to_benchmark_format(benchmark_dataset)
        # 
        # experiment = bt.init(
        #     project="Inference-Model-Benchmark",
        #     experiment=f"{model_config.provider} - {model_config.name} - Comprehensive",
        #     metadata={
        #         "model": model_config.model,
        #         "provider": model_config.provider,
        #         "modelName": model_config.name,
        #         "input_cost_per_1k": model_config.input_cost_per_1k,
        #         "output_cost_per_1k": model_config.output_cost_per_1k,
        #         "cost_per_1k_tokens": model_config.cost_per_1k_tokens,
        #         "type": "comprehensive",
        #     }
        # )
        # 
        # for data_point in all_data:
        #     try:
        #         result = await evaluate_factuality_with_metrics(
        #             model=model_config.model,
        #             input_text=data_point["input"]["input"],
        #             output_text=data_point["input"]["output"],
        #             expected_text=data_point["input"]["expected"]
        #         )
        #         
        #         # Calculate metrics with accurate input/output token costs
        #         latency = result["metadata"].get("latency_seconds", 0)
        #         input_tokens = result["metadata"].get("input_tokens", 0)
        #         output_tokens = result["metadata"].get("output_tokens", 0)
        #         total_tokens = result["metadata"].get("total_tokens", 0)
        #         
        #         # Calculate costs using input/output specific pricing
        #         input_cost = (input_tokens / 1000) * (model_config.input_cost_per_1k or model_config.cost_per_1k_tokens or 0)
        #         output_cost = (output_tokens / 1000) * (model_config.output_cost_per_1k or model_config.cost_per_1k_tokens or 0)
        #         total_cost = input_cost + output_cost
        #         
        #         # Calculate tokens per second
        #         tokens_per_second = total_tokens / latency if latency > 0 else 0
        #         
        #         # Ensure factuality score is within 0-1 range
        #         factuality_score = max(0, min(1, result["score"]))
        #         
        #         # Normalize values to 0-1 range for Braintrust scores
        #         max_cost = 0.1  # $0.10 max cost
        #         max_tokens = 2000  # 2000 tokens max
        #         max_latency = 20  # 20 seconds max
        #         
        #         normalized_input_cost = min(1.0, input_cost / max_cost)
        #         normalized_output_cost = min(1.0, output_cost / max_cost)
        #         normalized_total_cost = min(1.0, total_cost / max_cost)
        #         normalized_input_tokens = min(1.0, input_tokens / max_tokens)
        #         normalized_output_tokens = min(1.0, output_tokens / max_tokens)
        #         normalized_total_tokens = min(1.0, total_tokens / max_tokens)
        #         normalized_latency = min(1.0, latency / max_latency)
        #         
        #         experiment.log(
        #             input=data_point["input"],
        #             output=factuality_score,
        #             expected=data_point["expected"],
        #             scores={
        #                 "factuality": factuality_score,
        #                 "input_cost": normalized_input_cost,
        #                 "output_cost": normalized_output_cost,
        #                 "total_cost": normalized_total_cost,
        #                 "input_tokens": normalized_input_tokens,
        #                 "output_tokens": normalized_output_tokens,
        #                 "total_tokens": normalized_total_tokens,
        #                 "latency": normalized_latency,
        #             },
        #             metadata={
        #                 "input_cost_usd": input_cost,
        #                 "output_cost_usd": output_cost,
        #                 "total_cost_usd": total_cost,
        #                 "input_tokens_count": input_tokens,
        #                 "output_tokens_count": output_tokens,
        #                 "total_tokens_count": total_tokens,
        #                 "latency_seconds": latency,
        #                 "tokens_per_second": tokens_per_second,
        #                 "expected_output": data_point["expected"],
        #                 "actual_output": data_point["input"]["output"],
        #                 "choice": result["metadata"].get("choice"),
        #                 "rationale": result["metadata"].get("rationale"),
        #                 "error": result["metadata"].get("error"),
        #                 "type": "comprehensive",
        #             }
        #         )
        #         
        #         print(f"    - Score: {result['score']}, Latency: {latency:.3f}s, Tokens: {total_tokens}, Cost: ${total_cost:.4f}")
        #         
        #     except Exception as e:
        #         print(f"    Error: {e}")
        #         experiment.log(
        #             input=data_point["input"],
        #             output=0,
        #             expected=data_point["expected"],
        #             scores={
        #                 "factuality": 0,
        #                 "input_cost": 0,
        #                 "output_cost": 0,
        #                 "total_cost": 0,
        #                 "input_tokens": 0,
        #                 "output_tokens": 0,
        #                 "total_tokens": 0,
        #                 "latency": 0,
        #             },
        #             metadata={
        #                 "error": str(e),
        #                 "type": "comprehensive",
        #                 "input_cost_usd": 0,
        #                 "output_cost_usd": 0,
        #                 "total_cost_usd": 0,
        #                 "input_tokens_count": 0,
        #                 "output_tokens_count": 0,
        #                 "total_tokens_count": 0,
        #                 "latency_seconds": 0,
        #                 "tokens_per_second": 0,
        #             }
        #         )
    
    print("\nEnhanced benchmarking completed! Check your Braintrust dashboard for detailed results.")
    print("📈 Metrics tracked: Factuality, Latency, Token Usage, Cost, Success Rate")

if __name__ == "__main__":
    asyncio.run(run_enhanced_benchmark())
