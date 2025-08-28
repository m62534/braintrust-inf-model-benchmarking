#!/usr/bin/env python3
"""
Enhanced Braintrust Inference Model Benchmarking

This script benchmarks multiple inference models using Braintrust for evaluation,
including latency, accuracy, and comprehensive performance metrics.
"""

import os
import yaml
import json
import re
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import braintrust as bt
from openai import OpenAI
import anthropic
import google.generativeai as genai
from jinja2 import Template

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

warnings.filterwarnings("ignore", message=".*gRPC.*")
warnings.filterwarnings("ignore", message=".*absl.*")
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)

# Model enable/disable configuration
# Set these to True/False to enable/disable specific model providers
ENABLE_OPENAI_MODELS = os.getenv("ENABLE_OPENAI_MODELS", "True").lower() == "true"
ENABLE_ANTHROPIC_MODELS = os.getenv("ENABLE_ANTHROPIC_MODELS", "True").lower() == "true"
ENABLE_GOOGLE_MODELS = os.getenv("ENABLE_GOOGLE_MODELS", "True").lower() == "true"

# Individual model enable/disable configuration
# Set these to True/False to enable/disable specific models
ENABLE_GPT_5_MINI = os.getenv("ENABLE_GPT_5_MINI", "True").lower() == "true"
ENABLE_GPT_5 = os.getenv("ENABLE_GPT_5", "True").lower() == "true"
ENABLE_GPT_4_1 = os.getenv("ENABLE_GPT_4_1", "True").lower() == "true"
ENABLE_GPT_4_1_MINI = os.getenv("ENABLE_GPT_4_1_MINI", "True").lower() == "true"
ENABLE_CLAUDE_3_7_SONNET = os.getenv("ENABLE_CLAUDE_3_7_SONNET", "True").lower() == "true"
ENABLE_CLAUDE_SONNET_4 = os.getenv("ENABLE_CLAUDE_SONNET_4", "True").lower() == "true"
ENABLE_GEMINI_1_5_PRO = os.getenv("ENABLE_GEMINI_1_5_PRO", "True").lower() == "true"
ENABLE_GEMINI_2_5_PRO = os.getenv("ENABLE_GEMINI_2_5_PRO", "True").lower() == "true"

@dataclass
class ModelConfig:
    name: str
    provider: str
    model: str
    input_cost_per_1k: Optional[float] = None  # Cost per 1K input tokens
    output_cost_per_1k: Optional[float] = None  # Cost per 1K output tokens
    cost_per_1k_tokens: Optional[float] = None  # Legacy: cost per 1K tokens (used if input/output not specified)

# Enhanced model configurations with accurate pricing (input/output costs)
models = []

# Add OpenAI models if enabled
if ENABLE_OPENAI_MODELS:
    if ENABLE_GPT_5_MINI:
        models.append(ModelConfig("GPT-5-mini", "OpenAI", "openai:gpt-5-mini", 
                    input_cost_per_1k=0.01, output_cost_per_1k=0.03))
    if ENABLE_GPT_5:
        models.append(ModelConfig("GPT-5", "OpenAI", "openai:gpt-5", 
                    input_cost_per_1k=0.05, output_cost_per_1k=0.15))
    if ENABLE_GPT_4_1:
        models.append(ModelConfig("GPT-4.1", "OpenAI", "openai:gpt-4.1", 
                    input_cost_per_1k=0.03, output_cost_per_1k=0.06))
    if ENABLE_GPT_4_1_MINI:
        models.append(ModelConfig("GPT-4.1-mini", "OpenAI", "openai:gpt-4.1-mini", 
                    input_cost_per_1k=0.01, output_cost_per_1k=0.03))

# Add Anthropic models if enabled
if ENABLE_ANTHROPIC_MODELS:
    if ENABLE_CLAUDE_3_7_SONNET:
        models.append(ModelConfig("Claude-3.7-Sonnet", "Anthropic", "claude-3-7-sonnet-20250219", 
                    input_cost_per_1k=0.003, output_cost_per_1k=0.015))
    if ENABLE_CLAUDE_SONNET_4:
        models.append(ModelConfig("Claude-Sonnet-4", "Anthropic", "claude-sonnet-4-20250514", 
                    input_cost_per_1k=0.003, output_cost_per_1k=0.015))

# Add Google models if enabled
if ENABLE_GOOGLE_MODELS:
    if ENABLE_GEMINI_1_5_PRO:
        models.append(ModelConfig("Gemini-1.5-Pro", "Google", "gemini-1.5-pro", 
                    input_cost_per_1k=0.0025, output_cost_per_1k=0.0025))
    if ENABLE_GEMINI_2_5_PRO:
        models.append(ModelConfig("Gemini-2.5-Pro", "Google", "gemini-2.5-pro", 
                    input_cost_per_1k=0.0025, output_cost_per_1k=0.0025))

# Client initialization will be done inside functions to avoid import issues

async def generate_model_response(model: str, input_text: str) -> str:
    """Generate a response from the specified model for the given input."""
    start_time = time.time()
    
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
                    max_completion_tokens=2048,
                    temperature=0.1
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
            return response.choices[0].message.content
            
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
            return response.content[0].text
            
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
            
            model_instance = genai.GenerativeModel(model_id)
            response = model_instance.generate_content(
                input_text,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,
                    temperature=0.1
                )
            )
            return response.text
            
        else:
            raise ValueError(f"Unsupported model: {model}")
            
    except Exception as e:
        end_time = time.time()
        print(f"Error generating response for {model}: {e}")
        return f"Error: {str(e)}"

# Enhanced factuality evaluation template
template_yaml = """
prompt: |
  You are evaluating the factual accuracy and completeness of an AI model's response. Given the following:

  Input: {{input}}
  Model Output: {{output}}
  Expected Answer: {{expected}}

  Please evaluate the model's response based on both factual accuracy and how well it addresses the question.

  Choose one of the following options:
  A) Factually correct but very brief/incomplete (missing key details)
  B) Factually correct and reasonably complete (covers main points)
  C) Factually correct and comprehensive (detailed and thorough)
  D) Contains factual errors or contradictions

choice_scores:
  A: 0.4
  B: 0.7
  C: 1.0
  D: 0.0
"""

template = yaml.safe_load(template_yaml)

# Tool definition for structured evaluation
select_tool = {
    "name": "select_choice",
    "description": "Call this function to select a choice.",
    "parameters": {
        "properties": {
            "reasons": {
                "description": "Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.",
                "title": "Reasoning",
                "type": "string",
            },
            "choice": {
                "description": "The choice",
                "title": "Choice",
                "type": "string",
                "enum": list(template["choice_scores"].keys()),
            },
        },
        "required": ["reasons", "choice"],
        "title": "CoTResponse",
        "type": "object",
    },
}

def parse_tool_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse tool response from model output."""
    function_regex = r'<function=(\w+)>(.*?)(?:</function>|$)'
    match = re.search(function_regex, response, re.DOTALL)
    
    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return {
                "functionName": function_name,
                "args": args,
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing function arguments: {e}")
            return None
    
    return None

async def evaluate_factuality_with_metrics(model: str, input_text: str, output_text: str, expected_text: str) -> Dict[str, Any]:
    """Evaluate factuality of model response with comprehensive metrics."""
    tool_prompt = f"""You have access to the following functions:

Use the function '{select_tool["name"]}' to '{select_tool["description"]}':
{json.dumps(select_tool)}

If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

Here are a few examples:

"""

    start_time = time.time()
    
    try:
        # For now, let's use a simple approach without the proxy
        # We'll use the actual provider API keys directly
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        grok_api_key = os.getenv("GROK_API_KEY")  # Add Grok API key
        
        # Initialize OpenAI client
        openai_client = OpenAI(
            api_key=openai_api_key
        )

        # Initialize Anthropic client
        anthropic_client = anthropic.Anthropic(
            api_key=anthropic_api_key
        )

        # Configure Google AI
        genai.configure(
            api_key=google_api_key
        )

        # Render the prompt template
        prompt_template = Template(template["prompt"])
        user_content = prompt_template.render(
            input=input_text,
            output=output_text,
            expected=expected_text
        )

        # Make API call based on model provider
        if model.startswith("openai:"):
            actual_model = model.replace("openai:", "")
            
            # Use max_completion_tokens for GPT-5 models, max_tokens for others
            if "gpt-5" in actual_model.lower():
                response = openai_client.chat.completions.create(
                    model=actual_model,
                    messages=[
                        {"role": "system", "content": tool_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_completion_tokens=2048
                )
            else:
                response = openai_client.chat.completions.create(
                    model=actual_model,
                    messages=[
                        {"role": "system", "content": tool_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0,
                    max_tokens=2048
                )
            content = response.choices[0].message.content
            # Extract token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
        elif model.startswith("claude-") or model.startswith("anthropic:"):
            # Handle both claude-* and anthropic:claude-* formats
            actual_model = model.replace("anthropic:", "") if model.startswith("anthropic:") else model
            response = anthropic_client.messages.create(
                model=actual_model,
                max_tokens=2048,
                temperature=0,
                system=tool_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            content = response.content[0].text
            # Anthropic doesn't provide token usage in the same way
            input_tokens = len(user_content.split()) * 1.3  # Rough estimate
            output_tokens = len(content.split()) * 1.3
            total_tokens = input_tokens + output_tokens
            
        elif model.startswith("gemini-"):
            # Use the correct method for the current Google Generative AI library
            # Map model names to correct identifiers
            model_mapping = {
                "gemini-1.5-pro": "gemini-1.5-pro",
                "gemini-2.5-pro": "gemini-2.5-pro"
            }
            model_id = model_mapping.get(model, model)
            
            model_obj = genai.GenerativeModel(model_id)
            response = model_obj.generate_content(
                f"{tool_prompt}\n\n{user_content}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=2048
                )
            )
            content = response.text
            # Google doesn't provide token usage in the same way
            input_tokens = len(user_content.split()) * 1.3
            output_tokens = len(content.split()) * 1.3
            total_tokens = input_tokens + output_tokens
            
        else:
            raise ValueError(f"Unsupported model: {model}")

        end_time = time.time()
        latency = end_time - start_time

        parsed = parse_tool_response(content or "")
        
        return {
            "name": "Factuality",
            "score": template["choice_scores"].get(parsed["args"]["choice"], 0) if parsed else 0,
            "metadata": {
                "rationale": parsed["args"]["reasons"] if parsed else None,
                "choice": parsed["args"]["choice"] if parsed else None,
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": total_tokens / latency if latency > 0 else 0,
            },
        }
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        
        return {
            "name": "Factuality",
            "score": 0,
            "metadata": {
                "error": str(e),
                "latency_seconds": latency,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "tokens_per_second": 0,
            },
        }

def non_null_score(output: Optional[float]) -> Dict[str, Any]:
    """Check if output is not null."""
    return {
        "name": "NonNull",
        "score": 1 if output is not None and output != 0 else 0,
    }

async def correct_score(output: Optional[float], expected: Optional[float]) -> Dict[str, Any]:
    """Calculate correctness score."""
    if output is None or expected is None:
        return {
            "name": "CorrectScore",
            "score": 0,
            "metadata": {
                "error": "output is null" if output is None else "expected is null",
            },
        }
    
    # Simple numeric difference calculation
    diff = abs(float(output) - float(expected))
    score = max(0, 1 - diff)
    
    return {
        "name": "CorrectScore",
        "score": score,
        "metadata": {
            "difference": diff,
            "output": output,
            "expected": expected,
        },
    }

# Enhanced sample dataset for testing
sample_data = [
    {
        "input": {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "expected": "Paris is the capital of France."
        },
        "expected": 1.0
    },
    {
        "input": {
            "input": "What is 2 + 2?",
            "output": "2 + 2 equals 4.",
            "expected": "The answer is 4."
        },
        "expected": 1.0
    },
    {
        "input": {
            "input": "What is the weather like in Tokyo?",
            "output": "The weather in Tokyo is scorching.",
            "expected": "The weather in Tokyo is extremely hot."
        },
        "expected": 0.6
    },
    {
        "input": {
            "input": "Where is the White House located?",
            "output": "The White House is in Washington, D.C.",
            "expected": "The White House is located in Washington, D.C."
        },
        "expected": 1.0
    },
    {
        "input": {
            "input": "What is the largest planet in our solar system?",
            "output": "Jupiter is the largest planet.",
            "expected": "Jupiter is the largest planet in our solar system."
        },
        "expected": 0.6
    }
]

def print_model_configuration():
    """Print the current model configuration."""
    print("🔧 Model Configuration:")
    print(f"  OpenAI Models: {'✅ Enabled' if ENABLE_OPENAI_MODELS else '❌ Disabled'}")
    if ENABLE_OPENAI_MODELS:
        print(f"    - GPT-5-mini: {'✅' if ENABLE_GPT_5_MINI else '❌'}")
        print(f"    - GPT-5: {'✅' if ENABLE_GPT_5 else '❌'}")
        print(f"    - GPT-4.1: {'✅' if ENABLE_GPT_4_1 else '❌'}")
        print(f"    - GPT-4.1-mini: {'✅' if ENABLE_GPT_4_1_MINI else '❌'}")
    
    print(f"  Anthropic Models: {'✅ Enabled' if ENABLE_ANTHROPIC_MODELS else '❌ Disabled'}")
    if ENABLE_ANTHROPIC_MODELS:
        print(f"    - Claude-3.7-Sonnet: {'✅' if ENABLE_CLAUDE_3_7_SONNET else '❌'}")
        print(f"    - Claude-Sonnet-4: {'✅' if ENABLE_CLAUDE_SONNET_4 else '❌'}")
    
    print(f"  Google Models: {'✅ Enabled' if ENABLE_GOOGLE_MODELS else '❌ Disabled'}")
    if ENABLE_GOOGLE_MODELS:
        print(f"    - Gemini-1.5-Pro: {'✅' if ENABLE_GEMINI_1_5_PRO else '❌'}")
        print(f"    - Gemini-2.5-Pro: {'✅' if ENABLE_GEMINI_2_5_PRO else '❌'}")
    
    print(f"\n📊 Total Models Enabled: {len(models)}")
    if models:
        print("Models to test:", ", ".join([f"{m.name} ({m.provider})" for m in models]))
    else:
        print("❌ No models enabled! Please check your configuration.")
    print()

async def run_benchmark():
    """Run the main benchmarking function with enhanced metrics."""
    print("Starting enhanced inference model benchmarking...")
    print_model_configuration()
    
    if not models:
        print("❌ No models are enabled. Please set the appropriate environment variables to enable models.")
        return
    
    evals = []
    
    for model_config in models:
        print(f"\nRunning evaluation for {model_config.name}...")
        
        # Create Braintrust experiment
        experiment = bt.init(
            project="Enhanced-Inference-Model-Benchmark",
            experiment=f"{model_config.provider} - {model_config.name}",
            metadata={
                "model": model_config.model,
                "provider": model_config.provider,
                "modelName": model_config.name,
                "input_cost_per_1k": model_config.input_cost_per_1k,
                "output_cost_per_1k": model_config.output_cost_per_1k,
                "cost_per_1k_tokens": model_config.cost_per_1k_tokens,
            }
        )
        
        total_latency = 0
        total_tokens = 0
        total_cost = 0
        successful_calls = 0
        
        for data_point in sample_data:
            try:
                # Run enhanced factuality evaluation
                result = await evaluate_factuality_with_metrics(
                    model=model_config.model,
                    input_text=data_point["input"]["input"],
                    output_text=data_point["input"]["output"],
                    expected_text=data_point["input"]["expected"]
                )
                
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
                
                total_latency += latency
                total_tokens += result["metadata"].get("total_tokens", 0)
                total_cost += total_cost
                if result["score"] > 0:
                    successful_calls += 1
                
                # Ensure factuality score is within 0-1 range
                factuality_score = max(0, min(1, result["score"]))
                
                # Normalize values to 0-1 range for Braintrust scores
                # Use reasonable max values for normalization
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
                
                # Log results to Braintrust with raw values in metadata, only factuality in scores
                experiment.log(
                    input=data_point["input"],
                    output=factuality_score,
                    expected=data_point["expected"],
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
                        "expected_output": data_point["expected"],
                        "actual_output": data_point["input"]["output"],
                    }
                )
                
                print(f"  - Score: {result['score']}, Latency: {latency:.3f}s, Tokens: {int(total_tokens)}, Cost: ${total_cost:.4f}")
                
            except Exception as e:
                print(f"  - Error: {e}")
                experiment.log(
                    input=data_point["input"],
                    output=0,
                    expected=data_point["expected"],
                    scores={"factuality": 0},
                    metadata={"error": str(e)}
                )
        
        # Log summary metrics
        avg_latency = total_latency / len(sample_data) if sample_data else 0
        avg_tokens = total_tokens / len(sample_data) if sample_data else 0
        success_rate = successful_calls / len(sample_data) if sample_data else 0
        
        print(f"  📊 Summary: Avg Latency: {avg_latency:.3f}s, Avg Tokens: {int(avg_tokens)}, Success Rate: {success_rate:.1%}, Total Cost: ${total_cost:.4f}")
        
        # Don't call end() - let Braintrust handle experiment completion
        evals.append(experiment)
    
    print("\nEnhanced benchmarking completed! Check your Braintrust dashboard for detailed results.")
    print("📈 Metrics tracked: Factuality, Latency, Token Usage, Cost, Success Rate")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_benchmark())
