#!/usr/bin/env python3
"""
Braintrust Inference Model Benchmarking

This script benchmarks multiple inference models using Braintrust for evaluation.
"""

import os
import yaml
import json
import re
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

@dataclass
class ModelConfig:
    name: str
    provider: str
    model: str

# Model configurations
models = [
    ModelConfig("Claude-3.7-Sonnet", "Anthropic", "claude-3-7-sonnet-20250219"),
    ModelConfig("GPT-4.1", "OpenAI", "openai:gpt-4.1"),
    ModelConfig("GPT-5", "OpenAI", "openai:gpt-5"),
    ModelConfig("Gemini-2.5-Pro", "Google", "gemini-2.5-pro"),
]

# Client initialization will be done inside functions to avoid import issues

# Factuality evaluation template
template_yaml = """
prompt: |
  You are evaluating the factuality of an AI model's response. Given the following:

  Input: {{input}}
  Model Output: {{output}}
  Expected Answer: {{expected}}

  Please evaluate whether the model output is factually consistent with the expected answer.

  Choose one of the following options:
  A) The model output is a subset of the expected answer (contains less information)
  B) The model output is a superset of the expected answer (contains more information)
  C) The model output is equivalent to the expected answer (same information)
  D) The model output disagrees with the expected answer (contradictory information)

choice_scores:
  A: 0.3
  B: 0.6
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

async def evaluate_factuality(model: str, input_text: str, output_text: str, expected_text: str) -> Dict[str, Any]:
    """Evaluate factuality of model response."""
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

    try:
        # For now, let's use a simple approach without the proxy
        # We'll use the actual provider API keys directly
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
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
        elif model.startswith("claude-"):
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0,
                system=tool_prompt,
                messages=[{"role": "user", "content": user_content}]
            )
            content = response.content[0].text
        elif model.startswith("gemini-"):
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(
                f"{tool_prompt}\n\n{user_content}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=2048
                )
            )
            content = response.text
        else:
            raise ValueError(f"Unsupported model: {model}")

        parsed = parse_tool_response(content or "")
        
        return {
            "name": "Factuality",
            "score": template["choice_scores"].get(parsed["args"]["choice"], 0) if parsed else 0,
            "metadata": {
                "rationale": parsed["args"]["reasons"] if parsed else None,
                "choice": parsed["args"]["choice"] if parsed else None,
            },
        }
    except Exception as e:
        return {
            "name": "Factuality",
            "score": 0,
            "metadata": {
                "error": str(e),
            },
        }

def non_null_score(output: Optional[float]) -> Dict[str, Any]:
    """Check if output is not null."""
    return {
        "name": "NonNull",
        "score": 1 if output is not None and output != -1 else 0,
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

# Sample dataset for testing
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

async def run_benchmark():
    """Run the main benchmarking function."""
    print("Starting inference model benchmarking...")
    print("Models to test:", ", ".join([f"{m.name} ({m.provider})" for m in models]))
    
    evals = []
    
    for model_config in models:
        print(f"\nRunning evaluation for {model_config.name}...")
        
        # Create Braintrust experiment
        experiment = bt.init(
            project="Inference-Model-Benchmark",
            experiment=f"{model_config.provider} - {model_config.name}",
            metadata={
                "model": model_config.model,
                "provider": model_config.provider,
                "modelName": model_config.name,
            }
        )
        
        for data_point in sample_data:
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
                    }
                )
                
                print(f"  - Score: {result['score']}, Choice: {result['metadata'].get('choice', 'N/A')}")
                
            except Exception as e:
                print(f"  - Error: {e}")
                experiment.log(
                    input=data_point["input"],
                    output=0,
                    expected=data_point["expected"],
                    scores={"factuality": 0},
                    metadata={"error": str(e)}
                )
        
        # Don't call end() - let Braintrust handle experiment completion
        evals.append(experiment)
    
    print("\nBenchmarking completed! Check your Braintrust dashboard for results.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_benchmark())
