#!/usr/bin/env python3
"""
Test GPT-5 models specifically
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

def test_gpt5_models():
    """Test GPT-5 models with the same prompt as the benchmark."""
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    client = OpenAI(api_key=openai_key)
    
    # Same tool prompt as in benchmark
    tool_prompt = """You have access to the following functions:

Use the function 'select_choice' to 'Call this function to select a choice.':
{"name": "select_choice", "description": "Call this function to select a choice.", "parameters": {"properties": {"reasons": {"description": "Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.", "title": "Reasoning", "type": "string"}, "choice": {"description": "The choice", "title": "Choice", "type": "string", "enum": ["A", "B", "C", "D"]}}, "required": ["reasons", "choice"], "title": "CoTResponse", "type": "object"}}

If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

Here are a few examples:
<function=select_choice>{"reasons": "Step 1: Analyze the input...", "choice": "A"}</function>"""

    # Same user content as in benchmark
    user_content = """You are evaluating the factuality of an AI model's response. Given the following:

Input: What is the capital of France?
Model Output: The capital of France is Paris.
Expected Answer: Paris is the capital of France.

Please evaluate whether the model output is factually consistent with the expected answer.

Choose one of the following options:
A) The model output is a subset of the expected answer (contains less information)
B) The model output is a superset of the expected answer (contains more information)
C) The model output is equivalent to the expected answer (same information)
D) The model output disagrees with the expected answer (contradictory information)"""

    gpt5_models = ["gpt-5", "gpt-5-mini"]
    
    for model in gpt5_models:
        print(f"\n🧪 Testing {model}...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": tool_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_completion_tokens=2048
            )
            
            content = response.choices[0].message.content
            print(f"  📄 Raw response: {repr(content)}")
            print(f"  📄 Content: {content}")
            
            # Check if it's a function call
            if "<function=" in content and "</function>" in content:
                print("  ✅ Contains function call format")
            else:
                print("  ❌ No function call format found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    test_gpt5_models()
