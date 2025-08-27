#!/usr/bin/env python3
"""
Debug test for Braintrust benchmarking

This script helps debug API connectivity and evaluation issues.
"""

import os
import asyncio
from dotenv import load_dotenv
from benchmark import evaluate_factuality

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

async def test_evaluate_factuality():
    """Test the evaluate_factuality function with a simple case."""
    print("🧪 Testing evaluate_factuality function...")
    
    test_case = {
        "model": "openai:gpt-4o",  # Use a standard model
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris.",
        "expected": "Paris is the capital of France."
    }
    
    print("📝 Test case:", test_case)
    print("🚀 Making API call to model...")
    
    try:
        result = await evaluate_factuality(
            model=test_case["model"],
            input_text=test_case["input"],
            output_text=test_case["output"],
            expected_text=test_case["expected"]
        )
        
        print("✅ API call successful!")
        print("📄 Full result:", result)
        print("🎯 Score:", result.get("score"))
        print("🔍 Choice:", result.get("metadata", {}).get("choice"))
        print("💭 Rationale:", result.get("metadata", {}).get("rationale"))
        
        if result.get("metadata", {}).get("error"):
            print("❌ Error in result:", result["metadata"]["error"])
        
    except Exception as error:
        print("❌ Error during evaluation:", error)
        
        # Additional debugging
        print("🔍 Checking environment variables...")
        braintrust_key = os.getenv("BRAINTRUST_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if braintrust_key:
            print(f"  ✅ BRAINTRUST_API_KEY: {braintrust_key[:8]}...{braintrust_key[-4:]}")
        else:
            print("  ❌ BRAINTRUST_API_KEY: Not set")
            
        if openai_key:
            print(f"  ✅ OPENAI_API_KEY: {openai_key[:8]}...{openai_key[-4:]}")
        else:
            print("  ❌ OPENAI_API_KEY: Not set")

async def test_multiple_models():
    """Test multiple models to see which ones work."""
    print("\n🔍 Testing multiple models...")
    
    models_to_test = [
        "openai:gpt-4o",
        "openai:gpt-4o-mini", 
        "openai:gpt-4",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "gemini-2.0-flash-exp",
        "gemini-2.5-pro"
    ]
    
    test_input = {
        "input": "What is 2 + 2?",
        "output": "2 + 2 equals 4.",
        "expected": "The answer is 4."
    }
    
    for model in models_to_test:
        print(f"\n🧪 Testing model: {model}")
        try:
            result = await evaluate_factuality(
                model=model,
                input_text=test_input["input"],
                output_text=test_input["output"],
                expected_text=test_input["expected"]
            )
            
            print(f"  ✅ Success! Score: {result.get('score')}")
            print(f"  🔍 Choice: {result.get('metadata', {}).get('choice', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")

async def main():
    """Run all debug tests."""
    print("🔧 Braintrust Benchmarking Debug Test")
    print("=" * 50)
    
    # Test basic functionality
    await test_evaluate_factuality()
    
    # Test multiple models
    await test_multiple_models()
    
    print("\n" + "=" * 50)
    print("Debug test completed!")

if __name__ == "__main__":
    asyncio.run(main())
