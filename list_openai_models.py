#!/usr/bin/env python3
"""
List available OpenAI models to find GPT-5
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

def list_openai_models():
    """List all available OpenAI models."""
    print("🔍 Listing available OpenAI models...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        return
    
    try:
        client = OpenAI(api_key=openai_key)
        
        # List models
        models = client.models.list()
        
        print(f"📋 Found {len(models.data)} OpenAI models:")
        
        # Filter for GPT-5 models
        gpt5_models = []
        gpt4_models = []
        other_models = []
        
        for model in models.data:
            model_id = model.id
            if "gpt-5" in model_id.lower():
                gpt5_models.append(model_id)
            elif "gpt-4" in model_id.lower():
                gpt4_models.append(model_id)
            else:
                other_models.append(model_id)
        
        # Display GPT-5 models first
        if gpt5_models:
            print(f"\n🎯 GPT-5 models found ({len(gpt5_models)}):")
            for model in gpt5_models:
                print(f"  ✅ {model}")
        else:
            print("\n❌ No GPT-5 models found")
        
        # Display GPT-4 models
        if gpt4_models:
            print(f"\n📊 GPT-4 models found ({len(gpt4_models)}):")
            for model in gpt4_models:
                print(f"  📋 {model}")
        
        # Display other models (limit to first 10)
        if other_models:
            print(f"\n🔧 Other models found ({len(other_models)}):")
            for model in other_models[:10]:  # Limit to first 10
                print(f"  📋 {model}")
            if len(other_models) > 10:
                print(f"  ... and {len(other_models) - 10} more")
        
        return gpt5_models
        
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return []

def test_gpt5_models(gpt5_models):
    """Test GPT-5 models to see which ones work."""
    if not gpt5_models:
        print("\n❌ No GPT-5 models to test")
        return
    
    print(f"\n🧪 Testing {len(gpt5_models)} GPT-5 models...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_key)
    
    working_models = []
    
    for model in gpt5_models:
        print(f"\n  🧪 Testing {model}...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hello"}],
                max_completion_tokens=10
            )
            content = response.choices[0].message.content
            print(f"    ✅ Success! Response: {content}")
            working_models.append(model)
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
    
    return working_models

def main():
    """Main function."""
    print("🔧 OpenAI Model Finder")
    print("=" * 50)
    
    # List all models
    gpt5_models = list_openai_models()
    
    # Test GPT-5 models
    working_models = test_gpt5_models(gpt5_models)
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Total GPT-5 models found: {len(gpt5_models)}")
    print(f"  Working GPT-5 models: {len(working_models)}")
    
    if working_models:
        print(f"\n✅ Working GPT-5 models:")
        for model in working_models:
            print(f"  🎯 {model}")
            print(f"     Use in benchmark: openai:{model}")
    else:
        print("\n❌ No working GPT-5 models found")
        print("💡 This might be because:")
        print("   - GPT-5 is not available in your region")
        print("   - You need special access to GPT-5")
        print("   - The model name format has changed")

if __name__ == "__main__":
    main()
