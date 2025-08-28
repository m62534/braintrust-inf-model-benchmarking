#!/usr/bin/env python3
"""
List available Anthropic models
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

def list_anthropic_models():
    """List available Anthropic models."""
    print("🔍 Listing available Anthropic models...")
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return
    
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        
        # List models
        models = client.models.list()
        
        print("📋 Available Anthropic models:")
        for model in models.data:
            print(f"  - {model.id}")
            
        # Filter for Sonnet models
        sonnet_models = [model.id for model in models.data if "sonnet" in model.id.lower()]
        if sonnet_models:
            print(f"\n🎯 Sonnet models found: {sonnet_models}")
        else:
            print("\n⚠️  No Sonnet models found")
            
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

if __name__ == "__main__":
    list_anthropic_models()
