#!/usr/bin/env python3
"""
Simple API test to isolate the issue
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

def test_openai():
    """Test OpenAI API directly."""
    print("🧪 Testing OpenAI API...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    try:
        # Test with minimal configuration
        client = OpenAI(api_key=openai_key)
        
        # Try a simple completion
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        
        print(f"✅ OpenAI API works! Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False

def test_anthropic():
    """Test Anthropic API directly."""
    print("\n🧪 Testing Anthropic API...")
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return False
    
    try:
        # Test with minimal configuration
        client = anthropic.Anthropic(api_key=anthropic_key)
        
        # Try a simple completion
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say hello"}]
        )
        
        print(f"✅ Anthropic API works! Response: {response.content[0].text}")
        return True
        
    except Exception as e:
        print(f"❌ Anthropic API failed: {e}")
        return False

def test_google():
    """Test Google API directly."""
    print("\n🧪 Testing Google API...")
    
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("❌ GOOGLE_API_KEY not found")
        return False
    
    try:
        # Test with minimal configuration
        genai.configure(api_key=google_key)
        
        # Try a simple completion
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content("Say hello")
        
        print(f"✅ Google API works! Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Google API failed: {e}")
        return False

def main():
    """Run all API tests."""
    print("🔧 API Connection Test")
    print("=" * 50)
    
    openai_works = test_openai()
    anthropic_works = test_anthropic()
    google_works = test_google()
    
    print("\n" + "=" * 50)
    print("📊 API Test Results:")
    print(f"  OpenAI: {'✅ Working' if openai_works else '❌ Failed'}")
    print(f"  Anthropic: {'✅ Working' if anthropic_works else '❌ Failed'}")
    print(f"  Google: {'✅ Working' if google_works else '❌ Failed'}")
    
    if not any([openai_works, anthropic_works, google_works]):
        print("\n⚠️  No APIs are working. Check your API keys and internet connection.")
    else:
        print("\n✅ At least one API is working. The issue might be in the benchmark configuration.")

if __name__ == "__main__":
    main()
