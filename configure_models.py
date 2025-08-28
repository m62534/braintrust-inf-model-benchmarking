#!/usr/bin/env python3
"""
Model Configuration Script

This script helps you configure which models to enable/disable for benchmarking.
"""

import os
from pathlib import Path

def create_env_file():
    """Create or update .env.local file with model configuration."""
    
    # Check if .env.local exists
    env_file = Path(".env.local")
    if env_file.exists():
        print("📁 Found existing .env.local file")
        # Read existing content
        with open(env_file, 'r') as f:
            existing_content = f.read()
    else:
        print("📁 Creating new .env.local file")
        existing_content = ""
    
    # Parse existing variables
    existing_vars = {}
    for line in existing_content.split('\n'):
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            existing_vars[key.strip()] = value.strip()
    
    print("\n🔧 Model Configuration")
    print("=" * 50)
    
    # Provider-level configuration
    print("\n📊 Provider Groups:")
    print("Enable/disable entire provider groups:")
    
    enable_openai = input("Enable OpenAI models? (y/n, default: y): ").lower().strip()
    enable_openai = "True" if enable_openai != "n" else "False"
    
    enable_anthropic = input("Enable Anthropic models? (y/n, default: y): ").lower().strip()
    enable_anthropic = "True" if enable_anthropic != "n" else "False"
    
    enable_google = input("Enable Google models? (y/n, default: y): ").lower().strip()
    enable_google = "True" if enable_google != "n" else "False"
    
    # Individual model configuration
    print("\n🤖 Individual Models:")
    print("Enable/disable specific models:")
    
    model_config = {}
    
    if enable_openai == "True":
        print("\nOpenAI Models:")
        model_config['ENABLE_GPT_5_MINI'] = "True" if input("  Enable GPT-5-mini? (y/n, default: y): ").lower().strip() != "n" else "False"
        model_config['ENABLE_GPT_5'] = "True" if input("  Enable GPT-5? (y/n, default: y): ").lower().strip() != "n" else "False"
        model_config['ENABLE_GPT_4_1'] = "True" if input("  Enable GPT-4.1? (y/n, default: y): ").lower().strip() != "n" else "False"
        model_config['ENABLE_GPT_4_1_MINI'] = "True" if input("  Enable GPT-4.1-mini? (y/n, default: y): ").lower().strip() != "n" else "False"
    
    if enable_anthropic == "True":
        print("\nAnthropic Models:")
        model_config['ENABLE_CLAUDE_3_7_SONNET'] = "True" if input("  Enable Claude-3.7-Sonnet? (y/n, default: y): ").lower().strip() != "n" else "False"
        model_config['ENABLE_CLAUDE_SONNET_4'] = "True" if input("  Enable Claude-Sonnet-4? (y/n, default: y): ").lower().strip() != "n" else "False"
    
    if enable_google == "True":
        print("\nGoogle Models:")
        model_config['ENABLE_GEMINI_1_5_PRO'] = "True" if input("  Enable Gemini-1.5-Pro? (y/n, default: y): ").lower().strip() != "n" else "False"
        model_config['ENABLE_GEMINI_2_5_PRO'] = "True" if input("  Enable Gemini-2.5-Pro? (y/n, default: y): ").lower().strip() != "n" else "False"
    
    # Build new content
    new_content = []
    
    # Add API keys section
    new_content.append("# API Keys")
    new_content.append("OPENAI_API_KEY=" + existing_vars.get('OPENAI_API_KEY', 'your_openai_api_key_here'))
    new_content.append("ANTHROPIC_API_KEY=" + existing_vars.get('ANTHROPIC_API_KEY', 'your_anthropic_api_key_here'))
    new_content.append("GOOGLE_API_KEY=" + existing_vars.get('GOOGLE_API_KEY', 'your_google_api_key_here'))
    new_content.append("")
    
    # Add Braintrust configuration
    new_content.append("# Braintrust Configuration")
    new_content.append("BRAINTRUST_API_KEY=" + existing_vars.get('BRAINTRUST_API_KEY', 'your_braintrust_api_key_here'))
    new_content.append("")
    
    # Add model configuration
    new_content.append("# Model Enable/Disable Configuration")
    new_content.append("# Set to \"True\" or \"False\" to enable/disable entire provider groups")
    new_content.append(f"ENABLE_OPENAI_MODELS={enable_openai}")
    new_content.append(f"ENABLE_ANTHROPIC_MODELS={enable_anthropic}")
    new_content.append(f"ENABLE_GOOGLE_MODELS={enable_google}")
    new_content.append("")
    
    # Add individual model configuration
    new_content.append("# Individual Model Configuration")
    new_content.append("# Set to \"True\" or \"False\" to enable/disable specific models")
    for key, value in model_config.items():
        new_content.append(f"{key}={value}")
    
    # Write to file
    with open(env_file, 'w') as f:
        f.write('\n'.join(new_content))
    
    print(f"\n✅ Configuration saved to {env_file}")
    print("\n📋 Summary:")
    print(f"  OpenAI Models: {'✅ Enabled' if enable_openai == 'True' else '❌ Disabled'}")
    print(f"  Anthropic Models: {'✅ Enabled' if enable_anthropic == 'True' else '❌ Disabled'}")
    print(f"  Google Models: {'✅ Enabled' if enable_google == 'True' else '❌ Disabled'}")
    
    # Count enabled models
    enabled_count = sum(1 for value in model_config.values() if value == "True")
    print(f"\n📊 Total Models Enabled: {enabled_count}")
    
    print("\n💡 To run the benchmark:")
    print("  source venv/bin/activate && python run_benchmark.py enhanced")

if __name__ == "__main__":
    create_env_file()
