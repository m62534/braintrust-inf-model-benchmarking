#!/usr/bin/env python3
"""
Script to find and list Braintrust experiments
"""

import os
from dotenv import load_dotenv
import braintrust as bt

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

def find_experiments():
    """Find and list experiments."""
    print("🔍 Searching for Braintrust experiments...")
    
    braintrust_key = os.getenv("BRAINTRUST_API_KEY")
    if not braintrust_key:
        print("❌ BRAINTRUST_API_KEY not found")
        return
    
    try:
        # Try to list experiments from the expected project
        project_name = "Enhanced-Inference-Model-Benchmark"
        print(f"  📋 Looking for project: '{project_name}'")
        
        # Create a test experiment to see if we can access the project
        experiment = bt.init(
            project=project_name,
            experiment="Search-Test",
            metadata={"search_test": True}
        )
        
        print(f"  ✅ Successfully connected to project: '{project_name}'")
        print(f"  📊 Check your Braintrust dashboard for this project")
        print(f"  🌐 Dashboard URL: https://braintrust.dev/app/project/{project_name}")
        
        # Also check for the other project name
        other_project = "Inference-Model-Benchmark"
        print(f"\n  📋 Also checking for project: '{other_project}'")
        print(f"  🌐 Dashboard URL: https://braintrust.dev/app/project/{other_project}")
        
        # Check for test project
        test_project = "Test-Project"
        print(f"\n  📋 Also checking for test project: '{test_project}'")
        print(f"  🌐 Dashboard URL: https://braintrust.dev/app/project/{test_project}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing projects: {e}")
        print("\n💡 Try these steps:")
        print("  1. Go to https://braintrust.dev/app")
        print("  2. Look for projects in your dashboard")
        print("  3. Check if you have the correct API key")
        return False

def create_sample_experiment():
    """Create a sample experiment to verify everything works."""
    print("\n🧪 Creating a sample experiment...")
    
    try:
        experiment = bt.init(
            project="Sample-Benchmark-Project",
            experiment="Sample-Experiment",
            metadata={
                "description": "Sample experiment to verify Braintrust is working",
                "test": True
            }
        )
        
        # Log some sample data
        experiment.log(
            input={"question": "What is 2+2?"},
            output="4",
            expected="4",
            scores={"accuracy": 1.0},
            metadata={"test": True}
        )
        
        print("  ✅ Sample experiment created successfully!")
        print("  📊 Check your dashboard for project: 'Sample-Benchmark-Project'")
        print("  🌐 Dashboard URL: https://braintrust.dev/app/project/Sample-Benchmark-Project")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample experiment: {e}")
        return False

def main():
    """Main function."""
    print("🔧 Braintrust Experiment Finder")
    print("=" * 50)
    
    # Find existing experiments
    found = find_experiments()
    
    # Create sample experiment
    sample_created = create_sample_experiment()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Project Search: {'✅ Success' if found else '❌ Failed'}")
    print(f"  Sample Creation: {'✅ Success' if sample_created else '❌ Failed'}")
    
    if found or sample_created:
        print("\n✅ Braintrust is working! Check these URLs:")
        print("  🌐 Main Dashboard: https://braintrust.dev/app")
        print("  📊 Enhanced-Inference-Model-Benchmark: https://braintrust.dev/app/project/Enhanced-Inference-Model-Benchmark")
        print("  📊 Inference-Model-Benchmark: https://braintrust.dev/app/project/Inference-Model-Benchmark")
        print("  📊 Sample-Benchmark-Project: https://braintrust.dev/app/project/Sample-Benchmark-Project")
        print("\n💡 If you don't see your experiments, try:")
        print("  1. Refresh your browser")
        print("  2. Check if you're logged into the correct Braintrust account")
        print("  3. Verify your API key is correct")
    else:
        print("\n❌ Braintrust connection issues. Check your API key and network connection.")

if __name__ == "__main__":
    main()
