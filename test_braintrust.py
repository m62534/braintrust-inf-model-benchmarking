#!/usr/bin/env python3
"""
Simple Braintrust test to verify experiment creation and logging
"""

import os
from dotenv import load_dotenv
import braintrust as bt

# Load environment variables
if os.path.exists(".env.local"):
    load_dotenv(".env.local")
else:
    load_dotenv()

def test_braintrust_connection():
    """Test basic Braintrust connection and experiment creation."""
    print("🔍 Testing Braintrust connection...")
    
    braintrust_key = os.getenv("BRAINTRUST_API_KEY")
    if not braintrust_key:
        print("❌ BRAINTRUST_API_KEY not found")
        return False
    
    try:
        # Test experiment creation
        print("  🧪 Creating test experiment...")
        experiment = bt.init(
            project="Test-Project",
            experiment="Connection-Test",
            metadata={
                "test": True,
                "description": "Testing Braintrust connection"
            }
        )
        
        # Test logging
        print("  📝 Logging test data...")
        experiment.log(
            input={"test_input": "Hello world"},
            output="Test response",
            expected="Expected response",
            scores={"test_score": 1.0},
            metadata={"test": True}
        )
        
        print("  ✅ Braintrust connection successful!")
        print("  📊 Check your dashboard for project: 'Test-Project'")
        print("  📊 Experiment name: 'Connection-Test'")
        
        return True
        
    except Exception as e:
        print(f"❌ Braintrust connection failed: {e}")
        return False

def test_project_listing():
    """Test listing available projects."""
    print("\n🔍 Testing project listing...")
    
    try:
        # List projects
        projects = bt.list_projects()
        print(f"  📋 Found {len(projects)} projects:")
        for project in projects:
            print(f"    - {project.name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Project listing failed: {e}")
        return False

def main():
    """Run all Braintrust tests."""
    print("🔧 Braintrust Connection Test")
    print("=" * 50)
    
    # Test connection
    connection_works = test_braintrust_connection()
    
    # Test project listing
    listing_works = test_project_listing()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Connection: {'✅ Working' if connection_works else '❌ Failed'}")
    print(f"  Project Listing: {'✅ Working' if listing_works else '❌ Failed'}")
    
    if connection_works:
        print("\n✅ Braintrust is working! Check your dashboard for the test experiment.")
        print("   If you see the test experiment, the benchmark results should appear there too.")
    else:
        print("\n❌ Braintrust connection failed. Check your API key and network connection.")

if __name__ == "__main__":
    main()
