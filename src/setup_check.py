#!/usr/bin/env python3
"""
Setup check for Braintrust benchmarking project

Validates environment variables, dependencies, and project configuration.
"""

import os
import sys
import importlib
from typing import List, Dict, Any
from dotenv import load_dotenv

def check_python_version() -> Dict[str, Any]:
    """Check Python version compatibility."""
    version = sys.version_info
    is_compatible = version.major == 3 and version.minor >= 8
    
    return {
        "status": "✅" if is_compatible else "❌",
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "compatible": is_compatible,
        "message": f"Python {version.major}.{version.minor}.{version.micro} is {'compatible' if is_compatible else 'not compatible'}"
    }

def check_environment_variables() -> Dict[str, Any]:
    """Check if required environment variables are set."""
    # Load environment variables
    if os.path.exists(".env.local"):
        load_dotenv(".env.local")
    else:
        load_dotenv()
    
    required_vars = [
        "BRAINTRUST_API_KEY",
        "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY"
    ]
    
    results = {}
    all_present = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask the API key for security
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            results[var] = {
                "status": "✅",
                "present": True,
                "value": masked_value
            }
        else:
            results[var] = {
                "status": "❌",
                "present": False,
                "value": None
            }
            all_present = False
    
    return {
        "status": "✅" if all_present else "❌",
        "all_present": all_present,
        "variables": results,
        "message": f"Environment variables: {'All present' if all_present else 'Some missing'}"
    }

def check_dependencies() -> Dict[str, Any]:
    """Check if required Python packages are installed."""
    required_packages = [
        "braintrust",
        "openai", 
        "anthropic",
        "google.generativeai",
        "yaml",
        "jinja2",
        "dotenv"
    ]
    
    results = {}
    all_installed = True
    
    for package in required_packages:
        try:
            if package == "google.generativeai":
                import google.generativeai
                version = "installed"
            elif package == "yaml":
                import yaml
                version = "installed"
            else:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "installed")
            
            results[package] = {
                "status": "✅",
                "installed": True,
                "version": version
            }
        except ImportError:
            results[package] = {
                "status": "❌",
                "installed": False,
                "version": None
            }
            all_installed = False
    
    return {
        "status": "✅" if all_installed else "❌",
        "all_installed": all_installed,
        "packages": results,
        "message": f"Dependencies: {'All installed' if all_installed else 'Some missing'}"
    }

def check_source_files() -> Dict[str, Any]:
    """Check if required source files exist."""
    required_files = [
        "src/benchmark.py",
        "src/dataset.py", 
        "src/enhanced_benchmark.py",
        "src/setup_check.py",
        "requirements.txt"
    ]
    
    results = {}
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            results[file_path] = {
                "status": "✅",
                "exists": True
            }
        else:
            results[file_path] = {
                "status": "❌",
                "exists": False
            }
            all_exist = False
    
    return {
        "status": "✅" if all_exist else "❌",
        "all_exist": all_exist,
        "files": results,
        "message": f"Source files: {'All present' if all_exist else 'Some missing'}"
    }

def check_braintrust_configuration() -> Dict[str, Any]:
    """Check Braintrust configuration."""
    try:
        import braintrust as bt
        
        # Try to initialize a test experiment
        test_exp = bt.init(
            project="Setup-Test",
            experiment="Configuration-Test",
            metadata={"test": True}
        )
        test_exp.log(
            input={"test": "setup"},
            output=1.0,
            expected=1.0,
            scores={"test_score": 1.0}
        )
        # Don't call end() for setup test to avoid issues
        
        return {
            "status": "✅",
            "configured": True,
            "message": "Braintrust is properly configured and can create experiments"
        }
    except Exception as e:
        return {
            "status": "❌",
            "configured": False,
            "message": f"Braintrust configuration error: {str(e)}"
        }

def run_setup_check() -> Dict[str, Any]:
    """Run all setup checks."""
    print("🔍 Running Braintrust Benchmarking Setup Check")
    print("=" * 50)
    
    checks = {
        "python_version": check_python_version(),
        "environment_variables": check_environment_variables(),
        "dependencies": check_dependencies(),
        "source_files": check_source_files(),
        "braintrust_config": check_braintrust_configuration()
    }
    
    # Print results
    for check_name, result in checks.items():
        print(f"\n{result['status']} {result['message']}")
        
        if check_name == "environment_variables":
            for var_name, var_result in result["variables"].items():
                print(f"  {var_result['status']} {var_name}: {var_result['value'] or 'Not set'}")
        
        elif check_name == "dependencies":
            for pkg_name, pkg_result in result["packages"].items():
                print(f"  {pkg_result['status']} {pkg_name}: {pkg_result['version'] or 'Not installed'}")
        
        elif check_name == "source_files":
            for file_path, file_result in result["files"].items():
                print(f"  {file_result['status']} {file_path}")
    
    # Overall status
    all_passed = all(
        check.get("all_present", True) and check.get("all_installed", True) and 
        check.get("all_exist", True) and check.get("configured", True) and 
        check.get("compatible", True)
        for check in checks.values()
    )
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All checks passed! Your setup is ready for benchmarking.")
        print("\nNext steps:")
        print("1. Ensure your API keys are configured in Braintrust dashboard")
        print("2. Run: python src/benchmark.py")
        print("3. For detailed analysis: python src/enhanced_benchmark.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before running benchmarks.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Set up environment variables in .env.local file")
        print("3. Configure API keys in Braintrust dashboard")
    
    return {
        "all_passed": all_passed,
        "checks": checks
    }

if __name__ == "__main__":
    run_setup_check()
