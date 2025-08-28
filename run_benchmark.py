#!/usr/bin/env python3
"""
Simple runner script for Braintrust benchmarking

This script provides easy access to all benchmarking commands.
"""

import sys
import subprocess
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Command interrupted by user")
        return False

def main():
    """Main function to run benchmarks."""
    print("🎯 Braintrust Inference Model Benchmarking")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python run_benchmark.py <command>")
        print("\nAvailable commands:")
        print("  setup         - Check your setup and configuration")
        print("  basic         - Run basic benchmark")
        print("  enhanced      - Run enhanced benchmark with categories")
        print("  comparative   - Run comprehensive comparative analysis")
        print("  debug         - Run debug tests")
        print("  all           - Run setup check, then basic benchmark")
        print("\nExamples:")
        print("  python run_benchmark.py setup")
        print("  python run_benchmark.py basic")
        print("  python run_benchmark.py enhanced")
        print("  python run_benchmark.py comparative")
        return
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        success = run_command("python src/setup_check.py", "Running Setup Check")
        
    elif command == "basic":
        success = run_command("python src/benchmark.py", "Running Basic Benchmark")
        
    elif command == "enhanced":
        success = run_command("python src/enhanced_benchmark.py", "Running Enhanced Benchmark")
        
    elif command == "comparative":
        success = run_command("python src/comparative_analysis.py", "Running Comparative Analysis")
        
    elif command == "debug":
        success = run_command("python src/debug_test.py", "Running Debug Tests")
        
    elif command == "all":
        print("🔄 Running complete benchmark workflow...")
        setup_success = run_command("python src/setup_check.py", "Running Setup Check")
        
        if setup_success:
            print("\n✅ Setup check passed! Running basic benchmark...")
            benchmark_success = run_command("python src/benchmark.py", "Running Basic Benchmark")
            
            if benchmark_success:
                print("\n✅ Basic benchmark completed! Running enhanced benchmark...")
                enhanced_success = run_command("python src/enhanced_benchmark.py", "Running Enhanced Benchmark")
                
                if enhanced_success:
                    print("\n✅ Enhanced benchmark completed! Running comparative analysis...")
                    comparative_success = run_command("python src/comparative_analysis.py", "Running Comparative Analysis")
                    
                    if comparative_success:
                        print("\n🎉 All benchmarks completed successfully!")
                    else:
                        print("\n⚠️  Comparative analysis had issues, but other benchmarks completed.")
                else:
                    print("\n⚠️  Enhanced benchmark had issues, but basic benchmark completed.")
            else:
                print("\n❌ Basic benchmark failed. Check your setup.")
        else:
            print("\n❌ Setup check failed. Please fix issues before running benchmarks.")
            
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: setup, basic, enhanced, comparative, debug, all")
        return
    
    print(f"\n{'='*60}")
    print("🏁 Benchmark runner completed!")
    print("Check your Braintrust dashboard for results.")

if __name__ == "__main__":
    main()
