#!/usr/bin/env python3
"""
Test Runner - Run all tests in organized sequence
"""

import sys
import os
import subprocess
from pathlib import Path

def run_test(script_path, description):
    """Run a test script and report results"""
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING: {description}")
    print(f"📁 Script: {script_path}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              cwd=Path(__file__).parent,
                              capture_output=False)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run all tests in sequence"""
    print("🚀 TikTok Automata - Test Suite Runner")
    print("="*60)
    
    tests = [
        ("tests/test_cuda_setup.py", "CUDA Setup Validation"),
        ("src/scraper/tests/test_scraper_comprehensive.py", "Newsletter Scraper Tests"),
        ("src/summarizer/tests/test_llama_summarizer.py", "Llama Summarizer Tests"),
        ("tests/test_full_pipeline.py", "Full Pipeline Integration"),
    ]
    
    results = []
    
    for script_path, description in tests:
        if os.path.exists(script_path):
            success = run_test(script_path, description)
            results.append((description, success))
        else:
            print(f"⚠️  {description} - SKIPPED (file not found: {script_path})")
            results.append((description, None))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for description, result in results:
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        
        print(f"{status:<12} {description}")
    
    print(f"\n📈 SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print(f"\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print(f"\n🎉 All tests passed successfully!")

if __name__ == "__main__":
    main()
