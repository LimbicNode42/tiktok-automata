#!/usr/bin/env python3
"""
Quick test to verify the config access fix in production pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import config

def test_config_access():
    """Test that we can access TTS config properly."""
    print("🔧 Testing Config Access Fix...")
    
    try:
        # Test accessing TTS target duration (this was failing before)
        target_duration = config.tts.target_duration
        print(f"   ✅ config.tts.target_duration: {target_duration}s")
        
        # Test accessing TTS speed
        tts_speed = config.get_tts_speed()
        print(f"   ✅ config.get_tts_speed(): {tts_speed}x")
        
        # Test conversion to int (what production pipeline does)
        target_duration_int = int(config.tts.target_duration)
        print(f"   ✅ int(config.tts.target_duration): {target_duration_int}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error accessing config: {e}")
        return False

def test_production_config_import():
    """Test that production pipeline can import what it needs."""
    print("\n📦 Testing Production Pipeline Imports...")
    
    try:
        # Import the production pipeline module (this will test the imports)
        import production_pipeline
        print("   ✅ Production pipeline imports successfully")
        
        # Test that the config is accessible from the module
        if hasattr(production_pipeline, 'config'):
            target = production_pipeline.config.tts.target_duration
            print(f"   ✅ Config accessible from production pipeline: {target}s")
        else:
            print("   ⚠️ Config not directly accessible from production pipeline")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False

def main():
    """Run the config access tests."""
    print("🚀 Testing Config Access Fix")
    print("=" * 40)
    
    tests = [
        test_config_access,
        test_production_config_import,
    ]
    
    results = []
    for test_func in tests:
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Config access fix working! Production pipeline should run now.")
        return True
    else:
        print("⚠️ Some issues remain with config access")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
