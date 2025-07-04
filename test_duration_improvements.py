#!/usr/bin/env python3
"""
Test script to validate video duration improvements.

Tests:
1. TTS target duration configuration
2. Summarizer target duration settings  
3. Duration calculation accuracy
4. Production pipeline duration flow
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import config
from src.summarizer.llama_summarizer import TikTokSummaryConfig, LlamaSummarizer


def test_config_duration():
    """Test that config has correct target durations."""
    print("🎯 Testing Duration Configuration...")
    
    # Check TTS config
    tts_target = config.tts.target_duration
    print(f"   TTS target duration: {tts_target}s")
    
    if 40 <= tts_target <= 50:
        print("   ✅ TTS target duration in optimal range (40-50s)")
        return True
    else:
        print(f"   ❌ TTS target duration outside optimal range: {tts_target}s")
        return False


def test_summarizer_config():
    """Test summarizer default configuration."""
    print("\n📝 Testing Summarizer Configuration...")
    
    # Check default config
    default_config = TikTokSummaryConfig()
    print(f"   Default target duration: {default_config.target_duration}s")
    print(f"   Default max tokens: {default_config.max_tokens}")
    
    success = True
    if default_config.target_duration > 60:
        print(f"   ❌ Default target too long: {default_config.target_duration}s")
        success = False
    else:
        print(f"   ✅ Default target reasonable: {default_config.target_duration}s")
    
    if default_config.max_tokens > 2000:
        print(f"   ❌ Max tokens too high: {default_config.max_tokens}")
        success = False
    else:
        print(f"   ✅ Max tokens reasonable: {default_config.max_tokens}")
    
    return success


def test_duration_calculation():
    """Test duration calculation accuracy."""
    print("\n⏱️ Testing Duration Calculations...")
    
    # Test cases: [word_count, expected_duration_range]
    test_cases = [
        (100, (20, 35)),  # ~30s at 1.35x speed
        (150, (30, 50)),  # ~45s at 1.35x speed  
        (75, (15, 30)),   # ~22s at 1.35x speed
        (200, (40, 65)),  # ~60s at 1.35x speed
    ]
    
    tts_speed = config.get_tts_speed()
    print(f"   Using TTS speed: {tts_speed}x")
    
    success = True
    for word_count, expected_range in test_cases:
        # Calculate duration using the same formula as production pipeline
        base_duration = (word_count / 150) * 60  # seconds at normal speed
        calculated_duration = base_duration / tts_speed  # adjust for TTS speed
        
        min_expected, max_expected = expected_range
        
        print(f"   {word_count} words -> {calculated_duration:.1f}s (expected: {min_expected}-{max_expected}s)")
        
        if min_expected <= calculated_duration <= max_expected:
            print(f"     ✅ Duration calculation accurate")
        else:
            print(f"     ❌ Duration calculation off: {calculated_duration:.1f}s not in range {min_expected}-{max_expected}s")
            success = False
    
    return success


def test_word_target_calculation():
    """Test word target calculation for summarizer."""
    print("\n📊 Testing Word Target Calculation...")
    
    try:
        summarizer = LlamaSummarizer()
        
        # Test different target durations
        test_durations = [30, 45, 60, 90]
        
        success = True
        for duration in test_durations:
            word_target = summarizer._get_word_target(duration)
            print(f"   {duration}s target -> {word_target} words")
            
            # Parse the range to check reasonableness
            if "-" in word_target:
                min_words, max_words = map(int, word_target.split("-"))
                expected_words = (duration / 60) * 200  # Our target calculation
                
                if min_words <= expected_words <= max_words:
                    print(f"     ✅ Word target reasonable (expected ~{expected_words:.0f})")
                else:
                    print(f"     ❌ Word target unreasonable (expected ~{expected_words:.0f})")
                    success = False
            else:
                print(f"     ⚠️ Word target not in range format")
        
        return success
        
    except Exception as e:
        print(f"   ❌ Error testing word targets: {e}")
        return False


def test_production_config():
    """Test production configuration consistency."""
    print("\n🎬 Testing Production Configuration...")
    
    try:
        import json
        with open("production_config.json", "r") as f:
            prod_config = json.load(f)
        
        min_video = prod_config.get("min_video_duration", 0)
        max_video = prod_config.get("max_video_duration", 0)
        buffer_seconds = prod_config.get("video_buffer_seconds", 0)
        
        tts_target = config.tts.target_duration
        
        print(f"   TTS target: {tts_target}s")
        print(f"   Video range: {min_video}-{max_video}s")
        print(f"   Buffer: {buffer_seconds}s")
        
        # Check consistency
        success = True
        
        # Video range should accommodate TTS + buffer
        total_expected = tts_target + (buffer_seconds * 2)
        if min_video <= total_expected <= max_video:
            print(f"   ✅ Video range accommodates TTS + buffer ({total_expected:.1f}s)")
        else:
            print(f"   ❌ Video range doesn't fit TTS + buffer ({total_expected:.1f}s)")
            success = False
        
        # Check if ranges are reasonable for TikTok
        if 30 <= min_video <= 60 and 45 <= max_video <= 90:
            print(f"   ✅ Video duration ranges optimal for TikTok")
        else:
            print(f"   ⚠️ Video duration ranges may not be optimal for TikTok")
        
        return success
        
    except Exception as e:
        print(f"   ❌ Error checking production config: {e}")
        return False


def main():
    """Run all duration tests."""
    print("🚀 Testing Video Duration Improvements")
    print("=" * 50)
    
    tests = [
        ("Config Duration", test_config_duration),
        ("Summarizer Config", test_summarizer_config),
        ("Duration Calculation", test_duration_calculation),
        ("Word Target Calculation", test_word_target_calculation),
        ("Production Config", test_production_config),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 Duration improvements ready! Videos should now be 45-55 seconds.")
        return True
    else:
        print("⚠️ Some duration improvements need attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
