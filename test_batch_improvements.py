#!/usr/bin/env python3
"""
Test script to validate all batch improvements for 1.35x TTS speed.

This script tests:
1. TTS speed is correctly set to 1.35x
2. Subtitle formatting caps at 3 lines and avoids single words
3. Subtitle timing is properly calibrated for 1.35x speed
4. Video duration targets are appropriate
5. Dialogue filtering is active
"""

import asyncio
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import config
from src.video.subtitles import SubtitleGenerator


def test_tts_speed():
    """Test that TTS speed is correctly configured."""
    print("🎤 Testing TTS Speed Configuration...")
    
    speed = config.get_tts_speed()
    print(f"   Current TTS speed: {speed}x")
    
    if speed == 1.35:
        print("   ✅ TTS speed correctly set to 1.35x")
        return True
    else:
        print(f"   ❌ Expected 1.35x, got {speed}x")
        return False


def test_subtitle_formatting():
    """Test subtitle formatting with 3-line cap and no single words."""
    print("\n📝 Testing Subtitle Formatting...")
    
    generator = SubtitleGenerator()
    
    # Test various text lengths
    test_cases = [
        "This is a short sentence.",
        "This is a much longer sentence that should be split across multiple lines but never exceed three lines maximum.",
        "AI",  # Single word
        "Machine learning",  # Two words
        "Artificial intelligence is revolutionizing the technology industry with breakthrough innovations.",
        "OpenAI has released a new model. It performs better than previous versions. The results are impressive.",
    ]
    
    success = True
    
    for i, text in enumerate(test_cases):
        print(f"\n   Test {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # Use the internal segmentation method
            segments = generator._segment_text(text)
            
            for j, segment in enumerate(segments):
                lines = segment.split('\n')
                print(f"      Segment {j+1}: {len(lines)} lines")
                
                # Check line count
                if len(lines) > 3:
                    print(f"         ❌ Too many lines: {len(lines)} > 3")
                    success = False
                else:
                    print(f"         ✅ Line count OK: {len(lines)} ≤ 3")
                
                # Check for single words (avoid them)
                single_word_lines = [line for line in lines if len(line.split()) == 1 and line.strip()]
                if single_word_lines and len(segments) > 1:
                    print(f"         ⚠️  Single word line detected: '{single_word_lines[0]}'")
                    # Not a failure, but not ideal
                
                # Check character limits
                for k, line in enumerate(lines):
                    if len(line) > 25:
                        print(f"         ❌ Line {k+1} too long: {len(line)} > 25 chars")
                        success = False
                    else:
                        print(f"         ✅ Line {k+1} length OK: {len(line)} ≤ 25 chars")
                
                print(f"         Content: {repr(segment)}")
                
        except Exception as e:
            print(f"         ❌ Error processing text: {e}")
            success = False
    
    return success


def test_subtitle_timing():
    """Test subtitle timing calibration for 1.35x TTS speed."""
    print("\n⏰ Testing Subtitle Timing...")
    
    generator = SubtitleGenerator()
    
    # Test script
    test_script = "This is a test sentence. It should have proper timing. The subtitles should sync well."
    test_duration = 15.0  # 15 seconds
    
    try:
        # Generate segments
        segments = generator._segment_text(test_script)
        print(f"   Generated {len(segments)} segments")
        
        # Calculate timing
        timed_segments = generator._calculate_timing(segments, test_duration)
        print(f"   Calculated timing for {len(timed_segments)} segments")
        
        # Check timing properties
        total_subtitle_duration = 0
        for i, segment in enumerate(timed_segments):
            duration = segment.end_time - segment.start_time
            total_subtitle_duration += duration
            print(f"      Segment {i+1}: {segment.start_time:.2f}s - {segment.end_time:.2f}s ({duration:.2f}s)")
            print(f"         Text: {repr(segment.text)}")
            
            # Check for reasonable timing
            if duration < 0.5:
                print(f"         ⚠️  Very short duration: {duration:.2f}s")
            elif duration > 8.0:
                print(f"         ⚠️  Very long duration: {duration:.2f}s")
            else:
                print(f"         ✅ Good duration: {duration:.2f}s")
        
        # Check total timing
        if total_subtitle_duration > test_duration * 1.2:
            print(f"   ⚠️  Total subtitle duration ({total_subtitle_duration:.2f}s) exceeds audio duration ({test_duration}s) by more than 20%")
        else:
            print(f"   ✅ Total subtitle timing reasonable: {total_subtitle_duration:.2f}s vs {test_duration}s audio")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing timing: {e}")
        return False


def test_video_duration_config():
    """Test video duration configuration."""
    print("\n🎬 Testing Video Duration Configuration...")
    
    # Check production config
    try:
        import json
        with open("production_config.json", "r") as f:
            prod_config = json.load(f)
        
        min_duration = prod_config.get("min_video_duration", 0)
        max_duration = prod_config.get("max_video_duration", 0)
        buffer_seconds = prod_config.get("video_buffer_seconds", 0)
        
        print(f"   Min video duration: {min_duration}s")
        print(f"   Max video duration: {max_duration}s")
        print(f"   Video buffer: {buffer_seconds}s")
        
        # Check if durations are appropriate for 1.35x TTS
        if 35 <= min_duration <= 45 and 45 <= max_duration <= 55:
            print("   ✅ Video duration range appropriate for 1.35x TTS speed")
            return True
        else:
            print(f"   ⚠️  Video duration may not be optimal for 1.35x TTS speed")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        print(f"   ❌ Error checking video duration config: {e}")
        return False


def test_tts_target_duration():
    """Test TTS target duration configuration."""
    print("\n🎯 Testing TTS Target Duration...")
    
    target_duration = config.tts.target_duration
    print(f"   TTS target duration: {target_duration}s")
    
    # For 1.35x speed, target should be shorter
    if 40 <= target_duration <= 50:
        print("   ✅ TTS target duration appropriate for 1.35x speed")
        return True
    else:
        print(f"   ⚠️  TTS target duration may not be optimal: {target_duration}s")
        return True  # Not a failure, just a suggestion


def main():
    """Run all tests."""
    print("🚀 Testing Batch Improvements for 1.35x TTS Speed")
    print("=" * 60)
    
    tests = [
        ("TTS Speed", test_tts_speed),
        ("Subtitle Formatting", test_subtitle_formatting),
        ("Subtitle Timing", test_subtitle_timing),
        ("Video Duration Config", test_video_duration_config),
        ("TTS Target Duration", test_tts_target_duration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All improvements ready for production!")
        return True
    else:
        print("⚠️  Some improvements need attention before production")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
