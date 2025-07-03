#!/usr/bin/env python3
"""
Test script to validate TTS speed and subtitle sync for 1.35x configuration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import config
from video.subtitles import SubtitleGenerator

def test_tts_speed():
    """Test that TTS speed is correctly set to 1.35x"""
    print("🎤 Testing TTS Speed Configuration...")
    
    speed = config.get_tts_speed()
    print(f"   Configured TTS Speed: {speed}x")
    
    if speed == 1.35:
        print("   ✅ TTS Speed is correctly set to 1.35x")
        return True
    else:
        print(f"   ❌ TTS Speed is {speed}x, should be 1.35x")
        return False

def test_subtitle_timing():
    """Test subtitle timing calculations for 1.35x TTS"""
    print("\n📝 Testing Subtitle Timing for 1.35x TTS...")
    
    # Create a test subtitle generator
    generator = SubtitleGenerator()
    
    # Test text
    test_text = "This is a test sentence. It should be properly timed for TTS at 1.35x speed."
    
    # Test with typical TTS duration
    test_duration = 45.0  # 45 seconds
    
    try:
        # Generate subtitle segments using the correct async method
        import asyncio
        segments = asyncio.run(generator.generate_from_script(test_text, test_duration))
        
        print(f"   Generated {len(segments)} subtitle segments")
        
        # Check timing
        total_subtitle_time = 0
        for i, segment in enumerate(segments):
            print(f"   Segment {i+1}: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
            print(f"      Text: {repr(segment.text)}")
            total_subtitle_time = max(total_subtitle_time, segment.end_time)
        
        print(f"   Total subtitle duration: {total_subtitle_time:.1f}s")
        print(f"   Target duration: {test_duration:.1f}s")
        
        # Check if timing is reasonable
        if total_subtitle_time <= test_duration * 1.1:  # Within 10% is good
            print("   ✅ Subtitle timing looks good for 1.35x TTS")
            return True
        else:
            print("   ❌ Subtitle timing may be too long")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing subtitle timing: {e}")
        return False

def test_subtitle_formatting():
    """Test subtitle formatting with 3-line cap"""
    print("\n📋 Testing Subtitle Formatting...")
    
    generator = SubtitleGenerator()
    
    # Test various text lengths
    test_texts = [
        "Short text",
        "This is a medium length text that should fit well",
        "This is a longer text that should be formatted properly across multiple lines without exceeding the three line limit",
        "AI and machine learning are transforming how we work, live, and interact with technology in unprecedented ways."
    ]
    
    all_passed = True
    
    for i, test_text in enumerate(test_texts):
        print(f"\n   Test {i+1}: {test_text}")
        
        try:
            import asyncio
            segments = asyncio.run(generator.generate_from_script(test_text, 30.0))
            
            for j, segment in enumerate(segments):
                lines = segment.text.split('\n')
                line_count = len(lines)
                
                print(f"      Segment {j+1}: {line_count} lines")
                for k, line in enumerate(lines):
                    print(f"         Line {k+1}: '{line}' ({len(line)} chars)")
                
                # Check constraints
                if line_count > 3:
                    print(f"      ❌ Too many lines ({line_count} > 3)")
                    all_passed = False
                
                # Check for single-word lines
                for line in lines:
                    if line.strip() and len(line.strip().split()) == 1:
                        print(f"      ⚠️  Single word line: '{line.strip()}'")
                
                # Check line length
                for line in lines:
                    if len(line) > 28:
                        print(f"      ⚠️  Long line ({len(line)} chars): '{line}'")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            all_passed = False
    
    if all_passed:
        print("   ✅ Subtitle formatting tests passed")
    else:
        print("   ❌ Some subtitle formatting issues found")
    
    return all_passed

def main():
    """Run all tests"""
    print("🧪 Testing TTS Speed and Subtitle Sync for 1.35x Configuration")
    print("=" * 60)
    
    results = []
    
    # Test TTS speed
    results.append(test_tts_speed())
    
    # Test subtitle timing
    results.append(test_subtitle_timing())
    
    # Test subtitle formatting
    results.append(test_subtitle_formatting())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"   Passed: {passed}/{total} tests")
    
    if passed == total:
        print("   ✅ All tests passed! Configuration is ready for 1.35x TTS")
    else:
        print("   ❌ Some tests failed. Check the configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
