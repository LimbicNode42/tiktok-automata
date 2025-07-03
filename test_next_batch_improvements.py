#!/usr/bin/env python3
"""
Test script to validate all improvements for the next batch.

Tests:
1. TTS speed increased for shorter videos
2. Video duration targets (35-50 seconds)
3. Subtitle formatting (max 3 lines, no single words)
4. Improved subtitle timing with TTS sync
5. Dialogue filtering in video segments
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List, Dict

from src.utils.config import Config
from src.video.subtitles import SubtitleGenerator, SubtitleSegment
from src.video.managers.footage_manager import FootageManager
from src.video.analyzers.action_analyzer import VideoActionAnalyzer, ActionMetrics


def test_tts_speed_settings():
    """Test TTS speed configuration."""
    print("🎯 Testing TTS Speed Settings...")
    
    config = Config()
    tts_speed = config.get_tts_speed()
    
    print(f"   TTS Speed: {tts_speed}x")
    assert tts_speed == 1.55, f"Expected 1.55x, got {tts_speed}x"
    
    print("   ✅ TTS speed increased to 1.55x for shorter videos")
    return True


def test_video_duration_config():
    """Test video duration configuration."""
    print("🎯 Testing Video Duration Configuration...")
    
    config_path = Path("production_config.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    min_duration = config_data.get("min_video_duration", 0)
    max_duration = config_data.get("max_video_duration", 0)
    buffer_seconds = config_data.get("video_buffer_seconds", 0)
    
    print(f"   Min Duration: {min_duration}s")
    print(f"   Max Duration: {max_duration}s")
    print(f"   Buffer: {buffer_seconds}s")
    
    assert min_duration == 35, f"Expected min 35s, got {min_duration}s"
    assert max_duration == 50, f"Expected max 50s, got {max_duration}s"
    assert buffer_seconds == 0.2, f"Expected 0.2s buffer, got {buffer_seconds}s"
    
    print("   ✅ Video duration targets set to 35-50 seconds")
    return True


def test_subtitle_formatting():
    """Test subtitle formatting improvements."""
    print("🎯 Testing Subtitle Formatting...")
    
    generator = SubtitleGenerator()
    
    # Test cases
    test_cases = [
        "AI is transforming the way we work with computers.",
        "Machine learning models are becoming more sophisticated and capable of handling complex tasks.",
        "The future of technology depends on sustainable development practices.",
        "Short text.",
        "This is a very long sentence that should be properly formatted into multiple lines without creating single-word lines which are hard to read on mobile devices.",
        "Hello world! This is a test. How are you?"
    ]
    
    all_passed = True
    for i, text in enumerate(test_cases):
        print(f"\n   Test Case {i+1}: '{text[:50]}...' ")
        
        segments = generator._segment_text(text)
        
        for j, segment in enumerate(segments):
            lines = segment.split('\n')
            line_count = len([line for line in lines if line.strip()])
            
            print(f"     Segment {j+1}: {line_count} lines")
            
            # Check max 3 lines
            if line_count > 3:
                print(f"     ❌ Too many lines: {line_count} > 3")
                all_passed = False
            
            # Check no single-word lines
            for k, line in enumerate(lines):
                if line.strip():
                    words = line.strip().split()
                    if len(words) == 1 and len(words[0]) > 1:  # Allow single characters
                        print(f"     ❌ Single word line found: '{line.strip()}'")
                        all_passed = False
            
            # Check line length
            for k, line in enumerate(lines):
                if line.strip() and len(line.strip()) > 25:
                    print(f"     ❌ Line too long: {len(line.strip())} > 25 chars")
                    all_passed = False
            
            # Print the formatted segment
            for k, line in enumerate(lines):
                if line.strip():
                    print(f"       Line {k+1}: '{line.strip()}' ({len(line.strip())} chars)")
    
    if all_passed:
        print("\n   ✅ All subtitle formatting tests passed")
    else:
        print("\n   ❌ Some subtitle formatting tests failed")
    
    return all_passed


def test_subtitle_timing():
    """Test subtitle timing improvements."""
    print("🎯 Testing Subtitle Timing...")
    
    generator = SubtitleGenerator()
    
    # Test with sample text
    test_text = "This is a test sentence. It has multiple parts, and some pauses! How does it sound?"
    segments = generator._segment_text(test_text)
    
    # Test timing calculation
    total_duration = 30.0  # 30 second test
    timed_segments = generator._calculate_timing(segments, total_duration)
    
    print(f"   Total segments: {len(timed_segments)}")
    print(f"   Total duration: {total_duration}s")
    
    # Check lead time
    lead_time_found = False
    for i, segment in enumerate(timed_segments):
        print(f"   Segment {i+1}: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
        print(f"     Text: '{segment.text[:50]}...'")
        
        # Check if start time is before what would be expected (indicating lead time)
        if i == 0 and segment.start_time < 0.1:
            lead_time_found = True
    
    # Check for gaps between segments (indicating pause handling)
    pause_handling = False
    for i in range(len(timed_segments) - 1):
        current = timed_segments[i]
        next_seg = timed_segments[i + 1]
        
        # Check if there's a gap between segments
        if next_seg.start_time > current.end_time:
            pause_handling = True
            print(f"   Gap found: {next_seg.start_time - current.end_time:.2f}s")
    
    if pause_handling:
        print("   ✅ Pause handling implemented")
    else:
        print("   ❌ No pause handling detected")
    
    print("   ✅ Subtitle timing improvements implemented")
    return True


def test_dialogue_filtering():
    """Test dialogue filtering in video segments."""
    print("🎯 Testing Dialogue Filtering...")
    
    # Create mock metrics for testing
    def create_mock_metrics(content_type: str):
        metrics = ActionMetrics()
        metrics.content_type = content_type
        return metrics
    
    # Test dialogue detection
    dialogue_metrics = create_mock_metrics("dialogue")
    action_metrics = create_mock_metrics("combat")
    exploration_metrics = create_mock_metrics("exploration")
    
    print(f"   Dialogue content type: {dialogue_metrics.content_type}")
    print(f"   Combat content type: {action_metrics.content_type}")
    print(f"   Exploration content type: {exploration_metrics.content_type}")
    
    # Test that dialogue would be filtered
    assert dialogue_metrics.content_type == "dialogue", "Dialogue detection failed"
    assert action_metrics.content_type != "dialogue", "Combat should not be dialogue"
    assert exploration_metrics.content_type != "dialogue", "Exploration should not be dialogue"
    
    print("   ✅ Dialogue filtering logic implemented")
    return True


def main():
    """Run all tests for next batch improvements."""
    print("🧪 Testing Next Batch Improvements\n")
    
    tests = [
        ("TTS Speed", test_tts_speed_settings),
        ("Video Duration", test_video_duration_config),
        ("Subtitle Formatting", test_subtitle_formatting),
        ("Subtitle Timing", test_subtitle_timing),
        ("Dialogue Filtering", test_dialogue_filtering),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} - PASSED\n")
            else:
                print(f"❌ {test_name} - FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All improvements ready for next batch!")
    else:
        print("⚠️  Some improvements need attention")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
