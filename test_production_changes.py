#!/usr/bin/env python3
"""
Test script to validate the production changes made for the next batch.

Changes tested:
1. TTS speed reduced to 1.35x
2. Video duration targeting ~50 seconds
3. Subtitles forced to 2-line static display
4. Subtitles show 150ms before TTS audio
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import Config
from video.subtitles import SubtitleGenerator
from loguru import logger


async def test_tts_speed_config():
    """Test that TTS speed is correctly set to 1.35x."""
    logger.info("🎤 Testing TTS speed configuration")
    
    config = Config()
    expected_speed = 1.35
    actual_speed = config.tts.default_speed
    
    if abs(actual_speed - expected_speed) < 0.01:
        logger.success(f"✅ TTS speed correctly set to {actual_speed}x")
    else:
        logger.error(f"❌ TTS speed incorrect: expected {expected_speed}x, got {actual_speed}x")
    
    return actual_speed == expected_speed


async def test_subtitle_formatting():
    """Test that subtitles are formatted as 2-line static display."""
    logger.info("📝 Testing subtitle 2-line formatting")
    
    generator = SubtitleGenerator()
    
    test_cases = [
        "This is a short test.",
        "This is a medium length test that should be split across two lines perfectly.",
        "This is a much longer test case that demonstrates how the subtitle system handles longer content by splitting it optimally.",
        "Breaking! AI just achieved something incredible! Scientists are amazed!"
    ]
    
    all_passed = True
    
    for i, test_text in enumerate(test_cases, 1):
        logger.info(f"\n📝 Test case {i}: '{test_text[:40]}...'")
        
        segments = await generator.generate_from_script(
            test_text,
            3.0,  # 3 second duration
            style_name="default"
        )
        
        for j, segment in enumerate(segments):
            lines = segment.text.split('\n')
            line_count = len([line for line in lines if line.strip()])
            
            if line_count == 2:
                logger.success(f"  ✅ Segment {j+1}: 2 lines - '{segment.text.replace(chr(10), ' | ')}'")
            else:
                logger.error(f"  ❌ Segment {j+1}: {line_count} lines - '{segment.text.replace(chr(10), ' | ')}'")
                all_passed = False
            
            # Check character limits per line
            for line_idx, line in enumerate(lines):
                if line.strip() and len(line) > 28:
                    logger.warning(f"    ⚠️  Line {line_idx+1} too long: {len(line)} chars")
    
    if all_passed:
        logger.success("✅ All subtitle segments correctly formatted as 2 lines")
    else:
        logger.error("❌ Some subtitle segments not properly formatted")
    
    return all_passed


async def test_subtitle_timing():
    """Test that subtitles show before TTS audio."""
    logger.info("⏰ Testing subtitle timing (lead time)")
    
    generator = SubtitleGenerator()
    
    segments = await generator.generate_from_script(
        "First sentence here. Second sentence follows. Third sentence completes it.",
        6.0,  # 6 second duration to force multiple segments
        style_name="default"
    )
    
    # Check that segments show some lead time behavior
    # For a 6-second multi-segment scenario, later segments should show lead time
    total_segments = len(segments)
    has_lead_time = True  # Initialize the variable
    
    if total_segments == 1:
        # Single segment: lead time is clamped to 0, which is correct
        logger.info(f"  Single segment correctly starts at 0.00s (lead time clamped)")
    else:
        # Multiple segments: should see actual lead time in later segments
        for i, segment in enumerate(segments):
            if i > 0:  # Skip first segment which starts at 0
                # Later segments should show lead time
                segment_audio_start = i * (6.0 / total_segments)
                actual_start = segment.start_time
                lead_time = segment_audio_start - actual_start
                
                logger.info(f"  Segment {i+1}: Audio at {segment_audio_start:.2f}s, Subtitle at {actual_start:.2f}s, Lead: {lead_time:.3f}s")
                
                if lead_time < 0.1:  # Should have lead time
                    has_lead_time = False
                    break
    
    if has_lead_time:
        logger.success("✅ Subtitles have proper lead time before audio")
    else:
        logger.warning("⚠️  Subtitle lead time may need adjustment")
    
    return has_lead_time


async def test_video_duration_config():
    """Test video duration configuration."""
    logger.info("🎬 Testing video duration configuration")
    
    # Read production config
    import json
    
    try:
        with open("production_config.json", "r") as f:
            config = json.load(f)
        
        min_duration = config.get("min_video_duration", 60)
        max_duration = config.get("max_video_duration", 3600)
        buffer_seconds = config.get("video_buffer_seconds", 0.5)
        
        logger.info(f"  Min duration: {min_duration}s")
        logger.info(f"  Max duration: {max_duration}s")
        logger.info(f"  Buffer: {buffer_seconds}s")
        
        # Check if targeting ~50 seconds
        target_met = (45 <= min_duration <= 50) and (50 <= max_duration <= 60)
        buffer_reduced = buffer_seconds <= 0.3
        
        if target_met and buffer_reduced:
            logger.success("✅ Video duration configuration correctly targets ~50 seconds")
        else:
            logger.error(f"❌ Video duration config needs adjustment: min={min_duration}, max={max_duration}, buffer={buffer_seconds}")
        
        return target_met and buffer_reduced
        
    except Exception as e:
        logger.error(f"❌ Failed to read production config: {e}")
        return False


async def main():
    """Run all validation tests."""
    logger.info("🧪 Production Changes Validation")
    logger.info("=" * 50)
    
    tests = [
        ("TTS Speed Configuration", test_tts_speed_config),
        ("Subtitle 2-Line Formatting", test_subtitle_formatting),
        ("Subtitle Timing (Lead Time)", test_subtitle_timing),
        ("Video Duration Configuration", test_video_duration_config),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔧 Running: {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n📊 Test Results Summary")
    logger.info("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    if passed == total:
        logger.success(f"\n🎉 All {total} tests passed! Production changes are ready.")
        return 0
    else:
        logger.error(f"\n⚠️  {passed}/{total} tests passed. Please review failed tests.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
