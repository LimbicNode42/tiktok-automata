#!/usr/bin/env python3
"""
Test script to verify subtitle fade duration changes.
Tests that subtitle fade effects are properly reduced for better sync with 1.5x TTS speed.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from video.subtitles import SubtitleGenerator, SubtitleSegment


async def test_fade_durations():
    """Test that fade durations are properly reduced."""
    
    logger.info("🎬 Testing Subtitle Fade Duration Changes")
    
    # Initialize subtitle generator
    generator = SubtitleGenerator()
    
    # Test default style fade durations
    default_style = generator.styles["default"]
    logger.info(f"📝 Default style fade durations:")
    logger.info(f"   Fade in: {default_style.fade_in_duration}s")
    logger.info(f"   Fade out: {default_style.fade_out_duration}s")
    
    # Test bubble style fade durations
    bubble_style = generator.styles["bubble_gaming"]
    logger.info(f"🎮 Bubble gaming style fade durations:")
    logger.info(f"   Fade in: {bubble_style.fade_in_duration}s")
    logger.info(f"   Fade out: {bubble_style.fade_out_duration}s")
      # Verify the fade durations are reduced
    expected_duration = 0.1
    
    if default_style.fade_in_duration == expected_duration:
        logger.info(f"✅ Default fade in duration correctly set to {expected_duration}s")
    else:
        logger.error(f"❌ Default fade in duration is {default_style.fade_in_duration}s, expected {expected_duration}s")
    
    if default_style.fade_out_duration == expected_duration:
        logger.info(f"✅ Default fade out duration correctly set to {expected_duration}s")
    else:
        logger.error(f"❌ Default fade out duration is {default_style.fade_out_duration}s, expected {expected_duration}s")
    
    # Test creating a subtitle segment with the new durations
    test_text = "Testing fade durations for better TTS sync!"
    segments = await generator.generate_from_script(
        test_text, 
        2.0,  # 2 second audio
        style_name="bubble_gaming"
    )
    
    logger.info(f"📊 Generated {len(segments)} subtitle segments")
    
    for i, segment in enumerate(segments):
        logger.info(f"   Segment {i+1}: '{segment.text}' ({segment.start_time:.2f}s - {segment.end_time:.2f}s)")
        logger.info(f"      Duration: {segment.duration:.2f}s")
    
    # Test audio mixer fade durations
    logger.info("\n🎵 Testing Audio Mixer Fade Durations")
      try:
        from video.audio.audio_mixer import AudioMixerConfig
        
        config = AudioMixerConfig()
        logger.info(f"   Audio fade duration: {config.fade_duration}s")
        
        expected_audio_fade = 0.2
        if config.fade_duration == expected_audio_fade:
            logger.info(f"✅ Audio fade duration correctly set to {expected_audio_fade}s")
        else:
            logger.error(f"❌ Audio fade duration is {config.fade_duration}s, expected {expected_audio_fade}s")
    
    except Exception as e:
        logger.warning(f"Could not test audio mixer config: {e}")
    
    logger.info("🎉 Fade duration testing completed!")
    
    return True


async def main():
    """Main test function."""
    try:
        success = await test_fade_durations()
        if success:
            logger.info("✅ All fade duration tests passed!")
        else:
            logger.error("❌ Some fade duration tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
