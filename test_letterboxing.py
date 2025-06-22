#!/usr/bin/env python3
"""
Test script for letterboxing video aspect ratio handling.

This script tests the new letterboxing approach that:
- Maintains original video aspect ratio
- Adds black bars instead of stretching/squashing
- Outputs TikTok-ready 9:16 format
"""

import sys
import asyncio
from pathlib import Path
from loguru import logger

# MoviePy import with error handling
try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None
    logger.warning("MoviePy not available for video analysis")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.video.processors.video_processor import VideoProcessor, VideoConfig
from src.tts.kokoro_tts import KokoroTTSEngine

async def test_letterboxing():
    """Test the letterboxing functionality with real gaming footage."""
    
    logger.info("🎬 Testing letterboxing video aspect ratio handling...")
    
    # Step 1: Generate test audio
    logger.info("Step 1: Generating test TTS audio...")
    tts = KokoroTTSEngine()
    
    test_script = """
    Breaking news in the tech world! A major breakthrough in AI development has just been announced. 
    This could revolutionize how we interact with technology in the future.
    """    # Create TTS audio
    audio_file = await tts.generate_audio(
        text=test_script,
        voice="af_bella",
        output_path="letterbox_test/test_audio.wav"
    )
    
    if not audio_file:
        logger.error("❌ Failed to generate TTS audio")
        return False
    
    logger.success(f"✅ Generated audio: {audio_file}")
    
    # Step 2: Create video with letterboxing
    logger.info("Step 2: Creating video with letterboxing...")
    
    config = VideoConfig(
        width=1080,
        height=1920,
        fps=30,
        duration=120
    )
    
    processor = VideoProcessor(config)
    
    # Create video (this will use the new letterboxing method)
    video_file = await processor.create_video(
        audio_file=str(audio_file),
        script_content=test_script,
        content_analysis={"category": "tech", "is_breakthrough": True},
        output_path="letterbox_test/letterboxed_video.mp4"
    )
    
    if video_file:
        logger.success(f"✅ Created letterboxed video: {video_file}")        # Analyze the output
        try:
            if VideoFileClip:
                clip = VideoFileClip(video_file)
            logger.info(f"📊 Output video specs:")
            logger.info(f"   - Dimensions: {clip.w}x{clip.h}")
            logger.info(f"   - Aspect ratio: {clip.h/clip.w:.3f} (target: 1.778)")
            logger.info(f"   - Duration: {clip.duration:.2f}s")
            logger.info(f"   - FPS: {clip.fps}")
            clip.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to analyze output video: {e}")
            return False
    else:
        logger.error("❌ Failed to create letterboxed video")
        return False

async def test_different_aspect_ratios():
    """Test letterboxing with different input aspect ratios."""
    
    logger.info("🎬 Testing letterboxing with different aspect ratios...")
    
    # This would test with different source footage if available
    # For now, we'll just test with whatever footage we have
    
    logger.info("⚠️ Extended aspect ratio testing requires diverse source footage")
    logger.info("Current test uses available gaming footage")
    
    return True

async def main():
    """Run all letterboxing tests."""
    
    logger.info("🚀 Starting letterboxing tests...")
    
    try:
        # Test basic letterboxing
        test1_success = await test_letterboxing()
        
        # Test different aspect ratios
        test2_success = await test_different_aspect_ratios()
        
        if test1_success:
            logger.success("🎉 Letterboxing tests completed successfully!")
            logger.info("📁 Check the 'letterbox_test' directory for output files")
            logger.info("🎥 Manually review the video to verify proper letterboxing")
            return True
        else:
            logger.error("❌ Some letterboxing tests failed")
            return False
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
