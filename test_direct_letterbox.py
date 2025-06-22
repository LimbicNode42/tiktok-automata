#!/usr/bin/env python3
"""
Simple test to verify letterboxing with raw footage.
"""

import sys
import asyncio
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None

from src.video.processors.video_processor import VideoProcessor, VideoConfig

async def test_direct_letterboxing():
    """Test letterboxing directly on raw footage."""
    
    logger.info("🎬 Testing direct letterboxing on raw footage...")
    
    # Create processor
    config = VideoConfig(width=1080, height=1920)
    processor = VideoProcessor(config)
      # Get raw footage path
    raw_footage_path = Path("src/video/data/footage/raw/6nQtYwUqRxM_Squirrel Girl ｜ Marvel Rivals Free To Use Gameplay.mp4")
    
    if not raw_footage_path.exists():
        logger.error(f"❌ Raw footage not found: {raw_footage_path}")
        return False
    
    logger.info(f"📁 Using raw footage: {raw_footage_path}")
    
    # Load raw footage and apply letterboxing directly
    try:
        if VideoFileClip:
            # Load the raw video
            raw_video = VideoFileClip(str(raw_footage_path))
            logger.info(f"🎥 Raw video: {raw_video.w}x{raw_video.h} (aspect: {raw_video.h/raw_video.w:.3f})")
            
            # Apply letterboxing
            letterboxed_video = processor._apply_letterboxing(raw_video)
            logger.info(f"📐 Letterboxed: {letterboxed_video.w}x{letterboxed_video.h} (aspect: {letterboxed_video.h/letterboxed_video.w:.3f})")
            
            # Set duration to 10 seconds for testing
            test_video = letterboxed_video.subclipped(0, 10)
            
            # Save test output
            output_path = "letterbox_test/direct_letterbox_test.mp4"
            Path(output_path).parent.mkdir(exist_ok=True)
            
            logger.info(f"💾 Saving to: {output_path}")
            test_video.write_videofile(
                output_path,
                fps=30,
                codec='libx264',
                audio_codec='aac' if test_video.audio else None
            )
            
            # Cleanup
            raw_video.close()
            letterboxed_video.close()
            test_video.close()
            
            logger.success(f"✅ Direct letterboxing test completed: {output_path}")
            return True
            
        else:
            logger.error("❌ MoviePy not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def main():
    """Run the test."""
    
    logger.info("🚀 Starting direct letterboxing test...")
    
    success = await test_direct_letterboxing()
    
    if success:
        logger.success("🎉 Test completed! Check letterbox_test/direct_letterbox_test.mp4")
    else:
        logger.error("❌ Test failed")
    
    return success

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    # Run test
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
