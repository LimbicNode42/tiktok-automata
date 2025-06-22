#!/usr/bin/env python3
"""
Test the configurable letterboxing modes.
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

async def test_configurable_letterboxing():
    """Test the configurable letterboxing system."""
    
    logger.info("🎬 Testing configurable letterboxing modes...")
    
    # Get raw footage
    footage_dir = Path("src/video/data/footage/raw")
    footage_files = list(footage_dir.glob("*.mp4"))
    
    if not footage_files:
        logger.error("❌ No raw footage found")
        return False
    
    raw_footage = footage_files[0]
    logger.info(f"📁 Using raw footage: {raw_footage}")
    
    try:
        # Create output directory
        output_dir = Path("configurable_letterbox_test")
        output_dir.mkdir(exist_ok=True)
        
        # Test traditional mode
        logger.info("\n" + "="*50)
        logger.info("🔍 Testing Traditional Mode")
        logger.info("="*50)
        
        config_traditional = VideoConfig(
            letterbox_mode="traditional"
        )
        processor_traditional = VideoProcessor(config_traditional)
        
        raw_video = VideoFileClip(str(raw_footage)).subclipped(0, 5)  # 5 seconds
        result_traditional = processor_traditional._apply_letterboxing_with_config(raw_video)
        
        traditional_output = output_dir / "traditional_mode.mp4"
        result_traditional.write_videofile(str(traditional_output), 
                                         codec="libx264", audio_codec="aac", logger=None)
        
        # Test 75% percentage mode
        logger.info("\n" + "="*50)
        logger.info("🔍 Testing 75% Percentage Mode")
        logger.info("="*50)
        
        config_75 = VideoConfig(
            letterbox_mode="percentage",
            letterbox_crop_percentage=0.75
        )
        processor_75 = VideoProcessor(config_75)
        
        result_75 = processor_75._apply_letterboxing_with_config(raw_video)
        
        percentage_75_output = output_dir / "percentage_75_mode.mp4"
        result_75.write_videofile(str(percentage_75_output),
                                codec="libx264", audio_codec="aac", logger=None)
        
        # Test 60% percentage mode  
        logger.info("\n" + "="*50)
        logger.info("🔍 Testing 60% Percentage Mode")
        logger.info("="*50)
        
        config_60 = VideoConfig(
            letterbox_mode="percentage", 
            letterbox_crop_percentage=0.60
        )
        processor_60 = VideoProcessor(config_60)
        
        result_60 = processor_60._apply_letterboxing_with_config(raw_video)
        
        percentage_60_output = output_dir / "percentage_60_mode.mp4"
        result_60.write_videofile(str(percentage_60_output),
                                codec="libx264", audio_codec="aac", logger=None)
        
        # Cleanup
        raw_video.close()
        result_traditional.close()
        result_75.close()
        result_60.close()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.success("✅ Configurable letterboxing test completed!")
        logger.info("="*60)
        logger.info("📁 Output files:")
        logger.info(f"  • Traditional: {traditional_output}")
        logger.info(f"  • 75% Crop: {percentage_75_output}")
        logger.info(f"  • 60% Crop: {percentage_60_output}")
        logger.info("\n🎯 Usage:")
        logger.info("  config = VideoConfig(letterbox_mode='percentage', letterbox_crop_percentage=0.75)")
        logger.info("  processor = VideoProcessor(config)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if not VideoFileClip:
        logger.error("MoviePy not available - install with: pip install moviepy")
        sys.exit(1)
    
    success = asyncio.run(test_configurable_letterboxing())
    sys.exit(0 if success else 1)
