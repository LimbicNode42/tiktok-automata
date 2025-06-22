#!/usr/bin/env python3
"""
Test script to compare different letterboxing approaches.
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

async def test_letterboxing_comparison():
    """Test different letterboxing approaches side by side."""
    
    logger.info("🎬 Testing letterboxing comparison...")
    
    # Create processor
    config = VideoConfig(width=1080, height=1920)
    processor = VideoProcessor(config)
    
    # Get raw footage path
    footage_dir = Path("src/video/data/footage/raw")
    footage_files = list(footage_dir.glob("*.mp4"))
    
    if not footage_files:
        logger.error("❌ No raw footage found")
        return False
    
    raw_footage = footage_files[0]
    logger.info(f"📁 Using raw footage: {raw_footage}")
    
    try:
        # Load raw video
        raw_video = VideoFileClip(str(raw_footage))
        logger.info(f"🎥 Raw video: {raw_video.w}x{raw_video.h} (aspect: {raw_video.h/raw_video.w:.3f})")
        
        # Limit to 10 seconds for faster testing
        raw_video = raw_video.subclipped(0, min(10, raw_video.duration))
        
        # Create output directory
        output_dir = Path("letterbox_comparison")
        output_dir.mkdir(exist_ok=True)
        
        # Test 1: Traditional letterboxing (100% of source)
        logger.info("\n" + "="*50)
        logger.info("🔍 TEST 1: Traditional Letterboxing (100% source)")
        logger.info("="*50)
        traditional = processor._apply_letterboxing(raw_video)
        logger.info(f"📐 Traditional result: {traditional.w}x{traditional.h}")
        
        traditional_output = output_dir / "traditional_letterbox.mp4"
        logger.info(f"💾 Saving traditional: {traditional_output}")
        traditional.write_videofile(str(traditional_output), 
                                   codec="libx264", audio_codec="aac",
                                   temp_audiofile="temp-audio-traditional.m4a",
                                   remove_temp=True, logger=None)
        
        # Test 2: 75% crop letterboxing
        logger.info("\n" + "="*50)
        logger.info("🔍 TEST 2: 75% Crop Letterboxing")
        logger.info("="*50)
        crop_75 = processor._apply_percentage_letterboxing(raw_video, crop_percentage=0.75)
        logger.info(f"📐 75% crop result: {crop_75.w}x{crop_75.h}")
        
        crop_75_output = output_dir / "crop_75_letterbox.mp4"
        logger.info(f"💾 Saving 75% crop: {crop_75_output}")
        crop_75.write_videofile(str(crop_75_output),
                               codec="libx264", audio_codec="aac", 
                               temp_audiofile="temp-audio-crop75.m4a",
                               remove_temp=True, logger=None)
        
        # Test 3: 85% crop letterboxing
        logger.info("\n" + "="*50)
        logger.info("🔍 TEST 3: 85% Crop Letterboxing")
        logger.info("="*50)
        crop_85 = processor._apply_percentage_letterboxing(raw_video, crop_percentage=0.85)
        logger.info(f"📐 85% crop result: {crop_85.w}x{crop_85.h}")
        
        crop_85_output = output_dir / "crop_85_letterbox.mp4"
        logger.info(f"💾 Saving 85% crop: {crop_85_output}")
        crop_85.write_videofile(str(crop_85_output),
                               codec="libx264", audio_codec="aac",
                               temp_audiofile="temp-audio-crop85.m4a", 
                               remove_temp=True, logger=None)
        
        # Test 4: 60% crop letterboxing (more aggressive)
        logger.info("\n" + "="*50)
        logger.info("🔍 TEST 4: 60% Crop Letterboxing (Aggressive)")
        logger.info("="*50)
        crop_60 = processor._apply_percentage_letterboxing(raw_video, crop_percentage=0.60)
        logger.info(f"📐 60% crop result: {crop_60.w}x{crop_60.h}")
        
        crop_60_output = output_dir / "crop_60_letterbox.mp4"
        logger.info(f"💾 Saving 60% crop: {crop_60_output}")
        crop_60.write_videofile(str(crop_60_output),
                               codec="libx264", audio_codec="aac",
                               temp_audiofile="temp-audio-crop60.m4a",
                               remove_temp=True, logger=None)
        
        # Cleanup
        raw_video.close()
        traditional.close()
        crop_75.close()
        crop_85.close()
        crop_60.close()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.success("✅ Letterboxing comparison completed!")
        logger.info("="*60)
        logger.info("📁 Output files:")
        logger.info(f"  • Traditional (100%): {traditional_output}")
        logger.info(f"  • 75% Crop: {crop_75_output}")
        logger.info(f"  • 85% Crop: {crop_85_output}")
        logger.info(f"  • 60% Crop: {crop_60_output}")
        logger.info("\n🎯 Compare the videos to see the difference in black bars vs content shown")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if not VideoFileClip:
        logger.error("MoviePy not available - install with: pip install moviepy")
        sys.exit(1)
    
    success = asyncio.run(test_letterboxing_comparison())
    sys.exit(0 if success else 1)
