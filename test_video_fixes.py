#!/usr/bin/env python3
"""
Test script to validate video processing fixes, specifically:
1. Dimension issues (width not divisible by 2)
2. Text overlay parameter conflicts
3. Real gaming footage processing
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.video.processors.segment_processor import VideoSegmentProcessor

async def test_video_processing():
    """Test video segment creation with dimension fixes."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing video processing fixes...")
        
        # Set up directories
        base_dir = Path("src/video/data/footage")
        raw_dir = base_dir / "raw"
        processed_dir = base_dir / "processed"
        
        # Create processed directory if it doesn't exist
        processed_dir.mkdir(parents=True, exist_ok=True)
          # Initialize processors
        processor = VideoSegmentProcessor(processed_dir)
        
        # Get available videos directly from filesystem
        raw_footage_dir = base_dir / "raw"
        video_files = list(raw_footage_dir.glob("*.mp4"))
        
        if not video_files:
            logger.error("No gaming footage available for testing")
            return False
            
        logger.info(f"Found {len(video_files)} gaming videos")
        
        # Test with the first video
        video_path = video_files[0]
        video_id = video_path.stem  # filename without extension
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
            
        logger.info(f"Testing with video: {video_id}")
        logger.info(f"Video path: {video_path}")
        
        # Test dimension conversion
        from moviepy import VideoFileClip
        
        logger.info("Loading video clip...")
        clip = VideoFileClip(str(video_path))
        
        original_size = clip.size
        logger.info(f"Original video dimensions: {original_size[0]}x{original_size[1]}")
        
        # Test the conversion method
        logger.info("Testing TikTok format conversion...")
        tiktok_clip = processor._convert_to_tiktok_format(clip)
        
        final_size = tiktok_clip.size
        logger.info(f"Final video dimensions: {final_size[0]}x{final_size[1]}")
        
        # Verify dimensions are even
        if final_size[0] % 2 != 0:
            logger.error(f"Width is not even: {final_size[0]}")
            return False
        if final_size[1] % 2 != 0:
            logger.error(f"Height is not even: {final_size[1]}")
            return False
            
        logger.info("✅ Dimensions are even and compatible with x264/NVENC")
        
        # Test creating a short segment
        logger.info("Testing segment creation...")
        duration_info = {
            'index': 1,
            'title': 'test_segment',
            'duration': 10.0,  # 10 second test
            'start_time': 30.0  # Start 30 seconds in
        }
        
        segment_file = await processor._create_individual_segment(
            clip, video_id, 10.0, duration_info, clip.duration
        )
        
        if segment_file and segment_file.exists():
            file_size = segment_file.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Successfully created test segment: {segment_file.name} ({file_size:.1f}MB)")
            
            # Clean up test file
            segment_file.unlink()
            logger.info("🧹 Cleaned up test file")
            
            return True
        else:
            logger.error("❌ Failed to create test segment")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Full error: {traceback.format_exc()}")
        return False
    finally:
        # Cleanup
        if 'clip' in locals():
            clip.close()
        if 'tiktok_clip' in locals():
            tiktok_clip.close()

async def main():
    """Main test function."""
    success = await test_video_processing()
    if success:
        print("🎉 All video processing tests passed!")
        sys.exit(0)
    else:
        print("❌ Video processing tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
