#!/usr/bin/env python3
"""
Test script to verify the optimization improvements.
This script tests the optimized components in isolation.
"""

import asyncio
import time
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.video.managers.footage_manager import FootageManager

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_action_analysis_caching():
    """Test that action analysis uses caching properly."""
    logger.info("🧪 Testing action analysis caching...")
    
    manager = FootageManager()
    
    # Check if we have any videos to test with
    if not manager.metadata.get("videos"):
        logger.warning("⚠️ No videos available for testing - run the pipeline first")
        return
    
    video_id = list(manager.metadata["videos"].keys())[0]
    logger.info(f"📹 Testing with video: {video_id}")
    
    # First analysis (should be slow)
    start_time = time.time()
    result1 = await manager.analyze_video_action(video_id)
    first_time = time.time() - start_time
    logger.info(f"⏱️ First analysis: {first_time:.2f}s")
    
    # Second analysis (should be fast - cached)
    start_time = time.time()
    result2 = await manager.analyze_video_action(video_id)
    second_time = time.time() - start_time
    logger.info(f"⏱️ Second analysis (cached): {second_time:.2f}s")
      # Verify results are identical
    if result1 == result2:
        logger.info(f"✅ Caching works! Speedup: {first_time/second_time:.1f}x faster")
    else:
        logger.error("❌ Cache results differ from fresh analysis")
    
    return first_time, second_time

async def test_segment_caching():
    """Test that segment analysis uses caching properly."""
    logger.info("🧪 Testing segment caching...")
    
    manager = FootageManager()
    
    if not manager.metadata.get("videos"):
        logger.warning("⚠️ No videos available for testing")
        return
    
    video_id = list(manager.metadata["videos"].keys())[0]
    duration = 45.0
    
    # First segment analysis (should be slow)
    start_time = time.time()
    segments1 = await manager.get_best_action_segments(video_id, duration)
    first_time = time.time() - start_time
    logger.info(f"⏱️ First segment analysis: {first_time:.2f}s")
    
    # Second segment analysis (should be fast - cached)
    start_time = time.time()
    segments2 = await manager.get_best_action_segments(video_id, duration)
    second_time = time.time() - start_time
    logger.info(f"⏱️ Second segment analysis (cached): {second_time:.2f}s")
      # Verify results are identical
    if segments1 == segments2:
        logger.info(f"✅ Segment caching works! Speedup: {first_time/second_time:.1f}x faster")
    else:
        logger.error("❌ Cached segments differ from fresh analysis")
    
    return first_time, second_time

async def test_video_processing_speed():
    """Test a single video segment creation to measure encoding speed."""
    logger.info("🧪 Testing video processing speed...")
    
    try:
        from src.video.processors.segment_processor import VideoSegmentProcessor
        
        processor = VideoSegmentProcessor()
        
        # This is just a speed test - we'll create a small test segment
        # if we have footage available
        manager = FootageManager()
        
        if not manager.metadata.get("videos"):
            logger.warning("⚠️ No videos available for encoding speed test")
            return
        
        video_id = list(manager.metadata["videos"].keys())[0]
        video_info = manager.metadata["videos"][video_id]
        
        logger.info(f"🎬 Testing encoding speed with {video_id}")
        
        # Create a short test segment
        start_time = time.time()
        test_durations = [{"index": 0, "duration": 10.0, "start_time": 10.0}]  # 10 second test
        
        result = await processor.process_footage_for_tiktok(
            video_info,
            Path(video_info["file_path"]),
            video_id,
            duration_info_list=test_durations
        )
        
        encoding_time = time.time() - start_time
          if result:
            logger.info(f"✅ Encoding test successful: {encoding_time:.2f}s for 10s video")
            # Calculate encoding speed ratio
            speed_ratio = 10.0 / encoding_time
            logger.info(f"🚀 Encoding speed: {speed_ratio:.1f}x realtime")
        else:
            logger.error("❌ Encoding test failed")
        
        return encoding_time
        
    except Exception as e:
        logger.error(f"❌ Video processing test failed: {e}")
        return None

async def main():
    """Run all optimization tests."""
    logger.info("🚀 Starting optimization tests...")
    
    try:
        # Test action analysis caching
        action_times = await test_action_analysis_caching()
        
        # Test segment caching
        segment_times = await test_segment_caching()
        
        # Test video processing speed
        encoding_time = await test_video_processing_speed()
        
        # Summary
        logger.info("📊 OPTIMIZATION TEST SUMMARY:")
        
        if action_times:
            speedup = action_times[0] / action_times[1] if action_times[1] > 0 else 0
            logger.info(f"  🔍 Action Analysis Caching: {speedup:.1f}x speedup")
        
        if segment_times:
            speedup = segment_times[0] / segment_times[1] if segment_times[1] > 0 else 0
            logger.info(f"  🎯 Segment Caching: {speedup:.1f}x speedup")
        
        if encoding_time:
            realtime_ratio = 10.0 / encoding_time
            logger.info(f"  🎬 Video Encoding: {realtime_ratio:.1f}x realtime speed")
        
        logger.info(f"✅ All optimization tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Optimization tests failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
