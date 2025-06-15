#!/usr/bin/env python3
"""
Quick test script for action-based video segment categorization.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video import FootageManager
from loguru import logger

async def test_action_categorization():
    """Test action-based categorization of video segments."""
    logger.info("🎯 Testing Action-Based Video Categorization")
    
    try:
        fm = FootageManager()
        
        # Check available videos
        if not fm.metadata.get("videos"):
            logger.warning("⚠️ No videos available for testing")
            return
        
        videos = list(fm.metadata["videos"].keys())
        video_id = videos[0]
        
        logger.info(f"📹 Analyzing video: {video_id}")
        
        # Get action analysis
        action_data = await fm.analyze_video_action(video_id)
        
        # Show statistical breakdown
        total = len(action_data["high"]) + len(action_data["medium"]) + len(action_data["low"])
        
        logger.info(f"📊 Action Analysis Results:")
        logger.info(f"   Total segments: {total}")
        logger.info(f"   🔥 High action: {len(action_data['high'])} ({len(action_data['high'])/total*100:.1f}%)")
        logger.info(f"   🔸 Medium action: {len(action_data['medium'])} ({len(action_data['medium'])/total*100:.1f}%)")
        logger.info(f"   🔹 Low action: {len(action_data['low'])} ({len(action_data['low'])/total*100:.1f}%)")
        
        # Test getting best segments for different durations
        durations = [15, 30, 45, 60]
        
        for duration in durations:
            segments = fm.analyzer.get_best_action_segments(action_data, segment_duration=duration, max_segments=3)
            logger.info(f"   📽️ Best {duration}s segments: {len(segments)} found")
            
            for i, (start, end) in enumerate(segments[:2]):  # Show first 2
                logger.info(f"      Segment {i+1}: {start:.1f}s - {end:.1f}s")
        
        # Test adaptive vs fixed thresholds
        all_metrics = action_data["high"] + action_data["medium"] + action_data["low"]
        
        if all_metrics:
            logger.info(f"🎚️ Comparing Categorization Methods:")
            
            # Fixed thresholds (current default)
            fixed_stats = fm.analyzer.analyze_action_distribution(action_data)
            
            # Adaptive thresholds
            adaptive_categorized = fm.analyzer.categorize_with_adaptive_thresholds(all_metrics)
            adaptive_stats = fm.analyzer.analyze_action_distribution(adaptive_categorized)
            
            logger.info(f"   Fixed Thresholds: {fixed_stats['high_action_ratio']:.1%} high action")
            logger.info(f"   Adaptive Thresholds: {adaptive_stats['high_action_ratio']:.1%} high action")
            
            # Show content type distribution
            content_types = fixed_stats.get('content_distribution', {})
            if content_types:
                logger.info(f"   🎮 Content Types: {dict(content_types)}")
        
        logger.success("✅ Action categorization test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_action_categorization())
