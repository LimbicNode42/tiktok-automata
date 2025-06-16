#!/usr/bin/env python3
"""
Test the new JSON-based continuous segment analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video import FootageManager
from loguru import logger

async def test_json_segment_analysis():
    """Test analyzing video segments based on JSON durations."""
    logger.info("🎵 Testing JSON-Based Segment Analysis")
    
    try:
        fm = FootageManager()
        
        # Check for videos
        if not fm.metadata.get("videos"):
            logger.warning("⚠️ No videos available for testing")
            return
        
        # Get test files
        video_id = list(fm.metadata["videos"].keys())[0]
        video_info = fm.metadata["videos"][video_id]
        video_path = Path(video_info["file_path"])
        
        json_file = Path("src/tts/data/voice_recommendations_test/voice_recommendations_test_20250613_124154.json")
        
        if not json_file.exists():
            logger.error(f"❌ JSON file not found: {json_file}")
            return
        
        logger.info(f"🎥 Video: {video_id}")
        logger.info(f"📄 JSON: {json_file.name}")
        
        # Test the new functionality
        results = await fm.analyzer.analyze_segments_from_json(video_path, json_file, buffer_seconds=7.5)
        
        if results:
            logger.info(f"📊 Analysis Results for {len(results)} segments:")
            
            for segment_key, segment_data in list(results.items())[:3]:  # Show first 3
                info = segment_data['segment_info']
                best_segments = segment_data['best_video_segments']
                
                logger.info(f"\n🎬 {segment_key}:")
                logger.info(f"   Title: {info['title'][:50]}...")
                logger.info(f"   Audio Duration: {info['original_duration']:.1f}s + {7.5}s buffer = {info['buffered_duration']:.1f}s")
                logger.info(f"   Voice: {info['voice_name']} | Category: {info['category']}")
                
                if best_segments:
                    best = best_segments[0]
                    logger.info(f"   ✅ Best video segment: {best['start_time']:.1f}s - {best['end_time']:.1f}s")
                    logger.info(f"   📊 Action score: {best['avg_score']:.1f} (range: {best['min_score']:.1f} - {best['max_score']:.1f})")
                else:
                    logger.warning(f"   ⚠️ No suitable video segment found")
        else:
            logger.error("❌ No results returned")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_continuous_segments():
    """Test the continuous segment analysis with specific durations."""
    logger.info("⏱️ Testing Continuous Segment Analysis")
    
    try:
        fm = FootageManager()
        
        if not fm.metadata.get("videos"):
            logger.warning("⚠️ No videos available")
            return
        
        video_id = list(fm.metadata["videos"].keys())[0]
        video_info = fm.metadata["videos"][video_id]
        video_path = Path(video_info["file_path"])
        
        # Test with various durations
        test_durations = [30.0, 60.0, 90.0, 144.6]  # Including a typical audio duration
        
        logger.info(f"🎥 Testing durations: {test_durations}")
        
        results = await fm.analyzer.analyze_continuous_segments(video_path, test_durations)
        
        for duration, segments in results.items():
            logger.info(f"\n⏱️ Duration: {duration}s")
            if segments:
                best = segments[0]
                logger.info(f"   🥇 Best: {best['start_time']:.1f}s - {best['end_time']:.1f}s (score: {best['avg_score']:.1f})")
                logger.info(f"   📊 Variance: {best['score_variance']:.1f} | Samples: {best['sample_count']}")
                
                # Show top 3
                for i, seg in enumerate(segments[:3]):
                    logger.info(f"   #{i+1}: {seg['start_time']:.1f}s-{seg['end_time']:.1f}s (score: {seg['avg_score']:.1f})")
            else:
                logger.warning(f"   ⚠️ No segments found for {duration}s")
                
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

async def main():
    """Run all tests."""
    await test_continuous_segments()
    await test_json_segment_analysis()

if __name__ == "__main__":
    asyncio.run(main())
