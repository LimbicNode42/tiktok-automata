#!/usr/bin/env python3
"""
Test the gaming-optimized video analyzer on real gaming footage
using audio durations from voice recommendations JSON files.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.analyzers.action_analyzer import VideoActionAnalyzer
from loguru import logger

async def test_gaming_analyzer_with_json():
    """Test the gaming-optimized analyzer with real gaming videos and JSON durations."""
    logger.info("🎮 Testing Gaming-Optimized Video Analyzer")
    
    # Define test files
    video_files = [
        "src/video/data/footage/raw/6nQtYwUqRxM_Squirrel Girl ｜ Marvel Rivals Free To Use Gameplay.mp4",
        "src/video/data/footage/raw/fBVpT-stY70_Deadzone Rouge Gameplay - Free To Use Gameplay.mp4", 
        "src/video/data/footage/raw/ne_T-GNwObk_COD Black Ops 6 Gameplay - Free To Use.mp4"
    ]
    
    json_files = [
        "src/tts/data/voice_recommendations_test/voice_recommendations_test_20250613_123838.json",
        "src/tts/data/voice_recommendations_test/voice_recommendations_test_20250613_124154.json"
    ]
    
    analyzer = VideoActionAnalyzer()
    
    # Test each video
    for video_file in video_files:
        video_path = Path(video_file)
        
        if not video_path.exists():
            logger.warning(f"⚠️ Video not found: {video_path.name}")
            continue
            
        logger.info(f"\n🎥 Testing video: {video_path.name}")
        logger.info(f"📁 File size: {video_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Test basic action analysis
        start_time = time.time()
        try:
            action_results = await analyzer.analyze_video_action(video_path)
            analysis_duration = time.time() - start_time
            
            logger.info(f"⚡ Analysis completed in {analysis_duration:.1f} seconds")
            logger.info(f"📊 Results: {len(action_results['high'])} high, {len(action_results['medium'])} medium, {len(action_results['low'])} low action segments")
            
            # Show top action segments
            if action_results['high']:
                logger.info("🔥 Top HIGH action segments:")
                for i, segment in enumerate(action_results['high'][:3]):
                    logger.info(f"   #{i+1}: {segment.timestamp:.1f}s - Score: {segment.overall_score:.1f} - Type: {segment.content_type}")
            
            if action_results['medium']:
                logger.info("🔸 Top MEDIUM action segments:")
                for i, segment in enumerate(action_results['medium'][:2]):
                    logger.info(f"   #{i+1}: {segment.timestamp:.1f}s - Score: {segment.overall_score:.1f} - Type: {segment.content_type}")
                    
        except Exception as e:
            logger.error(f"❌ Basic analysis failed: {e}")
            continue
        
        # Test with JSON durations
        for json_file in json_files:
            json_path = Path(json_file)
            
            if not json_path.exists():
                logger.warning(f"⚠️ JSON not found: {json_path.name}")
                continue
                
            logger.info(f"\n📄 Testing with JSON: {json_path.name}")
            
            try:
                start_time = time.time()
                json_results = await analyzer.analyze_segments_from_json(video_path, json_path, buffer_seconds=7.5)
                json_duration = time.time() - start_time
                
                logger.info(f"⚡ JSON analysis completed in {json_duration:.1f} seconds")
                logger.info(f"📊 Found {len(json_results)} matching audio-video segment pairs")
                
                # Show results for first few segments
                for segment_key, segment_data in list(json_results.items())[:3]:
                    info = segment_data['segment_info']
                    best_segments = segment_data['best_video_segments']
                    
                    logger.info(f"\n🎬 {segment_key}:")
                    logger.info(f"   📰 Title: {info['title'][:60]}...")
                    logger.info(f"   🎵 Audio: {info['original_duration']:.1f}s + 7.5s buffer = {info['buffered_duration']:.1f}s")
                    logger.info(f"   🎤 Voice: {info.get('voice_name', 'Unknown')} | Category: {info.get('category', 'unknown')}")
                    
                    if best_segments:
                        best = best_segments[0]
                        logger.info(f"   ✅ Best match: {best['start_time']:.1f}s - {best['end_time']:.1f}s")
                        logger.info(f"   📊 Gaming score: {best['avg_score']:.1f} (variance: {best['score_variance']:.1f})")
                        
                        # Show top 3 options
                        if len(best_segments) > 1:
                            logger.info(f"   🎯 Alternative options:")
                            for i, seg in enumerate(best_segments[1:3], 2):
                                logger.info(f"      #{i}: {seg['start_time']:.1f}s-{seg['end_time']:.1f}s (score: {seg['avg_score']:.1f})")
                    else:
                        logger.warning(f"   ⚠️ No suitable video segment found for this duration")
                        
            except Exception as e:
                logger.error(f"❌ JSON analysis failed: {e}")
                import traceback
                traceback.print_exc()

async def test_gaming_metrics_comparison():
    """Compare old vs new gaming-optimized metrics on a sample."""
    logger.info("\n🔬 Testing Gaming Metrics - Quality Comparison")
    
    # Test with one video
    video_path = Path("src/video/data/footage/raw/6nQtYwUqRxM_Squirrel Girl ｜ Marvel Rivals Free To Use Gameplay.mp4")
    
    if not video_path.exists():
        logger.warning("⚠️ Test video not available for metrics comparison")
        return
    
    analyzer = VideoActionAnalyzer()
    
    try:
        # Sample analysis to show metric details
        logger.info(f"🎥 Analyzing: {video_path.name}")
        
        action_results = await analyzer.analyze_video_action(video_path)
        
        # Show detailed metrics for different action levels
        all_segments = action_results['high'] + action_results['medium'] + action_results['low']
        
        if all_segments:
            logger.info("\n📊 Gaming Metrics Breakdown:")
            
            # Show one example from each category
            for category in ['high', 'medium', 'low']:
                if action_results[category]:
                    segment = action_results[category][0]
                    logger.info(f"\n🎯 {category.upper()} Action Example (timestamp: {segment.timestamp:.1f}s):")
                    logger.info(f"   🏗️  Scene Complexity: {segment.scene_complexity:.1f}")
                    logger.info(f"   🎨 Color Variance: {segment.color_variance:.1f}")
                    logger.info(f"   🏃 Motion Intensity: {segment.motion_intensity:.1f}")
                    logger.info(f"   📐 Edge Density: {segment.edge_density:.1f}")
                    logger.info(f"   🎯 Overall Score: {segment.overall_score:.1f}")
                    logger.info(f"   🎮 Content Type: {segment.content_type}")
                    logger.info(f"   🔍 Confidence: {segment.confidence:.1f}")
            
            # Statistical summary
            scores = [s.overall_score for s in all_segments]
            logger.info(f"\n📈 Statistical Summary:")
            logger.info(f"   📊 Total segments analyzed: {len(all_segments)}")
            logger.info(f"   📈 Score range: {min(scores):.1f} - {max(scores):.1f}")
            logger.info(f"   📊 Average score: {sum(scores)/len(scores):.1f}")
            logger.info(f"   🎯 Action distribution: {len(action_results['high'])}/{len(action_results['medium'])}/{len(action_results['low'])} (H/M/L)")
            
            # Content type distribution
            content_types = {}
            for segment in all_segments:
                content_types[segment.content_type] = content_types.get(segment.content_type, 0) + 1
            
            logger.info(f"   🎮 Content types detected: {dict(content_types)}")
        
    except Exception as e:
        logger.error(f"❌ Metrics comparison failed: {e}")
        import traceback
        traceback.print_exc()

async def test_performance_benchmark():
    """Benchmark the performance improvements."""
    logger.info("\n⚡ Performance Benchmark")
    
    video_files = [
        "src/video/data/footage/raw/6nQtYwUqRxM_Squirrel Girl ｜ Marvel Rivals Free To Use Gameplay.mp4",
        "src/video/data/footage/raw/fBVpT-stY70_Deadzone Rouge Gameplay - Free To Use Gameplay.mp4",
        "src/video/data/footage/raw/ne_T-GNwObk_COD Black Ops 6 Gameplay - Free To Use.mp4"
    ]
    
    analyzer = VideoActionAnalyzer()
    total_time = 0
    successful_tests = 0
    
    for video_file in video_files:
        video_path = Path(video_file)
        
        if not video_path.exists():
            continue
            
        try:
            # Get video duration for context
            from moviepy import VideoFileClip
            clip = VideoFileClip(str(video_path))
            video_duration = clip.duration
            clip.close()
            
            logger.info(f"🎥 {video_path.name} ({video_duration/60:.1f} minutes)")
            
            start_time = time.time()
            await analyzer.analyze_video_action(video_path)
            analysis_time = time.time() - start_time
            
            total_time += analysis_time
            successful_tests += 1
            
            # Performance metrics
            time_ratio = analysis_time / video_duration
            logger.info(f"   ⚡ Analysis: {analysis_time:.1f}s ({time_ratio:.3f}x video duration)")
            
            if analysis_time > 300:  # 5 minutes
                logger.warning(f"   ⚠️ Analysis exceeded 5-minute target!")
            else:
                logger.success(f"   ✅ Under 5-minute target")
                
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")
    
    if successful_tests > 0:
        avg_time = total_time / successful_tests
        logger.info(f"\n📊 Performance Summary:")
        logger.info(f"   ✅ Successful tests: {successful_tests}")
        logger.info(f"   ⏱️ Average analysis time: {avg_time:.1f}s")
        logger.info(f"   🎯 Target achieved: {'✅ YES' if avg_time <= 300 else '❌ NO'}")

async def main():
    """Run all tests."""
    logger.info("🚀 Starting Gaming-Optimized Video Analyzer Tests")
    
    # Run tests in sequence
    await test_gaming_analyzer_with_json()
    await test_gaming_metrics_comparison() 
    await test_performance_benchmark()
    
    logger.info("\n🏁 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
