#!/usr/bin/env python3
"""
Quick test to check if full videos are being analyzed vs just time-limited portions.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.analyzers.action_analyzer import VideoActionAnalyzer
from loguru import logger

async def test_full_video_analysis():
    """Test if the analyzer covers the full video duration."""
    
    video_files = [
        "src/video/data/footage/raw/6nQtYwUqRxM_Squirrel Girl ｜ Marvel Rivals Free To Use Gameplay.mp4",
        "src/video/data/footage/raw/ne_T-GNwObk_COD Black Ops 6 Gameplay - Free To Use.mp4"
    ]
    
    analyzer = VideoActionAnalyzer()
    
    for video_file in video_files:
        video_path = Path(video_file)
        
        if not video_path.exists():
            continue
            
        # Get video duration
        try:
            from moviepy import VideoFileClip
            clip = VideoFileClip(str(video_path))
            video_duration = clip.duration
            clip.close()
            
            logger.info(f"\n🎥 Video: {video_path.name}")
            logger.info(f"📏 Total duration: {video_duration:.1f}s ({video_duration/60:.1f} minutes)")
            
            # Check analyzer settings
            logger.info(f"🔧 Analyzer settings:")
            logger.info(f"   Sample interval: {analyzer.sample_interval:.1f}s")
            logger.info(f"   Max samples per video: {analyzer.max_samples_per_video}")
            
            # Calculate what would be analyzed
            max_samples = min(analyzer.max_samples_per_video, int(video_duration / analyzer.sample_interval))
            if max_samples == 0:
                max_samples = 1
                
            # Check for long video adjustment
            if video_duration > 1800:  # 30+ minute videos
                adjusted_interval = video_duration / 20  # Max 20 samples
                logger.info(f"📹 Long video detected - would use {adjusted_interval:.1f}s intervals")
                actual_samples = 20
            else:
                actual_samples = max_samples
            
            # Calculate coverage
            import numpy as np
            timestamps = np.linspace(10, video_duration - 10, actual_samples)
            
            logger.info(f"📊 Analysis coverage:")
            logger.info(f"   Actual samples: {actual_samples}")
            logger.info(f"   First sample: {timestamps[0]:.1f}s")
            logger.info(f"   Last sample: {timestamps[-1]:.1f}s")
            logger.info(f"   Coverage: {((timestamps[-1] - timestamps[0]) / video_duration) * 100:.1f}% of video")
            logger.info(f"   Time between samples: {(timestamps[-1] - timestamps[0]) / (len(timestamps) - 1):.1f}s")
            
            # Test actual analysis
            logger.info(f"🔍 Running actual analysis...")
            import time
            start_time = time.time()
            
            results = await analyzer.analyze_video_action(video_path)
            
            analysis_time = time.time() - start_time
            total_segments = len(results['high']) + len(results['medium']) + len(results['low'])
            
            logger.info(f"⚡ Analysis completed:")
            logger.info(f"   Time taken: {analysis_time:.1f}s")
            logger.info(f"   Segments found: {total_segments}")
            logger.info(f"   Segments expected: {actual_samples}")
            logger.info(f"   Match: {'✅ YES' if total_segments == actual_samples else '❌ NO'}")
            
            # Check timestamp coverage
            all_segments = results['high'] + results['medium'] + results['low']
            if all_segments:
                actual_timestamps = [s.timestamp for s in all_segments]
                logger.info(f"📍 Actual timestamps:")
                logger.info(f"   First: {min(actual_timestamps):.1f}s")
                logger.info(f"   Last: {max(actual_timestamps):.1f}s")
                logger.info(f"   Full video analyzed: {'✅ YES' if max(actual_timestamps) > video_duration * 0.8 else '❌ NO'}")
                
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_full_video_analysis())
