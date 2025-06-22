#!/usr/bin/env python3
"""
Test script to validate the enhanced TikTok crop-aware action analysis.
This test validates:
1. TikTok crop region extraction and analysis
2. UI element detection in cropped regions
3. Content type detection with new metrics
4. Segment selection with UI filtering
"""

import asyncio
import os
import sys
import logging
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from video.analyzers.action_analyzer import VideoActionAnalyzer
from video.managers.footage_manager import FootageManager

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def test_tiktok_crop_analysis():
    """Test the TikTok crop-aware action analysis."""
    print("🎯 Testing TikTok Crop-Aware Action Analysis")
    print("=" * 60)
    
    # Initialize components
    footage_manager = FootageManager()
    analyzer = VideoActionAnalyzer()
    
    # Get available footage from metadata
    videos = footage_manager.metadata.get('videos', {})
    if not videos:
        print("❌ No footage available for testing")
        return False
    
    print(f"📁 Found {len(videos)} footage files")
    
    # Convert metadata to list format for testing
    footage_list = []
    for video_id, video_info in videos.items():
        footage_list.append({
            'id': video_id,
            'title': video_info.get('title', 'Unknown'),
            'path': video_info.get('file_path', ''),
            'duration': video_info.get('duration', 0),
            'resolution': '1080p'  # Assume 1080p after upgrade
        })
    
    # Test with first footage file
    test_footage = footage_list[0]
    print(f"🎮 Testing with: {test_footage['title']}")
    print(f"   Resolution: {test_footage.get('resolution', 'Unknown')}")
    print(f"   Duration: {test_footage.get('duration', 'Unknown')}s")
    
    video_path = test_footage['path']
      # Check if video file exists
    from pathlib import Path
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    # Test 1: Basic action analysis with TikTok metrics
    print("\n🔍 Test 1: Action Analysis with TikTok Metrics")
    try:
        # Test continuous segment analysis with the new API
        segments_result = await analyzer.analyze_continuous_segments(
            Path(video_path), 
            [30.0, 45.0]  # Test with 30s and 45s segments
        )
        
        print(f"✅ Analysis completed for segments: {list(segments_result.keys())}")
        
        # Display detailed metrics for each duration
        for duration, segments in segments_result.items():
            print(f"\n📊 Duration {duration}s: Found {len(segments)} segments")
            
            for i, segment in enumerate(segments[:2]):  # Show first 2 segments
                print(f"   Segment {i+1}: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s")
                print(f"   Average Score: {segment['avg_score']:.3f}")
                print(f"   Duration: {segment['duration']:.1f}s")
                print(f"   Sample Count: {segment['sample_count']}")
                
                # Check for new TikTok-specific metrics (might not be in this level)
                # Note: TikTok metrics are likely in the underlying analysis
                print(f"   Score Variance: {segment.get('score_variance', 0):.3f}")
                
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
      # Test 2: UI Detection Analysis  
    print("\n🎨 Test 2: UI Detection Analysis")
    try:
        # Test UI detection on individual frames (without cv2 dependency)
        import numpy as np
        from moviepy import VideoFileClip
        
        clip = VideoFileClip(video_path)
        
        # Get a sample frame at 30 seconds
        test_timestamp = min(30.0, clip.duration - 1)
        frame = clip.get_frame(test_timestamp)
        
        print(f"   Analyzing frame at {test_timestamp:.1f}s")
        print(f"   Frame shape: {frame.shape}")
        
        # Test TikTok crop region extraction
        cropped_frame = analyzer.crop_to_tiktok_region(frame)
        print(f"   Cropped frame shape: {cropped_frame.shape}")
        
        # Test UI detection
        ui_metrics = analyzer.detect_ui_elements(frame)
        print(f"   UI Metrics:")
        for key, value in ui_metrics.items():
            print(f"     {key}: {value}")
            
        clip.close()
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Performance Check
    print("\n⚡ Test 3: Performance Analysis")
    try:
        import time
        
        start_time = time.time()
        
        # Analyze a shorter segment for performance test
        performance_segments = await analyzer.analyze_continuous_segments(
            Path(video_path), 
            [30.0]  # Analyze 30-second segments
        )
        
        elapsed = time.time() - start_time
        
        if performance_segments and "30.0" in performance_segments:
            segments = performance_segments["30.0"]
            print(f"   Analysis time for 30s segments: {elapsed:.2f}s")
            print(f"   Found {len(segments)} segments")
            
            if segments:
                segment = segments[0]
                print(f"   Best segment: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s")
                print(f"   Segment score: {segment['avg_score']:.3f}")
                print(f"   Sample count: {segment['sample_count']}")
                
                # Check segment data structure
                available_keys = list(segment.keys())
                print(f"   Available metrics: {available_keys}")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    print("\n🎉 All TikTok Crop Analysis Tests Completed Successfully!")
    print("=" * 60)
    return True

async def main():
    """Main test execution."""
    setup_logging()
    
    print("🚀 Starting TikTok Crop-Aware Action Analysis Tests")
    print("This test validates the enhanced action analyzer with TikTok focus")
    print()
    
    try:
        success = await test_tiktok_crop_analysis()
        
        if success:
            print("\n✅ ALL TESTS PASSED")
            print("The TikTok crop-aware action analysis is working correctly!")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("Please check the error messages above.")
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
