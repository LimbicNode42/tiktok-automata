#!/usr/bin/env python3
"""
Test longer segment to ensure the optimization resize also works with even dimensions.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_long_segment():
    """Test dimension fixes with a longer segment that triggers optimization."""
    
    try:
        from src.video.processors.segment_processor import VideoSegmentProcessor
        
        print("📏 Testing long segment dimension fixes...")
        
        # Initialize processor
        processed_dir = Path("src/video/data/footage/processed")
        processor = VideoSegmentProcessor(processed_dir)
        
        # Find a gaming video
        raw_dir = Path("src/video/data/footage/raw")
        video_files = list(raw_dir.glob("*.mp4"))
        
        if not video_files:
            print("❌ No gaming videos found")
            return False
        
        video_file = video_files[0]
        print(f"🎮 Using gaming video: {video_file.stem}")
        
        # Load video clip
        print("📺 Loading video clip...")
        from moviepy import VideoFileClip
        
        with VideoFileClip(str(video_file)) as clip:
            print(f"🎯 Original video: {clip.size[0]}x{clip.size[1]}, {clip.duration:.1f}s")
            
            # Create a LONG segment (over 30s to trigger optimization)
            duration_info = {
                'index': 1,
                'title': 'Long_Segment_Test',
                'start_time': 30.0,
                'end_time': 65.0  # 35 second segment (> 30s threshold)
            }
            
            print("🎬 Creating long segment (35s) to test optimization resize...")
            
            # Create long segment
            segment_result = asyncio.run(processor._create_individual_segment(
                clip, 
                video_file.stem, 
                35.0,  # 35 second duration
                duration_info, 
                clip.duration
            ))
            
            if segment_result:
                file_size = segment_result.stat().st_size / (1024 * 1024)
                print(f"✅ Long segment created successfully: {segment_result.name}")
                print(f"📊 File size: {file_size:.1f}MB")
                
                # Clean up test file
                segment_result.unlink()
                print("🧹 Cleaned up test file")
                
                return True
            else:
                print("❌ Failed to create long segment")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_long_segment()
    if success:
        print("\n🎉 Long segment dimension test passed!")
    else:
        print("\n💥 Long segment dimension test failed!")
        sys.exit(1)
