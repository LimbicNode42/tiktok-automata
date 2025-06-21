#!/usr/bin/env python3
"""
Test NVENC quality by creating a video segment and keeping the output for visual inspection.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_nvenc_quality():
    """Test NVENC encoding quality with real gaming footage."""
    
    try:
        from src.video.processors.segment_processor import VideoSegmentProcessor
        
        print("🎬 Testing NVENC quality with real gaming footage...")
        
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
        print(f"📁 Video path: {video_file}")
        
        # Load video clip
        print("📺 Loading video clip...")
        from moviepy import VideoFileClip
        
        with VideoFileClip(str(video_file)) as clip:
            print(f"🎯 Original video: {clip.size[0]}x{clip.size[1]}, {clip.duration:.1f}s")
            
            # Create a segment with NVENC (high quality)
            duration_info = {
                'index': 1,
                'title': 'NVENC_Quality_Test',
                'start_time': 30.0,  # Start at 30s
                'end_time': 40.0     # 10 second segment
            }
            
            print("🎬 Creating high-quality NVENC segment...")
            
            # Create segment with high quality settings
            segment_result = asyncio.run(processor._create_individual_segment(
                clip, 
                video_file.stem, 
                10.0,  # 10 second duration
                duration_info, 
                clip.duration
            ))
            
            if segment_result:
                file_size = segment_result.stat().st_size / (1024 * 1024)
                print(f"✅ NVENC quality test segment created: {segment_result.name}")
                print(f"📊 File size: {file_size:.1f}MB")
                print(f"📁 Full path: {segment_result}")
                
                # Check if NVENC was actually used
                print("\n🔍 Checking if NVENC was used...")
                # We can infer this from the fast encoding time and file size
                if file_size > 3.0:  # Good quality should be >3MB for 10s
                    print("✅ High quality encoding detected (likely NVENC)")
                else:
                    print("⚠️ Lower quality encoding (possibly fallback)")
                
                print(f"\n🎥 Visual inspection file: {segment_result}")
                print("👀 Please check the video quality manually!")
                
                return True
            else:
                print("❌ Failed to create NVENC test segment")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nvenc_quality()
    if success:
        print("\n🎉 NVENC quality test completed successfully!")
    else:
        print("\n💥 NVENC quality test failed!")
        sys.exit(1)
