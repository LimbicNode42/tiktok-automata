#!/usr/bin/env python3
"""
Test script to verify that the video assignment fix works correctly.

This script tests:
1. Video assignment to articles
2. VideoProcessor using the assigned video ID
3. Metadata accuracy with actual video used
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.processors.video_processor import VideoProcessor, VideoConfig

async def test_video_assignment():
    """Test that video assignment works correctly."""
    
    print("🧪 Testing Video Assignment Fix")
    print("=" * 50)
    
    # Check if we have raw footage available
    storage_dir = Path("storage") / "raw"
    if not storage_dir.exists():
        storage_dir = Path("src") / "video" / "data" / "footage" / "raw"
    
    footage_files = list(storage_dir.glob("*.mp4"))
    
    if not footage_files:
        print("❌ No footage files found for testing")
        return
    
    print(f"📁 Found {len(footage_files)} footage files:")
    for f in footage_files[:3]:  # Show first 3
        print(f"  - {f.name}")
    
    # Test with assigned video ID
    test_assigned_video_id = footage_files[0].stem
    print(f"\n🎯 Testing with assigned video ID: {test_assigned_video_id}")
    
    # Create video processor
    video_config = VideoConfig(
        enable_subtitles=True,
        subtitle_style="highlight",
        output_quality="medium",
        export_srt=False,
        duration=30
    )
    
    processor = VideoProcessor(video_config)
    
    # Test create_video with assigned video ID
    # Note: We need a valid TTS audio file for this test
    # For now, just test the video selection logic
    
    # Test _select_raw_gaming_footage directly
    print("\n📹 Testing video selection with assigned ID...")
    
    try:
        gaming_video = await processor._select_raw_gaming_footage(
            duration=30.0,
            content_analysis=None,
            assigned_video_id=test_assigned_video_id
        )
        
        if gaming_video:
            print(f"✅ Video selected successfully!")
            print(f"📊 Video specs: {gaming_video.w}x{gaming_video.h}, {gaming_video.duration:.1f}s")
            
            # Check if the actual video used matches the assigned ID
            actual_video_used = getattr(processor, 'actual_video_used', None)
            if actual_video_used:
                print(f"🎯 Actual video used: {actual_video_used}")
                
                if actual_video_used == test_assigned_video_id:
                    print("✅ SUCCESS: Assigned video ID matches actual video used!")
                else:
                    print(f"⚠️  WARNING: Assigned video ID ({test_assigned_video_id}) != actual video used ({actual_video_used})")
            else:
                print("❌ No actual_video_used tracked")
            
            # Clean up
            gaming_video.close()
            
        else:
            print("❌ No video selected")
            
    except Exception as e:
        print(f"❌ Error testing video selection: {e}")
        import traceback
        traceback.print_exc()
    
    # Test without assigned video ID (random selection)
    print("\n🎲 Testing random video selection...")
    
    try:
        gaming_video = await processor._select_raw_gaming_footage(
            duration=30.0,
            content_analysis=None,
            assigned_video_id=None
        )
        
        if gaming_video:
            print(f"✅ Random video selected successfully!")
            print(f"📊 Video specs: {gaming_video.w}x{gaming_video.h}, {gaming_video.duration:.1f}s")
            
            actual_video_used = getattr(processor, 'actual_video_used', None)
            if actual_video_used:
                print(f"🎯 Random video used: {actual_video_used}")
            
            # Clean up
            gaming_video.close()
            
        else:
            print("❌ No random video selected")
            
    except Exception as e:
        print(f"❌ Error testing random selection: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_video_assignment())
