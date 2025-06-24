#!/usr/bin/env python3
"""
Test the bubble subtitle system with the full video pipeline.

This tests integration with video processing and ensures
the bubble effects work end-to-end.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.subtitles import SubtitleGenerator
from loguru import logger


async def test_pipeline_integration():
    """Test how bubble subtitles integrate with the video pipeline."""
    logger.info("🎥 Testing bubble subtitle pipeline integration")
    
    generator = SubtitleGenerator()
    
    # Gaming content script
    gaming_script = (
        "Welcome to the most insane gaming session ever! "
        "Today we're diving into the most challenging level. "
        "Watch as I pull off this incredible combo move. "
        "Don't forget to smash that like button!"
    )
    
    # Test different bubble styles with realistic video specs
    test_configs = [
        {
            "style": "bubble_gaming",
            "duration": 12.0,
            "description": "Gaming style with lime colors"
        },
        {
            "style": "bubble_neon",
            "duration": 12.0,
            "description": "Neon style with cyan colors"
        },
        {
            "style": "bubble_classic",
            "duration": 12.0,
            "description": "Classic yellow bubble style"
        }
    ]
    
    for config in test_configs:
        logger.info(f"\n🎨 Testing {config['description']}")
        
        try:
            # Generate segments
            segments = await generator.generate_from_script(
                gaming_script,
                config["duration"],
                style_name=config["style"]
            )
            
            logger.info(f"  📝 Generated {len(segments)} segments")
            
            # Show timing breakdown
            total_coverage = 0
            for i, segment in enumerate(segments, 1):
                logger.info(f"    {i}: '{segment.text[:30]}...' ({segment.duration:.1f}s)")
                total_coverage += segment.duration
            
            logger.info(f"  ⏱️  Total coverage: {total_coverage:.1f}s / {config['duration']:.1f}s")
            
            # Create video clips
            clips = await generator.create_subtitle_clips(
                segments, 
                video_width=1080, 
                video_height=1920
            )
            
            logger.success(f"  ✅ Created {len(clips)} video clips")
            
            # Export formats
            srt_file = await generator.export_srt(segments, f"pipeline_test_{config['style']}.srt")
            json_file = await generator.export_json(segments, f"pipeline_test_{config['style']}.json")
            
            logger.info(f"  📄 Exported: {Path(srt_file).name}")
            logger.info(f"  📊 Exported: {Path(json_file).name}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
    
    logger.success("🎉 Pipeline integration testing completed!")


async def test_tiktok_timing():
    """Test subtitle timing optimized for TikTok content."""
    logger.info("\n⏰ Testing TikTok-optimized timing")
    
    generator = SubtitleGenerator()
    
    # Short, punchy TikTok content
    tiktok_scripts = [
        {
            "text": "This gaming trick will blow your mind!",
            "duration": 3.0,
            "style": "bubble_gaming"
        },
        {
            "text": "Wait for it... BOOM! Did you see that incredible headshot?",
            "duration": 5.0,
            "style": "bubble_neon"
        },
        {
            "text": "Three simple steps to dominate any game. First, master your controls. Second, study your opponents. Third, practice every single day!",
            "duration": 8.0,
            "style": "bubble_classic"
        }
    ]
    
    for test in tiktok_scripts:
        logger.info(f"\n📱 Testing: '{test['text'][:40]}...'")
        
        segments = await generator.generate_from_script(
            test["text"],
            test["duration"],
            style_name=test["style"]
        )
        
        logger.info(f"  📝 Generated {len(segments)} segments for {test['duration']}s content")
        
        # Analyze timing
        for i, segment in enumerate(segments, 1):
            reading_speed = len(segment.text) / segment.duration if segment.duration > 0 else 0
            logger.info(f"    Segment {i}: {reading_speed:.1f} chars/sec - '{segment.text}'")
        
        # Check if timing is appropriate for TikTok (not too fast, not too slow)
        avg_speed = sum(len(s.text) / s.duration for s in segments if s.duration > 0) / len(segments)
        
        if 8 <= avg_speed <= 15:  # Optimal reading speed for mobile
            logger.success(f"  ✅ Good pacing: {avg_speed:.1f} chars/sec")
        elif avg_speed < 8:
            logger.warning(f"  ⚠️  Slow pacing: {avg_speed:.1f} chars/sec")
        else:
            logger.warning(f"  ⚠️  Fast pacing: {avg_speed:.1f} chars/sec")
    
    logger.success("🎉 TikTok timing tests completed!")


async def main():
    """Main test function."""
    try:
        await test_pipeline_integration()
        await test_tiktok_timing()
        
        logger.success("🎉 All bubble subtitle integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Integration tests failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
