#!/usr/bin/env python3
"""
Simple test to create MP4 video with bubble subtitles.

This creates a basic video with bubble subtitles embedded.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.subtitles import SubtitleGenerator
from loguru import logger

# Try importing MoviePy with error handling
try:
    from moviepy.editor import ColorClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
    logger.info("✅ MoviePy imported successfully")
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    logger.error(f"❌ MoviePy import failed: {e}")


async def create_simple_bubble_video():
    """Create a simple MP4 video with bubble subtitles."""
    if not MOVIEPY_AVAILABLE:
        logger.error("❌ Cannot create video without MoviePy")
        return
    
    logger.info("🎬 Creating simple bubble subtitle video")
    
    # Video settings
    width = 1080
    height = 1920
    duration = 5.0
    fps = 30
    
    # Test content
    test_text = "EPIC GAMING MOMENT! This trick will blow your mind completely!"
    style = "bubble_gaming"
    
    try:
        # Step 1: Generate subtitles
        logger.info("📝 Generating bubble subtitles...")
        generator = SubtitleGenerator()
        
        segments = await generator.generate_from_script(
            test_text,
            duration,
            style_name=style
        )
        
        logger.info(f"   ✅ Generated {len(segments)} subtitle segments")
        for i, segment in enumerate(segments, 1):
            logger.info(f"      {i}: '{segment.text}' ({segment.start_time:.1f}s-{segment.end_time:.1f}s)")
        
        # Step 2: Create subtitle clips
        logger.info("🎨 Creating subtitle clips...")
        subtitle_clips = await generator.create_subtitle_clips(segments, width, height)
        logger.info(f"   ✅ Created {len(subtitle_clips)} subtitle clips")
        
        # Step 3: Create background
        logger.info("🎬 Creating background...")
        background = ColorClip(
            size=(width, height),
            color=(20, 50, 20),  # Dark green for gaming
            duration=duration
        )
        logger.info("   ✅ Background created")
        
        # Step 4: Composite video
        logger.info("🔗 Compositing video...")
        all_clips = [background] + subtitle_clips
        final_video = CompositeVideoClip(all_clips, size=(width, height))
        logger.info("   ✅ Video composed")
        
        # Step 5: Export video
        output_dir = Path(__file__).parent / "output_videos"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"Bubble_Subtitle_Demo_{style}.mp4"
        
        logger.info(f"💾 Exporting video to: {output_file}")
        logger.info("   ⏳ This may take a moment...")
        
        final_video.write_videofile(
            str(output_file),
            fps=fps,
            codec='libx264',
            verbose=False,
            logger=None,
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        # Check result
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            logger.success(f"🎉 Video created successfully!")
            logger.info(f"   📁 File: {output_file}")
            logger.info(f"   📊 Size: {file_size:.1f} MB")
            logger.info(f"   📱 Resolution: {width}x{height}")
            logger.info(f"   ⏱️  Duration: {duration}s")
            logger.info(f"   🎨 Style: {style}")
        else:
            logger.error("❌ Video file was not created")
        
        # Clean up
        final_video.close()
        background.close()
        for clip in subtitle_clips:
            clip.close()
            
        # Also export subtitle files
        await generator.export_srt(segments, f"video_{style}.srt")
        logger.info(f"   📄 SRT file exported")
        
    except Exception as e:
        logger.error(f"❌ Failed to create video: {e}")
        import traceback
        logger.error(traceback.format_exc())


async def main():
    """Main function."""
    await create_simple_bubble_video()
    
    logger.success("🎉 Test completed!")
    logger.info("📁 Check the 'output_videos' folder for the MP4 file")


if __name__ == "__main__":
    asyncio.run(main())
