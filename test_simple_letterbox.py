#!/usr/bin/env python3
"""
Simple letterboxing test to isolate the issue.
Test the letterboxing method directly on a video file.
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from moviepy import VideoFileClip, ColorClip, CompositeVideoClip
except ImportError as e:
    logger.error(f"MoviePy import failed: {e}")
    sys.exit(1)

def test_letterbox_simple():
    """Test letterboxing directly on a video file."""
    
    logger.info("🎬 Testing simple letterboxing...")
      # Input video (landscape gaming footage)
    input_video = "src/video/data/footage/raw/6nQtYwUqRxM_Squirrel Girl ｜ Marvel Rivals Free To Use Gameplay.mp4"
    
    if not Path(input_video).exists():
        logger.error(f"❌ Input video not found: {input_video}")
        return False
    
    try:
        # Load source video
        source_clip = VideoFileClip(input_video)
        logger.info(f"📺 Source: {source_clip.w}x{source_clip.h} (aspect: {source_clip.h/source_clip.w:.3f})")
        
        # Target dimensions (TikTok format)
        target_width = 1080
        target_height = 1920
        target_aspect = target_height / target_width  # 1.778
        
        # Source dimensions
        source_width = source_clip.w
        source_height = source_clip.h
        source_aspect = source_height / source_width
        
        logger.info(f"🎯 Target: {target_width}x{target_height} (aspect: {target_aspect:.3f})")
        
        # Take only first 10 seconds for testing
        test_clip = source_clip.subclipped(0, 10)
        
        # Manual letterboxing implementation
        if source_aspect < target_aspect:
            # Source is wider - fit by height, add vertical bars
            logger.info("📏 Source is wider - fitting by height, adding vertical bars")
            
            # Scale to fit height
            scale_factor = target_height / source_height
            new_width = int(source_width * scale_factor)
            new_height = target_height
            
            logger.info(f"🔄 Scaling to {new_width}x{new_height}")
            
            # Resize the video
            scaled_clip = test_clip.resized(new_size=(new_width, new_height))
            
            # Calculate horizontal offset to center
            x_offset = (target_width - new_width) // 2
            
            logger.info(f"📍 Centering with x_offset: {x_offset}")
            
            # Create black background
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0),
                duration=test_clip.duration
            )
            
            # Composite
            if x_offset >= 0:
                # Normal letterboxing - video fits within target width
                final_clip = CompositeVideoClip([
                    background,
                    scaled_clip.with_position((x_offset, 0))
                ])
            else:
                # Video is still too wide - crop it
                logger.warning(f"⚠️ Video still too wide after scaling, cropping {abs(x_offset)} pixels from each side")
                crop_amount = abs(x_offset)
                cropped_clip = scaled_clip.cropped(x1=crop_amount, x2=new_width-crop_amount)
                final_clip = CompositeVideoClip([
                    background,
                    cropped_clip.with_position((0, 0))
                ])
                
        else:
            # Source is taller - fit by width, add horizontal bars
            logger.info("📏 Source is taller - fitting by width, adding horizontal bars")
            
            # Scale to fit width
            scale_factor = target_width / source_width
            new_width = target_width
            new_height = int(source_height * scale_factor)
            
            logger.info(f"🔄 Scaling to {new_width}x{new_height}")
            
            # Resize the video
            scaled_clip = test_clip.resized(new_size=(new_width, new_height))
            
            # Calculate vertical offset to center
            y_offset = (target_height - new_height) // 2
            
            logger.info(f"📍 Centering with y_offset: {y_offset}")
            
            # Create black background
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0),
                duration=test_clip.duration
            )
            
            # Composite
            final_clip = CompositeVideoClip([
                background,
                scaled_clip.with_position((0, y_offset))
            ])
        
        # Remove audio for simplicity
        final_clip = final_clip.without_audio()
        
        # Output
        output_path = "simple_letterbox_test.mp4"
        logger.info(f"💾 Exporting to {output_path}...")
        
        final_clip.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac' if final_clip.audio else None
        )
        
        # Verify output
        output_clip = VideoFileClip(output_path)
        logger.info(f"✅ Output: {output_clip.w}x{output_clip.h} (aspect: {output_clip.h/output_clip.w:.3f})")
        
        # Cleanup
        source_clip.close()
        test_clip.close()
        final_clip.close()
        output_clip.close()
        
        logger.success(f"🎉 Simple letterboxing test completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Letterboxing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    success = test_letterbox_simple()
    sys.exit(0 if success else 1)
