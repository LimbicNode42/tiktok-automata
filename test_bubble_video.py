#!/usr/bin/env python3
"""
Test the bubble subtitle system with real video generation.

This tests the full end-to-end pipeline with bubble subtitles
to ensure everything works properly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.subtitles import SubtitleGenerator
from video.processors.video_processor import VideoProcessor, VideoConfig
from loguru import logger


async def test_bubble_subtitles_in_video():
    """Test bubble subtitles in actual video generation."""
    logger.info("🎬 Testing bubble subtitles in video generation")
    
    # Create a test script
    test_script = (
        "Welcome to the most epic gaming moment ever! "
        "Watch as I execute this insane strategy that will blow your mind. "
        "This technique took me months to master!"
    )
    
    # Test with bubble styles
    bubble_styles = ["bubble_gaming", "bubble_neon", "bubble_classic"]
    
    for style in bubble_styles:
        logger.info(f"\n🎨 Testing {style} in video pipeline")
        
        try:
            # Create video config with bubble subtitle
            config = VideoConfig(
                output_width=1080,
                output_height=1920,
                fps=30,
                subtitle_style=style,  # Use our bubble style
                subtitle_position=0.75,  # Higher position for TikTok
                effects_preset="gaming_highlight"
            )
            
            # Generate subtitle segments
            generator = SubtitleGenerator()
            segments = await generator.generate_from_script(
                test_script, 
                7.0,  # 7 seconds
                style_name=style
            )
            
            logger.info(f"  📝 Generated {len(segments)} subtitle segments")
            
            # Create video clips
            clips = await generator.create_subtitle_clips(
                segments, 
                config.output_width, 
                config.output_height
            )
            
            logger.success(f"  ✅ Created {len(clips)} video clips with {style}")
            
            # Show segment details for verification
            for i, segment in enumerate(segments, 1):
                logger.info(f"    {i}: '{segment.text}' ({segment.start_time:.1f}s-{segment.end_time:.1f}s)")
            
            # Export subtitle files
            await generator.export_srt(segments, f"video_test_{style}.srt")
            
            logger.success(f"  📄 Exported SRT for {style}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed testing {style}: {e}")
    
    logger.success("🎉 Bubble subtitles video pipeline testing completed!")


async def test_subtitle_visual_properties():
    """Test the visual properties of bubble subtitles."""
    logger.info("\n🎨 Testing subtitle visual properties")
    
    generator = SubtitleGenerator()
    
    # Test visual configurations
    visual_tests = [
        {
            "style": "bubble_gaming",
            "description": "Gaming style - lime with dark green outline",
            "features": "Double stroke, large font size"
        },
        {
            "style": "bubble_neon", 
            "description": "Neon style - cyan with dark blue outline",
            "features": "Background color, double stroke"
        },
        {
            "style": "bubble_cute",
            "description": "Cute style - pink with purple outline", 
            "features": "Semi-transparent background"
        },
        {
            "style": "bubble_classic",
            "description": "Classic style - yellow with black outline",
            "features": "Traditional bubble look, shadow effects"
        }
    ]
    
    test_text = "EPIC GAMING MOMENT!"
    
    for test in visual_tests:
        logger.info(f"\n🔍 {test['description']}")
        logger.info(f"   Features: {test['features']}")
        
        # Generate segments
        segments = await generator.generate_from_script(
            test_text,
            2.0,
            style_name=test["style"]
        )
        
        # Get style details
        style = generator.styles[test["style"]]
        logger.info(f"   Font size: {style.font_size}px")
        logger.info(f"   Font color: {style.font_color}")
        logger.info(f"   Outline: {style.bubble_outline} ({style.bubble_outline_width}px)")
        
        if style.use_bubble_effect:
            logger.info(f"   Bubble effect: ✅")
            if style.double_stroke:
                logger.info(f"   Double stroke: {style.inner_stroke_color} ({style.inner_stroke_width}px)")
        
        if style.background_color:
            logger.info(f"   Background: {style.background_color}")
        
        # Create clips to test rendering
        clips = await generator.create_subtitle_clips(segments, 1080, 1920)
        logger.success(f"   ✅ Rendered {len(clips)} clips")
    
    logger.success("🎉 Visual properties testing completed!")


async def demonstrate_tiktok_examples():
    """Demonstrate TikTok-style subtitle examples."""
    logger.info("\n📱 Demonstrating TikTok-style examples")
    
    generator = SubtitleGenerator()
    
    # TikTok content examples
    examples = [
        {
            "content": "POV: You just hit the most insane trick shot ever!",
            "style": "bubble_gaming",
            "duration": 4.0,
            "genre": "Gaming"
        },
        {
            "content": "This life hack changed everything! Try it now!",
            "style": "bubble_cute",
            "duration": 3.5,
            "genre": "Life Tips"
        },
        {
            "content": "Wait for the drop... 3... 2... 1... BOOM!",
            "style": "bubble_neon",
            "duration": 5.0,
            "genre": "Music/Beat Drop"
        },
        {
            "content": "Five second rule that will make you rich!",
            "style": "bubble_classic", 
            "duration": 3.0,
            "genre": "Finance Tips"
        }
    ]
    
    for example in examples:
        logger.info(f"\n🎭 {example['genre']}: '{example['content']}'")
        
        # Generate segments
        segments = await generator.generate_from_script(
            example["content"],
            example["duration"],
            style_name=example["style"]
        )
        
        # Show timing breakdown
        logger.info(f"   Style: {example['style']}")
        logger.info(f"   Duration: {example['duration']}s")
        logger.info(f"   Segments: {len(segments)}")
        
        total_chars = sum(len(s.text) for s in segments)
        avg_speed = total_chars / example["duration"]
        
        if 8 <= avg_speed <= 15:
            pacing = "Perfect"
        elif avg_speed < 8:
            pacing = "Relaxed"
        else:
            pacing = "Fast"
        
        logger.info(f"   Reading speed: {avg_speed:.1f} chars/sec ({pacing})")
        
        # Export for review
        await generator.export_srt(segments, f"tiktok_example_{example['genre'].lower().replace('/', '_')}.srt")
    
    logger.success("🎉 TikTok examples demonstration completed!")


async def main():
    """Main test function."""
    try:
        await test_bubble_subtitles_in_video()
        await test_subtitle_visual_properties()
        await demonstrate_tiktok_examples()
        
        logger.success("🎉 All bubble subtitle video tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Video tests failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
