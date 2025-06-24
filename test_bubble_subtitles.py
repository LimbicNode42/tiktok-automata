#!/usr/bin/env python3
"""
Test script for the enhanced TikTok-style bubble subtitle system.

This script tests the new bubble effects and curved/bubbly fonts
for creating more visually appealing TikTok subtitles.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.subtitles import SubtitleGenerator, SubtitleStyle
from loguru import logger


async def test_bubble_styles():
    """Test the new bubble subtitle styles."""
    logger.info("🧪 Testing TikTok-style bubble subtitle system")
    
    # Initialize subtitle generator
    generator = SubtitleGenerator()
    
    # Test text
    test_script = (
        "This is an amazing gaming moment that will blow your mind! "
        "Check out this incredible play and don't forget to like and subscribe "
        "for more epic gaming content."
    )
    
    audio_duration = 8.0  # 8 seconds
    
    # Test all bubble styles
    bubble_styles = ["bubble", "bubble_blue", "bubble_gaming", "bubble_cute"]
    
    logger.info("Available subtitle styles:")
    for style_name in generator.get_style_names():
        logger.info(f"  - {style_name}")
    
    for style in bubble_styles:
        logger.info(f"\n🎨 Testing style: {style}")
        try:
            # Generate subtitle segments
            segments = await generator.generate_from_script(
                test_script, 
                audio_duration, 
                style_name=style
            )
            
            logger.success(f"✅ Generated {len(segments)} segments for {style}")
            
            # Display segment details
            for i, segment in enumerate(segments, 1):
                logger.info(f"  Segment {i}: '{segment.text}' ({segment.start_time:.1f}s - {segment.end_time:.1f}s)")
            
            # Export to SRT
            srt_path = await generator.export_srt(segments, f"test_{style}.srt")
            logger.info(f"  📄 Exported SRT: {srt_path}")
            
            # Test video clip creation (if MoviePy is available)
            try:
                clips = await generator.create_subtitle_clips(segments, 1080, 1920)
                logger.success(f"✅ Created {len(clips)} video clips for {style}")
            except Exception as e:
                logger.warning(f"  ⚠️ Video clip creation failed (MoviePy may not be available): {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to test style {style}: {e}")
    
    logger.success("🎉 Bubble subtitle testing completed!")


async def test_custom_bubble_style():
    """Test creating a custom bubble style."""
    logger.info("\n🎨 Testing custom bubble style creation")
    
    generator = SubtitleGenerator()
    
    # Create a custom gaming bubble style
    custom_style = SubtitleStyle(
        font_size=72,
        font_color="cyan",
        stroke_width=8,
        stroke_color="darkblue",
        font_family="Impact",
        use_bubble_effect=True,
        bubble_color="cyan",
        bubble_outline="darkblue",
        bubble_outline_width=10,
        double_stroke=True,
        inner_stroke_color="white",
        inner_stroke_width=3,
        background_color="rgba(0,100,200,0.8)"
    )
    
    # Add the custom style
    generator.add_custom_style("custom_gaming_bubble", custom_style)
    
    # Test the custom style
    test_script = "EPIC GAMING MOMENT! This is absolutely insane!"
    segments = await generator.generate_from_script(
        test_script, 
        4.0, 
        style_name="custom_gaming_bubble"
    )
    
    logger.success(f"✅ Custom style generated {len(segments)} segments")
    
    # Export to test
    await generator.export_srt(segments, "test_custom_gaming_bubble.srt")
    await generator.export_json(segments, "test_custom_gaming_bubble.json")
    
    logger.success("🎉 Custom bubble style testing completed!")


async def demonstrate_font_options():
    """Demonstrate different font options for bubble effects."""
    logger.info("\n🔤 Demonstrating font options for bubble effects")
    
    fonts_to_test = [
        "Comic Sans MS",    # Rounded, friendly
        "Cooper Black",     # Bold, rounded
        "Impact",           # Bold, gaming
        "Trebuchet MS",     # Clean, rounded
        "Arial Rounded MT Bold",  # Professional rounded
        "Chalkduster",      # Fun, textured (Mac)
        "Marker Felt",      # Casual, handwritten (Mac)
    ]
    
    generator = SubtitleGenerator()
    
    for font in fonts_to_test:
        logger.info(f"  📝 Font: {font}")
        
        # Create style with this font
        style = SubtitleStyle(
            font_size=66,
            font_color="white",
            stroke_width=6,
            stroke_color="black",
            font_family=font,
            use_bubble_effect=True,
            bubble_outline_width=8
        )
        
        style_name = f"test_{font.lower().replace(' ', '_')}"
        generator.add_custom_style(style_name, style)
    
    logger.info("📚 Font options demonstrated - ready for testing!")


async def main():
    """Main test function."""
    try:
        await test_bubble_styles()
        await test_custom_bubble_style()
        await demonstrate_font_options()
        
        logger.success("🎉 All bubble subtitle tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
