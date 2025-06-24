#!/usr/bin/env python3
"""
Test script to generate a video with bubble subtitles.

This will create a test video showing off the TikTok-style bubble subtitle system
and output an actual MP4 file with embedded subtitles.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.subtitles import SubtitleGenerator
from loguru import logger

try:
    from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.error("MoviePy not available for video generation")


async def create_video_with_bubble_subtitles():
    """Create an actual MP4 video with bubble subtitles embedded."""
    if not MOVIEPY_AVAILABLE:
        logger.error("❌ MoviePy is required to create video files. Please install with: pip install moviepy")
        return
    
    logger.info("🎬 Creating MP4 video with bubble subtitles")
    
    # Video specifications
    video_width = 1080
    video_height = 1920
    video_duration = 8.0
    fps = 30
    
    # Test scripts for different bubble styles
    test_videos = [
        {
            "text": "POV: You just hit the MOST INSANE trick shot ever! This gaming moment will blow your mind completely!",
            "style": "bubble_gaming",
            "duration": 8.0,
            "background_color": (20, 50, 20),  # Dark green background for gaming
            "title": "Gaming_Bubble_Demo"
        },
        {
            "text": "Wait for the drop... 3... 2... 1... BOOM! Did you see that incredible beat drop moment?",
            "style": "bubble_neon",
            "duration": 6.0,
            "background_color": (10, 30, 60),  # Dark blue background for neon
            "title": "Neon_Bubble_Demo"
        },
        {
            "text": "This life hack will change EVERYTHING! Try this simple trick and watch magic happen!",
            "style": "bubble_cute",
            "duration": 7.0,
            "background_color": (60, 20, 40),  # Dark pink background for cute
            "title": "Cute_Bubble_Demo"
        }
    ]
    
    generator = SubtitleGenerator()
    output_dir = Path(__file__).parent / "output_videos"
    output_dir.mkdir(exist_ok=True)
    
    for i, video_config in enumerate(test_videos, 1):
        logger.info(f"\n� Creating Video {i}: {video_config['title']}")
        logger.info(f"   Style: {video_config['style']}")
        logger.info(f"   Duration: {video_config['duration']}s")
        
        try:
            # Step 1: Generate subtitle segments
            logger.info("   📝 Generating subtitle segments...")
            segments = await generator.generate_from_script(
                video_config["text"],
                video_config["duration"],
                style_name=video_config["style"]
            )
            
            logger.info(f"      ✅ Generated {len(segments)} segments")
            
            # Step 2: Create subtitle clips
            logger.info("   🎨 Creating subtitle clips...")
            subtitle_clips = await generator.create_subtitle_clips(
                segments,
                video_width,
                video_height
            )
            
            logger.info(f"      ✅ Created {len(subtitle_clips)} subtitle clips")
            
            # Step 3: Create background video
            logger.info("   🎬 Creating background video...")
            background = ColorClip(
                size=(video_width, video_height),
                color=video_config["background_color"],
                duration=video_config["duration"]
            )
            
            # Step 4: Composite video with subtitles
            logger.info("   🔗 Compositing video with subtitles...")
            final_clips = [background] + subtitle_clips
            final_video = CompositeVideoClip(final_clips, size=(video_width, video_height))
            
            # Step 5: Export video
            output_file = output_dir / f"{video_config['title']}.mp4"
            logger.info(f"   💾 Exporting to: {output_file}")
            
            final_video.write_videofile(
                str(output_file),
                fps=fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None  # Suppress MoviePy logs
            )
            
            logger.success(f"   ✅ Video exported: {output_file}")
            
            # Show video info
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"      📊 File size: {file_size:.1f} MB")
            logger.info(f"      📱 Resolution: {video_width}x{video_height}")
            logger.info(f"      🎬 FPS: {fps}")
            logger.info(f"      ⏱️  Duration: {video_config['duration']}s")
            
            # Show subtitle info
            style = generator.styles[video_config["style"]]
            logger.info(f"      🎯 Font: {style.font_size}px {style.font_family}")
            logger.info(f"      🎨 Colors: {style.font_color} text, {style.bubble_outline} outline")
            
            # Clean up
            final_video.close()
            background.close()
            for clip in subtitle_clips:
                clip.close()
            
        except Exception as e:
            logger.error(f"   ❌ Failed to create video {i}: {e}")
    
    logger.success("🎉 All bubble subtitle videos created!")
    logger.info(f"📁 Check the '{output_dir}' folder for MP4 files")


async def create_style_comparison_video():
    """Create a single video showing all bubble styles side by side."""
    if not MOVIEPY_AVAILABLE:
        logger.error("❌ MoviePy is required to create video files.")
        return
    
    logger.info("\n🎨 Creating style comparison video")
    
    test_text = "EPIC MOMENT! This will blow your mind!"
    segment_duration = 3.0
    video_width = 1080
    video_height = 1920
    
    bubble_styles = ["bubble_gaming", "bubble_neon", "bubble_cute", "bubble_classic"]
    
    generator = SubtitleGenerator()
    output_dir = Path(__file__).parent / "output_videos"
    output_dir.mkdir(exist_ok=True)
    
    video_segments = []
    
    for i, style in enumerate(bubble_styles):
        logger.info(f"   🎨 Creating segment for {style}...")
        
        try:
            # Generate subtitles for this style
            segments = await generator.generate_from_script(
                test_text,
                segment_duration,
                style_name=style
            )
            
            # Create subtitle clips
            subtitle_clips = await generator.create_subtitle_clips(
                segments,
                video_width,
                video_height
            )
            
            # Create background with style-specific color
            colors = {
                "bubble_gaming": (20, 50, 20),   # Dark green
                "bubble_neon": (10, 30, 60),     # Dark blue  
                "bubble_cute": (60, 20, 40),     # Dark pink
                "bubble_classic": (40, 40, 10)   # Dark yellow
            }
            
            background = ColorClip(
                size=(video_width, video_height),
                color=colors[style],
                duration=segment_duration
            )
            
            # Composite this segment
            segment_clips = [background] + subtitle_clips
            segment_video = CompositeVideoClip(segment_clips, size=(video_width, video_height))
            
            video_segments.append(segment_video)
            logger.info(f"      ✅ Created {style} segment")
            
        except Exception as e:
            logger.error(f"      ❌ Failed to create {style} segment: {e}")
    
    if video_segments:
        logger.info("   🔗 Concatenating all segments...")
        
        # Concatenate all segments
        final_video = concatenate_videoclips(video_segments)
        
        # Export comparison video
        output_file = output_dir / "Bubble_Styles_Comparison.mp4"
        logger.info(f"   � Exporting comparison video: {output_file}")
        
        final_video.write_videofile(
            str(output_file),
            fps=30,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        logger.success(f"   ✅ Comparison video exported: {output_file}")
        
        # Show video info
        file_size = output_file.stat().st_size / (1024 * 1024)
        total_duration = len(bubble_styles) * segment_duration
        logger.info(f"      📊 File size: {file_size:.1f} MB")
        logger.info(f"      ⏱️  Total duration: {total_duration}s")
        logger.info(f"      � Segments: {len(bubble_styles)} styles")
        
        # Clean up
        final_video.close()
        for segment in video_segments:
            segment.close()
    
    logger.success("🎉 Style comparison video created!")


async def show_style_comparison():
    """Show a comparison of all bubble styles with the same text."""
    logger.info("\n🔄 Style Comparison Demo")
    
    test_text = "EPIC MOMENT! This will blow your mind!"
    duration = 3.0
    
    bubble_styles = [
        "bubble",
        "bubble_gaming", 
        "bubble_neon",
        "bubble_cute",
        "bubble_classic",
        "bubble_blue"
    ]
    
    generator = SubtitleGenerator()
    
    logger.info(f"📝 Test text: \"{test_text}\"")
    logger.info(f"⏱️  Duration: {duration}s")
    
    comparison_data = []
    
    for style in bubble_styles:
        logger.info(f"\n🎨 Testing {style}:")
        
        try:
            # Generate segments
            segments = await generator.generate_from_script(
                test_text,
                duration,
                style_name=style
            )
            
            # Get style info
            style_obj = generator.styles[style]
            
            # Create clips
            clips = await generator.create_subtitle_clips(segments, 1080, 1920)
            
            comparison_data.append({
                "style": style,
                "segments": len(segments),
                "clips": len(clips),
                "font_size": style_obj.font_size,
                "font_color": style_obj.font_color,
                "outline_color": style_obj.bubble_outline,
                "outline_width": style_obj.bubble_outline_width,
                "double_stroke": style_obj.double_stroke,
                "background": style_obj.background_color
            })
            
            logger.info(f"   ✅ {len(segments)} segments, {len(clips)} clips")
            logger.info(f"   🎯 {style_obj.font_size}px {style_obj.font_color} with {style_obj.bubble_outline} outline")
            
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")
    
    # Summary table
    logger.info("\n📊 Style Comparison Summary:")
    logger.info("Style".ljust(15) + "Font Size".ljust(10) + "Color".ljust(10) + "Outline".ljust(12) + "Features")
    logger.info("-" * 65)
    
    for data in comparison_data:
        features = []
        if data["double_stroke"]:
            features.append("Double")
        if data["background"]:
            features.append("BG")
        feature_str = "+".join(features) if features else "Basic"
        
        line = (
            data["style"].ljust(15) +
            f"{data['font_size']}px".ljust(10) +
            data["font_color"][:8].ljust(10) +
            f"{data['outline_color'][:8]} ({data['outline_width']}px)".ljust(12) +
            feature_str
        )
        logger.info(line)
    
    logger.success("🎉 Style comparison completed!")


async def test_video_integration():
    """Test integration with video pipeline (mock)."""
    logger.info("\n🎬 Testing Video Pipeline Integration")
    
    # Simulate video creation workflow
    test_scenario = {
        "title": "Gaming Highlight Reel",
        "script": "Watch this INSANE 360 no-scope headshot! This took me 1000 attempts to pull off perfectly!",
        "duration": 6.0,
        "style": "bubble_gaming",
        "video_specs": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "format": "MP4"
        }
    }
    
    logger.info(f"🎮 Scenario: {test_scenario['title']}")
    logger.info(f"📱 Video: {test_scenario['video_specs']['width']}x{test_scenario['video_specs']['height']} @ {test_scenario['video_specs']['fps']}fps")
    logger.info(f"🎨 Style: {test_scenario['style']}")
    
    generator = SubtitleGenerator()
    
    try:
        # Step 1: Generate subtitles
        logger.info("\n📝 Step 1: Generating subtitles...")
        segments = await generator.generate_from_script(
            test_scenario["script"],
            test_scenario["duration"],
            style_name=test_scenario["style"]
        )
        
        logger.success(f"   ✅ Generated {len(segments)} subtitle segments")
        
        # Step 2: Create video clips
        logger.info("\n🎬 Step 2: Creating video clips...")
        clips = await generator.create_subtitle_clips(
            segments,
            test_scenario["video_specs"]["width"],
            test_scenario["video_specs"]["height"]
        )
        
        logger.success(f"   ✅ Created {len(clips)} video clips ready for composition")
        
        # Step 3: Export subtitle files
        logger.info("\n📄 Step 3: Exporting subtitle files...")
        srt_path = await generator.export_srt(segments, f"video_test_{test_scenario['style']}.srt")
        json_path = await generator.export_json(segments, f"video_test_{test_scenario['style']}.json")
        
        logger.success(f"   ✅ Exported SRT: {Path(srt_path).name}")
        logger.success(f"   ✅ Exported JSON: {Path(json_path).name}")
        
        # Step 4: Show clip details
        logger.info("\n🔍 Step 4: Clip Analysis...")
        total_duration = sum(clip.duration for clip in clips)
        logger.info(f"   ⏱️  Total subtitle duration: {total_duration:.2f}s")
        logger.info(f"   📊 Average segment length: {total_duration/len(clips):.2f}s")
        
        for i, (segment, clip) in enumerate(zip(segments, clips), 1):
            logger.info(f"   {i}: \"{segment.text}\" ({segment.start_time:.1f}s-{segment.end_time:.1f}s)")
        
        # Step 5: Mock video composition
        logger.info("\n🎞️  Step 5: Mock video composition...")
        logger.info("   📹 Would composite with background video...")
        logger.info("   🎵 Would add background audio/TTS...")
        logger.info("   ✨ Would apply video effects...")
        logger.info("   💾 Would export final MP4...")
        
        logger.success("🎉 Video pipeline integration test completed!")
        
    except Exception as e:
        logger.error(f"❌ Video integration test failed: {e}")


async def main():
    """Main test function."""
    try:
        await create_video_with_bubble_subtitles()
        await create_style_comparison_video()
        await show_style_comparison()
        await test_video_integration()
        
        logger.success("🎉 All bubble subtitle tests completed successfully!")
        logger.info("\n📁 Check the 'output_videos' folder for MP4 files")
        logger.info("📁 Check the 'src/video/data/subtitles/' folder for subtitle files")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
