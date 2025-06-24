#!/usr/bin/env python3
"""
Test script for the complete TikTok video pipeline with bubble subtitles.

This script creates a full MP4 video using the video processor pipeline
with bubble-style subtitles integrated.
"""

import asyncio
import tempfile
from pathlib import Path
from loguru import logger

# Import the video processing pipeline
from src.video.processors.video_processor import VideoProcessor, VideoConfig
from src.tts.kokoro_tts import KokoroTTSEngine

# Test content for video generation
TEST_SCRIPT = """
OpenAI just released a massive ChatGPT update that's changing everything! 
The new reasoning model can solve complex problems step by step.
It's like having a personal AI assistant that actually thinks through problems.
This could revolutionize how we work with artificial intelligence.
The future of AI assistance is here and it's incredible!
"""

TEST_VOICE_INFO = {
    "voice_name": "af_sarah",
    "speed": 1.1,
    "emotion": "excited"
}

TEST_CONTENT_ANALYSIS = {
    "is_breakthrough": True,
    "controversy_score": 0,
    "has_funding": False,
    "is_partnership": False,
    "is_first_person": False,
    "excitement_level": "high"
}

async def test_full_pipeline_with_bubble_subtitles():
    """Test the complete video pipeline with bubble subtitles."""
    logger.info("🎬 Starting Full Pipeline Test with Bubble Subtitles")
    
    try:        # Step 1: Create TTS audio
        logger.info("🎙️ Step 1: Generating TTS audio...")
        tts = KokoroTTSEngine()
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_file = temp_audio.name
            
        # Generate TTS audio
        success = await tts.generate_tts(
            text=TEST_SCRIPT,
            output_file=audio_file,
            voice_name=TEST_VOICE_INFO["voice_name"],
            speed=TEST_VOICE_INFO["speed"]
        )
        
        if not success:
            logger.error("❌ TTS generation failed")
            return False
            
        logger.success(f"✅ TTS audio created: {audio_file}")
        
        # Step 2: Configure video processor with bubble subtitles
        logger.info("⚙️ Step 2: Configuring video processor...")
        config = VideoConfig(
            width=1080,
            height=1920,
            fps=30,
            enable_subtitles=True,
            subtitle_style="bubble",  # Use bubble style!
            subtitle_position=0.75,  # Higher position
            export_srt=True,
            letterbox_mode="blurred_background",
            background_opacity=0.7,
            use_gaming_footage=True,
            enable_zoom_effects=True,
            enable_transitions=True,
            output_quality="high"
        )
        
        processor = VideoProcessor(config)
        logger.success("✅ Video processor configured with bubble subtitles")
        
        # Step 3: Create output directory
        output_dir = Path("output_videos_bubble")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "tiktok_bubble_test.mp4"
        
        # Step 4: Generate the complete video
        logger.info("🎬 Step 3: Creating complete video with bubble subtitles...")
        
        video_file = await processor.create_video(
            audio_file=audio_file,
            script_content=TEST_SCRIPT,
            content_analysis=TEST_CONTENT_ANALYSIS,
            voice_info=TEST_VOICE_INFO,
            output_path=str(output_path)
        )
        
        if video_file:
            file_size = Path(video_file).stat().st_size / (1024 * 1024)
            logger.success(f"🎉 Full pipeline success! Video created: {video_file}")
            logger.success(f"📁 File size: {file_size:.1f}MB")
            logger.success(f"📍 Location: {Path(video_file).absolute()}")
            
            # Check if SRT file was also created
            srt_file = Path(video_file).with_suffix('.srt')
            if srt_file.exists():
                logger.success(f"📝 SRT file also created: {srt_file}")
            
            return True
        else:
            logger.error("❌ Video creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup temp audio file
        try:
            if 'audio_file' in locals():
                Path(audio_file).unlink(missing_ok=True)
                logger.info("🧹 Cleaned up temporary audio file")
        except:
            pass

async def test_multiple_bubble_styles():
    """Test different bubble subtitle styles."""
    logger.info("🎨 Testing Multiple Bubble Styles")
    
    bubble_styles = ["bubble", "bubble_blue", "bubble_gaming", "bubble_neon", "bubble_cute"]
    
    for style in bubble_styles:
        logger.info(f"🎭 Testing style: {style}")
        
        try:            # Create TTS audio (reuse for all styles)
            tts = KokoroTTSEngine()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_file = temp_audio.name
                
            success = await tts.generate_tts(
                text="This is a test of the bubble subtitle style system!",
                output_file=audio_file,
                voice_name="af_sarah",
                speed=1.0
            )
            
            if not success:
                logger.warning(f"⚠️ TTS failed for {style}, skipping...")
                continue
            
            # Configure processor for this style
            config = VideoConfig(
                width=1080,
                height=1920,
                fps=30,
                enable_subtitles=True,
                subtitle_style=style,  # Different style each time
                subtitle_position=0.75,
                export_srt=True,
                letterbox_mode="blurred_background"
            )
            
            processor = VideoProcessor(config)
            
            # Create output
            output_dir = Path("output_videos_bubble")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"bubble_test_{style}.mp4"
            
            # Generate video
            video_file = await processor.create_video(
                audio_file=audio_file,
                script_content="This is a test of the bubble subtitle style system!",
                content_analysis={"is_breakthrough": False},
                voice_info={"voice_name": "af_sarah"},
                output_path=str(output_path)
            )
            
            if video_file:
                logger.success(f"✅ {style} video created: {Path(video_file).name}")
            else:
                logger.error(f"❌ {style} video failed")
                
        except Exception as e:
            logger.error(f"❌ Style {style} failed: {e}")
        finally:
            # Cleanup
            try:
                if 'audio_file' in locals():
                    Path(audio_file).unlink(missing_ok=True)
            except:
                pass

if __name__ == "__main__":
    print("🎬 TikTok Video Pipeline with Bubble Subtitles Test")
    print("=" * 60)
    
    # Test 1: Full pipeline with bubble subtitles
    print("\n🚀 Test 1: Full Pipeline with Bubble Subtitles")
    success = asyncio.run(test_full_pipeline_with_bubble_subtitles())
    
    if success:
        print("\n🎨 Test 2: Multiple Bubble Styles")
        asyncio.run(test_multiple_bubble_styles())
    
    print("\n✨ Testing complete!")
