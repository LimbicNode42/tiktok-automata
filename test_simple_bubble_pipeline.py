#!/usr/bin/env python3
"""
Simple test for bubble subtitle integration into the video pipeline.

This test focuses on the video generation with bubble subtitles,
using a mock audio file to avoid TTS dependencies.
"""

import asyncio
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# Import the video processing pipeline
from src.video.processors.video_processor import VideoProcessor, VideoConfig

# Test content for video generation
TEST_SCRIPT = """
OpenAI just released ChatGPT's biggest update ever!
The new reasoning model can solve complex problems step by step.
This could revolutionize AI assistance forever.
"""

TEST_CONTENT_ANALYSIS = {
    "is_breakthrough": True,
    "controversy_score": 0,
    "has_funding": False,
    "is_partnership": False,
    "is_first_person": False,
    "excitement_level": "high"
}

def create_mock_audio(duration: float = 10.0, output_file: str = "mock_audio.wav"):
    """Create a simple mock audio file for testing."""
    sample_rate = 22050
    samples = int(sample_rate * duration)
    
    # Create simple sine wave audio for testing
    t = np.linspace(0, duration, samples, False)
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.1  # Low volume
    
    # Add some variation to make it more speech-like
    modulation = np.sin(2 * np.pi * 2 * t) * 0.5 + 0.5
    audio = audio * modulation
    
    # Write to file
    sf.write(output_file, audio, sample_rate)
    logger.info(f"🎵 Created mock audio file: {output_file} ({duration:.1f}s)")
    return output_file

async def test_bubble_video_generation():
    """Test video generation with bubble subtitles."""
    logger.info("🎬 Starting Bubble Subtitle Video Test")
    
    try:
        # Step 1: Create mock audio
        logger.info("🎙️ Step 1: Creating mock audio...")
        audio_file = create_mock_audio(duration=8.0)  # Short test video
        
        # Step 2: Configure video processor for bubble subtitles
        logger.info("⚙️ Step 2: Configuring video processor with bubble subtitles...")
        config = VideoConfig(
            width=1080,
            height=1920,
            fps=30,
            enable_subtitles=True,
            subtitle_style="bubble",  # Use main bubble style
            subtitle_position=0.75,
            export_srt=True,
            letterbox_mode="blurred_background",
            background_opacity=0.7,
            use_gaming_footage=True,
            enable_zoom_effects=False,  # Disable for faster processing
            enable_transitions=False,   # Disable for faster processing
            output_quality="medium"     # Medium quality for faster export
        )
        
        processor = VideoProcessor(config)
        logger.success("✅ Video processor configured with bubble subtitles")
        
        # Step 3: Create output directory
        output_dir = Path("output_videos_bubble")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "bubble_subtitle_test.mp4"
        
        # Step 4: Generate the video
        logger.info("🎬 Step 3: Creating video with bubble subtitles...")
        start_time = time.time()
        
        video_file = await processor.create_video(
            audio_file=audio_file,
            script_content=TEST_SCRIPT,
            content_analysis=TEST_CONTENT_ANALYSIS,
            voice_info={"voice_name": "test", "speed": 1.0},
            output_path=str(output_path)
        )
        
        processing_time = time.time() - start_time
        
        if video_file:
            file_size = Path(video_file).stat().st_size / (1024 * 1024)
            logger.success(f"🎉 Bubble subtitle video created successfully!")
            logger.success(f"📁 File: {video_file}")
            logger.success(f"📊 Size: {file_size:.1f}MB")
            logger.success(f"⏱️ Processing time: {processing_time:.1f}s")
            logger.success(f"📍 Full path: {Path(video_file).absolute()}")
            
            # Check if SRT file was created
            srt_file = Path(video_file).with_suffix('.srt')
            if srt_file.exists():
                logger.success(f"📝 SRT file created: {srt_file}")
                # Show first few lines of SRT
                with open(srt_file, 'r', encoding='utf-8') as f:
                    srt_content = f.read()[:300]
                    logger.info(f"📝 SRT preview:\n{srt_content}...")
            
            return True
        else:
            logger.error("❌ Video creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup mock audio file
        try:
            if 'audio_file' in locals():
                Path(audio_file).unlink(missing_ok=True)
                logger.info("🧹 Cleaned up mock audio file")
        except:
            pass

async def test_all_bubble_styles():
    """Test all available bubble styles."""
    logger.info("🎨 Testing All Bubble Styles")
    
    bubble_styles = ["bubble", "bubble_blue", "bubble_gaming", "bubble_neon", "bubble_cute", "bubble_classic"]
    results = {}
    
    for style in bubble_styles:
        logger.info(f"🎭 Testing style: {style}")
        
        try:
            # Create mock audio
            audio_file = create_mock_audio(duration=5.0, output_file=f"mock_{style}.wav")
            
            # Configure for this style
            config = VideoConfig(
                width=1080,
                height=1920,
                fps=30,
                enable_subtitles=True,
                subtitle_style=style,
                subtitle_position=0.75,
                export_srt=True,
                letterbox_mode="blurred_background",
                use_gaming_footage=True,
                enable_zoom_effects=False,
                enable_transitions=False,
                output_quality="medium"
            )
            
            processor = VideoProcessor(config)
            
            # Create output
            output_dir = Path("output_videos_bubble")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"bubble_test_{style}.mp4"
            
            # Generate video
            video_file = await processor.create_video(
                audio_file=audio_file,
                script_content=f"Testing the {style} subtitle style!",
                content_analysis={"is_breakthrough": False},
                voice_info={"voice_name": "test"},
                output_path=str(output_path)
            )
            
            if video_file:
                file_size = Path(video_file).stat().st_size / (1024 * 1024)
                results[style] = f"✅ Success ({file_size:.1f}MB)"
                logger.success(f"✅ {style}: {Path(video_file).name} ({file_size:.1f}MB)")
            else:
                results[style] = "❌ Failed"
                logger.error(f"❌ {style}: Failed")
                
        except Exception as e:
            results[style] = f"❌ Error: {str(e)[:50]}"
            logger.error(f"❌ {style}: {e}")
        finally:
            # Cleanup
            try:
                if 'audio_file' in locals():
                    Path(audio_file).unlink(missing_ok=True)
            except:
                pass
    
    # Summary
    logger.info("\n📊 Bubble Style Test Results:")
    for style, result in results.items():
        logger.info(f"  {style}: {result}")
    
    return results

if __name__ == "__main__":
    print("🎬 Bubble Subtitle Video Pipeline Test")
    print("=" * 50)
    
    # Test 1: Single bubble video
    print("\n🚀 Test 1: Bubble Subtitle Video Generation")
    success = asyncio.run(test_bubble_video_generation())
    
    if success:
        print("\n🎨 Test 2: All Bubble Styles")
        asyncio.run(test_all_bubble_styles())
    
    print("\n✨ Testing complete!")
    print("\n📁 Check the 'output_videos_bubble' folder for generated videos!")
