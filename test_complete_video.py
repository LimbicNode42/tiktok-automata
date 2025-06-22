#!/usr/bin/env python3
"""
Complete Video with Audio Test - Simplified Version

This creates a complete TikTok video with integrated audio by using a simplified
audio mixing approach that avoids the MoviePy effect chain issues.
"""

import asyncio
import sys
import os
from pathlib import Path
from loguru import logger
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.tts.kokoro_tts import KokoroTTSEngine
    import soundfile as sf
    from moviepy import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
    print("✅ Components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)


async def create_complete_video_with_audio():
    """Create a complete TikTok video with integrated audio."""
    logger.info("🎬 Creating Complete TikTok Video with Audio")
    
    test_dir = Path("complete_video_test")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Generate TTS audio
        logger.info("🎤 Step 1: Generating TTS audio...")
        tts = KokoroTTSEngine()
        
        script = """
        This AI breakthrough will change everything! Google just released Gemini 2.5 Pro, 
        and it's incredibly powerful. This new model can understand complex reasoning, 
        process multiple types of data at once, and even write code better than most humans.
        
        What makes this special? It can analyze images, understand context, and generate 
        creative content in seconds. The future of AI is happening right now!
        """
        
        tts_path = await tts.generate_audio(
            text=script.strip(),
            output_path=str(test_dir / "narration.wav"),
            voice='af_heart',
            speed=1.1
        )
        
        if not tts_path or not Path(tts_path).exists():
            logger.error("❌ TTS generation failed")
            return False
            
        logger.info(f"✅ TTS audio: {tts_path}")
        
        # Get TTS duration
        tts_audio_data, tts_sample_rate = sf.read(tts_path)
        tts_duration = len(tts_audio_data) / tts_sample_rate
        logger.info(f"   Duration: {tts_duration:.1f}s")
        
        # Step 2: Load gaming video
        logger.info("🎮 Step 2: Loading gaming video...")
        footage_dir = Path("src/video/data/footage/raw")
        video_files = list(footage_dir.glob("*.mp4"))
        
        if not video_files:
            logger.error("❌ No gaming video found")
            return False
            
        gaming_video_path = video_files[0]
        logger.info(f"   Using: {gaming_video_path.name}")
        
        # Step 3: Simple audio mixing using numpy
        logger.info("🎵 Step 3: Simple audio mixing...")
        
        # Load gaming video and extract audio
        video_clip = VideoFileClip(str(gaming_video_path))
        
        if video_clip.audio is None:
            logger.warning("⚠️ Gaming video has no audio, using silent background")
            background_audio = np.zeros(len(tts_audio_data), dtype=np.float32)
        else:
            # Extract background audio matching TTS duration
            bg_audio_clip = video_clip.audio.subclipped(0, min(tts_duration, video_clip.duration))
            background_audio = bg_audio_clip.to_soundarray(fps=tts_sample_rate)
            
            # Convert to mono if stereo
            if len(background_audio.shape) > 1:
                background_audio = np.mean(background_audio, axis=1)
            
            bg_audio_clip.close()
        
        # Ensure both audio tracks are the same length
        min_length = min(len(tts_audio_data), len(background_audio))
        tts_audio_data = tts_audio_data[:min_length]
        background_audio = background_audio[:min_length]
        
        # Mix audio with proper levels
        tts_volume = 0.85
        bg_volume = 0.25
        master_volume = 0.90
        
        mixed_audio = (tts_audio_data * tts_volume) + (background_audio * bg_volume)
        mixed_audio = mixed_audio * master_volume
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(mixed_audio))
        if peak > 0.95:
            mixed_audio = mixed_audio * (0.95 / peak)
        
        # Save mixed audio
        mixed_audio_path = test_dir / "mixed_audio.wav"
        sf.write(mixed_audio_path, mixed_audio, tts_sample_rate)
        
        logger.info(f"✅ Mixed audio: {mixed_audio_path}")
        logger.info(f"   Final duration: {len(mixed_audio) / tts_sample_rate:.1f}s")
        logger.info(f"   Peak level: {np.max(np.abs(mixed_audio)):.3f}")
        
        # Step 4: Create video with mixed audio
        logger.info("🎬 Step 4: Creating final video...")
        
        # Prepare video clip
        target_duration = len(mixed_audio) / tts_sample_rate
          # Resize video to TikTok dimensions (1080x1920)
        video_resized = video_clip.resized(new_size=(1080, 1920))        # Trim/loop video to match audio duration
        if video_resized.duration < target_duration:
            # Loop video if too short
            loops_needed = int(target_duration / video_resized.duration) + 1
            video_list = [video_resized] * loops_needed
            video_resized = concatenate_videoclips(video_list)
        
        video_resized = video_resized.subclipped(0, target_duration)
        
        # Replace audio with our mixed audio
        mixed_audio_clip = AudioFileClip(str(mixed_audio_path))
        final_video_clip = video_resized.with_audio(mixed_audio_clip)        # Add text overlay
        text_lines = "AI Breakthrough: Google Gemini 2.5 Pro"
        text_clip = TextClip(
            text=text_lines,
            font_size=64,
            color='white',
            stroke_color='black',
            stroke_width=3,
            duration=target_duration,
            size=(1000, None)
        ).with_position(('center', 100)).with_start(0)
        
        # Composite final video
        final_video = CompositeVideoClip([final_video_clip, text_clip])
        
        # Output path
        output_path = test_dir / f"tiktok_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
          # Render final video
        logger.info("🔄 Rendering final video... (this may take a few minutes)")
        final_video.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='medium',
            logger=None  # Disable MoviePy progress bar for cleaner output
        )
        
        # Cleanup clips
        video_clip.close()
        video_resized.close()
        mixed_audio_clip.close()
        final_video_clip.close()
        text_clip.close()
        final_video.close()
        
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Final video created: {output_path}")
            logger.info(f"   File size: {file_size_mb:.1f} MB")
            logger.info(f"   Duration: {target_duration:.1f} seconds")
            logger.info(f"   Resolution: 1080x1920 (TikTok format)")
            
            logger.info("\n🎉 COMPLETE VIDEO WITH AUDIO READY!")
            logger.info("🎧 This video contains:")
            logger.info("   - TTS narration (85% volume)")
            logger.info("   - Gaming background audio (25% volume)")
            logger.info("   - Text overlay")
            logger.info("   - TikTok 9:16 aspect ratio")
            
            logger.info(f"\n🎯 VIDEO READY FOR VALIDATION:")
            logger.info(f"   {output_path.absolute()}")
            logger.info("   Play this video to validate the audio integration!")
            
            return True
        else:
            logger.error("❌ Video creation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Video creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main entry."""
    print("🤖 TikTok Automata - Complete Video with Audio Test")
    print("🎬 Creating TikTok Video with Integrated Audio")
    print("=" * 60)
    
    try:
        success = await create_complete_video_with_audio()
        
        if success:
            print("\n✅ COMPLETE VIDEO CREATION SUCCESSFUL!")
            print("🎬 TikTok video with integrated audio is ready!")
            print("🎧 The video contains properly mixed audio:")
            print("   - Clear TTS narration")
            print("   - Subtle gaming background audio")
            print("   - Professional volume balancing")
            print("\n📂 Check the complete_video_test/ directory for the output")
            print("🎯 Play the video to validate the audio integration!")
        else:
            print("\n❌ COMPLETE VIDEO CREATION FAILED!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Video creation interrupted by user")
    except Exception as e:
        print(f"\n❌ Video creation failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    asyncio.run(main())
