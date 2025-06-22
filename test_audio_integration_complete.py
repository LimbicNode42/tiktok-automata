#!/usr/bin/env python3
"""
Complete Audio Integration Pipeline Test

This test validates the audio integration module by creating a complete TikTok video:
1. Uses sample text content 
2. Generates TTS audio
3. Creates/loads gaming video
4. Uses the NEW audio integration module to mix TTS + gaming audio
5. Outputs final video for audio validation

Follows coding guidelines: No interactive terminal commands, proper test scripts.
"""

import asyncio
import sys
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test imports
try:
    from src.tts.kokoro_tts import KokoroTTSEngine
    from src.video.audio.audio_mixer import AudioMixer, AudioConfig
    from src.utils.config import config
    print("✅ Core components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class AudioIntegrationTest:
    """Audio integration test orchestrator."""
    
    def __init__(self):
        self.test_dir = Path("audio_test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.tts = None
        self.audio_mixer = None
        
        logger.info("Audio integration test initialized")
    
    async def setup_components(self):
        """Initialize components."""
        logger.info("🔧 Setting up components...")
        
        # TTS system
        self.tts = KokoroTTSEngine()
        
        # Audio mixer (new integration module)
        audio_config = AudioConfig(
            tts_volume=0.85,
            background_volume=0.25,
            master_volume=0.90,
            fade_duration=1.0,
            enable_dynamic_range=True,
            apply_eq=True
        )
        self.audio_mixer = AudioMixer(audio_config)
        
        logger.info("✅ Components initialized")
    
    def get_sample_script(self) -> str:
        """Get sample TikTok script for testing."""
        return """
        This AI breakthrough will change everything! Google just released Gemini 2.5 Pro, 
        and it's incredibly powerful. This new model can understand complex reasoning, 
        process multiple types of data at once, and even write code better than most humans.
        
        What makes this special? It can analyze images, understand context, and generate 
        creative content in seconds. Tech companies are already integrating it into their 
        products, and early tests show it outperforms previous models by 40%.
        
        This could revolutionize how we work, learn, and create content. The future of AI 
        is happening right now!
        """
    
    async def generate_tts_audio(self, script: str) -> Optional[Path]:
        """Generate TTS audio from script."""
        logger.info("🎤 Generating TTS audio...")
        
        try:
            # Generate audio
            audio_path = await self.tts.generate_audio(
                text=script.strip(),
                output_path=str(self.test_dir / "tts_audio.wav"),
                voice='af_heart',  # Clear female voice
                speed=1.1         # Slightly faster for TikTok
            )
            
            if audio_path and Path(audio_path).exists():
                audio_file = Path(audio_path)
                logger.info(f"✅ TTS audio generated: {audio_file}")
                logger.info(f"   File size: {audio_file.stat().st_size / 1024:.1f} KB")
                return audio_file
            else:
                logger.error("❌ TTS audio generation failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ TTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def create_test_gaming_video(self) -> Optional[Path]:
        """Create or load gaming video footage."""
        logger.info("🎮 Creating test gaming video...")
        
        try:
            # Check for existing gaming footage
            footage_dir = Path("src/video/data/footage/raw")
            if footage_dir.exists():
                video_files = list(footage_dir.glob("*.mp4"))
                if video_files:
                    selected_video = video_files[0]
                    logger.info(f"✅ Using existing gaming footage: {selected_video}")
                    return selected_video
            
            # Create test video if none exists
            logger.info("Creating simple test video...")
            footage_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                from moviepy import ColorClip
                test_clip = ColorClip(
                    size=(1080, 1920), 
                    color=(50, 100, 150), 
                    duration=60
                ).with_fps(30)
                
                test_video_path = footage_dir / "test_gaming_footage.mp4"
                test_clip.write_videofile(
                    str(test_video_path),
                    codec='libx264',
                    audio=False,
                    verbose=False,
                    logger=None
                )
                test_clip.close()
                
                logger.info(f"✅ Created test gaming footage: {test_video_path}")
                return test_video_path
                
            except ImportError:
                logger.warning("MoviePy not available, creating audio-only test")
                # Create a simple audio file for testing
                import numpy as np
                import soundfile as sf
                
                # Generate simple ambient audio
                duration = 60
                sample_rate = 24000
                t = np.linspace(0, duration, duration * sample_rate)
                ambient_audio = 0.1 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.random.randn(len(t))
                
                test_audio_path = self.test_dir / "test_ambient.wav"
                sf.write(test_audio_path, ambient_audio, sample_rate)
                
                logger.info(f"✅ Created test ambient audio: {test_audio_path}")
                return test_audio_path
                
        except Exception as e:
            logger.error(f"❌ Test video creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def integrate_audio(self, tts_path: Path, video_path: Path) -> Optional[Path]:
        """Integrate TTS audio with gaming video audio using new audio module."""
        logger.info("🎵 Integrating audio streams with NEW audio module...")
        
        try:
            # Output path for mixed audio
            mixed_audio_path = self.test_dir / "mixed_audio.wav"
              # Use the NEW audio integration module
            result = await self.audio_mixer.mix_audio_streams(
                tts_audio_path=tts_path,
                background_video_path=video_path,
                output_path=mixed_audio_path
            )
            
            if result and mixed_audio_path.exists():
                logger.info(f"✅ Audio integration complete: {mixed_audio_path}")
                logger.info(f"   File size: {mixed_audio_path.stat().st_size / 1024:.1f} KB")
                return mixed_audio_path
            else:
                logger.error("❌ Audio integration failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Audio integration failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def create_final_video(self, video_path: Path, audio_path: Path, script: str) -> Optional[Path]:
        """Create final video with integrated audio."""
        logger.info("🎬 Creating final video...")
        
        try:
            # Try with MoviePy if available
            try:
                from moviepy import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
                
                # Load or create video
                if video_path.suffix == '.mp4':
                    video_clip = VideoFileClip(str(video_path))
                else:
                    # Audio-only fallback: create simple color video
                    from moviepy import ColorClip
                    video_clip = ColorClip(size=(1080, 1920), color=(50, 100, 150), duration=60).with_fps(30)
                
                # Get audio duration
                audio_clip = AudioFileClip(str(audio_path))
                target_duration = audio_clip.duration
                audio_clip.close()
                
                # Adjust video duration
                if video_clip.duration < target_duration:
                    loops_needed = int(target_duration / video_clip.duration) + 1
                    video_clip = video_clip.with_repeat(loops_needed)
                
                video_clip = video_clip.with_subclipped(0, target_duration)
                
                # Replace audio with mixed audio
                final_clip = video_clip.with_audio(str(audio_path))
                
                # Add simple text overlay
                text_lines = script.split('.')[0][:50] + "..."
                text_clip = TextClip(
                    text_lines,
                    font_size=48,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    duration=target_duration,
                    size=(1000, None)
                ).with_position(('center', 'center')).with_start(0)
                
                # Composite final video
                final_video = CompositeVideoClip([final_clip, text_clip])
                
                # Output path
                output_path = self.test_dir / f"tiktok_audio_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                
                # Render final video
                logger.info("🔄 Rendering final video...")
                final_video.write_videofile(
                    str(output_path),
                    codec='libx264',
                    audio_codec='aac',
                    fps=30,
                    preset='medium',
                    verbose=False,
                    logger=None
                )
                
                # Cleanup
                video_clip.close()
                final_clip.close()
                text_clip.close()
                final_video.close()
                
                if output_path.exists():
                    logger.info(f"✅ Final video created: {output_path}")
                    logger.info(f"   File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
                    logger.info(f"   Duration: {target_duration:.1f} seconds")
                    return output_path
                else:
                    logger.error("❌ Final video creation failed")
                    return None
                    
            except ImportError:
                logger.warning("MoviePy not available - audio-only output")
                # Just copy the mixed audio as the final output
                output_path = self.test_dir / f"tiktok_audio_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                import shutil
                shutil.copy2(audio_path, output_path)
                
                logger.info(f"✅ Audio-only output created: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"❌ Final video creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run_complete_test(self):
        """Run the complete audio integration test."""
        logger.info("🚀 Starting Audio Integration Test")
        logger.info("🎵 Testing NEW Audio Integration Module")
        logger.info("=" * 60)
        
        try:
            # Setup
            await self.setup_components()
            
            # Step 1: Get sample script
            logger.info("\n📝 Step 1: Using sample TikTok script...")
            script = self.get_sample_script()
            logger.info(f"✅ Script ready ({len(script)} chars)")
            
            # Step 2: Generate TTS audio
            logger.info("\n🎤 Step 2: Generating TTS audio...")
            tts_audio_path = await self.generate_tts_audio(script)
            if not tts_audio_path:
                logger.error("❌ Failed to generate TTS audio")
                return False
            
            # Step 3: Create/load gaming video
            logger.info("\n🎮 Step 3: Setting up gaming video...")
            video_path = await self.create_test_gaming_video()
            if not video_path:
                logger.error("❌ Failed to setup gaming video")
                return False
            
            # Step 4: Integrate audio (MAIN TEST)
            logger.info("\n🎵 Step 4: Testing NEW Audio Integration Module...")
            mixed_audio_path = await self.integrate_audio(tts_audio_path, video_path)
            if not mixed_audio_path:
                logger.error("❌ Failed to integrate audio")
                return False
            
            # Step 5: Create final video
            logger.info("\n🎬 Step 5: Creating final output...")
            final_output_path = await self.create_final_video(video_path, mixed_audio_path, script)
            if not final_output_path:
                logger.error("❌ Failed to create final output")
                return False
            
            # Success!
            logger.info("\n🎉 AUDIO INTEGRATION TEST COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"✅ TTS Audio: {tts_audio_path}")
            logger.info(f"✅ Mixed Audio: {mixed_audio_path}")
            logger.info(f"✅ Final Output: {final_output_path}")
            logger.info(f"\n🎯 Audio integration validation ready!")
            logger.info(f"   Play the output to hear the integrated audio:")
            logger.info(f"   - TTS narration (foreground)")
            logger.info(f"   - Gaming/ambient audio (background)")
            logger.info(f"   - Mobile-optimized mixing")
            logger.info(f"\n📂 Output location: {final_output_path.absolute()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Main test entry point."""
    print("🤖 TikTok Automata - Audio Integration Test")
    print("🎵 Testing NEW Audio Integration Module")
    print("=" * 60)
    
    test = AudioIntegrationTest()
    
    try:
        success = await test.run_complete_test()
        
        if success:
            print("\n✅ AUDIO INTEGRATION TEST SUCCESSFUL!")
            print("🎧 The output contains mixed audio:")
            print("   - TTS narration (foreground at 85% volume)")
            print("   - Gaming/ambient audio (background at 25% volume)")
            print("   - Master volume at 90% with mobile optimization")
            print("   - Fade transitions and EQ applied")
            print("\n🎯 Play the output file to validate audio quality!")
        else:
            print("\n❌ AUDIO INTEGRATION TEST FAILED!")
            print("Check the logs above for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # Run the test
    asyncio.run(main())
