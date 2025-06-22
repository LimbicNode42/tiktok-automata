#!/usr/bin/env python3
"""
Quick Audio Integration Test - Audio Files Only

This simplified test validates the audio integration module using just audio files
to isolate and validate the core audio mixing functionality.
"""

import asyncio
import sys
import os
from pathlib import Path
from loguru import logger
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.tts.kokoro_tts import KokoroTTSEngine
    from src.video.audio.audio_mixer import mix_tts_with_gaming_audio, AudioConfig
    import soundfile as sf
    print("✅ Components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import components: {e}")
    sys.exit(1)


async def test_audio_only():
    """Test audio integration with just audio files."""
    logger.info("🎵 Starting Audio-Only Integration Test")
    
    test_dir = Path("audio_quick_test")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Generate TTS audio
        logger.info("🎤 Generating TTS audio...")
        tts = KokoroTTSEngine()
        
        script = """
        This is a test of the new audio integration system. We're mixing TTS narration 
        with background gaming audio to create engaging TikTok content. The system 
        balances volume levels and applies mobile optimization for the best listening experience.
        """
        
        tts_path = await tts.generate_audio(
            text=script.strip(),
            output_path=str(test_dir / "test_tts.wav"),
            voice='af_heart',
            speed=1.1
        )
        
        if not tts_path or not Path(tts_path).exists():
            logger.error("❌ TTS generation failed")
            return False
            
        logger.info(f"✅ TTS audio: {tts_path}")
          # Step 2: Use existing gaming video instead of creating background audio
        logger.info("🎮 Using existing gaming video...")
        
        # Check for gaming footage
        footage_dir = Path("src/video/data/footage/raw")
        video_files = list(footage_dir.glob("*.mp4"))
        
        if not video_files:
            logger.error("❌ No gaming video found for testing")
            return False
            
        background_path = video_files[0]
        logger.info(f"✅ Gaming video: {background_path}")
        
        # Step 3: Test audio integration using standalone function
        logger.info("🎵 Testing audio integration...")
        
        config = AudioConfig(
            tts_volume=0.85,
            background_volume=0.25,
            master_volume=0.90,
            fade_duration=0.5,
            enable_dynamic_range=True,
            apply_eq=True
        )
        
        mixed_path = test_dir / "mixed_audio.wav"
          # Use the standalone function for simpler testing
        result_path = await mix_tts_with_gaming_audio(
            tts_audio_path=tts_path,
            gaming_video_path=background_path,  # Use gaming video file
            output_path=mixed_path,
            config=config
        )
        
        if result_path and Path(result_path).exists():
            logger.info(f"✅ Audio integration successful: {result_path}")
            logger.info(f"   File size: {Path(result_path).stat().st_size / 1024:.1f} KB")
            
            # Verify the output
            mixed_audio, output_sample_rate = sf.read(result_path)
            logger.info(f"   Duration: {len(mixed_audio) / output_sample_rate:.1f} seconds")
            logger.info(f"   Peak level: {np.max(np.abs(mixed_audio)):.3f}")
            
            logger.info("\n🎉 AUDIO INTEGRATION TEST SUCCESSFUL!")
            logger.info(f"🎧 Mixed audio ready for validation: {result_path.absolute()}")
            logger.info("   - TTS narration at 85% volume")
            logger.info("   - Background audio at 25% volume") 
            logger.info("   - Master volume at 90%")
            logger.info("   - Fade transitions and EQ applied")
            
            return True
        else:
            logger.error("❌ Audio integration failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test entry."""
    print("🤖 TikTok Automata - Quick Audio Integration Test")
    print("🎵 Testing NEW Audio Integration Module (Audio Files Only)")
    print("=" * 60)
    
    try:
        success = await test_audio_only()
        
        if success:
            print("\n✅ QUICK AUDIO TEST SUCCESSFUL!")
            print("🎵 The new audio integration module is working correctly!")
            print("📂 Check the audio_quick_test/ directory for output files")
        else:
            print("\n❌ QUICK AUDIO TEST FAILED!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    asyncio.run(main())
