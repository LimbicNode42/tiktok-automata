"""
Audio Integration Example - Shows how to use the audio mixing system.

This example demonstrates:
1. Basic audio mixing workflow
2. Integration with existing video pipeline
3. TTS + Gaming audio combination
"""

import asyncio
from pathlib import Path
from typing import Optional

# Example usage of the audio integration system
async def integrate_audio_example():
    """Example of complete audio integration workflow."""
    
    print("🎵 Audio Integration Example")
    print("=" * 50)
    
    # Step 1: Import modules (with error handling)
    try:
        from src.video.audio import AudioMixer, AudioConfig
        from src.tts.kokoro_tts import KokoroTTS
        print("✅ Modules imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Step 2: Configure audio processing
    audio_config = AudioConfig(
        tts_volume=0.85,          # TTS narration at 85% volume
        background_volume=0.25,   # Gaming audio at 25% volume (ambient)
        master_volume=0.9,        # Overall output at 90%
        enable_fade_transitions=True,
        normalize_audio=True,
        apply_eq=True            # Mobile-optimized EQ
    )
    
    print(f"🔧 Audio Config:")
    print(f"   TTS Volume: {audio_config.tts_volume}")
    print(f"   Background: {audio_config.background_volume}")
    print(f"   Master: {audio_config.master_volume}")
    
    # Step 3: Initialize audio mixer
    mixer = AudioMixer(audio_config)
    
    # Step 4: Example workflow
    print("\n📋 Audio Integration Workflow:")
    print("1. Generate TTS audio from text content")
    print("2. Select gaming video segment") 
    print("3. Extract background audio from gaming footage")
    print("4. Mix TTS (foreground) + Gaming (background)")
    print("5. Apply effects and mobile optimization")
    print("6. Synchronize with video timing")
    print("7. Output final mixed audio")
    
    # Step 5: Show configuration options
    print(f"\n⚙️ Available Audio Processing:")
    print(f"   - Voice enhancement for TTS clarity")
    print(f"   - Background ambience processing")
    print(f"   - Dynamic range compression for mobile")
    print(f"   - Audio-video synchronization")
    print(f"   - Fade transitions and effects")
    
    print("\n✅ Audio integration system ready!")
    return True

# Integration with video processor
class VideoWithAudioProcessor:
    """Enhanced video processor with audio integration."""
    
    def __init__(self):
        self.audio_config = AudioConfig()
        self.mixer = None
        
    async def create_tiktok_video_with_audio(
        self,
        text_content: str,
        gaming_video_path: str,
        video_start_time: float,
        video_duration: float,
        output_path: str
    ):
        """
        Create TikTok video with mixed audio (TTS + Gaming background).
        
        Workflow:
        1. Generate TTS audio from text
        2. Select and crop gaming video segment  
        3. Extract background audio from gaming video
        4. Mix TTS audio with gaming background
        5. Combine mixed audio with cropped video
        6. Output final TikTok-ready video
        """
        
        print(f"🎬 Creating TikTok video with audio integration")
        print(f"   Text: {text_content[:50]}...")
        print(f"   Gaming video: {gaming_video_path}")
        print(f"   Segment: {video_start_time}s - {video_start_time + video_duration}s")
        
        # Step 1: Generate TTS audio
        # (This would integrate with the existing TTS module)
        print("🎤 Step 1: Generate TTS audio")
        
        # Step 2: Process gaming video
        # (This would integrate with the existing video processor)
        print("🎮 Step 2: Process gaming video segment")
        
        # Step 3: Mix audio streams
        print("🎛️ Step 3: Mix TTS + Gaming audio")
        
        # Step 4: Combine audio with video
        print("🎬 Step 4: Combine audio with video")
        
        # Step 5: Apply final processing
        print("✨ Step 5: Apply final processing")
        
        print(f"✅ TikTok video created: {output_path}")
        
        return output_path

if __name__ == "__main__":
    # Run the example
    asyncio.run(integrate_audio_example())
