"""
Audio Mixer - Combines TTS narration with gaming background audio.

This module handles:
1. Mixing TTS audio (foreground) with gaming footage audio (background)
2. Audio level balancing and normalization
3. Audio synchronization with video segments
4. Audio effects and processing
"""

import asyncio
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
from loguru import logger
import numpy as np

try:
    from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips
    from moviepy.audio.fx import MultiplyVolume, AudioFadeIn, AudioFadeOut
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    AUDIO_LIBS_AVAILABLE = False
    logger.error(f"Audio libraries not available: {e}")
    logger.error("Install with: pip install moviepy soundfile")


@dataclass
class AudioConfig:
    """Configuration for audio mixing and processing."""
    
    # Volume levels (0.0 to 1.0)
    tts_volume: float = 0.85          # TTS foreground volume
    background_volume: float = 0.25   # Gaming audio background volume
    master_volume: float = 0.9        # Overall output volume
    
    # Audio processing
    sample_rate: int = 24000          # Target sample rate
    bit_depth: int = 16               # Audio bit depth
    channels: int = 1                 # Mono output for TikTok
    
    # Effects
    enable_fade_transitions: bool = True
    fade_duration: float = 0.5        # Fade in/out duration in seconds
    enable_dynamic_range: bool = True  # Compress dynamic range for mobile
    enable_noise_gate: bool = True    # Remove background noise gaps
    
    # Processing
    normalize_audio: bool = True      # Normalize final output
    apply_eq: bool = True             # Apply EQ for mobile speakers
    compression_ratio: float = 0.8    # Audio compression (0.0-1.0)
    
    # Output format
    output_format: str = 'wav'        # Output format (wav, mp3, m4a)
    output_quality: str = 'high'      # low, medium, high


class AudioMixer:
    """
    Advanced audio mixer for combining TTS narration with gaming background audio.
    
    Features:
    - Intelligent volume balancing
    - Audio synchronization
    - Mobile-optimized processing
    - High-quality output for TikTok
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize the audio mixer with configuration."""
        self.config = config or AudioConfig()
        self.temp_dir = Path(tempfile.gettempdir()) / "tiktok_audio_mixer"
        self.temp_dir.mkdir(exist_ok=True)
        
        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError("Required audio libraries not available. Install moviepy and soundfile.")
        
        logger.info(f"AudioMixer initialized with config: {self.config}")
    
    async def mix_audio_streams(
        self,
        tts_audio_path: Union[str, Path],
        background_video_path: Union[str, Path],
        output_path: Union[str, Path],
        video_start_time: float = 0.0,
        video_duration: Optional[float] = None
    ) -> Path:
        """
        Mix TTS audio with gaming background audio from video.
        
        Args:
            tts_audio_path: Path to TTS-generated audio file
            background_video_path: Path to gaming video (source of background audio)
            output_path: Path for the mixed audio output
            video_start_time: Start time in video for background audio extraction
            video_duration: Duration to extract from video (defaults to TTS duration)
            
        Returns:
            Path to the mixed audio file
        """
        try:
            logger.info(f"🎵 Mixing audio streams...")
            logger.info(f"   TTS: {tts_audio_path}")
            logger.info(f"   Background: {background_video_path}")
            logger.info(f"   Output: {output_path}")
            
            # Load TTS audio
            tts_audio = AudioFileClip(str(tts_audio_path))
            tts_duration = tts_audio.duration
            
            # Use TTS duration if video duration not specified
            target_duration = video_duration or tts_duration
            
            # Extract background audio from video
            background_audio = await self._extract_background_audio(
                background_video_path, video_start_time, target_duration
            )
            
            # Process and balance audio levels
            processed_tts = await self._process_tts_audio(tts_audio)
            processed_background = await self._process_background_audio(background_audio, target_duration)
            
            # Mix the audio streams
            mixed_audio = await self._mix_audio_tracks(processed_tts, processed_background)
              # Apply final processing and save
            final_audio = await self._finalize_audio(mixed_audio)
            final_audio.write_audiofile(str(output_path), logger=None)
            
            # Cleanup
            tts_audio.close()
            background_audio.close()
            mixed_audio.close()
            final_audio.close()
            
            logger.info(f"✅ Audio mixing completed: {output_path}")
            return Path(output_path)
            
        except Exception as e:
            logger.error(f"❌ Audio mixing failed: {e}")
            raise
    
    async def _extract_background_audio(
        self, 
        video_path: Union[str, Path], 
        start_time: float, 
        duration: float
    ) -> AudioFileClip:
        """Extract background audio from gaming video."""
        try:
            from moviepy import VideoFileClip
            
            # Load video and extract audio
            video = VideoFileClip(str(video_path))
            
            # Calculate end time, ensuring we don't exceed video duration
            end_time = min(start_time + duration, video.duration)
            actual_duration = end_time - start_time
            
            if actual_duration <= 0:
                raise ValueError(f"Invalid time range: {start_time}s to {end_time}s")
            
            # Extract audio segment
            audio = video.audio.subclipped(start_time, end_time)
            video.close()
            
            logger.info(f"   Extracted {actual_duration:.1f}s background audio from {start_time:.1f}s")
            return audio
            
        except Exception as e:
            logger.error(f"Background audio extraction failed: {e}")
            raise
    async def _process_tts_audio(self, tts_audio: AudioFileClip) -> AudioFileClip:
        """Process TTS audio for optimal foreground presence."""
        try:
            # Apply TTS volume using MoviePy 2.1.2 syntax
            from moviepy.audio.fx import MultiplyVolume, AudioFadeIn, AudioFadeOut
            
            audio = tts_audio.with_effects([MultiplyVolume(self.config.tts_volume)])
            
            # Add fade effects if enabled
            if self.config.enable_fade_transitions:
                fade_duration = min(self.config.fade_duration, audio.duration / 4)
                audio = audio.with_fps(self.config.sample_rate)
                
                # Apply fade in and fade out effects
                effects = [
                    AudioFadeIn(fade_duration),
                    AudioFadeOut(fade_duration)
                ]
                audio = audio.with_effects(effects)
            
            # Apply EQ for voice clarity (boost mid frequencies)
            if self.config.apply_eq:
                audio = await self._apply_voice_eq(audio)
            
            logger.info(f"   TTS audio processed: {audio.duration:.1f}s at volume {self.config.tts_volume}")
            return audio
            
        except Exception as e:
            logger.error(f"TTS audio processing failed: {e}")
            raise
    async def _process_background_audio(
        self, 
        background_audio: AudioFileClip, 
        target_duration: float
    ) -> AudioFileClip:
        """Process background gaming audio for ambient atmosphere."""
        try:
            # Ensure background audio matches target duration
            if background_audio.duration < target_duration:
                # Loop the background audio if it's shorter than needed
                loops_needed = int(np.ceil(target_duration / background_audio.duration))
                audio_clips = [background_audio] * loops_needed
                audio = concatenate_audioclips(audio_clips).subclipped(0, target_duration)
            else:
                audio = background_audio.subclipped(0, target_duration)
            
            # Apply background volume (lower for ambient effect) using MoviePy 2.1.2 syntax
            from moviepy.audio.fx import MultiplyVolume, AudioFadeIn, AudioFadeOut
            
            audio = audio.with_effects([MultiplyVolume(self.config.background_volume)])
            
            # Apply gentle fade for smooth background
            if self.config.enable_fade_transitions:
                fade_duration = min(self.config.fade_duration * 2, audio.duration / 6)
                effects = [
                    AudioFadeIn(fade_duration),
                    AudioFadeOut(fade_duration)
                ]
                audio = audio.with_effects(effects)
            
            # Apply low-pass filter for ambient background effect
            if self.config.apply_eq:
                audio = await self._apply_background_eq(audio)
            
            logger.info(f"   Background audio processed: {audio.duration:.1f}s at volume {self.config.background_volume}")
            return audio
            
        except Exception as e:
            logger.error(f"Background audio processing failed: {e}")
            raise
    
    async def _mix_audio_tracks(
        self, 
        tts_audio: AudioFileClip, 
        background_audio: AudioFileClip
    ) -> CompositeAudioClip:
        """Mix TTS and background audio tracks."""
        try:
            # Ensure both tracks have the same duration
            duration = min(tts_audio.duration, background_audio.duration)
            
            tts_trimmed = tts_audio.subclipped(0, duration)
            background_trimmed = background_audio.subclipped(0, duration)
            
            # Create composite audio with TTS in foreground and gaming audio in background
            mixed = CompositeAudioClip([background_trimmed, tts_trimmed])
            
            logger.info(f"   Mixed audio tracks: {duration:.1f}s duration")
            return mixed
            
        except Exception as e:
            logger.error(f"Audio mixing failed: {e}")
            raise
    async def _finalize_audio(self, mixed_audio: CompositeAudioClip) -> AudioFileClip:
        """Apply final processing to mixed audio."""
        try:
            # Apply master volume using MoviePy 2.1.2 syntax
            from moviepy.audio.fx import MultiplyVolume
            
            audio = mixed_audio.with_effects([MultiplyVolume(self.config.master_volume)])
            
            # Set target sample rate
            if hasattr(audio, 'with_fps'):
                audio = audio.with_fps(self.config.sample_rate)
            
            # Apply dynamic range compression for mobile playback
            if self.config.enable_dynamic_range:
                audio = await self._apply_compression(audio)
            
            # Normalize audio levels if enabled
            if self.config.normalize_audio:
                audio = await self._normalize_audio(audio)
            
            logger.info(f"   Final audio processing completed")
            return audio
            
        except Exception as e:
            logger.error(f"Audio finalization failed: {e}")
            raise
    
    async def _apply_voice_eq(self, audio: AudioFileClip) -> AudioFileClip:
        """Apply EQ optimized for voice clarity."""
        # Placeholder for EQ processing
        # In a full implementation, this would boost 1-4kHz range for voice clarity
        return audio
    
    async def _apply_background_eq(self, audio: AudioFileClip) -> AudioFileClip:
        """Apply EQ for background audio (low-pass filter)."""
        # Placeholder for EQ processing
        # In a full implementation, this would apply low-pass filter to reduce high frequencies
        return audio
    
    async def _apply_compression(self, audio: AudioFileClip) -> AudioFileClip:
        """Apply dynamic range compression for mobile playback."""
        # Placeholder for compression
        # In a full implementation, this would apply audio compression
        return audio
    
    async def _normalize_audio(self, audio: AudioFileClip) -> AudioFileClip:
        """Normalize audio levels to prevent clipping."""
        # Placeholder for normalization
        # In a full implementation, this would normalize peak levels
        return audio
    
    def get_audio_info(self, audio_path: Union[str, Path]) -> Dict:
        """Get detailed information about an audio file."""
        try:
            audio = AudioFileClip(str(audio_path))
            info = {
                'duration': audio.duration,
                'fps': audio.fps if hasattr(audio, 'fps') else None,
                'channels': getattr(audio, 'nchannels', None),
                'format': Path(audio_path).suffix.lower()
            }
            audio.close()
            return info
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {}
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("🧹 Cleaned up temporary audio files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")


# Helper function for convenience
async def mix_tts_with_gaming_audio(
    tts_audio_path: Union[str, Path],
    gaming_video_path: Union[str, Path],
    output_path: Union[str, Path],
    video_start_time: float = 0.0,
    config: Optional[AudioConfig] = None
) -> Path:
    """
    Convenience function to mix TTS audio with gaming background audio.
    
    Args:
        tts_audio_path: Path to TTS audio file
        gaming_video_path: Path to gaming video file  
        output_path: Path for mixed audio output
        video_start_time: Start time in gaming video
        config: Audio mixing configuration
        
    Returns:
        Path to mixed audio file
    """
    mixer = AudioMixer(config)
    try:
        return await mixer.mix_audio_streams(
            tts_audio_path, 
            gaming_video_path, 
            output_path,
            video_start_time
        )
    finally:
        mixer.cleanup_temp_files()
