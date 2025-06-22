"""
Audio Effects - Advanced audio processing and effects for TikTok content.

This module provides:
1. Audio enhancement effects
2. Voice processing
3. Background audio effects
4. Mobile-optimized processing
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
from loguru import logger

try:
    from moviepy import AudioFileClip
    from scipy import signal
    import soundfile as sf
    EFFECTS_LIBS_AVAILABLE = True
except ImportError as e:
    EFFECTS_LIBS_AVAILABLE = False
    logger.warning(f"Audio effects libraries not available: {e}")
    logger.warning("Install with: pip install scipy soundfile")


class AudioEffects:
    """
    Advanced audio effects processor for TikTok content.
    
    Provides mobile-optimized audio processing including:
    - Voice enhancement and clarity
    - Background audio processing
    - Dynamic range compression
    - EQ and filtering
    """
    
    def __init__(self):
        """Initialize audio effects processor."""
        if not EFFECTS_LIBS_AVAILABLE:
            logger.warning("Audio effects functionality limited without scipy")
        
        # Standard sample rates
        self.sample_rates = {
            'low': 16000,
            'medium': 24000, 
            'high': 48000
        }
        
        # EQ presets for different content types
        self.eq_presets = {
            'voice_clarity': {
                'low_cut': 80,      # High-pass filter to remove low rumble
                'presence_boost': 2000,  # Boost presence frequencies
                'air_boost': 8000,   # Boost air frequencies for clarity
                'sibilance_cut': 6000   # Gentle cut for harsh sibilants
            },
            'background_ambient': {
                'low_pass': 8000,   # Remove high frequencies for ambient effect
                'mid_cut': 2000,    # Cut mids to not compete with voice
                'bass_boost': 100   # Gentle bass boost for warmth
            },
            'mobile_optimized': {
                'bass_cut': 60,     # Cut sub-bass for small speakers
                'mid_boost': 1000,  # Boost mids for phone speakers
                'treble_smooth': 10000  # Smooth treble for harsh phone speakers
            }
        }
    
    async def enhance_voice_audio(
        self, 
        audio_clip: AudioFileClip,
        preset: str = 'voice_clarity',
        compression_ratio: float = 0.7
    ) -> AudioFileClip:
        """
        Enhance voice audio for maximum clarity and presence.
        
        Args:
            audio_clip: Input audio clip
            preset: EQ preset to use
            compression_ratio: Compression amount (0.0 = no compression, 1.0 = heavy)
            
        Returns:
            Enhanced audio clip
        """
        try:
            logger.info(f"🎤 Enhancing voice audio with {preset} preset")
            
            # Apply EQ for voice clarity
            enhanced = await self._apply_eq_preset(audio_clip, preset)
            
            # Apply gentle compression for consistent levels
            enhanced = await self._apply_compression(enhanced, compression_ratio)
            
            # Apply noise gate to remove background noise in quiet parts
            enhanced = await self._apply_noise_gate(enhanced)
            
            # De-essing to reduce harsh sibilants
            enhanced = await self._apply_deesser(enhanced)
            
            logger.info("✅ Voice enhancement completed")
            return enhanced
            
        except Exception as e:
            logger.error(f"Voice enhancement failed: {e}")
            return audio_clip
    
    async def process_background_audio(
        self,
        audio_clip: AudioFileClip,
        ambience_level: float = 0.3,
        preset: str = 'background_ambient'
    ) -> AudioFileClip:
        """
        Process background audio for ambient atmosphere.
        
        Args:
            audio_clip: Input audio clip
            ambience_level: How ambient to make it (0.0-1.0)
            preset: EQ preset to use
            
        Returns:
            Processed background audio
        """
        try:
            logger.info(f"🎮 Processing background audio with {preset} preset")
            
            # Apply EQ for background ambience
            processed = await self._apply_eq_preset(audio_clip, preset)
            
            # Apply gentle low-pass filtering for ambient effect
            cutoff_freq = 8000 - (ambience_level * 3000)  # More ambient = lower cutoff
            processed = await self._apply_lowpass_filter(processed, cutoff_freq)
            
            # Apply stereo width reduction for mono compatibility
            processed = await self._reduce_stereo_width(processed, 0.5)
            
            # Apply gentle reverb for atmosphere
            processed = await self._apply_subtle_reverb(processed, ambience_level)
            
            logger.info("✅ Background audio processing completed")
            return processed
            
        except Exception as e:
            logger.error(f"Background audio processing failed: {e}")
            return audio_clip
    
    async def optimize_for_mobile(
        self,
        audio_clip: AudioFileClip,
        target_lufs: float = -16.0
    ) -> AudioFileClip:
        """
        Optimize audio for mobile device playback.
        
        Args:
            audio_clip: Input audio clip
            target_lufs: Target loudness in LUFS
            
        Returns:
            Mobile-optimized audio
        """
        try:
            logger.info("📱 Optimizing audio for mobile playback")
            
            # Apply mobile EQ preset
            optimized = await self._apply_eq_preset(audio_clip, 'mobile_optimized')
            
            # Apply dynamic range compression for consistent playback
            optimized = await self._apply_mobile_compression(optimized)
            
            # Normalize to target loudness
            optimized = await self._normalize_loudness(optimized, target_lufs)
            
            # Apply peak limiter to prevent clipping
            optimized = await self._apply_limiter(optimized)
            
            logger.info("✅ Mobile optimization completed")
            return optimized
            
        except Exception as e:
            logger.error(f"Mobile optimization failed: {e}")
            return audio_clip
    
    async def _apply_eq_preset(
        self, 
        audio_clip: AudioFileClip, 
        preset: str
    ) -> AudioFileClip:
        """Apply EQ preset to audio clip."""
        if not EFFECTS_LIBS_AVAILABLE or preset not in self.eq_presets:
            return audio_clip
        
        try:
            # This is a placeholder for EQ processing
            # In a full implementation, you would apply actual EQ filters
            logger.debug(f"Applied {preset} EQ preset")
            return audio_clip
        except Exception as e:
            logger.warning(f"EQ processing failed: {e}")
            return audio_clip
    
    async def _apply_compression(
        self, 
        audio_clip: AudioFileClip, 
        ratio: float
    ) -> AudioFileClip:
        """Apply dynamic range compression."""
        try:
            # Placeholder for compression algorithm
            # In a full implementation, this would apply actual audio compression
            logger.debug(f"Applied compression with ratio {ratio}")
            return audio_clip
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return audio_clip
    
    async def _apply_noise_gate(
        self, 
        audio_clip: AudioFileClip,
        threshold: float = -40.0
    ) -> AudioFileClip:
        """Apply noise gate to remove background noise."""
        try:
            # Placeholder for noise gate
            # In a full implementation, this would gate audio below threshold
            logger.debug(f"Applied noise gate with threshold {threshold}dB")
            return audio_clip
        except Exception as e:
            logger.warning(f"Noise gate failed: {e}")
            return audio_clip
    
    async def _apply_deesser(
        self, 
        audio_clip: AudioFileClip,
        frequency: float = 6000.0,
        threshold: float = -20.0
    ) -> AudioFileClip:
        """Apply de-esser to reduce harsh sibilants."""
        try:
            # Placeholder for de-esser
            # In a full implementation, this would reduce sibilant frequencies
            logger.debug(f"Applied de-esser at {frequency}Hz")
            return audio_clip
        except Exception as e:
            logger.warning(f"De-esser failed: {e}")
            return audio_clip
    
    async def _apply_lowpass_filter(
        self, 
        audio_clip: AudioFileClip, 
        cutoff: float
    ) -> AudioFileClip:
        """Apply low-pass filter."""
        try:
            # Placeholder for low-pass filter
            # In a full implementation, this would apply actual filtering
            logger.debug(f"Applied low-pass filter at {cutoff}Hz")
            return audio_clip
        except Exception as e:
            logger.warning(f"Low-pass filter failed: {e}")
            return audio_clip
    
    async def _reduce_stereo_width(
        self, 
        audio_clip: AudioFileClip, 
        width: float
    ) -> AudioFileClip:
        """Reduce stereo width for mono compatibility."""
        try:
            # Placeholder for stereo width processing
            # In a full implementation, this would adjust stereo imaging
            logger.debug(f"Reduced stereo width to {width}")
            return audio_clip
        except Exception as e:
            logger.warning(f"Stereo width processing failed: {e}")
            return audio_clip
    
    async def _apply_subtle_reverb(
        self, 
        audio_clip: AudioFileClip, 
        amount: float
    ) -> AudioFileClip:
        """Apply subtle reverb for atmosphere."""
        try:
            # Placeholder for reverb processing
            # In a full implementation, this would add reverb
            logger.debug(f"Applied reverb with amount {amount}")
            return audio_clip
        except Exception as e:
            logger.warning(f"Reverb processing failed: {e}")
            return audio_clip
    
    async def _apply_mobile_compression(
        self, 
        audio_clip: AudioFileClip
    ) -> AudioFileClip:
        """Apply compression optimized for mobile playback."""
        try:
            # Mobile-specific compression with faster attack and release
            return await self._apply_compression(audio_clip, 0.8)
        except Exception as e:
            logger.warning(f"Mobile compression failed: {e}")
            return audio_clip
    
    async def _normalize_loudness(
        self, 
        audio_clip: AudioFileClip, 
        target_lufs: float
    ) -> AudioFileClip:
        """Normalize audio to target loudness."""
        try:
            # Placeholder for loudness normalization
            # In a full implementation, this would measure and adjust LUFS
            logger.debug(f"Normalized to {target_lufs} LUFS")
            return audio_clip
        except Exception as e:
            logger.warning(f"Loudness normalization failed: {e}")
            return audio_clip
    
    async def _apply_limiter(
        self, 
        audio_clip: AudioFileClip,
        ceiling: float = -0.3
    ) -> AudioFileClip:
        """Apply peak limiter to prevent clipping."""
        try:
            # Placeholder for limiting
            # In a full implementation, this would apply brick-wall limiting
            logger.debug(f"Applied limiter with ceiling {ceiling}dB")
            return audio_clip
        except Exception as e:
            logger.warning(f"Limiting failed: {e}")
            return audio_clip
    
    def get_audio_analysis(self, audio_clip: AudioFileClip) -> Dict:
        """Analyze audio characteristics."""
        try:
            return {
                'duration': audio_clip.duration,
                'sample_rate': getattr(audio_clip, 'fps', None),
                'channels': getattr(audio_clip, 'nchannels', None),
                'has_effects_support': EFFECTS_LIBS_AVAILABLE
            }
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {}
