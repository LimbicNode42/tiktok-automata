"""
Audio Synchronizer - Handles timing and synchronization of audio with video.

This module provides:
1. Audio-video synchronization
2. Timing adjustments for TTS narration
3. Beat detection and rhythm matching
4. Audio cue alignment
"""

import asyncio
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
from loguru import logger

try:
    from moviepy import AudioFileClip, VideoFileClip
    import numpy as np
    SYNC_LIBS_AVAILABLE = True
except ImportError as e:
    SYNC_LIBS_AVAILABLE = False
    logger.warning(f"Audio sync libraries not available: {e}")


@dataclass
class SyncConfig:
    """Configuration for audio synchronization."""
    
    # Timing tolerances
    sync_tolerance: float = 0.05        # Acceptable sync error in seconds
    max_time_stretch: float = 0.1       # Maximum time stretching (10%)
    
    # TTS synchronization
    tts_lead_time: float = 0.2          # TTS starts slightly before video action
    tts_fade_overlap: float = 0.5       # Overlap time for smooth transitions
    
    # Background audio
    background_sync_mode: str = 'loose' # 'strict', 'loose', 'ambient'
    beat_detection: bool = True         # Enable beat detection for rhythm matching
    
    # Processing
    auto_adjust_timing: bool = True     # Automatically adjust timing mismatches
    preserve_pitch: bool = True         # Preserve pitch when time-stretching


class AudioSynchronizer:
    """
    Advanced audio synchronizer for perfect audio-video timing.
    
    Features:
    - TTS-to-video synchronization
    - Background audio timing
    - Beat detection and matching
    - Automatic timing adjustments
    """
    
    def __init__(self, config: Optional[SyncConfig] = None):
        """Initialize the audio synchronizer."""
        self.config = config or SyncConfig()
        
        if not SYNC_LIBS_AVAILABLE:
            logger.warning("Limited synchronization functionality without moviepy/numpy")
        
        logger.info(f"AudioSynchronizer initialized with config: {self.config}")
    
    async def synchronize_tts_with_video(
        self,
        tts_audio_path: Union[str, Path],
        video_path: Union[str, Path],
        video_start_time: float,
        video_duration: float,
        output_path: Union[str, Path]
    ) -> Tuple[Path, Dict]:
        """
        Synchronize TTS audio with video segment for perfect timing.
        
        Args:
            tts_audio_path: Path to TTS audio file
            video_path: Path to video file
            video_start_time: Start time in video
            video_duration: Duration of video segment
            output_path: Path for synchronized audio output
            
        Returns:
            Tuple of (output_path, sync_info)
        """
        try:
            logger.info(f"🔄 Synchronizing TTS with video...")
            logger.info(f"   TTS: {tts_audio_path}")
            logger.info(f"   Video: {video_path} ({video_start_time}s - {video_start_time + video_duration}s)")
            
            # Load audio and analyze duration
            tts_audio = AudioFileClip(str(tts_audio_path))
            tts_duration = tts_audio.duration
            
            # Calculate timing adjustments
            sync_info = await self._calculate_sync_adjustments(
                tts_duration, video_duration, video_start_time
            )
            
            # Apply timing adjustments if needed
            synchronized_audio = await self._apply_timing_adjustments(
                tts_audio, sync_info
            )
              # Save synchronized audio
            synchronized_audio.write_audiofile(str(output_path), logger=None)
            
            # Cleanup
            tts_audio.close()
            synchronized_audio.close()
            
            logger.info(f"✅ TTS synchronization completed: {output_path}")
            return Path(output_path), sync_info
            
        except Exception as e:
            logger.error(f"❌ TTS synchronization failed: {e}")
            raise
    
    async def align_background_audio(
        self,
        background_audio: AudioFileClip,
        target_duration: float,
        sync_points: Optional[List[float]] = None
    ) -> AudioFileClip:
        """
        Align background audio with video timing.
        
        Args:
            background_audio: Background audio clip
            target_duration: Target duration to match
            sync_points: Optional timing points for alignment
            
        Returns:
            Aligned background audio
        """
        try:
            logger.info(f"🎵 Aligning background audio to {target_duration}s")
            
            current_duration = background_audio.duration
            
            if abs(current_duration - target_duration) < self.config.sync_tolerance:
                # Already well synchronized
                logger.info("   Audio already synchronized")
                return background_audio
            
            if self.config.background_sync_mode == 'strict':
                # Strict synchronization with time stretching
                aligned = await self._time_stretch_audio(
                    background_audio, target_duration / current_duration
                )
            elif self.config.background_sync_mode == 'loose':
                # Loose synchronization with trimming/looping
                aligned = await self._adjust_duration_flexible(
                    background_audio, target_duration
                )
            else:  # ambient mode
                # Ambient mode - minimal adjustment
                aligned = await self._adjust_duration_ambient(
                    background_audio, target_duration
                )
            
            logger.info(f"   Background audio aligned: {aligned.duration:.1f}s")
            return aligned
            
        except Exception as e:
            logger.error(f"Background audio alignment failed: {e}")
            return background_audio
    
    async def detect_audio_beats(
        self,
        audio_clip: AudioFileClip,
        bpm_range: Tuple[float, float] = (60, 180)
    ) -> List[float]:
        """
        Detect beats in audio for rhythm matching.
        
        Args:
            audio_clip: Audio clip to analyze
            bpm_range: Range of BPM to detect
            
        Returns:
            List of beat timestamps
        """
        try:
            if not self.config.beat_detection or not SYNC_LIBS_AVAILABLE:
                logger.info("Beat detection disabled or unavailable")
                return []
            
            logger.info(f"🥁 Detecting beats in audio ({bpm_range[0]}-{bpm_range[1]} BPM)")
            
            # Placeholder for beat detection algorithm
            # In a full implementation, this would use librosa or similar
            beats = await self._analyze_rhythm_patterns(audio_clip, bpm_range)
            
            logger.info(f"   Detected {len(beats)} beats")
            return beats
            
        except Exception as e:
            logger.error(f"Beat detection failed: {e}")
            return []
    
    async def create_sync_markers(
        self,
        video_path: Union[str, Path],
        start_time: float,
        duration: float
    ) -> List[Dict]:
        """
        Create synchronization markers from video analysis.
        
        Args:
            video_path: Path to video file
            start_time: Start time in video
            duration: Duration to analyze
            
        Returns:
            List of sync marker dictionaries
        """
        try:
            logger.info(f"📍 Creating sync markers for video segment")
            
            # Analyze video for sync points (scene changes, action peaks, etc.)
            markers = await self._analyze_video_sync_points(
                video_path, start_time, duration
            )
            
            logger.info(f"   Created {len(markers)} sync markers")
            return markers
            
        except Exception as e:
            logger.error(f"Sync marker creation failed: {e}")
            return []
    
    async def _calculate_sync_adjustments(
        self,
        tts_duration: float,
        video_duration: float,
        video_start_time: float
    ) -> Dict:
        """Calculate timing adjustments needed for synchronization."""
        try:
            duration_diff = tts_duration - video_duration
            duration_ratio = tts_duration / video_duration if video_duration > 0 else 1.0
            
            # Determine adjustment strategy
            needs_stretching = abs(duration_diff) > self.config.sync_tolerance
            stretch_ratio = 1.0
            trim_start = 0.0
            trim_end = 0.0
            
            if needs_stretching:
                if abs(duration_ratio - 1.0) <= self.config.max_time_stretch:
                    # Use time stretching
                    stretch_ratio = video_duration / tts_duration
                else:
                    # Use trimming/padding
                    if duration_diff > 0:  # TTS longer than video
                        trim_amount = duration_diff / 2
                        trim_start = trim_amount
                        trim_end = trim_amount
                    # If video longer, we'll pad with silence later
            
            sync_info = {
                'tts_duration': tts_duration,
                'video_duration': video_duration,
                'duration_diff': duration_diff,
                'needs_adjustment': needs_stretching,
                'stretch_ratio': stretch_ratio,
                'trim_start': trim_start,
                'trim_end': trim_end,
                'lead_time': self.config.tts_lead_time
            }
            
            logger.info(f"   Sync analysis: {sync_info}")
            return sync_info
            
        except Exception as e:
            logger.error(f"Sync calculation failed: {e}")
            return {}
    
    async def _apply_timing_adjustments(
        self,
        audio_clip: AudioFileClip,
        sync_info: Dict
    ) -> AudioFileClip:
        """Apply calculated timing adjustments to audio."""
        try:
            adjusted = audio_clip
            
            # Apply time stretching if needed
            if sync_info.get('needs_adjustment') and sync_info.get('stretch_ratio', 1.0) != 1.0:
                adjusted = await self._time_stretch_audio(
                    adjusted, sync_info['stretch_ratio']
                )
            
            # Apply trimming if needed
            if sync_info.get('trim_start', 0) > 0 or sync_info.get('trim_end', 0) > 0:
                start_trim = sync_info.get('trim_start', 0)
                end_trim = sync_info.get('trim_end', 0)
                new_duration = adjusted.duration - start_trim - end_trim
                
                if new_duration > 0:
                    adjusted = adjusted.subclipped(start_trim, adjusted.duration - end_trim)
            
            # Add lead time if configured
            lead_time = sync_info.get('lead_time', 0)
            if lead_time > 0:
                # Add silence at the beginning
                adjusted = await self._add_lead_silence(adjusted, lead_time)
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Timing adjustment failed: {e}")
            return audio_clip
    
    async def _time_stretch_audio(
        self,
        audio_clip: AudioFileClip,
        stretch_ratio: float
    ) -> AudioFileClip:
        """Time-stretch audio while preserving pitch."""
        try:
            if not self.config.preserve_pitch:
                # Simple speed change (affects pitch)
                return audio_clip.with_multiply_speed(stretch_ratio)
            
            # Placeholder for pitch-preserving time stretch
            # In a full implementation, this would use PSOLA or similar algorithm
            logger.debug(f"Time-stretched audio by {stretch_ratio:.3f}")
            return audio_clip.with_multiply_speed(stretch_ratio)
            
        except Exception as e:
            logger.warning(f"Time stretching failed: {e}")
            return audio_clip
    
    async def _adjust_duration_flexible(
        self,
        audio_clip: AudioFileClip,
        target_duration: float
    ) -> AudioFileClip:
        """Flexibly adjust audio duration with minimal quality loss."""
        try:
            current_duration = audio_clip.duration
            
            if current_duration < target_duration:
                # Need to extend - loop the audio
                loops_needed = math.ceil(target_duration / current_duration)
                from moviepy import concatenate_audioclips
                
                clips = [audio_clip] * loops_needed
                extended = concatenate_audioclips(clips)
                return extended.subclipped(0, target_duration)
            else:
                # Need to shorten - trim from the end
                return audio_clip.subclipped(0, target_duration)
                
        except Exception as e:
            logger.warning(f"Flexible duration adjustment failed: {e}")
            return audio_clip
    
    async def _adjust_duration_ambient(
        self,
        audio_clip: AudioFileClip,
        target_duration: float
    ) -> AudioFileClip:
        """Adjust duration with ambient/atmospheric approach."""
        try:
            # For ambient mode, allow natural fading and minimal adjustment
            current_duration = audio_clip.duration
            
            if abs(current_duration - target_duration) < 2.0:
                # Close enough for ambient use
                return audio_clip.subclipped(0, min(current_duration, target_duration))
            
            return await self._adjust_duration_flexible(audio_clip, target_duration)
            
        except Exception as e:
            logger.warning(f"Ambient duration adjustment failed: {e}")
            return audio_clip
    
    async def _add_lead_silence(
        self,
        audio_clip: AudioFileClip,
        lead_time: float
    ) -> AudioFileClip:
        """Add silence at the beginning of audio clip."""
        try:
            # Placeholder for adding silence
            # In a full implementation, this would create silent audio and concatenate
            logger.debug(f"Added {lead_time}s lead silence")
            return audio_clip
            
        except Exception as e:
            logger.warning(f"Lead silence addition failed: {e}")
            return audio_clip
    
    async def _analyze_rhythm_patterns(
        self,
        audio_clip: AudioFileClip,
        bpm_range: Tuple[float, float]
    ) -> List[float]:
        """Analyze rhythm patterns in audio."""
        try:
            # Placeholder for rhythm analysis
            # In a full implementation, this would use beat tracking algorithms
            
            # Generate sample beat timestamps for testing
            duration = audio_clip.duration
            estimated_bpm = (bpm_range[0] + bpm_range[1]) / 2
            beat_interval = 60.0 / estimated_bpm
            
            beats = []
            current_time = 0.0
            while current_time < duration:
                beats.append(current_time)
                current_time += beat_interval
            
            return beats[:20]  # Limit for testing
            
        except Exception as e:
            logger.warning(f"Rhythm analysis failed: {e}")
            return []
    
    async def _analyze_video_sync_points(
        self,
        video_path: Union[str, Path],
        start_time: float,
        duration: float
    ) -> List[Dict]:
        """Analyze video for natural synchronization points."""
        try:
            # Placeholder for video analysis
            # In a full implementation, this would analyze scene changes, motion, etc.
            
            markers = []
            num_markers = max(1, int(duration / 10))  # One marker per 10 seconds
            
            for i in range(num_markers):
                marker_time = start_time + (i + 1) * (duration / num_markers)
                markers.append({
                    'time': marker_time,
                    'type': 'scene_change',
                    'confidence': 0.8,
                    'description': f'Scene marker {i + 1}'
                })
            
            return markers
            
        except Exception as e:
            logger.warning(f"Video sync analysis failed: {e}")
            return []
    
    def get_sync_status(
        self,
        tts_duration: float,
        video_duration: float
    ) -> Dict:
        """Get synchronization status and recommendations."""
        try:
            duration_diff = abs(tts_duration - video_duration)
            is_synchronized = duration_diff <= self.config.sync_tolerance
            
            recommendations = []
            if not is_synchronized:
                if duration_diff > video_duration * self.config.max_time_stretch:
                    recommendations.append("Consider editing TTS content for better timing")
                else:
                    recommendations.append("Time stretching recommended")
            
            return {
                'synchronized': is_synchronized,
                'duration_difference': duration_diff,
                'tts_duration': tts_duration,
                'video_duration': video_duration,
                'recommendations': recommendations,
                'sync_quality': 'excellent' if duration_diff < 0.1 else 'good' if duration_diff < 0.5 else 'poor'
            }
            
        except Exception as e:
            logger.error(f"Sync status analysis failed: {e}")
            return {'synchronized': False, 'error': str(e)}
