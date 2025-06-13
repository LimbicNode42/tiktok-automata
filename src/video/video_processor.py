"""
Main Video Processor for TikTok content generation.

Handles the complete video pipeline:
1. Source footage selection and preparation
2. Audio synchronization 
3. Text overlay generation
4. Effects and transitions
5. Final video assembly
"""

import asyncio
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from loguru import logger
import time

try:
    from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip, concatenate_videoclips
    # Import video effects from the video.fx module
    from moviepy.video.fx import Resize, MultiplySpeed, FadeIn, FadeOut
except ImportError as e:
    logger.error(f"MoviePy not installed or import failed: {e}")
    logger.error("Run: pip install --upgrade moviepy")
    raise


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    # TikTok specifications
    width: int = 1080           # TikTok width
    height: int = 1920          # TikTok height (9:16 ratio)
    fps: int = 30               # Frame rate
    duration: int = 120         # Target duration in seconds
    
    # Content settings
    use_gaming_footage: bool = True
    footage_intensity: str = "medium"  # low, medium, high
    text_overlay_style: str = "modern"  # modern, classic, minimal
    
    # Effects
    enable_zoom_effects: bool = True
    enable_transitions: bool = True
    enable_beat_sync: bool = True
    
    # Output
    output_quality: str = "high"  # low, medium, high
    output_format: str = "mp4"
    max_file_size_mb: int = 30   # TikTok limit


class VideoProcessor:
    """
    Main video processor for creating TikTok videos from gaming footage.
    
    Features:
    - Gaming footage selection and editing
    - Audio-synchronized editing
    - Dynamic text overlays
    - TikTok-optimized output
    """
    
    def __init__(self, config: VideoConfig = None):
        """Initialize the video processor."""
        self.config = config or VideoConfig()
        
        # Module paths
        self.footage_dir = Path(__file__).parent / "data" / "footage"
        self.temp_dir = Path(__file__).parent / "data" / "temp"
        self.output_dir = Path(__file__).parent / "data" / "output"
        
        # Create directories
        for dir_path in [self.footage_dir, self.temp_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load gaming footage metadata
        self.footage_metadata = self._load_footage_metadata()
        
        logger.info(f"VideoProcessor initialized for {self.config.width}x{self.config.height} @ {self.config.fps}fps")
    
    def _load_footage_metadata(self) -> Dict:
        """Load metadata about available gaming footage."""
        metadata_file = self.footage_dir / "footage_metadata.json"
        
        try:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load footage metadata: {e}")
        
        # Default metadata structure
        return {
            "sources": [],
            "categories": {
                "high_action": [],
                "medium_action": [],
                "ambient": []
            },
            "durations": {},
            "last_updated": None
        }
    
    async def create_video(
        self, 
        audio_file: str, 
        script_content: str,
        content_analysis: Dict = None,
        voice_info: Dict = None,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a complete TikTok video from audio and script.
        
        Args:
            audio_file: Path to the TTS-generated audio file
            script_content: The TikTok script text for overlays
            content_analysis: Analysis of content type for footage selection
            voice_info: Voice information for styling
            output_path: Optional custom output path
            
        Returns:
            Path to the generated video file or None if failed
        """
        try:
            start_time = time.time()
            logger.info("Starting video creation process...")
            
            # Step 1: Prepare audio
            audio_clip = await self._prepare_audio(audio_file)
            if not audio_clip:
                return None
            
            actual_duration = audio_clip.duration
            logger.info(f"Audio duration: {actual_duration:.2f}s")
            
            # Step 2: Select and prepare gaming footage
            background_video = await self._select_gaming_footage(
                duration=actual_duration,
                content_analysis=content_analysis
            )
            if not background_video:
                return None
            
            # Step 3: Create text overlays
            text_overlays = await self._create_text_overlays(
                script_content, 
                actual_duration,
                voice_info
            )
            
            # Step 4: Apply effects and transitions
            enhanced_video = await self._apply_effects_and_transitions(
                background_video,
                audio_clip,
                content_analysis
            )
            
            # Step 5: Composite final video
            final_video = await self._composite_final_video(
                enhanced_video,
                text_overlays,
                audio_clip
            )
              # Step 6: Export video
            output_file = await self._export_video(final_video, output_path)
            
            # Cleanup
            await self._cleanup_temp_files()
            
            processing_time = time.time() - start_time
            logger.success(f"Video created successfully in {processing_time:.2f}s: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            await self._cleanup_temp_files()
            return None
    
    async def _prepare_audio(self, audio_file: str) -> Optional[AudioFileClip]:
        """Prepare the audio clip for video synchronization."""
        try:
            audio_clip = AudioFileClip(audio_file)
              # For now, just return the audio clip as-is
            # Volume normalization can be added later with proper MoviePy fx imports
            return audio_clip
            
        except Exception as e:
            logger.error(f"Audio preparation failed: {e}")
            return None
    
    async def _select_gaming_footage(
        self, 
        duration: float, 
        content_analysis: Dict = None
    ) -> Optional[VideoFileClip]:
        """
        Select appropriate gaming footage based on content analysis.
        Now uses REAL gaming footage when available!
        """
        try:
            # Determine footage intensity based on content
            intensity = self._determine_footage_intensity(content_analysis)
            
            logger.info(f"Selecting {intensity} intensity gaming footage for {duration:.2f}s")
            
            # Try to get real gaming footage first
            from .footage_manager import FootageManager
            manager = FootageManager()
            
            # Try to get real footage
            footage_path = await manager.get_footage_for_content(
                content_type=content_analysis.get('category', 'tech') if content_analysis else 'tech',
                duration=duration,
                intensity=intensity
            )
            
            if footage_path and footage_path.exists():
                logger.success(f"✅ Using real gaming footage: {footage_path.name}")
                
                # Load the real gaming footage
                gaming_video = VideoFileClip(str(footage_path))
                
                # Crop/resize to TikTok format (9:16 aspect ratio)
                target_aspect = self.config.height / self.config.width  # 1920/1080 = 1.78
                current_aspect = gaming_video.h / gaming_video.w
                
                if current_aspect > target_aspect:
                    # Video is taller than target, crop height
                    new_height = int(gaming_video.w * target_aspect)
                    y_center = gaming_video.h // 2
                    y_start = max(0, y_center - new_height // 2)
                    gaming_video = gaming_video.cropped(y1=y_start, y2=y_start + new_height)
                else:
                    # Video is wider than target, crop width
                    new_width = int(gaming_video.h / target_aspect)
                    x_center = gaming_video.w // 2
                    x_start = max(0, x_center - new_width // 2)
                    gaming_video = gaming_video.cropped(x1=x_start, x2=x_start + new_width)
                  # Resize to exact TikTok dimensions
                gaming_video = gaming_video.resized((self.config.width, self.config.height))
                # Set duration to match audio
                gaming_video = gaming_video.with_duration(duration)
                  # If the video is shorter than needed, loop it
                if gaming_video.duration < duration:
                    loops_needed = int(duration / gaming_video.duration) + 1
                    gaming_video = concatenate_videoclips([gaming_video] * loops_needed)
                    gaming_video = gaming_video.subclipped(0, duration)
                
                # Keep gaming footage audio for now (comment out removal for testing)
                # gaming_video = gaming_video.without_audio()  # TODO: Uncomment when using TTS
                
                logger.success(f"✅ Processed real gaming footage to {self.config.width}x{self.config.height}")
                return gaming_video
            
            else:
                logger.warning("⚠️ No real gaming footage available, using placeholder")
                  # Fallback to placeholder colored background (but warn user)
                placeholder_color = self._get_color_for_intensity(intensity)
                background = ColorClip(
                    size=(self.config.width, self.config.height),
                    color=placeholder_color,
                    duration=duration
                )
                
                return background
            
        except Exception as e:
            logger.error(f"Gaming footage selection failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _determine_footage_intensity(self, content_analysis: Dict = None) -> str:
        """Determine appropriate footage intensity based on content."""
        if not content_analysis:
            return "medium"
        
        # High intensity for breakthroughs, controversy, major funding
        if (content_analysis.get('is_breakthrough') or 
            content_analysis.get('controversy_score', 0) > 0 or
            content_analysis.get('has_funding')):
            return "high"
        
        # Low intensity for partnerships, first-person content
        if (content_analysis.get('is_partnership') or 
            content_analysis.get('is_first_person')):
            return "low"
        
        return "medium"
    
    def _get_color_for_intensity(self, intensity: str) -> Tuple[int, int, int]:
        """Get background color based on intensity (placeholder)."""
        colors = {
            "high": (220, 20, 60),    # Crimson - high energy
            "medium": (70, 130, 180),  # Steel blue - balanced
            "low": (47, 79, 79)        # Dark slate gray - calm
        }
        return colors.get(intensity, colors["medium"])
    
    async def _create_text_overlays(
        self, 
        script_content: str, 
        duration: float,
        voice_info: Dict = None
    ) -> List[TextClip]:
        """Create dynamic text overlays for the video."""
        try:
            overlays = []
            
            # Split script into segments for overlays
            segments = self._split_script_for_overlays(script_content, duration)
            
            for segment in segments:
                text_clip = self._create_text_clip(
                    segment['text'],
                    segment['start_time'],
                    segment['duration'],
                    segment['style']
                )
                if text_clip:
                    overlays.append(text_clip)
            
            logger.info(f"Created {len(overlays)} text overlay segments")
            return overlays
            
        except Exception as e:
            logger.error(f"Text overlay creation failed: {e}")
            return []
    
    def _split_script_for_overlays(self, script: str, duration: float) -> List[Dict]:
        """Split script into segments for text overlays."""
        # Simple implementation - split by sentences
        sentences = [s.strip() for s in script.split('.') if s.strip()]
        if not sentences:
            return []
        
        segments = []
        segment_duration = duration / len(sentences)
        
        for i, sentence in enumerate(sentences):
            # Skip very short segments
            if len(sentence) < 10:
                continue
                
            start_time = i * segment_duration
            
            # Determine style based on content
            style = "emphasis" if any(word in sentence.lower() for word in 
                                   ['breaking', 'insane', 'crazy', 'amazing']) else "normal"
            
            segments.append({
                'text': sentence,
                'start_time': start_time,
                'duration': min(segment_duration, duration - start_time),
                'style': style
            })
        
        return segments
    
    def _create_text_clip(
        self, 
        text: str, 
        start_time: float, 
        duration: float, 
        style: str = "normal"
    ) -> Optional[TextClip]:
        """Create a single text clip with styling."""
        try:            # Style configurations
            styles = {
                "normal": {
                    "fontsize": 60,  # Use fontsize instead of font_size
                    "color": "white",
                    "font": "Arial-Bold",
                    "stroke_color": "black",
                    "stroke_width": 2
                },
                "emphasis": {
                    "fontsize": 70,  # Use fontsize instead of font_size
                    "color": "yellow",
                    "font": "Arial-Bold",
                    "stroke_color": "red",
                    "stroke_width": 3
                }
            }
            
            style_config = styles.get(style, styles["normal"])
            
            # Create text clip
            txt_clip = TextClip(
                text,
                fontsize=style_config["fontsize"],  # Use fontsize instead of font_size
                color=style_config["color"],
                font=style_config["font"],
                stroke_color=style_config["stroke_color"],
                stroke_width=style_config["stroke_width"]
            ).with_duration(duration).with_start(start_time)
              # Position text (centered with some offset for mobile viewing)
            txt_clip = txt_clip.with_position(('center', 0.7))  # 70% down the screen
              # Add fade effects
            txt_clip = txt_clip.with_effects([FadeIn(0.5), FadeOut(0.5)])
            
            return txt_clip
            
        except Exception as e:
            logger.error(f"Text clip creation failed: {e}")
            return None
    
    async def _apply_effects_and_transitions(
        self, 
        video_clip: VideoFileClip, 
        audio_clip: AudioFileClip,
        content_analysis: Dict = None
    ) -> VideoFileClip:
        """Apply effects and transitions to the video."""
        try:
            enhanced_video = video_clip            # Apply speed variations based on content analysis
            if content_analysis and content_analysis.get('urgency_level') == 'high':
                # Slight speed increase for urgent content
                enhanced_video = enhanced_video.with_effects([MultiplySpeed(1.05)])
              # Ensure video matches audio duration exactly
            enhanced_video = enhanced_video.with_duration(audio_clip.duration)
            
            return enhanced_video
            
        except Exception as e:
            logger.error(f"Effects application failed: {e}")
            return video_clip
    
    async def _composite_final_video(
        self, 
        background_video: VideoFileClip,
        text_overlays: List[TextClip],
        audio_clip: AudioFileClip    ) -> CompositeVideoClip:
        """Composite all elements into the final video."""
        try:            # Combine background with text overlays
            all_clips = [background_video] + text_overlays
            final_video = CompositeVideoClip(all_clips, size=(self.config.width, self.config.height))
            
            # Set audio - prefer gaming footage audio if TTS audio is silent/minimal
            if hasattr(background_video, 'audio') and background_video.audio is not None:
                # Use gaming footage audio if available
                final_video = final_video.with_audio(background_video.audio)
                logger.info("🔊 Using gaming footage audio")
            else:
                # Fall back to provided TTS audio
                final_video = final_video.with_audio(audio_clip)
                logger.info("🔊 Using TTS audio")
            
            # Ensure correct duration
            final_video = final_video.with_duration(audio_clip.duration)
            
            return final_video
            
        except Exception as e:
            logger.error(f"Video composition failed: {e}")
            raise
    
    async def _export_video(
        self, 
        video_clip: CompositeVideoClip, 
        output_path: Optional[str] = None
    ) -> str:
        """Export the final video to file."""
        try:
            if not output_path:
                timestamp = int(time.time())
                output_path = str(self.output_dir / f"tiktok_video_{timestamp}.mp4")
              # Export settings based on quality config
            codec_settings = self._get_export_settings()
            
            logger.info(f"Exporting video to: {output_path}")
            
            video_clip.write_videofile(
                output_path,
                **codec_settings,
                logger=None  # Suppress moviepy logs
            )
            
            # Verify file size
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            logger.info(f"Exported video: {file_size_mb:.1f}MB")
            
            if file_size_mb > self.config.max_file_size_mb:
                logger.warning(f"Video size ({file_size_mb:.1f}MB) exceeds TikTok limit ({self.config.max_file_size_mb}MB)")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Video export failed: {e}")
            raise
    
    def _get_export_settings(self) -> Dict:
        """Get export settings based on quality configuration."""
        settings = {
            "low": {
                "codec": "libx264",
                "bitrate": "1000k",
                "fps": 24
            },
            "medium": {
                "codec": "libx264", 
                "bitrate": "2000k",
                "fps": 30
            },
            "high": {
                "codec": "libx264",
                "bitrate": "4000k", 
                "fps": 30
            }
        }
        
        return settings.get(self.config.output_quality, settings["medium"])
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            import shutil
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob("*"):
                    temp_file.unlink()
            logger.debug("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Temp cleanup failed: {e}")
    
    def get_available_footage(self) -> List[Dict]:
        """Get list of available gaming footage."""
        return self.footage_metadata.get("sources", [])
    
    def get_processing_info(self) -> Dict:
        """Get information about the video processor."""
        return {
            "config": {
                "resolution": f"{self.config.width}x{self.config.height}",
                "fps": self.config.fps,
                "quality": self.config.output_quality
            },
            "footage_available": len(self.footage_metadata.get("sources", [])),
            "output_dir": str(self.output_dir)
        }
