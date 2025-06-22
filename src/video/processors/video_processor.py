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
    
    # Letterboxing settings
    letterbox_mode: str = "traditional"  # "traditional" or "percentage"
    letterbox_crop_percentage: float = 0.75  # Only used when mode is "percentage"
    
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
            "categories": {                "high_action": [],
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
        output_path: Optional[str] = None,
        json_file_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a complete TikTok video from audio and script.
        
        Args:
            audio_file: Path to the TTS-generated audio file
            script_content: The TikTok script text for overlays
            content_analysis: Analysis of content type for footage selection
            voice_info: Voice information for styling
            output_path: Optional custom output path
            json_file_path: Optional path to JSON file with custom audio durations
            
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
            
            # Handle custom durations from JSON if provided
            if json_file_path:
                logger.info(f"Using custom durations from JSON file: {json_file_path}")
                # TODO: Implement JSON-based duration handling with VideoSegmentProcessor
                # For now, we'll use the standard flow
              # Step 2: Select and prepare gaming footage
            if json_file_path:
                # Use custom durations from JSON
                background_video = await self._select_gaming_footage_with_json(
                    json_file_path=json_file_path,
                    content_analysis=content_analysis
                )
            else:
                # Use standard duration
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
            
            logger.info(f"Selecting {intensity} intensity gaming footage for {duration:.2f}s")            # Try to get real gaming footage first
            from ..managers.footage_manager import FootageManager
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
                  # Apply letterboxing for TikTok format (preserves aspect ratio)
                gaming_video = self._apply_letterboxing(gaming_video)
                
                # If the video is shorter than needed, loop it first
                if gaming_video.duration < duration:
                    loops_needed = int(duration / gaming_video.duration) + 1
                    gaming_video = concatenate_videoclips([gaming_video] * loops_needed)
                  # Now set exact duration to match audio - this is critical for sync
                gaming_video = gaming_video.subclipped(0, duration)
                
                # TEMPORARILY DISABLE gaming audio to avoid duration issues
                gaming_video = gaming_video.without_audio()
                logger.info("🔇 Disabled gaming footage audio for testing")
                
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
            # TEMPORARILY DISABLED: Skip text overlays to focus on video core functionality
            logger.info("⚠️ Text overlays temporarily disabled for testing")
            return []
            
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
            
            segments.append({                'text': sentence,
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
        try:
            # Style configurations (simplified to avoid conflicts)
            styles = {
                "normal": {
                    "font_size": 60,
                    "color": "white",
                    "stroke_color": "black",
                    "stroke_width": 2
                },
                "emphasis": {
                    "font_size": 70,
                    "color": "yellow",
                    "stroke_color": "red",
                    "stroke_width": 3
                }
            }
            
            style_config = styles.get(style, styles["normal"])
            
            # Create text clip (simplified to avoid parameter conflicts)
            # Remove font parameter to use default system font
            txt_clip = TextClip(
                text,
                font_size=style_config["font_size"],
                color=style_config["color"],
                stroke_color=style_config["stroke_color"],
                stroke_width=style_config["stroke_width"]
            ).with_duration(duration).with_start(start_time)
            
            # Position text (centered with some offset for mobile viewing)
            txt_clip = txt_clip.with_position(('center', 0.7))  # 70% down the screen
            
            # Add fade effects
            txt_clip = txt_clip.with_effects([FadeIn(0.5), FadeOut(0.5)])
            
            return txt_clip
            
        except Exception as e:
            logger.error(f"Text clip creation failed: Invalid font {text}, {e}")
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
                enhanced_video = enhanced_video.with_effects([MultiplySpeed(1.05)])            # Ensure video matches audio duration exactly
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
        try:
            # Use the shorter of background video or audio duration, with safety buffer
            safety_buffer = 0.1  # 100ms safety buffer to avoid duration issues
            final_duration = min(background_video.duration, audio_clip.duration) - safety_buffer
            logger.info(f"🎬 Final video duration: {final_duration:.3f}s (background: {background_video.duration:.3f}s, audio: {audio_clip.duration:.3f}s)")
            
            # Ensure background video doesn't exceed final duration
            if background_video.duration > final_duration:
                background_video = background_video.subclipped(0, final_duration)
              # Combine background with text overlays
            all_clips = [background_video] + text_overlays
            final_video = CompositeVideoClip(all_clips, size=(self.config.width, self.config.height))
            
            # Always use the provided audio clip (TTS)
            final_video = final_video.with_audio(audio_clip.subclipped(0, final_duration))
            logger.info("🔊 Using TTS audio")
            
            # Set exact final duration
            final_video = final_video.with_duration(final_duration)
            
            return final_video
            
        except Exception as e:
            logger.error(f"Video composition failed: {e}")
            raise
    
    async def _export_video(
        self, 
        video_clip: CompositeVideoClip, 
        output_path: Optional[str] = None    ) -> str:
        """Export the final video to file."""
        try:
            if not output_path:
                timestamp = int(time.time())
                output_path = str(self.output_dir / f"tiktok_video_{timestamp}.mp4")
            
            # Export settings based on quality config
            codec_settings = self._get_export_settings()
            
            logger.info(f"Exporting video to: {output_path}")
            
            # Prepare export parameters
            export_params = dict(codec_settings)
            
            # Build ffmpeg_params based on codec
            if codec_settings.get("codec") == "h264_nvenc":
                # NVENC-specific parameters
                ffmpeg_params = [
                    '-movflags', '+faststart',
                    '-pix_fmt', 'yuv420p',
                    '-maxrate', f'{int(codec_settings["bitrate"][:-1]) * 1.2}k',
                    '-bufsize', f'{int(codec_settings["bitrate"][:-1]) * 2}k'
                ]
                # Add NVENC-specific parameters
                if "nvenc_params" in codec_settings:
                    ffmpeg_params.extend(codec_settings["nvenc_params"])
                    export_params.pop("nvenc_params", None)  # Remove from main params
            else:
                # CPU encoding (x264) parameters
                ffmpeg_params = [
                    '-movflags', '+faststart', 
                    '-tune', 'fastdecode',
                    '-x264-params', 'ref=1:bframes=0:me=dia:subme=0:cabac=0'
                ]
            
            export_params['ffmpeg_params'] = ffmpeg_params
            
            video_clip.write_videofile(
                output_path,
                **export_params,
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
        """Get export settings with NVENC hardware acceleration and 1080p quality."""
        
        # NVENC settings for RTX 3070 - much faster than CPU encoding
        nvenc_settings = {
            "low": {
                "codec": "h264_nvenc",
                "bitrate": "3000k",     # Good quality for 1080p
                "fps": 30,
                "preset": "p4",         # Fast NVENC preset
                "audio_codec": "aac",
                "audio_bitrate": "128k",
                "nvenc_params": ["-rc", "vbr", "-cq", "25", "-spatial_aq", "1", "-temporal_aq", "1"]
            },
            "medium": {
                "codec": "h264_nvenc", 
                "bitrate": "5000k",     # High quality for 1080p
                "fps": 30,
                "preset": "p4",         # Fast NVENC preset
                "audio_codec": "aac",
                "audio_bitrate": "128k",
                "nvenc_params": ["-rc", "vbr", "-cq", "23", "-spatial_aq", "1", "-temporal_aq", "1"]
            },
            "high": {
                "codec": "h264_nvenc",
                "bitrate": "8000k",     # Very high quality for 1080p
                "fps": 30,
                "preset": "p2",         # Higher quality NVENC preset
                "audio_codec": "aac",
                "audio_bitrate": "192k",
                "nvenc_params": ["-rc", "vbr", "-cq", "20", "-spatial_aq", "1", "-temporal_aq", "1"]
            }
        }
        
        # CPU fallback settings (if NVENC fails)
        cpu_settings = {
            "low": {
                "codec": "libx264",
                "bitrate": "3000k",
                "fps": 30,
                "preset": "fast",
                "threads": 8,
                "audio_codec": "aac",
                "audio_bitrate": "128k"
            },
            "medium": {
                "codec": "libx264", 
                "bitrate": "5000k",
                "fps": 30,
                "preset": "medium",
                "threads": 8,
                "audio_codec": "aac",
                "audio_bitrate": "128k"
            },
            "high": {
                "codec": "libx264",
                "bitrate": "8000k",
                "fps": 30,
                "preset": "slow",
                "threads": 8,
                "audio_codec": "aac",
                "audio_bitrate": "192k"
            }
        }
        
        # Try NVENC first (hardware acceleration)
        try:
            # Quick test to see if NVENC works
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=0.1:size=320x240:rate=1', 
                '-c:v', 'h264_nvenc', '-preset', 'p4', '-f', 'null', '-'
            ], capture_output=True, timeout=3, text=True)
            
            if result.returncode == 0:
                logger.success("🚀 Using NVENC hardware acceleration")
                return nvenc_settings.get(self.config.output_quality, nvenc_settings["medium"])
            else:
                logger.warning(f"NVENC test failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"NVENC test error: {e}")
        
        # Fallback to optimized CPU encoding
        logger.info("💻 Using CPU encoding (NVENC not available)")
        return cpu_settings.get(self.config.output_quality, cpu_settings["medium"])
    
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
    
    async def _select_gaming_footage_with_json(
        self, 
        json_file_path: str, 
        content_analysis: Dict = None
    ) -> Optional[VideoFileClip]:
        """
        Select gaming footage and create custom segments based on JSON file durations.        """
        try:
            from pathlib import Path
            from ..managers.footage_manager import FootageManager
            
            logger.info(f"Creating custom gaming footage segments from JSON: {json_file_path}")
            
            manager = FootageManager()
            
            # First, find a suitable video to create segments from
            video_id = None
            if manager.metadata.get("videos"):
                # Find a video suitable for the content analysis
                intensity = self._determine_footage_intensity(content_analysis)
                
                for vid_id, vid_info in manager.metadata["videos"].items():
                    if vid_info.get("content_type") == intensity or intensity == "medium":
                        video_id = vid_id
                        break
                
                # If no perfect match, use the first available video
                if not video_id:
                    video_id = list(manager.metadata["videos"].keys())[0]
            
            if not video_id:
                logger.warning("No videos available for custom segmentation")
                return await self._select_gaming_footage(60, content_analysis)
            
            logger.info(f"Using video {video_id} for custom segmentation")
            
            # Create segments using the JSON file
            segments = await manager.create_segments_from_json(
                video_id=video_id,
                json_file_path=Path(json_file_path),
                buffer_seconds=7.5  # Use default buffer
            )
            
            if not segments:
                logger.warning("No custom segments created, falling back to standard footage")
                return await self._select_gaming_footage(60, content_analysis)
            
            logger.success(f"✅ Created {len(segments)} custom gaming footage segments")
            
            # For now, return the first segment as the main footage
            # In a more advanced implementation, this could composite multiple segments
            if segments and segments[0].exists():
                video_clip = VideoFileClip(str(segments[0]))
                logger.info(f"Using first custom segment: {segments[0].name} ({video_clip.duration:.1f}s)")
                return video_clip
            else:
                logger.warning("First custom segment not accessible, falling back to standard footage")
                return await self._select_gaming_footage(60, content_analysis)
        except Exception as e:
            logger.error(f"Failed to create custom gaming footage: {e}")
            logger.info("Falling back to standard footage selection")
            return await self._select_gaming_footage(60, content_analysis)

    def _apply_letterboxing(self, video_clip: VideoFileClip) -> VideoFileClip:
        """
        Apply letterboxing to maintain aspect ratio while fitting TikTok format.
        
        ALWAYS fits by WIDTH to ensure proper letterboxing:
        1. Scales the video to fit the target width (1080px)
        2. Maintains original aspect ratio
        3. Adds black bars (top/bottom) if needed
        
        Args:
            video_clip: Source video clip to letterbox
            
        Returns:
            Letterboxed video clip with TikTok dimensions
        """
        try:
            target_width = self.config.width   # 1080
            target_height = self.config.height # 1920
            
            current_width = video_clip.w
            current_height = video_clip.h
            
            logger.info(f"🎬 LETTERBOXING: Source {current_width}x{current_height} -> Target {target_width}x{target_height}")
            
            # ALWAYS scale to fit WIDTH, maintaining aspect ratio
            new_width = target_width
            new_height = int(target_width * (current_height / current_width))
            
            logger.info(f"📏 Fitting by width: scaling to {new_width}x{new_height}")
            
            # Scale video to fit width while maintaining aspect ratio
            scaled_video = video_clip.resized(new_size=(new_width, new_height))
            
            # Calculate vertical centering (black bars top/bottom)
            y_offset = (target_height - new_height) // 2
            
            if y_offset > 0:
                logger.info(f"📦 Adding {y_offset}px black bars on top and bottom")
            elif y_offset < 0:
                logger.warning(f"⚠️ Video height ({new_height}px) exceeds target ({target_height}px) - will be cropped")
                # If video is too tall, crop it to fit
                crop_amount = abs(y_offset)
                scaled_video = scaled_video.cropped(y1=crop_amount, y2=new_height-crop_amount)
                y_offset = 0
                logger.info(f"✂️ Cropped {crop_amount*2}px from top/bottom")
            else:
                logger.info("✅ Perfect fit, no black bars needed")
            
            # Create black background
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0),  # Black
                duration=video_clip.duration
            )
            
            # Composite scaled video on background
            letterboxed = CompositeVideoClip([
                background,
                scaled_video.with_position(('center', y_offset))
            ])
            
            logger.success(f"✅ Letterboxed: {current_width}x{current_height} -> {new_width}x{new_height} with {y_offset}px offset")
            return letterboxed
            
        except Exception as e:
            logger.error(f"❌ Letterboxing failed: {e}")
            logger.warning("Falling back to simple resize (may cause distortion)")
            return video_clip.resized(new_size=(self.config.width, self.config.height))

    def _apply_percentage_letterboxing(self, video_clip: VideoFileClip, crop_percentage: float = 0.75) -> VideoFileClip:
        """
        Apply percentage-based letterboxing that crops some of the source to reduce black bars.
        
        Instead of fitting 100% of source width, this method fits a percentage (e.g. 75%) 
        of the source width to the target width, effectively zooming in and cropping
        the edges to show more content with smaller black bars.
        
        Args:
            video_clip: Source video clip to letterbox
            crop_percentage: Percentage of source width to fit to target (0.5-1.0)
                           0.75 = fit 75% of source width, crop 25%
                           
        Returns:
            Letterboxed video clip with TikTok dimensions
        """
        try:
            target_width = self.config.width   # 1080
            target_height = self.config.height # 1920
            
            current_width = video_clip.w
            current_height = video_clip.h
            
            # Clamp crop percentage to reasonable bounds
            crop_percentage = max(0.5, min(1.0, crop_percentage))
            
            logger.info(f"🎬 PERCENTAGE LETTERBOXING: Source {current_width}x{current_height} -> Target {target_width}x{target_height}")
            logger.info(f"🔍 Crop percentage: {crop_percentage:.1%} (showing {crop_percentage:.1%} of source width)")
            
            # Calculate effective source dimensions (the portion we'll use)
            effective_source_width = current_width * crop_percentage
            effective_source_height = current_height  # Keep full height for now
            
            # Calculate scaling to fit effective source width to target width
            scale_factor = target_width / effective_source_width
            new_width = int(current_width * scale_factor)
            new_height = int(current_height * scale_factor)
            
            logger.info(f"📏 Scaling to {new_width}x{new_height} (scale factor: {scale_factor:.3f})")
            
            # Scale the video
            scaled_video = video_clip.resized(new_size=(new_width, new_height))
            
            # Calculate cropping/positioning for horizontal centering
            x_overflow = (new_width - target_width) // 2
            x_crop_start = max(0, x_overflow)
            x_crop_end = min(new_width, x_overflow + target_width)
            
            # Calculate vertical positioning
            y_offset = (target_height - new_height) // 2
            
            if x_overflow > 0:
                logger.info(f"✂️ Cropping {x_overflow}px from left and right edges")
                # Crop horizontally to fit width
                scaled_video = scaled_video.cropped(x1=x_crop_start, x2=x_crop_end)
                new_width = target_width
            
            if y_offset > 0:
                logger.info(f"📦 Adding {y_offset}px black bars on top and bottom")
            elif y_offset < 0:
                logger.warning(f"⚠️ Video height ({new_height}px) exceeds target ({target_height}px)")
                # Crop vertically to fit height
                crop_amount = abs(y_offset)
                scaled_video = scaled_video.cropped(y1=crop_amount, y2=new_height-crop_amount)
                y_offset = 0
                new_height = target_height
                logger.info(f"✂️ Cropped {crop_amount*2}px from top/bottom")
            else:
                logger.info("✅ Perfect vertical fit, no black bars needed")
            
            # Create final composition
            if y_offset > 0:
                # Create black background and composite
                background = ColorClip(
                    size=(target_width, target_height),
                    color=(0, 0, 0),  # Black
                    duration=video_clip.duration
                )
                
                letterboxed = CompositeVideoClip([
                    background,
                    scaled_video.with_position(('center', y_offset))
                ])
            else:
                # Video fills the entire frame
                letterboxed = scaled_video
            
            logger.success(f"✅ Percentage letterboxed: {current_width}x{current_height} -> {new_width}x{new_height}")
            logger.success(f"📊 Result: {crop_percentage:.1%} source shown, {100-crop_percentage*100:.1f}% cropped, {y_offset}px black bars")
            
            return letterboxed
            
        except Exception as e:
            logger.error(f"❌ Percentage letterboxing failed: {e}")
            logger.warning("Falling back to simple resize (may cause distortion)")
            return video_clip.resized(new_size=(self.config.width, self.config.height))
    
    async def _select_raw_gaming_footage(
        self, 
        duration: float, 
        content_analysis: Dict = None
    ) -> Optional[VideoFileClip]:
        """
        Select raw gaming footage WITHOUT TikTok format conversion for letterboxing.
        This bypasses the segment processor's automatic conversion.
        """
        try:
            # Determine footage intensity based on content
            intensity = self._determine_footage_intensity(content_analysis)
            
            logger.info(f"Selecting RAW {intensity} intensity gaming footage for letterboxing...")
            
            # Get footage manager
            from ..managers.footage_manager import FootageManager
            manager = FootageManager()
            
            # Get the raw footage path directly
            footage_files = list(manager.base_manager.footage_dir.glob("raw/*.mp4"))
            
            if not footage_files:
                logger.warning("No raw footage files found")
                return None
            
            # Select a random footage file (or based on intensity later)
            import random
            selected_file = random.choice(footage_files)
            
            logger.info(f"📹 Selected raw footage: {selected_file.name}")
            
            # Load the raw video without any processing
            raw_video = VideoFileClip(str(selected_file))
            
            logger.info(f"📊 Raw footage specs: {raw_video.w}x{raw_video.h}, duration: {raw_video.duration:.1f}s")
            
            # If the video is shorter than needed, loop it
            if raw_video.duration < duration:
                loops_needed = int(duration / raw_video.duration) + 1
                raw_video = concatenate_videoclips([raw_video] * loops_needed)
            
            # Trim to exact duration
            raw_video = raw_video.subclipped(0, duration)
            
            # Remove audio to avoid conflicts
            raw_video = raw_video.without_audio()
            
            logger.success(f"✅ Raw footage prepared: {raw_video.w}x{raw_video.h}, {raw_video.duration:.1f}s")
            
            return raw_video
            
        except Exception as e:
            logger.error(f"Raw footage selection failed: {e}")
            return None

    def _apply_letterboxing_with_config(self, video_clip: VideoFileClip) -> VideoFileClip:
        """
        Apply letterboxing based on the configuration settings.
        
        Args:
            video_clip: Source video clip to letterbox
            
        Returns:
            Letterboxed video clip using the configured method
        """
        if self.config.letterbox_mode == "percentage":
            logger.info(f"🔍 Using percentage letterboxing ({self.config.letterbox_crop_percentage:.1%})")
            return self._apply_percentage_letterboxing(video_clip, self.config.letterbox_crop_percentage)
        else:
            logger.info("📦 Using traditional letterboxing (100% source)")
            return self._apply_letterboxing(video_clip)
