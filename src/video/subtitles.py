"""
Subtitle Generator for TikTok Videos with TTS Synchronization.

This module creates synchronized subtitles that match TTS audio timing,
with TikTok-optimized styling and automatic text segmentation.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
from loguru import logger

try:
    from moviepy import TextClip, CompositeVideoClip
    from moviepy.video.fx import FadeIn, FadeOut
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.error("MoviePy not available for subtitle generation")


@dataclass
class SubtitleSegment:
    """A single subtitle segment with timing and styling."""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    style: str = "default"  # Style preset name
    position: Tuple[str, float] = ("center", 0.85)  # (x, y) position
    max_chars_per_line: int = 25  # Max characters per line for mobile
    
    @property
    def duration(self) -> float:
        """Get the duration of this subtitle segment."""
        return self.end_time - self.start_time
    
    def to_srt_entry(self, index: int) -> str:
        """Convert to SRT format entry."""
        start_srt = self._seconds_to_srt_time(self.start_time)
        end_srt = self._seconds_to_srt_time(self.end_time)
        return f"{index}\n{start_srt} --> {end_srt}\n{self.text}\n\n"
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


@dataclass
class SubtitleStyle:
    """Styling configuration for subtitles."""
    font_size: int = 48
    font_color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 3
    background_color: Optional[str] = None  # Semi-transparent background
    background_opacity: float = 0.7
    font_family: str = "Arial"
    alignment: str = "center"
    
    # TikTok-specific styling
    shadow_offset: Tuple[int, int] = (2, 2)
    shadow_color: str = "black"
    
    # Animation effects
    fade_in_duration: float = 0.2
    fade_out_duration: float = 0.2
    
    # Mobile optimization
    line_spacing: float = 1.2
    margin_horizontal: int = 40  # Pixels from screen edge


class SubtitleGenerator:
    """
    Generate synchronized subtitles for TikTok videos.
    
    Features:
    - TTS audio synchronization
    - TikTok-optimized styling
    - Multiple subtitle styles
    - Automatic text segmentation
    - SRT file export
    - MoviePy integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the subtitle generator."""
        self.config = config or {}        # Default subtitle styles
        self.styles = {
            "default": SubtitleStyle(),
            "bold": SubtitleStyle(
                font_size=52,
                stroke_width=4,
                font_family="Arial"
            ),
            "minimal": SubtitleStyle(
                font_size=44,
                stroke_width=2,
                background_color="rgba(0,0,0,0.5)",
                font_family="Arial"
            ),
            "highlight": SubtitleStyle(
                font_size=50,
                font_color="yellow",
                stroke_color="red",
                stroke_width=4,
                font_family="Arial"
            ),
            "modern": SubtitleStyle(
                font_size=46,
                font_color="white",
                background_color="rgba(0,0,0,0.6)",
                stroke_width=2,
                font_family="Arial"
            )
        }
        
        # Path for subtitle cache and exports
        self.output_dir = Path(__file__).parent / "data" / "subtitles"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SubtitleGenerator initialized")
    
    async def generate_from_script(
        self, 
        script_text: str, 
        audio_duration: float,
        tts_info: Optional[Dict] = None,
        style_name: str = "default"
    ) -> List[SubtitleSegment]:
        """
        Generate subtitle segments from script text and audio duration.
        
        Args:
            script_text: The text to be converted to subtitles
            audio_duration: Total audio duration in seconds
            tts_info: Optional TTS timing information
            style_name: Style preset to use
            
        Returns:
            List of SubtitleSegment objects
        """
        try:
            logger.info(f"Generating subtitles for {audio_duration:.2f}s audio")
            
            # Clean and segment the text
            segments = self._segment_text(script_text)
            
            # Calculate timing for each segment
            timed_segments = self._calculate_timing(segments, audio_duration, tts_info)
            
            # Apply styling
            styled_segments = self._apply_style(timed_segments, style_name)
            
            logger.success(f"Generated {len(styled_segments)} subtitle segments")
            return styled_segments
            
        except Exception as e:
            logger.error(f"Subtitle generation failed: {e}")
            return []
    
    def _segment_text(self, text: str) -> List[str]:
        """
        Segment text into subtitle-appropriate chunks.
        
        Optimizes for:
        - Mobile readability (short lines)
        - Natural speech pauses
        - TikTok attention spans
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        segments = []
        
        for sentence in sentences:
            # If sentence is short enough, use as-is
            if len(sentence) <= 50:
                segments.append(sentence)
            else:
                # Split long sentences by phrases/clauses
                phrases = re.split(r'[,;:]+', sentence)
                current_segment = ""
                
                for phrase in phrases:
                    phrase = phrase.strip()
                    if not phrase:
                        continue
                    
                    # Check if adding this phrase would exceed optimal length
                    if current_segment and len(current_segment + " " + phrase) > 45:
                        if current_segment:
                            segments.append(current_segment)
                        current_segment = phrase
                    else:
                        if current_segment:
                            current_segment += " " + phrase
                        else:
                            current_segment = phrase
                
                # Add any remaining segment
                if current_segment:
                    segments.append(current_segment)
        
        # Further split overly long segments by words
        final_segments = []
        for segment in segments:
            if len(segment) <= 50:
                final_segments.append(segment)
            else:
                words = segment.split()
                current_line = ""
                
                for word in words:
                    if current_line and len(current_line + " " + word) > 45:
                        final_segments.append(current_line)
                        current_line = word
                    else:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                
                if current_line:
                    final_segments.append(current_line)
        
        logger.info(f"Segmented text into {len(final_segments)} parts")
        return final_segments
    
    def _calculate_timing(
        self, 
        segments: List[str], 
        total_duration: float,
        tts_info: Optional[Dict] = None
    ) -> List[SubtitleSegment]:
        """
        Calculate timing for subtitle segments.
        
        Uses character-based estimation with pauses for natural reading.
        """
        if not segments:
            return []
        
        # Estimate reading speed (characters per second)
        # Adjusted for TTS speech rate - typically slower than normal reading
        base_cps = 12  # Conservative estimate for TTS
        
        # Calculate duration for each segment based on character count
        segment_durations = []
        total_chars = sum(len(s) for s in segments)
        
        for segment in segments:
            # Base duration from character count
            char_duration = len(segment) / base_cps
            
            # Add pause time for punctuation
            pause_time = 0
            pause_time += segment.count('.') * 0.3
            pause_time += segment.count(',') * 0.2
            pause_time += segment.count('!') * 0.4
            pause_time += segment.count('?') * 0.4
            
            # Minimum duration for readability
            min_duration = max(1.5, len(segment) / 15)  # At least 1.5s, or enough for reading
            
            duration = max(min_duration, char_duration + pause_time)
            segment_durations.append(duration)
        
        # Scale to fit total duration
        total_estimated = sum(segment_durations)
        if total_estimated > total_duration:
            scale_factor = (total_duration * 0.95) / total_estimated  # Use 95% to leave some buffer
            segment_durations = [d * scale_factor for d in segment_durations]
        
        # Create timed segments
        timed_segments = []
        current_time = 0
        
        for i, (segment, duration) in enumerate(zip(segments, segment_durations)):
            start_time = current_time
            end_time = min(current_time + duration, total_duration)
            
            timed_segments.append(SubtitleSegment(
                text=segment,
                start_time=start_time,
                end_time=end_time
            ))
            
            current_time = end_time
            
            # Small gap between segments for natural pacing
            if i < len(segments) - 1:
                current_time += 0.1
        
        return timed_segments
    
    def _apply_style(
        self, 
        segments: List[SubtitleSegment], 
        style_name: str
    ) -> List[SubtitleSegment]:
        """Apply styling to subtitle segments."""
        style = self.styles.get(style_name, self.styles["default"])
        
        # Apply style properties to segments (stored for later MoviePy rendering)
        for segment in segments:
            segment.style = style_name
        
        return segments
    
    async def create_subtitle_clips(
        self, 
        segments: List[SubtitleSegment],
        video_width: int = 1080,
        video_height: int = 1920
    ) -> List[TextClip]:
        """
        Create MoviePy TextClip objects for subtitle segments.
        
        Args:
            segments: List of subtitle segments
            video_width: Video width in pixels
            video_height: Video height in pixels
            
        Returns:
            List of TextClip objects ready for composition
        """
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy not available for subtitle clip creation")
            return []
        
        clips = []
        
        for segment in segments:
            try:
                style = self.styles[segment.style]
                
                # Format text for mobile display
                formatted_text = self._format_text_for_mobile(segment.text, style)                # Create text clip
                txt_clip = TextClip(
                    text=formatted_text,
                    font_size=style.font_size,
                    color=style.font_color,
                    stroke_color=style.stroke_color,
                    stroke_width=style.stroke_width
                ).with_duration(segment.duration).with_start(segment.start_time)
                
                # Position the text
                y_position = segment.position[1]
                if isinstance(y_position, float) and y_position <= 1.0:
                    # Convert relative position to pixels
                    y_pixels = int(video_height * y_position)
                else:
                    y_pixels = y_position
                
                txt_clip = txt_clip.with_position((segment.position[0], y_pixels))
                
                # Add fade effects
                if style.fade_in_duration > 0:
                    txt_clip = txt_clip.with_effects([FadeIn(style.fade_in_duration)])
                if style.fade_out_duration > 0:
                    txt_clip = txt_clip.with_effects([FadeOut(style.fade_out_duration)])
                
                clips.append(txt_clip)
                
            except Exception as e:
                logger.error(f"Failed to create subtitle clip for segment '{segment.text}': {e}")
                continue
        
        logger.success(f"Created {len(clips)} subtitle clips")
        return clips
    
    def _format_text_for_mobile(self, text: str, style: SubtitleStyle) -> str:
        """Format text for optimal mobile display."""
        # Split long lines for better mobile readability
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if current_line and len(current_line + " " + word) > 25:  # Mobile-optimized line length
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    async def export_srt(
        self, 
        segments: List[SubtitleSegment], 
        filename: Optional[str] = None
    ) -> str:
        """
        Export subtitles to SRT format.
        
        Args:
            segments: List of subtitle segments
            filename: Optional output filename
            
        Returns:
            Path to the exported SRT file
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"subtitles_{timestamp}.srt"
        
        output_path = self.output_dir / filename
        
        srt_content = ""
        for i, segment in enumerate(segments, 1):
            srt_content += segment.to_srt_entry(i)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        logger.success(f"Exported SRT subtitles to: {output_path}")
        return str(output_path)
    
    async def export_json(
        self, 
        segments: List[SubtitleSegment], 
        filename: Optional[str] = None
    ) -> str:
        """
        Export subtitles to JSON format for debugging/analysis.
        
        Args:
            segments: List of subtitle segments
            filename: Optional output filename
            
        Returns:
            Path to the exported JSON file
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"subtitles_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert segments to JSON-serializable format
        segments_data = []
        for segment in segments:
            segments_data.append({
                "text": segment.text,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "style": segment.style,
                "position": segment.position
            })
        
        export_data = {
            "segments": segments_data,
            "total_segments": len(segments),
            "total_duration": segments[-1].end_time if segments else 0,
            "export_timestamp": time.time()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.success(f"Exported JSON subtitles to: {output_path}")
        return str(output_path)
    
    def get_style_names(self) -> List[str]:
        """Get list of available subtitle styles."""
        return list(self.styles.keys())
    
    def add_custom_style(self, name: str, style: SubtitleStyle) -> None:
        """Add a custom subtitle style."""
        self.styles[name] = style
        logger.info(f"Added custom subtitle style: {name}")


# Convenience functions for quick usage

async def generate_subtitles(
    script_text: str,
    audio_duration: float,
    style: str = "default"
) -> List[SubtitleSegment]:
    """Quick function to generate subtitles."""
    generator = SubtitleGenerator()
    return await generator.generate_from_script(script_text, audio_duration, style_name=style)


async def create_subtitle_video_clips(
    script_text: str,
    audio_duration: float,
    video_width: int = 1080,
    video_height: int = 1920,
    style: str = "default"
) -> List[TextClip]:
    """Quick function to create subtitle video clips."""
    generator = SubtitleGenerator()
    segments = await generator.generate_from_script(script_text, audio_duration, style_name=style)
    return await generator.create_subtitle_clips(segments, video_width, video_height)
