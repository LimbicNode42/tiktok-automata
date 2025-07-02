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
    position: Tuple[str, float] = ("center", 0.75)  # (x, y) position - moved higher up
    max_chars_per_line: int = 28  # Optimized for 2-line static display
    force_two_lines: bool = True  # Always display as 2 lines for consistency
    
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
    font_size: int = 64  # Increased from 48 for better readability
    font_color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 4  # Increased for better contrast
    background_color: Optional[str] = None  # Semi-transparent background
    background_opacity: float = 0.7
    font_family: str = "Arial"
    alignment: str = "center"
    
    # TikTok-specific styling
    shadow_offset: Tuple[int, int] = (3, 3)  # Slightly larger shadow
    shadow_color: str = "black"
    
    # Bubble/TikTok effects
    use_bubble_effect: bool = False  # Enable bubble-like styling
    bubble_color: str = "white"  # Main bubble color
    bubble_outline: str = "black"  # Bubble border color
    bubble_outline_width: int = 6  # Thick outline for bubble effect
    double_stroke: bool = False  # Double stroke for extra bubble effect
    inner_stroke_color: str = "white"  # Inner stroke color for double effect
    inner_stroke_width: int = 2  # Inner stroke width
      # Animation effects - Reduced for better sync with 1.5x TTS speed
    fade_in_duration: float = 0.1
    fade_out_duration: float = 0.1
    
    # Mobile optimization  
    line_spacing: float = 1.2
    margin_horizontal: int = 20  # Reduced from 40 to use more screen width


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
        self.config = config or {}        # Enhanced subtitle styles with larger fonts and better positioning
        self.styles = {
            "default": SubtitleStyle(
                font_size=64,
                font_family="Arial"
            ),
            "bold": SubtitleStyle(
                font_size=68,
                stroke_width=5,
                font_family="Arial"
            ),
            "minimal": SubtitleStyle(
                font_size=60,
                stroke_width=3,
                background_color="rgba(0,0,0,0.6)",
                font_family="Arial"
            ),
            "highlight": SubtitleStyle(
                font_size=70,
                font_color="yellow",
                stroke_color="red",
                stroke_width=5,
                font_family="Arial"
            ),
            "modern": SubtitleStyle(
                font_size=66,
                font_color="white",
                background_color="rgba(0,0,0,0.7)",
                stroke_width=4,
                font_family="Arial"
            ),
            # New font test styles
            "impact": SubtitleStyle(
                font_size=68,
                font_color="white",
                stroke_width=5,
                font_family="Impact"
            ),
            "roboto": SubtitleStyle(
                font_size=64,
                font_color="white",
                stroke_width=4,
                font_family="Roboto"
            ),            "montserrat": SubtitleStyle(
                font_size=64,
                font_color="white",
                stroke_width=4,
                font_family="Montserrat"
            ),            # TikTok-style bubble effects
            "bubble": SubtitleStyle(
                font_size=68,
                font_color="white",
                stroke_width=6,
                stroke_color="black",
                font_family="Arial Black",  # More commonly available bold font
                use_bubble_effect=True,
                bubble_color="white",
                bubble_outline="black",
                bubble_outline_width=8,
                double_stroke=True,
                inner_stroke_color="yellow",
                inner_stroke_width=3
            ),
            "bubble_blue": SubtitleStyle(
                font_size=66,
                font_color="white",
                stroke_width=5,
                stroke_color="navy",
                font_family="Arial Black",  # Fallback to Arial Black
                use_bubble_effect=True,
                bubble_color="lightblue",
                bubble_outline="navy",
                bubble_outline_width=7,
                background_color="rgba(173,216,230,0.8)"  # Light blue background
            ),
            "bubble_gaming": SubtitleStyle(
                font_size=70,
                font_color="lime",
                stroke_width=6,
                stroke_color="darkgreen",
                font_family="Arial Black",  # Use Arial Black instead of Impact
                use_bubble_effect=True,
                bubble_color="lime",
                bubble_outline="darkgreen",
                bubble_outline_width=8,
                double_stroke=True,
                inner_stroke_color="white",
                inner_stroke_width=2
            ),
            "bubble_cute": SubtitleStyle(
                font_size=64,
                font_color="hotpink",
                stroke_width=5,
                stroke_color="purple",
                font_family="Arial",  # Fallback to standard Arial
                use_bubble_effect=True,
                bubble_color="pink",
                bubble_outline="purple",
                bubble_outline_width=6,
                background_color="rgba(255,192,203,0.7)"  # Pink background
            ),
            # Additional bubble styles with different approaches
            "bubble_classic": SubtitleStyle(
                font_size=66,
                font_color="yellow",
                stroke_width=8,
                stroke_color="black",
                font_family="Arial Black",
                use_bubble_effect=True,
                bubble_color="yellow",
                bubble_outline="black",
                bubble_outline_width=10,
                shadow_offset=(4, 4)
            ),
            "bubble_neon": SubtitleStyle(
                font_size=68,
                font_color="cyan",
                stroke_width=6,
                stroke_color="darkblue",
                font_family="Arial Black",
                use_bubble_effect=True,
                bubble_color="cyan",
                bubble_outline="darkblue",
                bubble_outline_width=8,
                double_stroke=True,
                inner_stroke_color="white",
                inner_stroke_width=2,
                background_color="rgba(0,50,100,0.8)"
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
        Segment text into subtitle-appropriate chunks with forced 2-line display.
        
        Optimizes for:
        - Static 2-line display for consistency
        - Mobile readability (balanced lines)
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
            # Format for exactly 2 lines by finding optimal split point
            words = sentence.split()
            if len(words) <= 3:
                # Very short sentence - pad to ensure 2 lines
                segments.append(self._format_to_two_lines(sentence))
            elif len(sentence) <= 50:  # Reduced from 56 to ensure 28 chars per line
                segments.append(self._format_to_two_lines(sentence))
            else:
                # Split long sentences by phrases/clauses and format each
                phrases = re.split(r'[,;:]+', sentence)
                current_segment = ""
                
                for phrase in phrases:
                    phrase = phrase.strip()
                    if not phrase:
                        continue
                    
                    # Check if adding this phrase would exceed 2-line capacity (50 chars total)
                    if current_segment and len(current_segment + " " + phrase) > 50:
                        if current_segment:
                            segments.append(self._format_to_two_lines(current_segment))
                        current_segment = phrase
                    else:
                        if current_segment:
                            current_segment += " " + phrase
                        else:
                            current_segment = phrase
                
                # Add any remaining segment
                if current_segment:
                    segments.append(self._format_to_two_lines(current_segment))
        
        # Filter out empty segments and ensure all are formatted for 2 lines
        final_segments = []
        for segment in segments:
            if segment and segment.strip():
                final_segments.append(self._format_to_two_lines(segment))
        
        return final_segments
    
    def _format_to_two_lines(self, text: str) -> str:
        """
        Format text to display exactly as 2 lines with optimal word wrapping.
        
        Args:
            text: Input text to format
            
        Returns:
            Text formatted with newline for 2-line display
        """
        words = text.split()
        if len(words) <= 1:
            # Single word - add filler text to create 2 lines
            return f"{text}\n "
        elif len(words) == 2:
            # Two words - put one per line
            return f"{words[0]}\n{words[1]}"
        
        # Find the optimal split point for balanced lines under 28 chars each
        best_split = len(words) // 2
        best_balance = float('inf')
        
        # Try different split points to find the most balanced within char limits
        for i in range(1, len(words)):
            first_line = ' '.join(words[:i])
            second_line = ' '.join(words[i:])
            
            # Skip if either line exceeds character limit
            if len(first_line) > 28 or len(second_line) > 28:
                continue
            
            # Prefer balanced line lengths
            line_balance = abs(len(first_line) - len(second_line))
            
            if line_balance < best_balance:
                best_balance = line_balance
                best_split = i
        
        first_line = ' '.join(words[:best_split])
        second_line = ' '.join(words[best_split:])
        
        return f"{first_line}\n{second_line}"
    
    def _legacy_segment_words(self, segment: str) -> List[str]:
        """Legacy word-based segmentation for fallback."""
        words = segment.split()
        final_segments = []
        current_line = ""
        
        for word in words:
            if current_line and len(current_line + " " + word) > 50:
                final_segments.append(self._format_to_two_lines(current_line))
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        # Add any remaining text
        if current_line:
            final_segments.append(self._format_to_two_lines(current_line))
        
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
        
        # Create timed segments with lead time for better sync
        timed_segments = []
        subtitle_lead_time = 0.15  # Show subtitles 150ms before TTS audio
        
        # Calculate base timing first
        current_audio_time = 0
        
        for i, (segment, duration) in enumerate(zip(segments, segment_durations)):
            # Subtitle starts before the audio for this segment
            start_time = max(0, current_audio_time - subtitle_lead_time)
            # Subtitle ends when audio segment ends
            end_time = min(current_audio_time + duration, total_duration)
            
            timed_segments.append(SubtitleSegment(
                text=segment,
                start_time=start_time,
                end_time=end_time
            ))
            
            # Move to next audio segment
            current_audio_time += duration
            
            # Small gap between segments for natural pacing
            if i < len(segments) - 1:
                current_audio_time += 0.1
        
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
                
                # Format text for mobile display with padding consideration
                # Reduce max chars to account for side padding
                effective_max_chars = max(20, segment.max_chars_per_line - 4)  # Ensure minimum 20 chars, subtract 4 for padding
                formatted_text = self._format_text_for_mobile(segment.text, style, effective_max_chars)
                
                # Create base text clip with bubble styling if enabled
                if style.use_bubble_effect:
                    txt_clip = self._create_bubble_text_clip(
                        formatted_text, style, video_width, video_height, segment
                    )
                else:
                    # Standard text clip creation
                    txt_clip = TextClip(
                        text=formatted_text,
                        font_size=style.font_size,
                        color=style.font_color,
                        stroke_color=style.stroke_color,
                        stroke_width=style.stroke_width,
                        method='caption',  # Use caption method for proper multi-line centering
                        size=(video_width - 80, None),  # Add 40px padding on each side (80px total)
                        text_align='center',  # Center align the text
                        vertical_align='center'  # Center vertically within the text box
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
    
    def _create_bubble_text_clip(
        self, 
        text: str, 
        style: SubtitleStyle, 
        video_width: int, 
        video_height: int, 
        segment: SubtitleSegment
    ) -> TextClip:
        """
        Create a text clip with TikTok-style bubble effects.
        
        This method creates enhanced text styling with:
        - Double stroke effects
        - Bubble-like borders
        - Enhanced visual appeal for TikTok content
        """        # Define font fallback chain for better compatibility
        font_fallbacks = [
            style.font_family,
            "Arial Black",
            "Arial",
            "Helvetica",
            None  # System default
        ]
        
        txt_clip = None
        working_font = None
        
        for font in font_fallbacks:
            try:
                # Base text clip with primary styling
                txt_clip = TextClip(
                    text=text,
                    font_size=style.font_size,
                    color=style.font_color,
                    stroke_color=style.bubble_outline,
                    stroke_width=style.bubble_outline_width,
                    font=font,
                    method='caption',
                    size=(video_width - 80, None),  # 40px padding on each side
                    text_align='center',
                    vertical_align='center'
                ).with_duration(segment.duration).with_start(segment.start_time)
                
                # If we get here, the font worked
                working_font = font
                logger.debug(f"Using font: {font or 'system default'}")
                break
                
            except Exception as e:
                logger.debug(f"Font '{font}' failed: {e}")
                continue
        
        if txt_clip is None:
            logger.error("All fonts failed, using fallback text creation")
            # Last resort fallback
            txt_clip = TextClip(
                text=text,
                font_size=style.font_size,
                color=style.font_color,
                stroke_color=style.bubble_outline,
                stroke_width=style.bubble_outline_width,
                method='caption',
                size=(video_width - 80, None),
                text_align='center',
                vertical_align='center'
            ).with_duration(segment.duration).with_start(segment.start_time)
          # Create double stroke effect if enabled
        if style.double_stroke and txt_clip is not None:
            try:
                # Create inner stroke layer with the same font that worked
                inner_clip = TextClip(
                    text=text,
                    font_size=style.font_size,
                    color=style.font_color,
                    stroke_color=style.inner_stroke_color,
                    stroke_width=style.inner_stroke_width,
                    font=working_font,  # Use the same font that worked above
                    method='caption',
                    size=(video_width - 80, None),
                    text_align='center',
                    vertical_align='center'
                ).with_duration(segment.duration).with_start(segment.start_time)
                
                # Position both clips at the same location
                y_position = segment.position[1]
                if isinstance(y_position, float) and y_position <= 1.0:
                    y_pixels = int(video_height * y_position)
                else:
                    y_pixels = y_position
                
                txt_clip = txt_clip.with_position((segment.position[0], y_pixels))
                inner_clip = inner_clip.with_position((segment.position[0], y_pixels))
                
                # Return the inner clip (layered effect)
                return inner_clip
                
            except Exception as e:
                logger.warning(f"Double stroke effect failed: {e}")
                # Fall back to single stroke
        
        return txt_clip

    def _format_text_for_mobile(self, text: str, style: SubtitleStyle, max_chars_per_line: int = 35) -> str:
        """Format text for optimal mobile display."""
        # Split long lines for better mobile readability
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if current_line and len(current_line + " " + word) > max_chars_per_line:  # Use configurable line length
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
