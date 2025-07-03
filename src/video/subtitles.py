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
    max_chars_per_line: int = 25  # Reduced to accommodate 3 lines
    force_three_lines: bool = True  # Force 3 lines for consistency
    max_lines: int = 3  # Hard cap at 3 lines maximum
    
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
        Segment text into subtitle-appropriate chunks with forced 3-line maximum.
        
        Optimizes for:
        - Maximum 3 lines to avoid bottom screen placement
        - No single-word lines to maintain readability
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
            # Check if sentence is too long for 3 lines (approximately 75 characters)
            if len(sentence) > 65:  # Conservative limit
                # Split long sentences into smaller chunks
                chunks = self._split_long_sentence(sentence)
                for chunk in chunks:
                    if chunk and chunk.strip():
                        segments.append(self._format_to_three_lines(chunk))
            else:
                # Format shorter sentences directly
                if sentence and sentence.strip():
                    segments.append(self._format_to_three_lines(sentence))
        
        # Filter out empty segments
        final_segments = []
        for segment in segments:
            if segment and segment.strip():
                final_segments.append(segment)
        
        return final_segments
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into smaller chunks suitable for subtitles."""
        words = sentence.split()
        if len(words) <= 6:
            return [sentence]
        
        chunks = []
        current_chunk = ""
        
        for word in words:
            # Check if adding this word would make the chunk too long
            test_chunk = current_chunk + " " + word if current_chunk else word
            
            # Estimate if this chunk can fit in 3 lines (roughly 65 characters)
            if len(test_chunk) <= 65:
                current_chunk = test_chunk
            else:
                # Current chunk is ready, start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _format_to_three_lines(self, text: str, force_two_lines: bool = False) -> str:
        """
        Format text to display in maximum 3 lines with no single-word lines.
        
        Args:
            text: Input text to format
            force_two_lines: Force into 2 lines even if 3 would be better
            
        Returns:
            Text formatted with newlines for multi-line display
        """
        words = text.split()
        
        # Single word - create 2 lines with padding
        if len(words) == 1:
            return f"{text}\n "  # Add space to create second line
        
        # Two words - decide based on length
        if len(words) == 2:
            total_length = len(words[0]) + len(words[1])
            if total_length <= 25:
                # Short enough for one line, but we need 2 lines
                return f"{words[0]}\n{words[1]}"
            else:
                # Too long for single line anyway
                return f"{words[0]}\n{words[1]}"
        
        # Three words - handle carefully to avoid single-word lines
        if len(words) == 3:
            # Try different combinations
            option1 = f"{words[0]} {words[1]}\n{words[2]}"  # 2+1
            option2 = f"{words[0]}\n{words[1]} {words[2]}"  # 1+2
            
            # Check lengths
            if len(words[0]) + len(words[1]) + 1 <= 25:
                return option1  # 2+1 words
            elif len(words[1]) + len(words[2]) + 1 <= 25:
                return option2  # 1+2 words
            else:
                # All words are long, use 2 lines
                return self._balance_two_lines(words)
        
        # Four or more words - use smart balancing
        if force_two_lines or len(words) <= 6:
            # Try to fit in 2 lines
            return self._balance_two_lines(words)
        else:
            # Use up to 3 lines for longer content
            return self._balance_three_lines(words)
    
    def _balance_two_lines(self, words: List[str]) -> str:
        """Balance words across 2 lines, ensuring no single-word lines and proper length."""
        if len(words) <= 2:
            # For 2 words, check if they fit on one line
            if len(words) == 2:
                combined = ' '.join(words)
                if len(combined) <= 25:
                    # They fit on one line, but we need 2 lines for consistency
                    return f"{words[0]}\n{words[1]}"
                else:
                    # Too long for one line
                    return f"{words[0]}\n{words[1]}"
            return ' '.join(words)
        
        # Force minimum 2 words per line for short texts
        if len(words) <= 4:
            if len(words) == 3:
                # Try 2+1 split
                line1 = f"{words[0]} {words[1]}"
                line2 = words[2]
                if len(line1) <= 25 and len(line2) <= 25:
                    return f"{line1}\n{line2}"
                else:
                    # Fall back to 1+2 split
                    line1 = words[0]
                    line2 = f"{words[1]} {words[2]}"
                    if len(line1) <= 25 and len(line2) <= 25:
                        return f"{line1}\n{line2}"
                    else:
                        # Words are too long individually
                        return f"{words[0]}\n{words[1]}\n{words[2]}"
            else:  # 4 words
                # Try 2+2 split
                line1 = f"{words[0]} {words[1]}"
                line2 = f"{words[2]} {words[3]}"
                if len(line1) <= 25 and len(line2) <= 25:
                    return f"{line1}\n{line2}"
                else:
                    # Try 3+1 or 1+3 split
                    line1 = f"{words[0]} {words[1]} {words[2]}"
                    line2 = words[3]
                    if len(line1) <= 25:
                        return f"{line1}\n{line2}"
                    else:
                        # Try 1+3
                        line1 = words[0]
                        line2 = f"{words[1]} {words[2]} {words[3]}"
                        if len(line2) <= 25:
                            return f"{line1}\n{line2}"
                        else:
                            # Use 3 lines
                            return f"{words[0]} {words[1]}\n{words[2]}\n{words[3]}"
        
        # For longer texts, find optimal split point
        best_split = 2
        best_balance = float('inf')
        
        for i in range(2, len(words) - 1):
            first_line = ' '.join(words[:i])
            second_line = ' '.join(words[i:])
            
            # Must fit in character limit
            if len(first_line) > 25 or len(second_line) > 25:
                continue
            
            # Prefer balanced line lengths
            line_balance = abs(len(first_line) - len(second_line))
            
            if line_balance < best_balance:
                best_balance = line_balance
                best_split = i
        
        # If no good split found, use 3 lines
        if best_balance == float('inf'):
            return self._force_three_lines(words)
        
        first_line = ' '.join(words[:best_split])
        second_line = ' '.join(words[best_split:])
        
        return f"{first_line}\n{second_line}"
    
    def _force_three_lines(self, words: List[str]) -> str:
        """Force text into 3 lines when 2 lines won't work."""
        if len(words) <= 3:
            return '\n'.join(words)
        
        # Try to balance across 3 lines
        words_per_line = len(words) // 3
        remainder = len(words) % 3
        
        split1 = words_per_line + (1 if remainder > 0 else 0)
        split2 = split1 + words_per_line + (1 if remainder > 1 else 0)
        
        line1 = ' '.join(words[:split1])
        line2 = ' '.join(words[split1:split2])
        line3 = ' '.join(words[split2:])
        
        # Check if lines fit
        if len(line1) <= 25 and len(line2) <= 25 and len(line3) <= 25:
            return f"{line1}\n{line2}\n{line3}"
        else:
            # Emergency fallback: just split at word boundaries
            line1 = ' '.join(words[:len(words)//3])
            line2 = ' '.join(words[len(words)//3:2*len(words)//3])
            line3 = ' '.join(words[2*len(words)//3:])
            return f"{line1}\n{line2}\n{line3}"
    
    def _balance_three_lines(self, words: List[str]) -> str:
        """Balance words across 3 lines, ensuring no single-word lines and proper length."""
        if len(words) <= 6:
            return self._balance_two_lines(words)
        
        # For longer texts, try to split into 3 balanced lines
        total_chars = sum(len(word) + 1 for word in words) - 1  # +1 for spaces, -1 for last space
        
        if total_chars > 75:  # Too long for 3 lines (25 chars each)
            return self._balance_two_lines(words)
        
        # Try to distribute words evenly across 3 lines
        words_per_line = len(words) // 3
        remainder = len(words) % 3
        
        # Calculate split points ensuring minimum 2 words per line
        split1 = max(2, words_per_line + (1 if remainder > 0 else 0))
        split2 = max(split1 + 2, split1 + words_per_line + (1 if remainder > 1 else 0))
        
        # Ensure we don't exceed the word count
        if split2 >= len(words):
            return self._balance_two_lines(words)
        
        # Create lines
        line1 = ' '.join(words[:split1])
        line2 = ' '.join(words[split1:split2])
        line3 = ' '.join(words[split2:])
        
        # Check line lengths and word counts
        if (len(line1) > 25 or len(line2) > 25 or len(line3) > 25 or
            len(words[split2:]) < 2):  # Ensure at least 2 words in last line
            return self._balance_two_lines(words)
        
        return f"{line1}\n{line2}\n{line3}"
    
    def _legacy_segment_words(self, segment: str) -> List[str]:
        """Legacy word-based segmentation for fallback."""
        words = segment.split()
        final_segments = []
        current_line = ""
        
        for word in words:
            if current_line and len(current_line + " " + word) > 70:  # Increased for 3-line capacity
                final_segments.append(self._format_to_three_lines(current_line))
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        # Add any remaining text
        if current_line:
            final_segments.append(self._format_to_three_lines(current_line))
        
        return final_segments
    
    def _calculate_timing(
        self, 
        segments: List[str], 
        total_duration: float,
        tts_info: Optional[Dict] = None
    ) -> List[SubtitleSegment]:
        """
        Calculate timing for subtitle segments optimized for 1.35x TTS speed.
        
        Uses character-based estimation with calibrated timing for better sync.
        """
        if not segments:
            return []
        
        # Get TTS speed from config for dynamic adjustment
        try:
            from ..utils.config import config
            tts_speed = config.get_tts_speed()
        except:
            tts_speed = 1.35  # Default fallback
        
        # Estimate reading speed (characters per second) adjusted for TTS speed
        # At 1.35x speed, speech is faster, so we need to adjust timing
        base_cps = 12 * tts_speed  # Scale with TTS speed
        
        # Calculate duration for each segment based on character count
        segment_durations = []
        total_chars = sum(len(s) for s in segments)
        
        for segment in segments:
            # Base duration from character count (adjusted for TTS speed)
            char_duration = len(segment) / base_cps
            
            # Add pause time for punctuation (reduced for faster TTS)
            pause_time = 0
            pause_time += segment.count('.') * (0.4 / tts_speed)  # Scale pauses with speed
            pause_time += segment.count(',') * (0.25 / tts_speed)
            pause_time += segment.count('!') * (0.35 / tts_speed)
            pause_time += segment.count('?') * (0.35 / tts_speed)
            pause_time += segment.count(';') * (0.3 / tts_speed)
            pause_time += segment.count(':') * (0.2 / tts_speed)
            
            # Add pause for line breaks (scaled for TTS speed)
            pause_time += segment.count('\n') * (0.15 / tts_speed)
            
            # Minimum duration for readability (adjusted for TTS speed)
            min_duration = max(1.5, len(segment) / (15 * tts_speed))
            
            duration = max(min_duration, char_duration + pause_time)
            segment_durations.append(duration)
        
        # Scale to fit total duration
        total_estimated = sum(segment_durations)
        if total_estimated > total_duration:
            scale_factor = (total_duration * 0.95) / total_estimated  # Use 95% to leave buffer
            segment_durations = [d * scale_factor for d in segment_durations]
        
        # Create timed segments with optimized lead time for 1.35x speed
        timed_segments = []
        subtitle_lead_time = 0.15  # Optimized for 1.35x TTS speed
        
        # Calculate base timing with better pause handling
        current_audio_time = 0
        
        for i, (segment, duration) in enumerate(zip(segments, segment_durations)):
            # Subtitle starts before the audio for this segment
            start_time = max(0, current_audio_time - subtitle_lead_time)
            
            # For segments with long pauses (sentence endings), extend subtitle duration
            extension = 0
            if segment.endswith('.') or segment.endswith('!') or segment.endswith('?'):
                extension = 0.3  # Keep subtitle visible during pause
            
            # Subtitle ends when audio segment ends plus any extension
            end_time = min(current_audio_time + duration + extension, total_duration)
            
            timed_segments.append(SubtitleSegment(
                text=segment,
                start_time=start_time,
                end_time=end_time
            ))
            
            # Move to next audio segment
            current_audio_time += duration
            
            # Adaptive gap between segments based on punctuation
            if i < len(segments) - 1:
                # Longer gaps after sentence endings
                if segment.endswith('.') or segment.endswith('!') or segment.endswith('?'):
                    current_audio_time += 0.4
                else:
                    current_audio_time += 0.15
        
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
