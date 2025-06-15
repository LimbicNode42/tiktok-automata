"""
Video Segment Processor - Handles processing video segments with custom durations.
"""

import asyncio
import json
import traceback
from pathlib import Path
from typing import Optional, List
from loguru import logger


class VideoSegmentProcessor:
    """
    Handles processing raw footage into TikTok-ready segments.
    Supports both default durations and custom durations from JSON files.
    """
    
    def __init__(self, processed_footage_dir: Path):
        """Initialize the video segment processor."""
        self.processed_footage_dir = processed_footage_dir
    def load_audio_durations_from_json(self, json_file_path: Path, buffer_seconds: float = 7.5) -> List[dict]:
        """
        Load audio durations from a voice recommendations JSON file with buffer.
        
        Args:
            json_file_path: Path to the JSON file containing voice recommendations
            buffer_seconds: Buffer to add to each duration (default 7.5s for 5-10s range)
            
        Returns:
            List of dictionaries with duration info and metadata
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            duration_info = []
            if 'results' in data:
                for i, result in enumerate(data['results']):
                    if 'duration' in result and result.get('success', False):
                        original_duration = float(result['duration'])
                        buffered_duration = original_duration + buffer_seconds
                        
                        duration_data = {
                            'index': i + 1,
                            'title': result.get('title', 'Unknown'),
                            'original_duration': original_duration,
                            'buffered_duration': buffered_duration,
                            'buffer_added': buffer_seconds,
                            'audio_file': result.get('audio_file', ''),
                            'voice_id': result.get('voice_id', ''),
                            'category': result.get('category', 'unknown')
                        }
                        duration_info.append(duration_data)
                        
                        logger.info(f"Loaded duration {i+1}: {original_duration:.1f}s + {buffer_seconds:.1f}s = {buffered_duration:.1f}s for '{result.get('title', 'Unknown')[:50]}...'")
            
            logger.info(f"Loaded {len(duration_info)} audio durations with {buffer_seconds}s buffer from {json_file_path.name}")
            return duration_info
            
        except Exception as e:
            logger.error(f"Failed to load audio durations from {json_file_path}: {e}")
            return []

    async def create_individual_segments_from_durations(self, video_info: dict, raw_file: Path, 
                                                      video_id: str, duration_info_list: List[dict]) -> Optional[List[Path]]:
        """
        Create individual video segments for each duration (not sequential).
        Each segment starts from the beginning of the video with the specified duration.
        
        Args:
            video_info: Video metadata dictionary
            raw_file: Path to the raw video file
            video_id: ID of the video to process
            duration_info_list: List of duration dictionaries with metadata
            
        Returns:
            List of paths to created video segments
        """
        try:
            if not raw_file.exists():
                logger.error(f"Raw footage file not found: {raw_file}")
                return None
            
            logger.info(f"Creating {len(duration_info_list)} individual segments from: {video_info['title']}")
            
            # Import moviepy here to avoid startup delay
            try:
                from moviepy import VideoFileClip
            except ImportError:
                logger.error("MoviePy not installed. Run: pip install moviepy")
                return None
            
            # Load video
            clip = VideoFileClip(str(raw_file))
            total_duration = clip.duration
            segments = []
            
            logger.info(f"Source video duration: {total_duration:.1f}s")
            
            for duration_info in duration_info_list:
                segment_duration = duration_info['buffered_duration']
                index = duration_info['index']
                title = duration_info['title']
                
                # Skip if requested duration is longer than available footage
                if segment_duration > total_duration:
                    logger.warning(f"Skipping segment {index} - requested {segment_duration:.1f}s but only {total_duration:.1f}s available")
                    continue
                
                logger.info(f"Creating segment {index}: {segment_duration:.1f}s for '{title[:30]}...'")
                
                # Create segment from start of video with the specified duration
                segment_file = await self._create_individual_segment(
                    clip, video_id, segment_duration, duration_info, total_duration
                )
                
                if segment_file:
                    segments.append(segment_file)
                else:
                    logger.warning(f"Failed to create segment {index}")
            
            clip.close()
            
            logger.info(f"Successfully created {len(segments)} individual segments")
            return segments
            
        except Exception as e:
            logger.error(f"Individual segment creation failed: {e}")
            traceback.print_exc()
            return None

    async def _create_individual_segment(self, clip, video_id: str, segment_duration: float, 
                                       duration_info: dict, total_duration: float) -> Optional[Path]:
        """Create a single individual segment from the start of the video."""
        segment_clip = None
        tiktok_segment = None
        
        try:
            index = duration_info['index']
            title_safe = "".join(c for c in duration_info['title'][:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            title_safe = title_safe.replace(' ', '_')
            
            # Find a good starting point (avoid very beginning which might be black)
            start_offset = min(5.0, total_duration * 0.1)  # Start 5s in or 10% into video, whichever is smaller
            actual_start = start_offset
            actual_end = min(actual_start + segment_duration, total_duration)
            
            # Adjust if we don't have enough footage after the offset
            if actual_end - actual_start < segment_duration * 0.8:  # If less than 80% of desired duration
                actual_start = max(0, total_duration - segment_duration)  # Start from end and work backwards
                actual_end = total_duration
            
            logger.info(f"Segment {index}: extracting {actual_start:.1f}s to {actual_end:.1f}s (duration: {actual_end - actual_start:.1f}s)")
            
            # Extract segment
            segment_clip = clip.subclipped(actual_start, actual_end)
            
            # Convert to TikTok format
            try:
                tiktok_segment = self._convert_to_tiktok_format(segment_clip)
            except Exception as e:
                logger.warning(f"TikTok format conversion failed, using original: {e}")
                tiktok_segment = segment_clip
            
            # Create filename with more metadata
            segment_file = self.processed_footage_dir / f"{video_id}_segment_{index:02d}_{duration_info['buffered_duration']:.1f}s_{title_safe}.mp4"
            
            # Remove existing file if it exists
            if segment_file.exists():
                segment_file.unlink()
                logger.info(f"🗑️ Removed existing segment: {segment_file.name}")
            
            logger.info(f"🎬 Writing individual segment {index}: {segment_file.name}")
            
            # Write video file
            tiktok_segment.write_videofile(
                str(segment_file),
                codec='libx264',
                audio_codec='aac',
                fps=30,
                preset='medium',
                logger=None,
                temp_audiofile=None,
                remove_temp=True
            )
            
            # Verify file was created
            if segment_file.exists() and segment_file.stat().st_size > 0:
                file_size = segment_file.stat().st_size / (1024 * 1024)
                logger.success(f"✅ Created individual segment {index}: {segment_file.name} ({file_size:.1f}MB)")
                return segment_file
            else:
                logger.error(f"❌ Segment file not created or is empty: {segment_file.name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to create individual segment {duration_info['index']}: {e}")
            logger.error(f"   Full error: {traceback.format_exc()}")
            return None
        
        finally:
            # Cleanup
            if tiktok_segment is not None:
                try:
                    tiktok_segment.close()
                except:
                    pass
            
            if segment_clip is not None:
                try:
                    segment_clip.close()
                except:
                    pass
            
            # Force garbage collection and small delay
            import gc
            gc.collect()
            await asyncio.sleep(0.2)

    async def process_footage_for_tiktok(self, video_info: dict, raw_file: Path, video_id: str, 
                                       custom_durations: Optional[List[float]] = None,
                                       duration_info_list: Optional[List[dict]] = None) -> Optional[List[Path]]:
        """
        Process raw footage into TikTok-ready segments.
        
        Args:
            video_info: Video metadata dictionary
            raw_file: Path to the raw video file
            video_id: ID of the video to process
            custom_durations: Optional list of custom durations for segments (in seconds).
                             If provided, creates sequential segments with these durations.
            duration_info_list: Optional list of duration dictionaries with metadata.
                               If provided, creates individual segments for each duration.
                               Takes precedence over custom_durations.
        """
        try:
            if not raw_file.exists():
                logger.error(f"Raw footage file not found: {raw_file}")
                return None
            
            logger.info(f"Processing footage for TikTok: {video_info['title']}")
            
            # Import moviepy here to avoid startup delay
            try:
                from moviepy import VideoFileClip
            except ImportError:
                logger.error("MoviePy not installed. Run: pip install moviepy")
                return None
            
            # If duration_info_list is provided, use the new individual segment creation
            if duration_info_list:
                logger.info("Using individual segment creation mode")
                return await self.create_individual_segments_from_durations(
                    video_info, raw_file, video_id, duration_info_list
                )
            
            # Otherwise, use the existing logic for sequential segments
            # Load video
            clip = VideoFileClip(str(raw_file))
            duration = clip.duration
            segments = []
            
            if custom_durations:
                # Create segments with custom durations (sequential)
                logger.info(f"Creating {len(custom_durations)} sequential segments with custom durations")
                segments = await self._create_custom_duration_segments(
                    clip, video_id, custom_durations, duration
                )
            else:
                # Create segments with default 45-second duration
                logger.info("Creating segments with default 45-second duration")
                segments = await self._create_default_duration_segments(
                    clip, video_id, duration
                )
            
            clip.close()
            
            logger.info(f"Created {len(segments)} TikTok segments")
            return segments
            
        except Exception as e:
            logger.error(f"TikTok processing failed: {e}")
            traceback.print_exc()
            return None

    async def _create_custom_duration_segments(self, clip, video_id: str, 
                                             custom_durations: List[float], total_duration: float) -> List[Path]:
        """Create segments with custom durations."""
        segments = []
        start_time = 0
        
        for i, segment_duration in enumerate(custom_durations):
            end_time = min(start_time + segment_duration, total_duration)
            
            if end_time - start_time < 10:  # Skip segments shorter than 10 seconds
                logger.warning(f"Skipping segment {i+1} - too short ({end_time - start_time:.1f}s)")
                continue
            
            if start_time >= total_duration:  # No more footage available
                logger.warning(f"Ran out of footage at segment {i+1}")
                break
            
            # Create and save segment
            segment_file = await self._create_single_segment(
                clip, video_id, start_time, end_time, segment_index=i+1
            )
            
            if segment_file:
                segments.append(segment_file)
            
            # Move to next segment position
            start_time = end_time
        
        return segments

    async def _create_default_duration_segments(self, clip, video_id: str, total_duration: float) -> List[Path]:
        """Create segments with default 45-second duration."""
        segments = []
        segment_duration = 45  # 45 second segments
        
        for start_time in range(0, int(total_duration), segment_duration):
            end_time = min(start_time + segment_duration, total_duration)
            
            if end_time - start_time < 20:  # Skip segments shorter than 20 seconds
                continue
            
            # Create and save segment
            segment_file = await self._create_single_segment(
                clip, video_id, start_time, end_time
            )
            
            if segment_file:
                segments.append(segment_file)
        
        return segments

    async def _create_single_segment(self, clip, video_id: str, start_time: float, 
                                   end_time: float, segment_index: Optional[int] = None) -> Optional[Path]:
        """Create a single video segment."""
        segment_clip = None
        tiktok_segment = None
        
        try:
            # Extract segment with fresh video clip reference
            segment_clip = clip.subclipped(start_time, end_time)
            
            # Convert to TikTok format (9:16 aspect ratio) with error handling
            try:
                tiktok_segment = self._convert_to_tiktok_format(segment_clip)
            except Exception as e:
                logger.warning(f"TikTok format conversion failed, using original: {e}")
                tiktok_segment = segment_clip
            
            # Create filename
            if segment_index is not None:
                # Custom duration segment
                segment_file = self.processed_footage_dir / f"{video_id}_segment_{segment_index}_{start_time:.1f}_{end_time:.1f}.mp4"
                logger.info(f"🎬 Writing custom segment {segment_index}: {start_time:.1f}-{end_time:.1f}s...")
            else:
                # Default duration segment
                segment_file = self.processed_footage_dir / f"{video_id}_segment_{int(start_time)}_{int(end_time)}.mp4"
                logger.info(f"🎬 Writing segment {int(start_time)}-{int(end_time)}s...")
            
            # Remove existing file if it exists
            if segment_file.exists():
                segment_file.unlink()
                logger.info(f"🗑️ Removed existing segment: {segment_file.name}")
            
            # Write video file with isolated clip
            tiktok_segment.write_videofile(
                str(segment_file),
                codec='libx264',
                audio_codec='aac',
                fps=30,
                preset='medium',
                logger=None,
                temp_audiofile=None,
                remove_temp=True
            )
            
            # Verify file was actually created
            if segment_file.exists() and segment_file.stat().st_size > 0:
                file_size = segment_file.stat().st_size / (1024 * 1024)
                if segment_index is not None:
                    logger.success(f"✅ Created custom segment {segment_index}: {segment_file.name} ({file_size:.1f}MB)")
                else:
                    logger.success(f"✅ Created segment: {segment_file.name} ({file_size:.1f}MB)")
                return segment_file
            else:
                logger.error(f"❌ Segment file not created or is empty: {segment_file.name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to save segment: {e}")
            logger.error(f"   Full error: {traceback.format_exc()}")
            return None
        
        finally:
            # Proper cleanup of clips - close in reverse order
            if tiktok_segment is not None:
                try:
                    tiktok_segment.close()
                except:
                    pass
            
            if segment_clip is not None:
                try:
                    segment_clip.close()
                except:
                    pass
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            # Small delay to prevent resource conflicts
            await asyncio.sleep(0.2)

    def _convert_to_tiktok_format(self, clip):
        """Convert video clip to TikTok format (9:16 aspect ratio, 1080x1920)."""
        try:
            # Target TikTok dimensions
            target_width = 1080
            target_height = 1920
            target_ratio = target_height / target_width  # 16:9 = 1.777...
            
            # Get current dimensions
            current_width, current_height = clip.size
            current_ratio = current_height / current_width
            
            if current_ratio > target_ratio:
                # Video is taller than target - crop height
                new_height = int(current_width * target_ratio)
                y_offset = (current_height - new_height) // 2
                cropped = clip.cropped(y1=y_offset, y2=y_offset + new_height)
            else:
                # Video is wider than target - crop width
                new_width = int(current_height / target_ratio)
                x_offset = (current_width - new_width) // 2
                cropped = clip.cropped(x1=x_offset, x2=x_offset + new_width)
            
            # Resize to exact TikTok dimensions
            resized = cropped.resized((target_width, target_height))
            return resized
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            # Return original clip if conversion fails
            return clip
