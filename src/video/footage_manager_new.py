"""
Main Footage Manager - Orchestrates all footage management functionality.
"""

import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger

from .base_footage_manager import BaseFootageManager
from .youtube_downloader import YouTubeDownloader, FootageSource
from .segment_processor import VideoSegmentProcessor


class FootageManager(BaseFootageManager):
    """
    Main footage manager that orchestrates downloading, processing, and serving footage.
    
    Features:
    - YouTube channel monitoring and downloading
    - Automatic footage categorization
    - Video preprocessing for TikTok use
    - Metadata management and caching
    - Custom duration support from JSON files
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize the footage manager."""
        super().__init__(storage_dir)
        
        # Initialize components
        self.downloader = YouTubeDownloader(self.raw_footage_dir)
        self.processor = VideoSegmentProcessor(self.processed_footage_dir)
        
        logger.info(f"FootageManager initialized with modular components")
    
    async def add_footage_source(self, source: FootageSource) -> bool:
        """Add a new footage source for monitoring."""
        try:
            source_id = hashlib.md5(source.channel_url.encode()).hexdigest()[:8]
            
            self.metadata["sources"][source_id] = {
                "channel_url": source.channel_url,
                "channel_name": source.channel_name,
                "content_type": source.content_type,
                "max_videos": source.max_videos,
                "min_duration": source.min_duration,
                "max_duration": source.max_duration,
                "quality_preference": source.quality_preference,
                "videos_downloaded": 0,
                "last_check": None
            }
            
            self._save_metadata()
            logger.info(f"Added footage source: {source.channel_name} ({source.content_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add footage source: {e}")
            return False

    async def download_footage_from_source(self, source_id: str, max_new_videos: int = 5) -> List[str]:
        """Download new footage from a specific source."""
        try:
            if source_id not in self.metadata["sources"]:
                logger.error(f"Unknown source ID: {source_id}")
                return []
            
            source_info = self.metadata["sources"][source_id]
            
            # Use the downloader to get files
            downloaded_files = await self.downloader.download_footage_from_source(
                source_info, source_id, max_new_videos
            )
            
            # Update metadata for downloaded files
            for downloaded_file in downloaded_files:
                file_path = Path(downloaded_file)
                # Extract video ID from filename (assumes yt-dlp naming convention)
                parts = file_path.stem.split('_', 1)
                if len(parts) >= 1:
                    video_id = parts[0]
                    
                    # Add basic metadata (will be updated with proper info later)
                    self.metadata["videos"][video_id] = {
                        "title": parts[1] if len(parts) > 1 else "Unknown",
                        "duration": 0,  # Will be updated when processing
                        "source_id": source_id,
                        "content_type": source_info["content_type"],
                        "file_path": str(file_path),
                        "processed": False,
                        "download_date": time.time()
                    }
                    
                    # Update category
                    category = source_info["content_type"]
                    if category in self.metadata["categories"]:
                        if video_id not in self.metadata["categories"][category]:
                            self.metadata["categories"][category].append(video_id)
            
            # Update source metadata
            if downloaded_files:
                self.metadata["sources"][source_id]["videos_downloaded"] += len(downloaded_files)
                self.metadata["sources"][source_id]["last_check"] = time.time()
                self.metadata["total_videos"] += len(downloaded_files)
                self._save_metadata()
            
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Footage download failed: {e}")
            return []

    async def process_footage_for_tiktok(self, video_id: str, custom_durations: Optional[List[float]] = None,
                                       duration_info_list: Optional[List[dict]] = None) -> Optional[List[Path]]:
        """
        Process raw footage into TikTok-ready segments.
        
        Args:
            video_id: ID of the video to process
            custom_durations: Optional list of custom durations for segments (in seconds).
                             If provided, creates sequential segments with these durations.
            duration_info_list: Optional list of duration dictionaries with metadata.
                               If provided, creates individual segments for each duration.
                               Takes precedence over custom_durations.
        """
        try:
            if video_id not in self.metadata["videos"]:
                logger.error(f"Unknown video ID: {video_id}")
                return None
            
            video_info = self.metadata["videos"][video_id]
            raw_file = Path(video_info["file_path"])
            
            # Use the processor to create segments
            segments = await self.processor.process_footage_for_tiktok(
                video_info, raw_file, video_id, custom_durations, duration_info_list
            )
            
            if segments:
                # Update metadata
                self.metadata["videos"][video_id]["processed"] = True
                self.metadata["videos"][video_id]["segments"] = [str(s) for s in segments]
                self._save_metadata()
            
            return segments
            
        except Exception as e:
            logger.error(f"TikTok processing failed: {e}")
            return None

    async def create_segments_from_json(self, video_id: str, json_file_path: Path, 
                                      buffer_seconds: float = 7.5) -> Optional[List[Path]]:
        """
        Create video segments matching audio durations from a JSON file with buffer.
        
        Args:
            video_id: ID of the video to process
            json_file_path: Path to the JSON file containing voice recommendations with durations
            buffer_seconds: Buffer to add to each duration (5-10 seconds, default 7.5s)
            
        Returns:
            List of paths to created video segments
        """
        duration_info_list = self.processor.load_audio_durations_from_json(json_file_path, buffer_seconds)
        if not duration_info_list:
            logger.warning(f"No valid durations found in {json_file_path}, falling back to default segments")
            return await self.process_footage_for_tiktok(video_id)
        
        logger.info(f"Creating {len(duration_info_list)} individual segments with {buffer_seconds}s buffer")
        return await self.process_footage_for_tiktok(video_id, duration_info_list=duration_info_list)

    async def get_footage_for_content(self, content_type: str, duration: float, intensity: str = "medium", 
                                    json_file_path: Optional[Path] = None, buffer_seconds: float = 7.5) -> Optional[Path]:        """
        Get suitable footage for given content parameters.
        
        Args:
            content_type: Type of content ("ai", "tech", "business", etc.)
            duration: Required duration in seconds (not used if json_file_path provided)
            intensity: Content intensity ("low", "medium", "high")
            json_file_path: Optional path to JSON file with custom durations
            buffer_seconds: Buffer to add to each JSON duration (5-10 seconds, default 7.5s)
        """
        try:
            # Map intensity levels to footage action categories
            intensity_mapping = {
                "high": "high_action",
                "medium": "medium_action",
                "low": "ambient"
            }
            
            # Map content types to default intensities (fallback)
            content_type_mapping = {
                "ai": "high_action",
                "tech": "medium_action", 
                "business": "ambient",
                "gaming": "medium_action",
                "default": "medium_action"
            }
            
            # Use intensity first, then fallback to content type mapping
            category = intensity_mapping.get(intensity)
            if not category:
                category = content_type_mapping.get(content_type, "medium_action")
                
            logger.info(f"🎯 Selecting footage: content_type={content_type}, intensity={intensity} → category={category}")
            
            # Get videos in this category
            if category not in self.metadata["categories"]:
                logger.warning(f"No footage available for category: {category}")
                return None
            
            video_ids = self.metadata["categories"][category]
            if not video_ids:
                logger.warning(f"No videos in category: {category}")
                return None
            
            # First, check if we have any existing processed segments
            for video_id in video_ids:
                if video_id not in self.metadata["videos"]:
                    continue
                    
                video_info = self.metadata["videos"][video_id]
                
                # Check for existing segments on disk (regardless of metadata)
                existing_segments = list(self.processed_footage_dir.glob(f"{video_id}_segment_*.mp4"))
                if existing_segments:
                    # Use the first existing segment
                    segment_path = existing_segments[0]
                    logger.info(f"✅ Using existing footage segment: {segment_path.name}")
                    return segment_path
                
                # If no segments exist but video is marked as processed, check metadata segments
                if video_info.get("processed", False):
                    segments = video_info.get("segments", [])
                    for segment_path_str in segments:
                        segment_path = Path(segment_path_str)
                        if segment_path.exists():
                            logger.info(f"✅ Using metadata footage segment: {segment_path.name}")
                            return segment_path
            
            # If no existing segments, try to process the first video
            for video_id in video_ids:
                if video_id not in self.metadata["videos"]:
                    continue
                
                video_info = self.metadata["videos"][video_id]
                raw_file = Path(video_info["file_path"])
                
                if raw_file.exists():
                    logger.info(f"Processing video {video_id} to create segments...")
                    
                    # Use custom durations from JSON if provided
                    if json_file_path:
                        segments = await self.create_segments_from_json(video_id, json_file_path)
                    else:
                        segments = await self.process_footage_for_tiktok(video_id)
                    
                    if segments:
                        logger.info(f"✅ Successfully created {len(segments)} segments")
                        return segments[0]  # Return first segment
                    else:
                        logger.warning(f"Failed to process video {video_id}")
                else:
                    logger.warning(f"Raw file not found for video {video_id}: {raw_file}")
            
            logger.warning(f"No suitable footage found for {content_type}/{intensity}")
            return None
            
        except Exception as e:
            logger.error(f"Footage selection failed: {e}")
            return None
