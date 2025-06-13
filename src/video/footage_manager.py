"""
Footage Manager for downloading and managing gaming content from YouTube.

Handles:
1. YouTube video downloading from copyright-free channels
2. Footage categorization and metadata management
3. Video preprocessing for TikTok use
4. Smart caching and storage management
"""

import asyncio
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from loguru import logger
import time
import hashlib
import random
import traceback

try:
    import yt_dlp
    import urllib.request
except ImportError:
    logger.error("yt-dlp not installed. Run: pip install yt-dlp")
    raise


@dataclass
class FootageSource:
    """Configuration for a gaming footage source."""
    channel_url: str
    channel_name: str
    content_type: str  # "high_action", "medium_action", "ambient"
    max_videos: int = 50
    min_duration: int = 300  # 5 minutes minimum
    max_duration: int = 3600  # 1 hour maximum
    quality_preference: str = "720p"  # Balance quality vs file size


class FootageManager:
    """
    Manager for gaming footage sourcing and preprocessing.
    
    Features:
    - YouTube channel monitoring and downloading
    - Automatic footage categorization
    - Video preprocessing for TikTok use
    - Metadata management and caching
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize the footage manager."""
        self.storage_dir = storage_dir or Path(__file__).parent / "data" / "footage"
        self.raw_footage_dir = self.storage_dir / "raw"
        self.processed_footage_dir = self.storage_dir / "processed"
        self.metadata_file = self.storage_dir / "footage_metadata.json"
        
        # Create directories
        for dir_path in [self.storage_dir, self.raw_footage_dir, self.processed_footage_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
          # Configure yt-dlp with network-friendly settings
        self.ydl_opts = {
            'format': 'best[height<=720][ext=mp4][acodec!=none]/best[height<=720][acodec!=none]/best[ext=mp4][acodec!=none]/best[acodec!=none]/best',
            'outtmpl': str(self.raw_footage_dir / '%(id)s_%(title)s.%(ext)s'),
            'writeinfojson': True,
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'ignoreerrors': False,
            # Network stability improvements
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 5,
            'file_access_retries': 3,
            'extractor_retries': 3,
            # Chunked downloading for better reliability
            'http_chunk_size': 1048576,  # 1MB chunks
            'external_downloader_args': {
                'ffmpeg': ['-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5']
            }
        }
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        logger.info(f"FootageManager initialized with storage: {self.storage_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load footage metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
        
        return {
            "sources": {},
            "videos": {},
            "categories": {
                "high_action": [],
                "medium_action": [],
                "ambient": []
            },
            "last_updated": None,
            "total_duration": 0,
            "total_videos": 0
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            self.metadata["last_updated"] = time.time()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
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
            url = source_info["channel_url"]
            
            logger.info(f"Downloading footage from: {source_info['channel_name']}")
              # Check if this is a single video URL or a channel URL
            if "watch?v=" in url or "youtu.be/" in url:
                # This is a single video URL
                logger.info(f"Detected single video URL: {url}")
                return await self._download_single_video_url(url, source_info, source_id)
            else:
                # This is a channel URL - use the original logic
                return await self._download_from_channel(url, source_info, source_id, max_new_videos)
            
        except Exception as e:
            logger.error(f"Footage download failed: {e}")
            return []
    
    async def _check_network_connectivity(self) -> bool:
        """Check if network connection is stable."""
        try:
            import urllib.request
            urllib.request.urlopen('https://www.youtube.com', timeout=10)
            return True
        except Exception:
            return False
    
    async def _download_with_retries(self, video_url: str, opts: Dict, max_attempts: int = 3) -> bool:
        """Download with exponential backoff retry logic."""
        for attempt in range(max_attempts):
            try:
                # Check network connectivity first
                if not await self._check_network_connectivity():
                    logger.warning(f"Network connectivity issue on attempt {attempt + 1}")
                    if attempt < max_attempts - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"⏳ Waiting {delay}s for network recovery...")
                        await asyncio.sleep(delay)
                        continue
                    return False
                
                # Adjust timeout for this attempt
                current_opts = opts.copy()
                current_opts['socket_timeout'] = 30 + (attempt * 15)
                current_opts['http_chunk_size'] = max(524288, 1048576 // (2 ** attempt))  # Smaller chunks on retry
                
                logger.info(f"📥 Download attempt {attempt + 1}/{max_attempts} (timeout: {current_opts['socket_timeout']}s)")
                
                with yt_dlp.YoutubeDL(current_opts) as ydl:
                    ydl.download([video_url])
                return True
                
            except Exception as e:
                logger.warning(f"❌ Download attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ All {max_attempts} download attempts failed")
        
        return False

    async def _download_single_video_url(self, video_url: str, source_info: Dict, source_id: str) -> List[str]:
        """Download a single video from a direct URL with enhanced retry logic."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📡 Attempt {attempt + 1}/{self.max_retries} - Getting video info for: {video_url}")
                
                # Get video info first with retry-friendly options
                info_opts = {'quiet': True, 'socket_timeout': 20 + (attempt * 5)}
                with yt_dlp.YoutubeDL(info_opts) as ydl:
                    try:
                        video_info = ydl.extract_info(video_url, download=False)
                        if not video_info:
                            logger.error(f"Failed to get video info for: {video_url}")
                            continue
                        
                        video_id = video_info.get('id')
                        if not video_id:
                            logger.error("No video ID found")
                            continue
                          # Check if already downloaded and file actually exists
                        if video_id in self.metadata["videos"]:
                            existing_file = Path(self.metadata["videos"][video_id]["file_path"])
                            if existing_file.exists():
                                logger.info(f"Video {video_id} already downloaded and file exists")
                                return [str(existing_file)]
                            else:
                                logger.warning(f"Video {video_id} in metadata but file missing: {existing_file}")
                                logger.info("Re-downloading the video...")
                        
                        # Check duration constraints
                        duration = video_info.get('duration', 0)
                        logger.info(f"Video duration: {duration} seconds")
                        
                        if duration < source_info["min_duration"]:
                            logger.warning(f"Video too short: {duration}s < {source_info['min_duration']}s")
                            # Download anyway for testing
                        
                        if duration > source_info["max_duration"]:
                            logger.warning(f"Video too long: {duration}s > {source_info['max_duration']}s")
                            # Download anyway for testing
                        
                    except Exception as e:
                        logger.error(f"Failed to extract video info on attempt {attempt + 1}: {e}")
                        if attempt < self.max_retries - 1:
                            delay = self.retry_delay * (2 ** attempt)
                            logger.info(f"⏳ Waiting {delay}s before retry...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            return []
                
                # Download the video
                logger.info(f"🎬 Downloading video: {video_info.get('title', 'Unknown')}")
                downloaded_file = await self._download_single_video(video_info, source_info)
                
                if downloaded_file:
                    # Add to metadata
                    self.metadata["videos"][video_id] = {
                        "title": video_info.get('title', 'Unknown'),
                        "duration": duration,
                        "source_id": source_id,
                        "content_type": source_info["content_type"],
                        "file_path": str(downloaded_file),
                        "processed": False,
                        "download_date": time.time()
                    }
                    
                    # Update category
                    category = source_info["content_type"]
                    if category not in self.metadata["categories"]:
                        self.metadata["categories"][category] = []
                    self.metadata["categories"][category].append(video_id)
                    
                    # Update source metadata
                    self.metadata["sources"][source_id]["videos_downloaded"] = 1
                    self.metadata["sources"][source_id]["last_check"] = time.time()
                    self.metadata["total_videos"] += 1
                    
                    self._save_metadata()
                    
                    logger.success(f"✅ Successfully downloaded video: {downloaded_file.name}")
                    return [str(downloaded_file)]
                else:
                    logger.warning(f"⚠️ Download attempt {attempt + 1} returned no file")
                    
            except Exception as e:
                logger.error(f"❌ Single video URL download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                else:
                    import traceback
                    traceback.print_exc()
        
        logger.error("❌ All download attempts failed")
        return []

    async def _download_from_channel(self, channel_url: str, source_info: Dict, source_id: str, max_new_videos: int) -> List[str]:
        """Download videos from a channel URL."""
        try:
            # Get video list from channel
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                try:
                    channel_info = ydl.extract_info(channel_url, download=False)
                    if not channel_info or 'entries' not in channel_info:
                        logger.error(f"Failed to get channel info for: {channel_url}")
                        return []
                    
                    videos = list(channel_info['entries'])[:max_new_videos]
                    
                except Exception as e:
                    logger.error(f"Failed to extract channel info: {e}")
                    return []
            
            downloaded_files = []
            
            for video_info in videos:
                if not video_info:
                    continue
                
                video_id = video_info.get('id')
                if not video_id:
                    continue
                
                # Check if already downloaded
                if video_id in self.metadata["videos"]:
                    continue
                
                # Check duration constraints
                duration = video_info.get('duration', 0)
                if duration < source_info["min_duration"] or duration > source_info["max_duration"]:
                    continue
                
                # Download video
                downloaded_file = await self._download_single_video(video_info, source_info)
                if downloaded_file:
                    downloaded_files.append(downloaded_file)
                    
                    # Add to metadata
                    self.metadata["videos"][video_id] = {
                        "title": video_info.get('title', 'Unknown'),
                        "duration": duration,
                        "source_id": source_id,
                        "content_type": source_info["content_type"],
                        "file_path": str(downloaded_file),
                        "processed": False,
                        "download_date": time.time()
                    }
                      # Update category
                    category = source_info["content_type"]
                    if category in self.metadata["categories"]:
                        self.metadata["categories"][category].append(video_id)
            
            # Update source metadata
            self.metadata["sources"][source_id]["videos_downloaded"] += len(downloaded_files)
            self.metadata["sources"][source_id]["last_check"] = time.time()
            self.metadata["total_videos"] += len(downloaded_files)
            
            self._save_metadata()
            
            logger.info(f"Downloaded {len(downloaded_files)} new videos from {source_info['channel_name']}")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Channel download failed: {e}")
            return []
    async def _download_single_video(self, video_info: Dict, source_info: Dict) -> Optional[Path]:
        """Download a single video file with robust retry logic."""
        video_url = video_info.get('webpage_url') or f"https://youtube.com/watch?v={video_info['id']}"
        video_id = video_info.get('id')
        title = video_info.get('title', 'Unknown')
        
        logger.info(f"🔄 Downloading complete file: {title}")
        logger.info(f"📍 URL: {video_url}")
        
        for attempt in range(self.max_retries):
            try:
                # Clean up any partial downloads from previous attempts
                partial_files = list(self.raw_footage_dir.glob("*.part*"))
                for partial_file in partial_files:
                    try:
                        partial_file.unlink()
                        logger.info(f"🧹 Cleaned up partial file: {partial_file.name}")
                    except Exception:
                        pass
                
                # Check network connectivity before attempting download
                if not await self._check_network_connectivity():
                    logger.warning(f"Network connectivity issue on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"⏳ Waiting {delay}s for network recovery...")
                        await asyncio.sleep(delay)
                        continue
                    return None
                
                # Create download options optimized for complete file download
                download_opts = self.ydl_opts.copy()
                download_opts.update({
                    'socket_timeout': 60 + (attempt * 30),  # Progressive timeout increase
                    'retries': 5,  # Internal retries for network issues
                    'fragment_retries': 10,  # More fragment retries
                    'file_access_retries': 5,
                    'http_chunk_size': 2097152,  # 2MB chunks for better stability
                })
                
                logger.info(f"📥 Download attempt {attempt + 1}/{self.max_retries} (timeout: {download_opts['socket_timeout']}s)")
                
                # Get list of files before download
                files_before = set(self.raw_footage_dir.glob("*"))
                
                # Download with yt-dlp
                with yt_dlp.YoutubeDL(download_opts) as ydl:
                    ydl.download([video_url])
                
                # Get list of files after download
                files_after = set(self.raw_footage_dir.glob("*"))
                new_files = files_after - files_before
                
                # Find the downloaded video file
                video_files = [f for f in new_files if f.suffix.lower() in ['.mp4', '.mkv', '.webm']]
                if video_files:
                    downloaded_file = video_files[0]
                    file_size_mb = downloaded_file.stat().st_size / (1024 * 1024)
                    logger.success(f"✅ Downloaded complete file: {downloaded_file.name} ({file_size_mb:.1f}MB)")
                    return downloaded_file
                
                # Fallback: try to find by video ID or title matching
                for file_path in self.raw_footage_dir.glob("*.mp4"):
                    if video_id in file_path.name:
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        logger.success(f"✅ Found downloaded file: {file_path.name} ({file_size_mb:.1f}MB)")
                        return file_path
                
                logger.warning(f"⚠️ No video file found after download attempt {attempt + 1}")
                
            except Exception as e:
                logger.warning(f"❌ Download attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"⏳ Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ All download attempts failed for: {title}")
        
        logger.error(f"❌ Complete file download failed after all attempts for: {title}")
        return None

    async def process_footage_for_tiktok(self, video_id: str) -> Optional[List[Path]]:
        """
        Process raw footage into TikTok-ready segments.
        
        Creates multiple shorter clips optimized for TikTok use.
        """
        try:
            if video_id not in self.metadata["videos"]:
                logger.error(f"Unknown video ID: {video_id}")
                return None
            
            video_info = self.metadata["videos"][video_id]
            raw_file = Path(video_info["file_path"])
            
            if not raw_file.exists():
                logger.error(f"Raw footage file not found: {raw_file}")
                return None
            
            logger.info(f"Processing footage for TikTok: {video_info['title']}")            # Import moviepy here to avoid startup delay
            try:
                from moviepy import VideoFileClip
            except ImportError:
                logger.error("MoviePy not installed. Run: pip install moviepy")
                return None
            
            # Load video
            clip = VideoFileClip(str(raw_file))
            duration = clip.duration
            
            # Create segments (30-60 second clips)
            segment_duration = 45  # 45 second segments
            segments = []
            
            for start_time in range(0, int(duration), segment_duration):
                end_time = min(start_time + segment_duration, duration)
                
                if end_time - start_time < 20:  # Skip segments shorter than 20 seconds
                    continue                # Extract segment
                segment = clip.subclipped(start_time, end_time)
                
                # Convert to TikTok format (9:16 aspect ratio) with error handling
                try:
                    tiktok_segment = self._convert_to_tiktok_format(segment)
                except Exception as e:
                    logger.warning(f"TikTok format conversion failed, using original: {e}")
                    tiktok_segment = segment
                
                # Save segment
                segment_file = self.processed_footage_dir / f"{video_id}_segment_{start_time}_{end_time}.mp4"
                try:
                    tiktok_segment.write_videofile(
                        str(segment_file),
                        codec='libx264',
                        audio_codec='aac',
                        fps=30,
                        preset='medium',
                        logger=None
                    )
                    segments.append(segment_file)
                    logger.info(f"✅ Created segment: {segment_file.name}")
                except Exception as e:
                    logger.error(f"Failed to save segment {segment_file.name}: {e}")
                
                tiktok_segment.close()
            
            clip.close()
              # Update metadata
            self.metadata["videos"][video_id]["processed"] = True
            self.metadata["videos"][video_id]["segments"] = [str(s) for s in segments]
            self._save_metadata()
            
            logger.info(f"Created {len(segments)} TikTok segments")
            return segments
            
        except Exception as e:
            logger.error(f"TikTok processing failed: {e}")
            return None
    
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
            logger.error(f"Format conversion failed: {e}")            # Return original clip if conversion fails
            return clip

    async def get_footage_for_content(self, content_type: str, duration: float, intensity: str = "medium") -> Optional[Path]:
        """
        Get suitable footage for given content parameters.
        
        Args:
            content_type: Type of content ("ai", "tech", "business", etc.)
            duration: Required duration in seconds
            intensity: Content intensity ("low", "medium", "high")
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

    async def get_available_footage_stats(self) -> Dict:
        """Get statistics about available footage."""
        try:
            stats = {
                "total_videos": self.metadata["total_videos"],
                "categories": {},
                "processed_videos": 0,
                "total_segments": 0
            }
            
            for category, video_ids in self.metadata["categories"].items():
                stats["categories"][category] = len(video_ids)
            
            for video_info in self.metadata["videos"].values():
                if video_info.get("processed", False):
                    stats["processed_videos"] += 1
                    segments = video_info.get("segments", [])
                    stats["total_segments"] += len(segments)
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats calculation failed: {e}")
            return {}
