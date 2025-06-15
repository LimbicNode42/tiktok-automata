"""
YouTube Downloader - Handles downloading footage from YouTube channels and videos.
"""

import asyncio
import time
import hashlib
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from loguru import logger

try:
    import yt_dlp
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


class YouTubeDownloader:
    """
    Handles downloading footage from YouTube channels and videos.
    """
    
    def __init__(self, raw_footage_dir: Path):
        """Initialize the YouTube downloader."""
        self.raw_footage_dir = raw_footage_dir
        self.max_retries = 3
        self.retry_delay = 5
        
        # Configure yt-dlp with network-friendly settings
        self.ydl_opts = {
            'format': 'best[height<=720][ext=mp4][acodec!=none]/best[height<=720][acodec!=none]/best[ext=mp4][acodec!=none]/best[acodec!=none]/best',
            'outtmpl': str(self.raw_footage_dir / '%(id)s_%(title)s.%(ext)s'),
            'writeinfojson': True,
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'ignoreerrors': False,
            'socket_timeout': 15,
            'retries': 15,
            'fragment_retries': 20,
            'file_access_retries': 15,
            'extractor_retries': 15,
            'http_chunk_size': 1048576,  # 1MB chunks
            'external_downloader_args': {
                'ffmpeg': ['-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '2']
            }
        }
    
    async def download_footage_from_source(self, source_info: Dict, source_id: str, max_new_videos: int = 5) -> List[str]:
        """Download new footage from a specific source."""
        try:
            url = source_info["channel_url"]
            logger.info(f"Downloading footage from: {source_info['channel_name']}")
            
            # Check if this is a single video URL or a channel URL
            if "watch?v=" in url or "youtu.be/" in url:
                # This is a single video URL
                logger.info(f"Detected single video URL: {url}")
                return await self._download_single_video_url(url, source_info, source_id)
            else:
                # This is a channel URL
                return await self._download_from_channel(url, source_info, source_id, max_new_videos)
            
        except Exception as e:
            logger.error(f"Footage download failed: {e}")
            return []

    async def _download_single_video_url(self, video_url: str, source_info: Dict, source_id: str) -> List[str]:
        """Download a single video from a direct URL with enhanced retry logic."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📡 Attempt {attempt + 1}/{self.max_retries} - Getting video info for: {video_url}")
                
                # Get video info first with retry-friendly options
                info_opts = {'quiet': True, 'socket_timeout': 20 + (attempt * 5)}
                with yt_dlp.YoutubeDL(info_opts) as ydl:
                    video_info = ydl.extract_info(video_url, download=False)
                
                # Download the video
                logger.info(f"🎬 Downloading video: {video_info.get('title', 'Unknown')}")
                downloaded_file = await self._download_single_video(video_info, source_info)
                
                if downloaded_file:
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
                    logger.error("❌ All attempts failed - printing traceback for debugging:")
                    traceback.print_exc()
        
        logger.error("❌ All download attempts failed")
        return []

    async def _download_from_channel(self, channel_url: str, source_info: Dict, source_id: str, max_new_videos: int) -> List[str]:
        """Download videos from a channel URL."""
        try:
            logger.info(f"🔍 Extracting video list from channel: {channel_url}")
            
            # Configure yt-dlp for channel extraction
            channel_opts = {
                'quiet': False,
                'ignoreerrors': True,
                'extract_flat': True,
                'playlistend': max_new_videos * 2,
            }
            
            # Get video list from channel
            with yt_dlp.YoutubeDL(channel_opts) as ydl:
                try:
                    logger.info("📋 Getting channel video list...")
                    channel_info = ydl.extract_info(channel_url, download=False)
                    
                    if not channel_info or 'entries' not in channel_info:
                        logger.error(f"No video entries found in channel: {channel_url}")
                        return []
                    
                    # Filter out None entries and get video info
                    raw_entries = [entry for entry in channel_info['entries'] if entry is not None]
                    logger.info(f"📹 Found {len(raw_entries)} videos in channel")
                    
                    if not raw_entries:
                        logger.warning("No accessible videos found in channel")
                        return []
                    
                    # Get detailed info for videos
                    videos = []
                    for entry in raw_entries[:max_new_videos * 2]:
                        try:
                            if len(videos) >= max_new_videos:
                                break
                                
                            video_id = entry.get('id')
                            if not video_id:
                                continue
                                
                            # Get detailed video info
                            logger.info(f"🔍 Getting detailed info for video: {video_id}")
                            detailed_opts = {'quiet': True, 'ignoreerrors': True}
                            
                            with yt_dlp.YoutubeDL(detailed_opts) as detail_ydl:
                                video_info = detail_ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
                                
                                if video_info and video_info.get('availability') != 'subscriber_only':
                                    # Check duration constraints
                                    duration = video_info.get('duration', 0)
                                    if source_info["min_duration"] <= duration <= source_info["max_duration"]:
                                        videos.append(video_info)
                                        logger.success(f"✅ Added video to download queue: {video_info.get('title', 'Unknown')[:50]}...")
                                    else:
                                        logger.info(f"⏭️ Skipping video {video_id}: duration {duration}s outside range")
                                else:
                                    logger.info(f"⏭️ Skipping members-only or unavailable video: {video_id}")
                                    
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get info for video {entry.get('id', 'unknown')}: {e}")
                            continue
                    
                    logger.info(f"📥 Selected {len(videos)} videos for download")
                    
                except Exception as e:
                    logger.error(f"Failed to extract channel info: {e}")
                    return []
            
            downloaded_files = []
            
            for i, video_info in enumerate(videos, 1):
                if not video_info:
                    continue
                
                video_id = video_info.get('id')
                if not video_id:
                    continue
                
                logger.info(f"📥 Downloading video {i}/{len(videos)}: {video_info.get('title', 'Unknown')}")
                
                # Download video
                downloaded_file = await self._download_single_video(video_info, source_info)
                if downloaded_file:
                    downloaded_files.append(downloaded_file)
            
            logger.info(f"Downloaded {len(downloaded_files)} new videos from {source_info['channel_name']}")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Channel download failed: {e}")
            return []

    async def _download_single_video(self, video_info: Dict, source_info: Dict) -> Optional[Path]:
        """Download a single video file."""
        video_url = video_info.get('webpage_url') or f"https://youtube.com/watch?v={video_info['id']}"
        video_id = video_info.get('id')
        title = video_info.get('title', 'Unknown')
        
        logger.info(f"🔄 Downloading complete file: {title}")
        logger.info(f"📍 URL: {video_url}")
        
        try:
            # Clean up any partial downloads from previous attempts
            partial_files = list(self.raw_footage_dir.glob("*.part*"))
            for partial_file in partial_files:
                try:
                    partial_file.unlink()
                    logger.info(f"🧹 Cleaned up partial file: {partial_file.name}")
                except Exception:
                    pass
            
            # Create download options optimized for complete file download
            download_opts = self.ydl_opts.copy()
            download_opts.update({
                'socket_timeout': 60,
                'retries': 20,
                'fragment_retries': 20,
                'file_access_retries': 15,
                'http_chunk_size': 2097152,  # 2MB chunks
            })
            
            logger.info(f"📥 Starting download with yt-dlp internal retries (20 retries)")
            
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
            
            # Fallback: try to find by video ID
            for file_path in self.raw_footage_dir.glob("*.mp4"):
                if video_id in file_path.name:
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.success(f"✅ Found downloaded file: {file_path.name} ({file_size_mb:.1f}MB)")
                    return file_path
            
            logger.error(f"⚠️ No video file found after download")
            return None
            
        except Exception as e:
            logger.error(f"❌ Download failed for {title}: {e}")
            return None
