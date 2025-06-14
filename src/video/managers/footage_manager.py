"""
Simplified Footage Manager - Orchestrates video operations with clear separation of concerns.

Single responsibility: Coordinate between downloaders, analyzers, and processors.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

# Import from organized modules
from ..downloaders.youtube_downloader import YouTubeDownloader, FootageSource
from ..analyzers.action_analyzer import VideoActionAnalyzer
from ..processors.segment_processor import VideoSegmentProcessor
from ..base_footage_manager import BaseFootageManager


class FootageManager(BaseFootageManager):
    """
    Simplified footage manager with clear delegation to specialized modules.
    
    Responsibilities:
    - Coordinate operations between modules
    - Maintain metadata and storage
    - Provide simple public interface
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        super().__init__(storage_dir)
        
        # Initialize specialized modules
        self.downloader = YouTubeDownloader(self.raw_footage_dir)
        self.analyzer = VideoActionAnalyzer() 
        self.processor = VideoSegmentProcessor(self.processed_footage_dir)
        
        logger.info("FootageManager initialized with modular components")
    
    # === DOWNLOAD OPERATIONS ===
    
    async def add_footage_source(self, source: FootageSource) -> bool:
        """Add a new footage source for downloading."""
        return await self.downloader.add_source(source, self.metadata)
    
    async def download_footage_from_source(self, source_id: str, max_new_videos: int = 5) -> List[str]:
        """Download footage from a specific source."""
        return await self.downloader.download_from_source(source_id, self.metadata, max_new_videos)
    
    # === ANALYSIS OPERATIONS ===
    
    async def analyze_video_action(self, video_id: str) -> Dict:
        """Analyze action intensity in a downloaded video."""
        if video_id not in self.metadata["videos"]:
            logger.error(f"Video {video_id} not found")
            return {}
        
        video_info = self.metadata["videos"][video_id]
        video_path = Path(video_info["file_path"])
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return {}
        
        logger.info(f"🔍 Analyzing action in video: {video_id}")
        action_data = await self.analyzer.analyze_video_action(video_path)
        
        # Store analysis results in metadata
        self.metadata["videos"][video_id]["action_analysis"] = {
            "high_action_count": len(action_data["high"]),
            "medium_action_count": len(action_data["medium"]),
            "low_action_count": len(action_data["low"]),
            "analyzed": True
        }
        self._save_metadata()
        
        return action_data
    
    async def get_best_action_segments(self, video_id: str, segment_duration: float = 45.0) -> List:
        """Get the best action segments from a video."""
        action_data = await self.analyze_video_action(video_id)
        return self.analyzer.get_best_action_segments(action_data, segment_duration)
    
    # === PROCESSING OPERATIONS ===
    
    async def process_footage_for_tiktok(self, video_id: str, 
                                       custom_durations: Optional[List[float]] = None,
                                       duration_info_list: Optional[List[Dict]] = None) -> List[Path]:
        """Process footage into TikTok-ready segments."""
        if video_id not in self.metadata["videos"]:
            logger.error(f"Unknown video ID: {video_id}")
            return []
        
        video_info = self.metadata["videos"][video_id]
        raw_file = Path(video_info["file_path"])
        
        # Delegate to processor
        return await self.processor.process_footage_for_tiktok(
            video_info, raw_file, video_id, custom_durations, duration_info_list
        )
    
    async def create_segments_from_json(self, video_id: str, json_file_path: Path, 
                                      buffer_seconds: float = 7.5) -> List[Path]:
        """Create segments with custom durations from JSON file."""
        duration_info_list = self.processor.load_audio_durations_from_json(json_file_path, buffer_seconds)
        if not duration_info_list:
            logger.warning(f"No valid durations found in {json_file_path}")
            return []
        
        logger.info(f"Creating {len(duration_info_list)} individual segments with {buffer_seconds}s buffer")
        return await self.process_footage_for_tiktok(video_id, duration_info_list=duration_info_list)
    
    # === CONTENT SELECTION ===
    
    async def get_footage_for_content(self, content_type: str, duration: float, 
                                    intensity: str = "medium", 
                                    json_file_path: Optional[Path] = None, 
                                    buffer_seconds: float = 7.5) -> Optional[Path]:
        """
        Get suitable footage for given content parameters.
        
        Now uses action analysis to select the best segments!
        """
        try:
            # Find available videos matching the intensity
            suitable_videos = []
            
            for video_id, video_info in self.metadata["videos"].items():
                if video_info.get("content_type") == intensity + "_action":
                    suitable_videos.append(video_id)
            
            if not suitable_videos:
                logger.warning(f"No {intensity} intensity footage available")
                return None
            
            # Use the first suitable video (could be improved with better selection logic)
            video_id = suitable_videos[0]
            
            # If JSON file provided, create custom segments
            if json_file_path and json_file_path.exists():
                segments = await self.create_segments_from_json(video_id, json_file_path, buffer_seconds)
                return segments[0] if segments else None
            
            # Otherwise, analyze for best action segments
            best_segments = await self.get_best_action_segments(video_id, duration)
            
            if best_segments:
                # Create a segment from the best action timestamp
                start_time, end_time = best_segments[0]
                segments = await self.process_footage_for_tiktok(
                    video_id, 
                    custom_durations=[end_time - start_time]
                )
                return segments[0] if segments else None
            
            # Fallback to regular processing
            segments = await self.process_footage_for_tiktok(video_id)
            return segments[0] if segments else None
            
        except Exception as e:
            logger.error(f"Error getting footage for content: {e}")
            return None
