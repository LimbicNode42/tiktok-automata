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
        
        # Add video selection rotation to ensure variety
        self._video_selection_counter = 0
        
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
        """Analyze action intensity in a downloaded video with caching."""
        if video_id not in self.metadata["videos"]:
            logger.error(f"Video {video_id} not found")
            return {}
        
        video_info = self.metadata["videos"][video_id]
        video_path = Path(video_info["file_path"])
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return {}
          # 🚀 OPTIMIZATION: Check if already analyzed
        if "action_analysis" in video_info and video_info["action_analysis"].get("analyzed"):
            # Check if we have cached detailed results
            cache_file = self.storage_dir / f"{video_id}_action_cache.json"
            if cache_file.exists():
                try:
                    import json
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    logger.info(f"⚡ Using cached action analysis for {video_id}")
                    return cached_data
                except Exception as e:
                    logger.warning(f"Cache read failed, re-analyzing: {e}")
        
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
          # 🚀 OPTIMIZATION: Cache detailed results for future use
        cache_file = self.storage_dir / f"{video_id}_action_cache.json"
        try:
            import json            # Convert ActionMetrics objects to dict for JSON serialization
            cacheable_data = {}
            for level, metrics_list in action_data.items():
                cacheable_data[level] = []
                for metrics in metrics_list:
                    cacheable_data[level].append({
                        'timestamp': float(metrics.timestamp),
                        'motion_intensity': float(metrics.motion_intensity),
                        'color_variance': float(metrics.color_variance),
                        'edge_density': float(metrics.edge_density),
                        'scene_complexity': float(metrics.scene_complexity),
                        'action_type': getattr(metrics, 'action_type', 'unknown')
                    })
            
            with open(cache_file, 'w') as f:
                json.dump(cacheable_data, f, indent=2)
            logger.info(f"💾 Cached action analysis for {video_id}")
        except Exception as e:
            logger.warning(f"Failed to cache analysis: {e}")
        
        return action_data
    
    async def get_best_action_segments(self, video_id: str, segment_duration: float = 45.0) -> List:
        """Get the best action segments from a video with caching."""        # 🚀 OPTIMIZATION: Check for cached segment results
        cache_key = f"{video_id}_segments_{segment_duration:.1f}s"
        cache_file = self.storage_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    cached_segments = json.load(f)
                logger.info(f"⚡ Using cached action segments for {video_id}")
                return [(seg['start'], seg['end']) for seg in cached_segments]
            except Exception as e:
                logger.warning(f"Segment cache read failed, recomputing: {e}")
        
        # Get action data (this is now cached too)
        action_data = await self.analyze_video_action(video_id)
        segments = self.analyzer.get_best_action_segments(action_data, segment_duration)
        
        # 🚀 OPTIMIZATION: Cache segment results
        try:
            import json
            cacheable_segments = [{'start': seg[0], 'end': seg[1]} for seg in segments]
            with open(cache_file, 'w') as f:
                json.dump(cacheable_segments, f, indent=2)
            logger.info(f"💾 Cached action segments for {video_id}")
        except Exception as e:
            logger.warning(f"Failed to cache segments: {e}")
        
        return segments
    
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
        try:            # Find available videos matching the intensity
            suitable_videos = []
            
            for video_id, video_info in self.metadata["videos"].items():
                if video_info.get("content_type") == intensity + "_action":
                    suitable_videos.append(video_id)
            
            # If no exact match found, try fallback intensities
            if not suitable_videos:
                logger.warning(f"No {intensity} intensity footage available, trying fallbacks...")
                
                # Fallback hierarchy: high -> medium -> low
                fallback_intensities = []
                if intensity == "high":
                    fallback_intensities = ["medium", "low"]
                elif intensity == "medium":
                    fallback_intensities = ["high", "low"]
                elif intensity == "low":
                    fallback_intensities = ["medium", "high"]
                
                for fallback in fallback_intensities:
                    for video_id, video_info in self.metadata["videos"].items():
                        if video_info.get("content_type") == fallback + "_action":
                            suitable_videos.append(video_id)
                            logger.info(f"🔄 Using {fallback} intensity footage as fallback")
                            break
                    if suitable_videos:
                        break
            
            if not suitable_videos:
                logger.warning(f"No suitable footage available for any intensity level")
                return None
              # Better video selection: rotate through available videos for variety
            self._video_selection_counter = (self._video_selection_counter + 1) % len(suitable_videos)
            video_id = suitable_videos[self._video_selection_counter]
            logger.info(f"📹 Selected video {self._video_selection_counter + 1}/{len(suitable_videos)}: {video_id}")
            
            # Also add content-based selection as secondary variety
            import hashlib
            content_hash = hashlib.md5(f"{video_id}_{duration}".encode()).hexdigest()
            segment_seed = int(content_hash[:8], 16)
            
            # If JSON file provided, create custom segments
            if json_file_path and json_file_path.exists():
                segments = await self.create_segments_from_json(video_id, json_file_path, buffer_seconds)
                return segments[0] if segments else None
            
            # Use action analysis to find the BEST high-action segment for this duration
            best_segments = await self.get_best_action_segments(video_id, duration)
            
            if best_segments:
                # Get the best action segment (start_time, end_time)
                start_time, end_time = best_segments[0]
                logger.info(f"🎯 Using best action segment: {start_time:.1f}s - {end_time:.1f}s")
                
                # Create a segment with the specific start and end times from action analysis
                segments = await self.process_footage_for_tiktok(
                    video_id, 
                    duration_info_list=[{
                        'start_time': start_time,
                        'duration': duration,
                        'end_time': min(start_time + duration, end_time)
                    }]
                )
                return segments[0] if segments else None
            
            # Fallback to regular processing starting from a random point
            import random
            video_info = self.metadata["videos"][video_id]
            # Start from a random point in the first 60% of the video to ensure we have enough footage
            max_start = max(0, video_info.get("duration", 300) * 0.6 - duration)
            random_start = random.uniform(0, max_start) if max_start > 0 else 0
            
            segments = await self.process_footage_for_tiktok(
                video_id,
                duration_info_list=[{
                    'start_time': random_start,
                    'duration': duration
                }]
            )
            return segments[0] if segments else None
            
        except Exception as e:
            logger.error(f"Error getting footage for content: {e}")
            return None
