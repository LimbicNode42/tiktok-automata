"""
Video Action Analyzer - Analyzes gaming footage for action intensity.

This module focuses solely on detecting and quantifying action levels in video content.
Used to categorize video segments by intensity for better TikTok content selection.
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger


@dataclass
class ActionMetrics:
    """Enhanced data class for action analysis results."""
    motion_intensity: float = 0.0
    color_variance: float = 0.0
    edge_density: float = 0.0
    scene_complexity: float = 0.0
    audio_energy: float = 0.0
    overall_score: float = 0.0
    timestamp: float = 0.0
    category: str = "low"  # low, medium, high, extreme
    content_type: str = "unknown"  # gaming, action, dialogue, transition
    confidence: float = 0.0


class ContentTypeDetector:
    """Detects the type of content in video segments."""
    
    @staticmethod
    def detect_content_type(metrics: ActionMetrics) -> str:
        """Detect content type based on metrics."""
        # Gaming content typically has high motion + high edge density
        if metrics.motion_intensity > 12 and metrics.edge_density > 8:
            return "gaming"
        
        # Action scenes have high motion + moderate complexity
        elif metrics.motion_intensity > 8 and metrics.scene_complexity > 5:
            return "action"
        
        # Dialogue scenes have low motion + low complexity
        elif metrics.motion_intensity < 3 and metrics.edge_density < 4:
            return "dialogue"
        
        # Transition scenes have moderate motion + high color variance
        elif 3 <= metrics.motion_intensity <= 8 and metrics.color_variance > 2000:
            return "transition"
        
        return "unknown"


class AdvancedAnalyzer:
    """Advanced analysis algorithms for better action detection."""
    
    @staticmethod
    def calculate_scene_complexity(frames: List[np.ndarray]) -> float:
        """Calculate scene complexity using texture analysis."""
        try:
            complexities = []
            for frame in frames:
                # Convert to grayscale
                gray = np.mean(frame, axis=2)
                
                # Calculate local variance (texture measure)
                kernel_size = 3
                h, w = gray.shape
                local_vars = []
                
                for i in range(kernel_size, h - kernel_size, 5):  # Sample every 5 pixels
                    for j in range(kernel_size, w - kernel_size, 5):
                        patch = gray[i-kernel_size:i+kernel_size+1, j-kernel_size:j+kernel_size+1]
                        local_vars.append(np.var(patch))
                
                avg_complexity = np.mean(local_vars) if local_vars else 0.0
                complexities.append(avg_complexity)
            
            return np.mean(complexities)
        except Exception as e:
            logger.warning(f"Scene complexity calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def calculate_optical_flow_approximation(frames: List[np.ndarray]) -> float:
        """Approximate optical flow using frame differences."""
        try:
            if len(frames) < 2:
                return 0.0
                
            flow_magnitudes = []
            for i in range(1, len(frames)):
                # Convert to grayscale
                gray1 = np.mean(frames[i-1], axis=2)
                gray2 = np.mean(frames[i], axis=2)
                
                # Calculate gradient-based flow approximation
                dx = np.abs(np.diff(gray2, axis=1))
                dy = np.abs(np.diff(gray2, axis=0))
                
                # Temporal difference
                dt = np.abs(gray2[1:, 1:] - gray1[1:, 1:])
                
                # Flow magnitude approximation
                flow_mag = np.sqrt(dx[1:, :]**2 + dy[:, 1:]**2) * dt
                flow_magnitudes.append(np.mean(flow_mag))
            
            return np.mean(flow_magnitudes)
        except Exception as e:
            logger.warning(f"Optical flow calculation failed: {e}")
            return 0.0


class VideoActionAnalyzer:
    """
    Analyzes video content for action intensity with statistical categorization.
      Single responsibility: Detect and quantify action levels in video segments.
    Features:
    - Multi-metric action detection (motion, edges, complexity)
    - Adaptive thresholding based on video statistics
    - Statistical ranking and percentile-based categorization    """
    
    def __init__(self):
        self.sample_interval = 30.0  # Analyze every 30 seconds
        self.frame_skip = 5  # Skip frames for faster processing
        self.max_frames_per_sample = 3  # Reduce frames per sample for speed
        
        # Performance optimization settings
        self.enable_optical_flow = True  # Can be disabled for speed
        self.enable_scene_complexity = True  # Can be disabled for speed
        self.adaptive_thresholds = {}  # Store per-video thresholds
        
    async def analyze_video_action(self, video_path: Path) -> Dict[str, List[ActionMetrics]]:
        """
        Analyze a video file for action intensity throughout its duration.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with action metrics categorized by intensity
        """
        try:
            # Import moviepy here to avoid startup delay
            from moviepy import VideoFileClip
        except ImportError:
            logger.error("MoviePy not installed. Run: pip install moviepy")
            return {"high": [], "medium": [], "low": []}
        
        logger.info(f"🔍 Analyzing action levels in: {video_path.name}")
        
        try:
            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            
            # Analyze video in chunks
            metrics_list = []
            timestamps = np.arange(0, duration, self.sample_interval)
            
            for i, timestamp in enumerate(timestamps):
                if timestamp + 1 >= duration:  # Need at least 1 second of video
                    break
                
                # Extract 1-second sample
                sample = clip.subclipped(timestamp, min(timestamp + 1, duration))
                metrics = await self._analyze_sample(sample, timestamp)
                metrics_list.append(metrics)
                
                sample.close()
                
                # Progress logging
                if i % 10 == 0:
                    progress = (i / len(timestamps)) * 100
                    logger.info(f"   Analysis progress: {progress:.1f}%")
            
            clip.close()
            
            # Categorize by intensity
            categorized = self._categorize_metrics(metrics_list)
            
            logger.success(f"✅ Action analysis complete: {len(categorized['high'])} high, {len(categorized['medium'])} medium, {len(categorized['low'])} low action segments")            
            return categorized
            
        except Exception as e:
            logger.error(f"❌ Action analysis failed: {e}")
            return {"high": [], "medium": [], "low": []}
    
    async def _analyze_sample(self, sample_clip, timestamp: float) -> ActionMetrics:
        """Analyze a short video sample for action metrics."""
        metrics = ActionMetrics()
        metrics.timestamp = timestamp
        
        try:
            # Get frames for analysis (reduced number for performance)
            sample_duration = min(sample_clip.duration, 1.0)  # Use up to 1 second
            frame_times = np.linspace(0, sample_duration * 0.9, self.max_frames_per_sample)  # Avoid end of clip
            frames = [sample_clip.get_frame(t) for t in frame_times]
            
            # Calculate motion intensity (frame differences)
            motion_scores = []
            for i in range(1, len(frames)):
                diff = np.mean(np.abs(frames[i] - frames[i-1]))
                motion_scores.append(diff)
            
            metrics.motion_intensity = np.mean(motion_scores) if motion_scores else 0.0
            
            # Calculate color variance
            color_variances = []
            for frame in frames:
                # Variance across all channels
                var = np.var(frame.flatten())
                color_variances.append(var)
            
            metrics.color_variance = np.mean(color_variances)
            
            # Calculate edge density (using Sobel-like approximation)
            edge_densities = []
            for frame in frames:
                gray = np.mean(frame, axis=2)  # Convert to grayscale
                # Simple edge detection
                edges_x = np.abs(np.diff(gray, axis=1))
                edges_y = np.abs(np.diff(gray, axis=0))
                edge_density = np.mean(edges_x) + np.mean(edges_y)
                edge_densities.append(edge_density)
            
            metrics.edge_density = np.mean(edge_densities)
              # Calculate scene complexity
            metrics.scene_complexity = AdvancedAnalyzer.calculate_scene_complexity(frames)
            
            # Calculate optical flow approximation (enhance motion detection)
            optical_flow_score = AdvancedAnalyzer.calculate_optical_flow_approximation(frames)
            
            # Combine motion metrics for better accuracy
            metrics.motion_intensity = (metrics.motion_intensity * 0.7 + optical_flow_score * 0.3)
            
            # Calculate overall score (weighted combination)
            metrics.overall_score = (
                metrics.motion_intensity * 0.4 +
                metrics.color_variance * 0.0001 +  # Scale down color variance
                metrics.edge_density * 0.35 +
                metrics.scene_complexity * 0.25
            )
            
            # Determine category
            if metrics.overall_score > 20:
                metrics.category = "high"
            elif metrics.overall_score > 10:
                metrics.category = "medium"
            else:
                metrics.category = "low"
                
            # Detect content type
            metrics.content_type = ContentTypeDetector.detect_content_type(metrics)
            
        except Exception as e:
            logger.warning(f"⚠️ Sample analysis failed at {timestamp}s: {e}")
            metrics.category = "low"
        
        return metrics
    
    def _categorize_metrics(self, metrics_list: List[ActionMetrics]) -> Dict[str, List[ActionMetrics]]:
        """Categorize metrics by action intensity."""
        categorized = {"high": [], "medium": [], "low": []}
        
        for metrics in metrics_list:
            categorized[metrics.category].append(metrics)
        
        return categorized
    
    def get_best_action_segments(self, categorized_metrics: Dict[str, List[ActionMetrics]], 
                                segment_duration: float = 45.0, 
                                max_segments: int = 5) -> List[Tuple[float, float]]:
        """
        Get the best action segments for TikTok content.
        
        Args:
            categorized_metrics: Results from analyze_video_action
            segment_duration: Desired segment length in seconds
            max_segments: Maximum number of segments to return
            
        Returns:
            List of (start_time, end_time) tuples for best segments
        """
        # Prioritize high action, then medium
        all_metrics = categorized_metrics["high"] + categorized_metrics["medium"]
        
        if not all_metrics:
            # Fallback to low action if nothing else available
            all_metrics = categorized_metrics["low"]
        
        # Sort by overall score (highest first)
        all_metrics.sort(key=lambda m: m.overall_score, reverse=True)
        
        # Extract non-overlapping segments
        segments = []
        used_times = set()
        
        for metrics in all_metrics:
            start_time = metrics.timestamp
            end_time = start_time + segment_duration
            
            # Check for overlap with existing segments
            overlap = any(
                (start_time < existing_end and end_time > existing_start)
                for existing_start, existing_end in segments            )            
            if not overlap and len(segments) < max_segments:
                segments.append((start_time, end_time))
        
        logger.info(f"🎯 Selected {len(segments)} high-action segments")
        return segments
    
    async def analyze_continuous_segments(self, video_path: Path, segment_durations: List[float]) -> Dict[str, List[Dict]]:
        """
        Analyze video for best continuous segments of specific durations.
        
        Args:
            video_path: Path to the video file
            segment_durations: List of desired segment durations in seconds
            
        Returns:
            Dictionary mapping duration to best segment info
        """
        try:
            from moviepy import VideoFileClip
        except ImportError:
            logger.error("MoviePy not installed. Run: pip install moviepy")
            return {}
        
        logger.info(f"� Analyzing continuous segments for durations: {segment_durations}")
        
        try:
            clip = VideoFileClip(str(video_path))
            video_duration = clip.duration
            
            results = {}
            
            for target_duration in segment_durations:
                logger.info(f"🔍 Finding best {target_duration:.1f}s continuous segment...")
                
                if target_duration >= video_duration:
                    logger.warning(f"⚠️ Target duration {target_duration:.1f}s >= video duration {video_duration:.1f}s")
                    continue
                
                best_segments = await self._find_best_continuous_segment(clip, target_duration)
                results[str(target_duration)] = best_segments
                
                if best_segments:
                    best = best_segments[0]
                    logger.info(f"✅ Best {target_duration:.1f}s segment: {best['start_time']:.1f}s - {best['end_time']:.1f}s (score: {best['avg_score']:.1f})")
            
            clip.close()
            return results
            
        except Exception as e:
            logger.error(f"❌ Continuous segment analysis failed: {e}")
            return {}
    
    async def _find_best_continuous_segment(self, clip, target_duration: float) -> List[Dict]:
        """Find the best continuous segment of specified duration."""
        video_duration = clip.duration
        sample_interval = 2.0  # Analyze every 2 seconds
        
        # Calculate possible start times (with some buffer at the end)
        max_start_time = video_duration - target_duration - 1.0
        if max_start_time <= 0:
            return []
        
        start_times = np.arange(0, max_start_time, sample_interval)
        segment_scores = []
        
        logger.info(f"   Evaluating {len(start_times)} possible segment positions...")
        
        for i, start_time in enumerate(start_times):
            end_time = start_time + target_duration
            
            # Sample the segment at regular intervals
            sample_points = np.linspace(start_time, end_time - 1, min(int(target_duration / 2), 20))
            segment_metrics = []
            
            for sample_time in sample_points:
                try:
                    # Create mini clip for analysis
                    mini_clip = clip.subclipped(sample_time, min(sample_time + 1, end_time))
                    metrics = await self._analyze_sample(mini_clip, sample_time)
                    segment_metrics.append(metrics)
                    mini_clip.close()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to analyze sample at {sample_time:.1f}s: {e}")
                    continue
            
            # Calculate average score for this segment
            if segment_metrics:
                avg_score = np.mean([m.overall_score for m in segment_metrics])
                min_score = np.min([m.overall_score for m in segment_metrics])
                max_score = np.max([m.overall_score for m in segment_metrics])
                
                segment_scores.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': target_duration,
                    'avg_score': avg_score,
                    'min_score': min_score,
                    'max_score': max_score,
                    'score_variance': np.var([m.overall_score for m in segment_metrics]),
                    'sample_count': len(segment_metrics)
                })
            
            # Progress update every 10 segments
            if i % 10 == 0:
                progress = (i / len(start_times)) * 100
                logger.info(f"     Progress: {progress:.1f}%")
        
        # Sort by average score (highest first)
        segment_scores.sort(key=lambda x: x['avg_score'], reverse=True)
        
        return segment_scores[:5]  # Return top 5 segments
    
    def analyze_action_distribution(self, categorized_metrics: Dict[str, List[ActionMetrics]]) -> Dict[str, float]:
        """
        Analyze the statistical distribution of action intensities.
        
        Returns:
            Dictionary with statistical insights about action distribution
        """
        all_metrics = categorized_metrics["high"] + categorized_metrics["medium"] + categorized_metrics["low"]
        
        if not all_metrics:
            return {"action_ratio": 0.0, "avg_intensity": 0.0, "intensity_variance": 0.0}
        
        # Calculate key statistics
        overall_scores = [m.overall_score for m in all_metrics]
        motion_scores = [m.motion_intensity for m in all_metrics]
        
        stats = {
            "total_segments": len(all_metrics),
            "high_action_ratio": len(categorized_metrics["high"]) / len(all_metrics),
            "medium_action_ratio": len(categorized_metrics["medium"]) / len(all_metrics),
            "low_action_ratio": len(categorized_metrics["low"]) / len(all_metrics),
            "avg_overall_score": np.mean(overall_scores),
            "avg_motion_intensity": np.mean(motion_scores),
            "intensity_variance": np.var(overall_scores),
            "peak_intensity": np.max(overall_scores),
            "min_intensity": np.min(overall_scores),
            "intensity_range": np.max(overall_scores) - np.min(overall_scores)
        }
        
        # Content type distribution
        content_types = {}
        for metrics in all_metrics:
            content_type = metrics.content_type
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        stats["content_distribution"] = content_types
        
        logger.info(f"📊 Action Distribution: {stats['high_action_ratio']:.1%} high, {stats['medium_action_ratio']:.1%} medium, {stats['low_action_ratio']:.1%} low")
        
        return stats
    
    def get_adaptive_thresholds(self, all_metrics: List[ActionMetrics]) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on video content statistics.
        
        Args:
            all_metrics: List of all action metrics from a video
            
        Returns:
            Dictionary with adaptive threshold values
        """
        if len(all_metrics) < 10:  # Need minimum data for statistics
            return {"high_threshold": 20.0, "medium_threshold": 10.0}
        
        overall_scores = [m.overall_score for m in all_metrics]
        
        # Use percentiles for adaptive thresholding
        p90 = np.percentile(overall_scores, 90)
        p70 = np.percentile(overall_scores, 70)
        p50 = np.percentile(overall_scores, 50)
        
        # Ensure minimum separation between thresholds
        high_threshold = max(p90, p50 + 5.0)
        medium_threshold = max(p70, p50 + 2.0)
        
        logger.info(f"🎚️ Adaptive thresholds: High>{high_threshold:.1f}, Medium>{medium_threshold:.1f}")
        
        return {
            "high_threshold": high_threshold,
            "medium_threshold": medium_threshold,
            "percentile_90": p90,
            "percentile_70": p70,
            "median": p50
        }
    
    def categorize_with_adaptive_thresholds(self, metrics_list: List[ActionMetrics]) -> Dict[str, List[ActionMetrics]]:
        """
        Categorize metrics using adaptive thresholds based on video statistics.
        
        Args:
            metrics_list: List of action metrics to categorize
              Returns:
            Dictionary with categorized metrics using adaptive thresholds
        """
        thresholds = self.get_adaptive_thresholds(metrics_list)
        high_threshold = thresholds["high_threshold"]
        medium_threshold = thresholds["medium_threshold"]
        
        categorized = {"high": [], "medium": [], "low": []}
        
        for metrics in metrics_list:
            if metrics.overall_score >= high_threshold:
                metrics.category = "high"
                categorized["high"].append(metrics)
            elif metrics.overall_score >= medium_threshold:
                metrics.category = "medium"
                categorized["medium"].append(metrics)
            else:
                metrics.category = "low"
                categorized["low"].append(metrics)
        
        return categorized
    
    async def analyze_segments_from_json(self, video_path: Path, json_file_path: Path, buffer_seconds: float = 7.5) -> Dict[str, Dict]:
        """
        Analyze video segments based on durations from JSON file.
        
        Args:
            video_path: Path to the video file
            json_file_path: Path to JSON file with audio durations
            buffer_seconds: Extra seconds to add to each duration
            
        Returns:
            Dictionary with results for each audio segment
        """
        import json
        
        logger.info(f"🎵 Analyzing segments from JSON: {json_file_path.name}")
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            # Extract durations from JSON
            durations = []
            segment_info = []
            
            for i, result in enumerate(data.get('results', [])):
                if 'duration' in result:
                    original_duration = result['duration']
                    buffered_duration = original_duration + buffer_seconds
                    durations.append(buffered_duration)
                    
                    segment_info.append({
                        'index': i,
                        'title': result.get('title', f'Segment {i}'),
                        'original_duration': original_duration,
                        'buffered_duration': buffered_duration,
                        'voice_name': result.get('voice_name', 'Unknown'),
                        'category': result.get('category', 'unknown')
                    })
            
            logger.info(f"📊 Found {len(durations)} audio segments with durations: {[f'{d:.1f}s' for d in durations[:3]]}...")
            
            # Analyze continuous segments for these durations
            results = await self.analyze_continuous_segments(video_path, durations)
            
            # Combine with segment info
            combined_results = {}
            for i, info in enumerate(segment_info):
                duration_key = str(info['buffered_duration'])
                if duration_key in results:
                    combined_results[f"segment_{i}"] = {
                        'segment_info': info,
                        'best_video_segments': results[duration_key]
                    }
            
            logger.success(f"✅ Analyzed {len(combined_results)} segments from JSON")
            return combined_results
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze segments from JSON: {e}")
            return {}
