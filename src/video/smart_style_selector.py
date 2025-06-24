"""
Smart Subtitle Style Selector for TikTok Videos.

This module analyzes video content to automatically choose the best
subtitle style for optimal contrast and readability.
"""

import asyncio
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logger.error("MoviePy not available for video analysis")


@dataclass
class VideoAnalysis:
    """Results of video content analysis for subtitle styling."""
    average_brightness: float  # 0.0 to 1.0
    dominant_colors: List[Tuple[int, int, int]]  # RGB colors
    contrast_score: float  # How much contrast is available
    subtitle_region_brightness: float  # Brightness in subtitle area
    subtitle_region_colors: List[Tuple[int, int, int]]  # Colors in subtitle area
    recommended_style: str  # Best subtitle style for this video
    confidence: float  # Confidence in the recommendation (0.0 to 1.0)


@dataclass
class StyleContrastScore:
    """Contrast score for a specific subtitle style."""
    style_name: str
    contrast_score: float
    readability_score: float
    visual_appeal_score: float
    total_score: float


class SmartStyleSelector:
    """
    Analyzes video content to choose optimal subtitle styles.
    
    Features:
    - Brightness analysis in subtitle regions
    - Dominant color detection
    - Contrast scoring for different styles
    - Automatic style recommendation
    """
    
    def __init__(self):
        """Initialize the style selector."""
        # Define subtitle styles with their visual characteristics
        self.style_characteristics = {
            "bubble": {
                "text_color": (255, 255, 255),  # White
                "outline_color": (0, 0, 0),     # Black
                "outline_width": 8,
                "background": None,
                "best_for": "medium_contrast"
            },
            "bubble_blue": {
                "text_color": (255, 255, 255),  # White
                "outline_color": (0, 0, 128),   # Navy
                "outline_width": 7,
                "background": (173, 216, 230, 0.8),  # Light blue background
                "best_for": "warm_videos"
            },
            "bubble_gaming": {
                "text_color": (0, 255, 0),      # Lime
                "outline_color": (0, 100, 0),   # Dark green
                "outline_width": 8,
                "background": None,
                "best_for": "dark_videos"
            },
            "bubble_neon": {
                "text_color": (0, 255, 255),    # Cyan
                "outline_color": (0, 0, 139),   # Dark blue
                "outline_width": 8,
                "background": (0, 50, 100, 0.8),  # Dark blue background
                "best_for": "colorful_videos"
            },
            "bubble_cute": {
                "text_color": (255, 20, 147),   # Hot pink
                "outline_color": (128, 0, 128), # Purple
                "outline_width": 6,
                "background": (255, 192, 203, 0.7),  # Pink background
                "best_for": "cool_videos"
            },
            "bubble_classic": {
                "text_color": (255, 255, 0),    # Yellow
                "outline_color": (0, 0, 0),     # Black
                "outline_width": 10,
                "background": None,
                "best_for": "high_contrast"
            }
        }
        
        logger.info("SmartStyleSelector initialized with 6 bubble styles")
    
    async def analyze_video_for_subtitles(
        self, 
        video_clip: VideoFileClip,
        subtitle_position: float = 0.75
    ) -> VideoAnalysis:
        """
        Analyze video content to determine best subtitle style.
        
        Args:
            video_clip: The video to analyze
            subtitle_position: Vertical position where subtitles appear (0.0-1.0)
            
        Returns:
            VideoAnalysis with recommended subtitle style
        """
        try:
            logger.info("🔍 Analyzing video content for optimal subtitle style...")
            
            # Sample frames from the video for analysis
            sample_frames = await self._sample_video_frames(video_clip, num_samples=10)
            
            # Analyze subtitle region specifically
            subtitle_region_data = await self._analyze_subtitle_region(
                sample_frames, 
                video_clip.h, 
                subtitle_position
            )
            
            # Analyze overall video characteristics
            overall_data = await self._analyze_overall_video(sample_frames)
            
            # Score all subtitle styles
            style_scores = await self._score_all_styles(subtitle_region_data, overall_data)
            
            # Choose the best style
            best_style = max(style_scores, key=lambda x: x.total_score)
            
            analysis = VideoAnalysis(
                average_brightness=overall_data["brightness"],
                dominant_colors=overall_data["colors"],
                contrast_score=overall_data["contrast"],
                subtitle_region_brightness=subtitle_region_data["brightness"],
                subtitle_region_colors=subtitle_region_data["colors"],
                recommended_style=best_style.style_name,
                confidence=best_style.total_score / 100.0  # Normalize to 0-1
            )
            
            logger.success(f"✅ Recommended style: {best_style.style_name} (confidence: {analysis.confidence:.2f})")
            logger.info(f"   📊 Subtitle area brightness: {subtitle_region_data['brightness']:.2f}")
            logger.info(f"   🎨 Dominant colors: {len(overall_data['colors'])} detected")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            # Return default analysis
            return VideoAnalysis(
                average_brightness=0.5,
                dominant_colors=[(128, 128, 128)],
                contrast_score=0.5,
                subtitle_region_brightness=0.5,
                subtitle_region_colors=[(128, 128, 128)],
                recommended_style="bubble",  # Safe default
                confidence=0.5
            )
    
    async def _sample_video_frames(
        self, 
        video_clip: VideoFileClip, 
        num_samples: int = 10
    ) -> List[np.ndarray]:
        """Sample frames evenly throughout the video."""
        try:
            duration = video_clip.duration
            sample_times = np.linspace(0.5, duration - 0.5, num_samples)
            
            frames = []
            for t in sample_times:
                frame = video_clip.get_frame(t)
                frames.append(frame)
            
            logger.info(f"📸 Sampled {len(frames)} frames for analysis")
            return frames
            
        except Exception as e:
            logger.error(f"Frame sampling failed: {e}")
            return []
    
    async def _analyze_subtitle_region(
        self, 
        frames: List[np.ndarray], 
        video_height: int,
        subtitle_position: float
    ) -> Dict:
        """Analyze the region where subtitles will appear."""
        try:
            # Define subtitle region (bottom 30% of video around subtitle position)
            region_height = int(video_height * 0.3)
            center_y = int(video_height * subtitle_position)
            start_y = max(0, center_y - region_height // 2)
            end_y = min(video_height, center_y + region_height // 2)
            
            region_brightnesses = []
            region_colors = []
            
            for frame in frames:
                # Extract subtitle region
                subtitle_region = frame[start_y:end_y, :, :]
                
                # Calculate brightness (grayscale mean)
                gray_region = cv2.cvtColor(subtitle_region, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray_region) / 255.0
                region_brightnesses.append(brightness)
                
                # Extract dominant colors
                reshaped = subtitle_region.reshape(-1, 3)
                colors = self._extract_dominant_colors(reshaped, k=3)
                region_colors.extend(colors)
            
            avg_brightness = np.mean(region_brightnesses)
            unique_colors = self._cluster_similar_colors(region_colors, threshold=30)
            
            logger.info(f"   📍 Subtitle region analysis: brightness={avg_brightness:.2f}")
            
            return {
                "brightness": avg_brightness,
                "colors": unique_colors[:5],  # Top 5 colors
                "brightness_variance": np.std(region_brightnesses)
            }
            
        except Exception as e:
            logger.error(f"Subtitle region analysis failed: {e}")
            return {"brightness": 0.5, "colors": [(128, 128, 128)], "brightness_variance": 0.1}
    
    async def _analyze_overall_video(self, frames: List[np.ndarray]) -> Dict:
        """Analyze overall video characteristics."""
        try:
            overall_brightnesses = []
            overall_colors = []
            
            for frame in frames:
                # Overall brightness
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray_frame) / 255.0
                overall_brightnesses.append(brightness)
                
                # Sample colors from the frame
                reshaped = frame.reshape(-1, 3)
                # Sample every 100th pixel to speed up processing
                sampled = reshaped[::100]
                colors = self._extract_dominant_colors(sampled, k=5)
                overall_colors.extend(colors)
            
            avg_brightness = np.mean(overall_brightnesses)
            contrast_score = np.std(overall_brightnesses) * 2  # Higher std = more contrast
            unique_colors = self._cluster_similar_colors(overall_colors, threshold=40)
            
            logger.info(f"   🌍 Overall video analysis: brightness={avg_brightness:.2f}, contrast={contrast_score:.2f}")
            
            return {
                "brightness": avg_brightness,
                "colors": unique_colors[:8],  # Top 8 colors
                "contrast": min(contrast_score, 1.0)  # Cap at 1.0
            }
            
        except Exception as e:
            logger.error(f"Overall video analysis failed: {e}")
            return {"brightness": 0.5, "colors": [(128, 128, 128)], "contrast": 0.5}
    
    def _extract_dominant_colors(self, pixels: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using k-means clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Use k-means to find dominant colors
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = []
            for center in kmeans.cluster_centers_:
                color = tuple(int(c) for c in center)
                colors.append(color)
            
            return colors
            
        except ImportError:
            # Fallback without sklearn
            logger.warning("sklearn not available, using simple color extraction")
            return self._simple_color_extraction(pixels, k)
        except Exception as e:
            logger.warning(f"K-means color extraction failed: {e}")
            return self._simple_color_extraction(pixels, k)
    
    def _simple_color_extraction(self, pixels: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Simple color extraction without sklearn."""
        try:
            # Sample random pixels
            if len(pixels) > 1000:
                indices = np.random.choice(len(pixels), 1000, replace=False)
                sampled_pixels = pixels[indices]
            else:
                sampled_pixels = pixels
            
            # Group similar colors manually
            colors = []
            for pixel in sampled_pixels[::50]:  # Sample every 50th pixel
                color = tuple(int(c) for c in pixel)
                colors.append(color)
            
            # Return unique colors
            unique_colors = list(set(colors))
            return unique_colors[:k]
            
        except Exception as e:
            logger.warning(f"Simple color extraction failed: {e}")
            return [(128, 128, 128)]  # Gray fallback
    
    def _cluster_similar_colors(
        self, 
        colors: List[Tuple[int, int, int]], 
        threshold: int = 30
    ) -> List[Tuple[int, int, int]]:
        """Group similar colors together."""
        try:
            if not colors:
                return [(128, 128, 128)]
            
            clustered = []
            
            for color in colors:
                # Check if this color is similar to any existing clustered color
                is_similar = False
                for clustered_color in clustered:
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(color, clustered_color)))
                    if distance < threshold:
                        is_similar = True
                        break
                
                if not is_similar:
                    clustered.append(color)
            
            return clustered
            
        except Exception as e:
            logger.warning(f"Color clustering failed: {e}")
            return colors[:5]  # Return first 5 if clustering fails
    
    async def _score_all_styles(
        self, 
        subtitle_region_data: Dict, 
        overall_data: Dict
    ) -> List[StyleContrastScore]:
        """Score all available subtitle styles for this video."""
        scores = []
        
        for style_name, characteristics in self.style_characteristics.items():
            score = await self._score_single_style(
                style_name, 
                characteristics, 
                subtitle_region_data, 
                overall_data
            )
            scores.append(score)
        
        # Sort by total score
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        logger.info(f"   📊 Style scores: {', '.join([f'{s.style_name}({s.total_score:.1f})' for s in scores[:3]])}")
        
        return scores
    
    async def _score_single_style(
        self, 
        style_name: str, 
        characteristics: Dict,
        subtitle_region_data: Dict, 
        overall_data: Dict
    ) -> StyleContrastScore:
        """Score a single subtitle style for this video."""
        try:
            # Get style colors
            text_color = characteristics["text_color"]
            outline_color = characteristics["outline_color"]
            
            # Calculate contrast with subtitle region
            region_brightness = subtitle_region_data["brightness"]
            region_colors = subtitle_region_data["colors"]
            
            # Contrast score: how well does text stand out?
            text_luminance = self._calculate_luminance(text_color)
            contrast_score = abs(text_luminance - region_brightness) * 100
            
            # Readability score: outline contrast with background
            readability_score = 0
            for bg_color in region_colors:
                bg_luminance = self._calculate_luminance(bg_color)
                outline_luminance = self._calculate_luminance(outline_color)
                outline_contrast = abs(outline_luminance - bg_luminance)
                readability_score = max(readability_score, outline_contrast * 100)
            
            # Visual appeal: color harmony with video
            visual_appeal_score = self._calculate_color_harmony(
                text_color, 
                overall_data["colors"]
            )
            
            # Bonus scoring for special conditions
            bonus_score = 0
            if characteristics["best_for"] == "dark_videos" and region_brightness < 0.3:
                bonus_score += 20
            elif characteristics["best_for"] == "high_contrast" and overall_data["contrast"] > 0.7:
                bonus_score += 15
            elif characteristics["best_for"] == "medium_contrast" and 0.3 <= region_brightness <= 0.7:
                bonus_score += 10
            
            # Calculate total score
            total_score = (
                contrast_score * 0.4 +           # 40% weight on contrast
                readability_score * 0.3 +        # 30% weight on readability
                visual_appeal_score * 0.2 +      # 20% weight on visual appeal
                bonus_score * 0.1                # 10% weight on bonus conditions
            )
            
            return StyleContrastScore(
                style_name=style_name,
                contrast_score=contrast_score,
                readability_score=readability_score,
                visual_appeal_score=visual_appeal_score,
                total_score=total_score
            )
            
        except Exception as e:
            logger.warning(f"Style scoring failed for {style_name}: {e}")
            return StyleContrastScore(
                style_name=style_name,
                contrast_score=50,
                readability_score=50,
                visual_appeal_score=50,
                total_score=50
            )
    
    def _calculate_luminance(self, rgb_color: Tuple[int, int, int]) -> float:
        """Calculate relative luminance of a color (0.0 to 1.0)."""
        try:
            r, g, b = [c / 255.0 for c in rgb_color]
            
            # Apply gamma correction
            def gamma_correct(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            
            r = gamma_correct(r)
            g = gamma_correct(g)
            b = gamma_correct(b)
            
            # Calculate luminance using standard weights
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
            
        except Exception as e:
            logger.warning(f"Luminance calculation failed: {e}")
            return 0.5  # Middle gray fallback
    
    def _calculate_color_harmony(
        self, 
        text_color: Tuple[int, int, int], 
        video_colors: List[Tuple[int, int, int]]
    ) -> float:
        """Calculate how well the text color harmonizes with video colors."""
        try:
            if not video_colors:
                return 50.0
            
            harmony_scores = []
            
            for video_color in video_colors:
                # Calculate color distance in RGB space
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(text_color, video_color)))
                max_distance = np.sqrt(3 * 255 ** 2)  # Maximum possible distance
                
                # Convert to harmony score (closer = more harmonious up to a point)
                normalized_distance = distance / max_distance
                
                # Sweet spot: not too similar (poor contrast) but not too different (clashing)
                if 0.3 <= normalized_distance <= 0.7:
                    harmony = 100 * (1 - abs(normalized_distance - 0.5) * 2)
                else:
                    harmony = 50 * (1 - abs(normalized_distance - 0.5))
                
                harmony_scores.append(harmony)
            
            return max(harmony_scores) if harmony_scores else 50.0
            
        except Exception as e:
            logger.warning(f"Color harmony calculation failed: {e}")
            return 50.0


# Quick access function for integration
async def analyze_and_recommend_style(
    video_clip: VideoFileClip, 
    subtitle_position: float = 0.75
) -> str:
    """
    Quick function to analyze video and get recommended subtitle style.
    
    Args:
        video_clip: The video to analyze
        subtitle_position: Where subtitles will appear (0.0-1.0)
        
    Returns:
        Recommended subtitle style name
    """
    selector = SmartStyleSelector()
    analysis = await selector.analyze_video_for_subtitles(video_clip, subtitle_position)
    return analysis.recommended_style
