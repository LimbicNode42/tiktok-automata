"""
Video Processing Module for TikTok Automata - Now with organized modular architecture!

This module handles video generation using gaming footage, text overlays,
and synchronized editing for TikTok content creation.

Updated with modular architecture for better maintainability and custom duration support.

Architecture:
- downloaders/: YouTube downloading logic
- analyzers/:   Video action analysis
- processors/:  Video and segment processing  
- managers/:    High-level coordination
"""

# Import from organized modules
from .managers import FootageManager
from .downloaders import YouTubeDownloader  
from .analyzers import VideoActionAnalyzer, ActionMetrics
from .processors import VideoProcessor, VideoConfig, VideoSegmentProcessor

# Legacy imports for backward compatibility
from .base_footage_manager import BaseFootageManager
from .downloaders.youtube_downloader import FootageSource
# TODO: Add back when ready
# from .overlay_generator import OverlayGenerator, OverlayConfig
# from .video_effects import VideoEffects, EffectConfig

__all__ = [
    # Main interfaces
    'FootageManager',
    'VideoProcessor', 
    'VideoConfig',
    
    # Specialized modules  
    'YouTubeDownloader',
    'VideoActionAnalyzer',
    'ActionMetrics',
    'VideoSegmentProcessor',
    
    # Legacy/base classes
    'BaseFootageManager',
    'FootageSource',
    # TODO: Add back when ready
    # 'OverlayGenerator',
    # 'OverlayConfig',
    # 'VideoEffects',
    # 'EffectConfig'
]
