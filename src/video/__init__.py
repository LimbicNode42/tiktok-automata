"""
Video Processing Module for TikTok Automata

This module handles video generation using gaming footage, text overlays,
and synchronized editing for TikTok content creation.

Updated with modular architecture for better maintainability and custom duration support.
"""

from .video_processor import VideoProcessor, VideoConfig
from .footage_manager_new import FootageManager
from .youtube_downloader import FootageSource
from .segment_processor import VideoSegmentProcessor
from .base_footage_manager import BaseFootageManager
# TODO: Add back when ready
# from .overlay_generator import OverlayGenerator, OverlayConfig
# from .video_effects import VideoEffects, EffectConfig

__all__ = [
    'VideoProcessor',
    'VideoConfig', 
    'FootageManager',
    'FootageSource',
    'VideoSegmentProcessor',
    'BaseFootageManager',
    # TODO: Add back when ready
    # 'OverlayGenerator',
    # 'OverlayConfig',
    # 'VideoEffects',
    # 'EffectConfig'
]
