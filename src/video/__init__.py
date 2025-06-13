"""
Video Processing Module for TikTok Automata

This module handles video generation using gaming footage, text overlays,
and synchronized editing for TikTok content creation.
"""

from .video_processor import VideoProcessor, VideoConfig
from .footage_manager import FootageManager, FootageSource
# TODO: Add back when ready
# from .overlay_generator import OverlayGenerator, OverlayConfig
# from .video_effects import VideoEffects, EffectConfig

__all__ = [
    'VideoProcessor',
    'VideoConfig', 
    'FootageManager',
    'FootageSource',
    # TODO: Add back when ready
    # 'OverlayGenerator',
    # 'OverlayConfig',
    # 'VideoEffects',
    # 'EffectConfig'
]
