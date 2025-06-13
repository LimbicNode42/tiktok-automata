"""
TTS (Text-to-Speech) module for TikTok Automata.
Provides high-quality audio generation using Kokoro TTS.
"""

from .kokoro_tts import KokoroTTSEngine, TTSConfig

__all__ = ['KokoroTTSEngine', 'TTSConfig']
