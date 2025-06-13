"""
Unit tests for Kokoro TTS Engine.
Run with: python -m pytest src/tts/tests/test_kokoro_tts.py -v
"""

import asyncio
import os
import sys
from pathlib import Path
import tempfile

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from src.tts.kokoro_tts import KokoroTTSEngine, TTSConfig, create_tts_engine, quick_tts
    from src.scraper.newsletter_scraper import Article
    print("✓ Successfully imported TTS components for testing")
except ImportError as e:
    print(f"✗ Failed to import TTS components: {e}")


def test_tts_config_defaults():
    """Test TTSConfig default values."""
    config = TTSConfig()
    
    assert config.voice == 'af_heart'
    assert config.speed == 1.0
    assert config.language == 'a'
    assert config.sample_rate == 24000
    assert config.output_format == 'wav'
    assert config.target_duration == 60.0
    assert config.normalize_audio == True


def test_kokoro_engine_creation():
    """Test creating KokoroTTSEngine."""
    engine = KokoroTTSEngine()
    
    assert engine.config is not None
    assert not engine.initialized
    assert engine.pipeline is None
    assert len(engine.available_voices) > 0
    assert 'af_heart' in engine.available_voices


def test_voice_management():
    """Test voice listing and selection."""
    engine = KokoroTTSEngine()
    
    # Test voice listing
    voices = engine.list_voices()
    assert isinstance(voices, dict)
    assert len(voices) > 0
    
    # Test setting valid voice
    original_voice = engine.config.voice
    engine.set_voice('am_adam')
    assert engine.config.voice == 'am_adam'
    
    # Test setting invalid voice (should not change)
    engine.set_voice('invalid_voice_name')
    assert engine.config.voice == 'am_adam'


def test_text_preparation():
    """Test text cleaning and preparation functions."""
    engine = KokoroTTSEngine()
    
    # Test basic cleaning
    dirty_text = "**Bold** text with *italic* and `code` and #hashtag"
    clean_text = engine._prepare_text(dirty_text)
    
    assert "**" not in clean_text
    assert "*" not in clean_text
    assert "`" not in clean_text
    assert "hashtag hashtag" in clean_text
    
    # Test timestamp removal
    tiktok_text = "**[0s-3s]** This is a TikTok script [5s-10s] with timestamps"
    clean_tiktok = engine._prepare_text(tiktok_text)
    
    assert "[0s-3s]" not in clean_tiktok
    assert "[5s-10s]" not in clean_tiktok
    assert "This is a TikTok script" in clean_tiktok
    
    # Test whitespace normalization
    spaced_text = "Too    many   \t\n  spaces"
    clean_spaced = engine._prepare_text(spaced_text)
    assert "Too many spaces" == clean_spaced


def test_text_splitting():
    """Test intelligent text splitting functionality."""
    engine = KokoroTTSEngine()
    engine.config.max_length = 100  # Set short limit for testing
    
    # Test short text (should not split)
    short_text = "This is a short text that fits in one chunk."
    chunks = engine._split_text_intelligently(short_text)
    assert len(chunks) == 1
    assert chunks[0] == short_text
    
    # Test long text (should split)
    long_text = "This is a very long sentence that definitely exceeds our maximum length limit and should be split into multiple chunks for processing. " * 2
    chunks = engine._split_text_intelligently(long_text)
    assert len(chunks) > 1
    
    # Verify no chunk exceeds max length
    for chunk in chunks:
        assert len(chunk) <= engine.config.max_length


def test_audio_info_error_handling():
    """Test audio info function with non-existent file."""
    engine = KokoroTTSEngine()
    
    # Test with non-existent file
    info = engine.get_audio_info("/path/that/does/not/exist.wav")
    assert "error" in info
    assert info["error"] == "File not found"


async def test_engine_lifecycle():
    """Test engine initialization and cleanup."""
    try:
        engine = KokoroTTSEngine()
        assert not engine.initialized
        
        # Initialize
        await engine.initialize()
        assert engine.initialized
        
        # Cleanup
        await engine.cleanup()
        assert not engine.initialized
        
    except ImportError:
        # Kokoro not available, skip this test
        pass


async def test_quick_tts_function():
    """Test the quick TTS convenience function."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "quick_test.wav"
            
            result = await quick_tts(
                "Quick test of TTS function",
                str(output_path),
                voice='af_heart',
                speed=1.0
            )
            
            # If Kokoro is available, result should be the output path
            # If not available, result should be None
            if result is not None:
                assert Path(result).exists()
                
    except ImportError:
        # Kokoro not available, test should handle gracefully
        pass


def run_async_test(coro):
    """Helper to run async tests."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


# Run async tests
def test_engine_lifecycle_sync():
    """Sync wrapper for async engine lifecycle test."""
    run_async_test(test_engine_lifecycle())


def test_quick_tts_sync():
    """Sync wrapper for async quick TTS test."""
    run_async_test(test_quick_tts_function())


if __name__ == "__main__":
    print("Running TTS unit tests...")
    
    # Run all test functions
    test_functions = [
        test_tts_config_defaults,
        test_kokoro_engine_creation,
        test_voice_management,
        test_text_preparation,
        test_text_splitting,
        test_audio_info_error_handling,
        test_engine_lifecycle_sync,
        test_quick_tts_sync
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above.")
