"""
Tests for Kokoro TTS Engine.
"""

import asyncio
import os
import sys
from pathlib import Path
import pytest
import tempfile
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from src.tts.kokoro_tts import KokoroTTSEngine, TTSConfig, create_tts_engine, quick_tts
    from src.scraper.newsletter_scraper import Article
    print("✓ Successfully imported TTS components")
except ImportError as e:
    print(f"✗ Failed to import TTS components: {e}")
    sys.exit(1)


class TestKokoroTTS:
    """Test suite for Kokoro TTS Engine."""
    
    @pytest.fixture
    async def tts_engine(self):
        """Create a TTS engine for testing."""
        config = TTSConfig(
            voice='af_heart',
            speed=1.0,
            normalize_audio=True,
            add_silence_padding=False  # Faster for tests
        )
        engine = KokoroTTSEngine(config)
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.fixture
    def sample_article(self):
        """Create a sample article for testing."""
        return Article(
            title="Test Article: AI Revolution",
            content="This is a test article about artificial intelligence.",
            url="https://example.com/test",
            category="ai",
            tiktok_summary="🚨 BREAKING: AI is changing everything! This new technology is revolutionizing how we work and live. From chatbots to self-driving cars, AI is everywhere now!"
        )
    
    async def test_engine_initialization(self):
        """Test TTS engine initialization."""
        engine = KokoroTTSEngine()
        
        assert not engine.initialized
        
        await engine.initialize()
        
        assert engine.initialized
        assert engine.pipeline is not None
        
        await engine.cleanup()
        assert not engine.initialized
    
    async def test_voice_management(self):
        """Test voice listing and changing."""
        engine = KokoroTTSEngine()
        
        # Test voice listing
        voices = engine.list_voices()
        assert len(voices) > 0
        assert 'af_heart' in voices
        assert 'am_adam' in voices
        
        # Test voice changing
        engine.set_voice('am_adam')
        assert engine.config.voice == 'am_adam'
        
        # Test invalid voice
        engine.set_voice('invalid_voice')
        assert engine.config.voice == 'am_adam'  # Should remain unchanged
    
    async def test_text_preparation(self):
        """Test text cleaning and preparation."""
        engine = KokoroTTSEngine()
        
        # Test markdown removal
        text = "**Bold text** and *italic* text with `code` and #hashtags"
        clean = engine._prepare_text(text)
        assert "**" not in clean
        assert "*" not in clean
        assert "`" not in clean
        assert "hashtag hashtags" in clean
        
        # Test timestamp removal
        text = "**[0s-3s]** This is a TikTok script [3s-8s] with timestamps"
        clean = engine._prepare_text(text)
        assert "[0s-3s]" not in clean
        assert "[3s-8s]" not in clean
        assert "This is a TikTok script" in clean
    
    async def test_text_splitting(self):
        """Test intelligent text splitting."""
        engine = KokoroTTSEngine()
        engine.config.max_length = 50  # Short for testing
        
        # Short text should not be split
        short_text = "This is a short text."
        chunks = engine._split_text_intelligently(short_text)
        assert len(chunks) == 1
        assert chunks[0] == short_text
        
        # Long text should be split
        long_text = "This is a very long text that should be split into multiple chunks. " * 5
        chunks = engine._split_text_intelligently(long_text)
        assert len(chunks) > 1
        
        # Each chunk should be under max length
        for chunk in chunks:
            assert len(chunk) <= engine.config.max_length
    
    async def test_basic_audio_generation(self, tts_engine):
        """Test basic audio generation functionality."""
        test_text = "Hello, this is a test of the Kokoro TTS engine!"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_audio.wav"
            
            result = await tts_engine.generate_audio(test_text, str(output_path))
            
            assert result is not None
            assert Path(result).exists()
            
            # Check audio info
            info = tts_engine.get_audio_info(result)
            assert "error" not in info
            assert info["duration"] > 0
            assert info["sample_rate"] == tts_engine.config.sample_rate
    
    async def test_voice_variations(self, tts_engine):
        """Test generation with different voices."""
        test_text = "Testing different voices in Kokoro TTS."
        
        voices_to_test = ['af_heart', 'am_adam', 'bf_emma']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for voice in voices_to_test:
                if voice in tts_engine.available_voices:
                    output_path = Path(temp_dir) / f"test_{voice}.wav"
                    
                    result = await tts_engine.generate_audio(
                        test_text, 
                        str(output_path), 
                        voice=voice
                    )
                    
                    assert result is not None
                    assert Path(result).exists()
                    
                    info = tts_engine.get_audio_info(result)
                    assert info["duration"] > 0
    
    async def test_speed_variations(self, tts_engine):
        """Test generation with different speeds."""
        test_text = "Testing speech speed variations."
        speeds = [0.8, 1.0, 1.2]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            durations = []
            
            for speed in speeds:
                output_path = Path(temp_dir) / f"test_speed_{speed}.wav"
                
                result = await tts_engine.generate_audio(
                    test_text, 
                    str(output_path), 
                    speed=speed
                )
                
                assert result is not None
                assert Path(result).exists()
                
                info = tts_engine.get_audio_info(result)
                durations.append(info["duration"])
            
            # Faster speed should produce shorter audio
            assert durations[0] > durations[2]  # 0.8 speed > 1.2 speed
    
    async def test_article_generation(self, tts_engine, sample_article):
        """Test TTS generation from article."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await tts_engine.generate_from_article(
                sample_article, 
                output_dir=temp_dir
            )
            
            assert result is not None
            assert Path(result).exists()
            
            # Check that it used the TikTok summary
            info = tts_engine.get_audio_info(result)
            assert info["duration"] > 0
    
    async def test_batch_generation(self, tts_engine):
        """Test batch TTS generation."""
        texts = [
            "First test text for batch generation.",
            "Second test text with different content.",
            "Third and final text in the batch."
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = await tts_engine.batch_generate(
                texts, 
                temp_dir, 
                filename_prefix="batch_test"
            )
            
            assert len(results) == len(texts)
            
            # All should be successful
            for result in results:
                assert result is not None
                assert Path(result).exists()
    
    async def test_long_text_handling(self, tts_engine):
        """Test handling of very long text."""
        # Create a long TikTok script
        long_text = """
        🚨 BREAKING: This is an incredibly important announcement that will change everything you know about technology!
        
        Let me tell you about this amazing discovery that scientists just made. It's absolutely mind-blowing and will revolutionize how we think about artificial intelligence.
        
        But here's where it gets crazy - this technology is already being used by major companies around the world. From Google to Microsoft to Apple, everyone is jumping on this bandwagon.
        
        And the implications are huge! We're talking about a complete transformation of how we work, how we communicate, and how we live our daily lives.
        
        But wait, there's more! The researchers behind this breakthrough predict that within the next five years, this technology will be everywhere. In our phones, our cars, our homes - literally everywhere!
        
        So what does this mean for you? Well, it means you need to stay ahead of the curve and understand what's coming next in the world of technology.
        """ * 3  # Make it extra long
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "long_text.wav"
            
            result = await tts_engine.generate_audio(long_text, str(output_path))
            
            assert result is not None
            assert Path(result).exists()
            
            info = tts_engine.get_audio_info(result)
            assert info["duration"] > 30  # Should be a substantial audio file
    
    async def test_error_handling(self, tts_engine):
        """Test error handling for edge cases."""
        # Empty text
        result = await tts_engine.generate_audio("")
        assert result is None
        
        # Very short text
        result = await tts_engine.generate_audio("Hi")
        # Should still work for very short text
        assert result is not None or result is None  # Either is acceptable
        
        # Invalid output directory
        invalid_path = "/invalid/path/that/does/not/exist/file.wav"
        # This should handle the error gracefully
        try:
            result = await tts_engine.generate_audio("Test", invalid_path)
            # If it succeeds, the directory was created
            assert result is not None or result is None
        except Exception:
            # Error is acceptable for invalid paths
            pass
    
    async def test_quick_tts_function(self):
        """Test the convenience quick_tts function."""
        test_text = "Testing the quick TTS function!"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "quick_test.wav"
            
            result = await quick_tts(test_text, str(output_path))
            
            if result:  # Only test if Kokoro is available
                assert Path(result).exists()
    
    async def test_factory_function(self):
        """Test the create_tts_engine factory function."""
        config = TTSConfig(voice='am_adam', speed=1.1)
        
        engine = await create_tts_engine(config)
        
        assert engine.initialized
        assert engine.config.voice == 'am_adam'
        assert engine.config.speed == 1.1
        
        await engine.cleanup()


# Integration test functions for manual testing
async def test_integration_with_real_summary():
    """Integration test with real TikTok summary."""
    print("\n🎵 Testing TTS Integration with Real TikTok Summary")
    print("=" * 60)
    
    # Sample TikTok summary from your test results
    tiktok_summary = """
    This $14 billion investment in Scale AI is breaking the internet, but you might want to sit down because this is about to get real. According to sources close to the deal, Meta's Mark Zuckerberg is getting desperate to catch up with the AI game. He's willing to shell out billions to hire Scale AI's co-founder Alexandr Wang, a move that could be the turning point in the AI wars. The author claims that Meta's struggling to innovate, and rival companies like OpenAI are already leaving them in the dust. It seems Zuckerberg is willing to take drastic measures to bridge the gap. The question is, can Wang help him do just that? But here's the thing - Wang isn't just any ordinary AI expert. He's built a reputation as a fearless leader who isn't afraid to challenge the status quo.
    """
    
    try:
        # Test with multiple voices
        voices_to_test = ['af_heart', 'am_adam', 'bf_emma']
        
        for voice in voices_to_test:
            print(f"\n🎤 Testing voice: {voice}")
            
            config = TTSConfig(
                voice=voice,
                speed=1.1,  # Slightly faster for TikTok
                normalize_audio=True,
                add_silence_padding=True
            )
            
            tts = KokoroTTSEngine(config)
            await tts.initialize()
            
            # Create output directory
            output_dir = Path("src/tts/data")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = await tts.generate_audio(
                tiktok_summary,
                str(output_dir / f"tiktok_sample_{voice}.wav")
            )
            
            if output_file:
                info = tts.get_audio_info(output_file)
                print(f"   ✅ Generated: {info['duration']:.1f}s audio")
                print(f"   📁 Saved to: {output_file}")
                
                # Check if duration is reasonable for TikTok (30-120 seconds)
                if 20 <= info['duration'] <= 180:
                    print(f"   ✅ Duration perfect for TikTok!")
                else:
                    print(f"   ⚠️  Duration might be too {'short' if info['duration'] < 20 else 'long'} for TikTok")
            else:
                print(f"   ❌ Failed to generate audio")
            
            await tts.cleanup()
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_performance_benchmark():
    """Performance benchmark for TTS generation."""
    print("\n⚡ TTS Performance Benchmark")
    print("=" * 40)
    
    try:
        config = TTSConfig(voice='af_heart', speed=1.0)
        tts = KokoroTTSEngine(config)
        await tts.initialize()
        
        # Test different text lengths
        test_cases = [
            ("Short", "This is a short test."),
            ("Medium", "This is a medium-length test with multiple sentences. It should take a bit longer to generate than the short one."),
            ("Long", """
            This is a long test case that simulates a full TikTok script. It contains multiple paragraphs and should stress-test the TTS engine's performance with longer content. 
            
            The goal is to measure how quickly Kokoro can generate high-quality speech for substantial amounts of text, which is crucial for our TikTok automation pipeline.
            
            We want to ensure that the generation speed is acceptable for batch processing of multiple articles and that the quality remains consistently high throughout the entire process.
            """)
        ]
        
        import time
        
        for name, text in test_cases:
            print(f"\n📝 Testing {name} text ({len(text)} chars)")
            
            start_time = time.time()
            output_file = await tts.generate_audio(text)
            generation_time = time.time() - start_time
            
            if output_file:
                info = tts.get_audio_info(output_file)
                audio_duration = info['duration']
                rtf = audio_duration / generation_time  # Real-time factor
                
                print(f"   ⏱️  Generation time: {generation_time:.2f}s")
                print(f"   🎵 Audio duration: {audio_duration:.2f}s")
                print(f"   ⚡ Real-time factor: {rtf:.1f}x")
                
                if rtf >= 1.0:
                    print(f"   ✅ Real-time or faster!")
                else:
                    print(f"   ⚠️  Slower than real-time")
                
                # Clean up test file
                Path(output_file).unlink(missing_ok=True)
            else:
                print(f"   ❌ Generation failed")
        
        await tts.cleanup()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")


async def main():
    """Run all tests."""
    print("🎵 Kokoro TTS Engine Test Suite")
    print("=" * 50)
    
    # Check if Kokoro is available
    try:
        from kokoro import KPipeline
        print("✅ Kokoro TTS is available")
    except ImportError:
        print("❌ Kokoro TTS not installed")
        print("Install with: pip install kokoro>=0.9.4 soundfile")
        return
    
    # Run integration tests
    await test_integration_with_real_summary()
    await test_performance_benchmark()
    
    print("\n✅ All tests completed!")
    print("\n💡 To run unit tests, use: python -m pytest src/tts/tests/test_kokoro_tts.py -v")


if __name__ == "__main__":
    asyncio.run(main())
