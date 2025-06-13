"""
Unit tests for Kokoro TTS Engine.
Run with: python -m pytest src/tts/tests/test_kokoro_tts.py -v
"""

import asyncio
import os
import sys
import json
from pathlib import Path
import tempfile
from datetime import datetime

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
    
    # Verify no chunk exceeds max length (allowing some buffer for punctuation)
    for chunk in chunks:
        assert len(chunk) <= engine.config.max_length + 10  # Allow small buffer for punctuation


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


def load_llama_test_results():
    """Load the Llama test results with voice recommendations."""
    try:
        results_file = Path("src/summarizer/data/llama_test_results_20250613_122625.json")
        if not results_file.exists():
            print(f"❌ Test results file not found: {results_file}")
            return None
        
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('results', [])
    except Exception as e:
        print(f"❌ Failed to load test results: {e}")
        return None


async def test_voice_recommendations_with_real_summaries():
    """Test TTS generation using real summaries with recommended voices."""
    print("\n🎤 Testing Voice Recommendations with Real Summaries")
    print("=" * 60)
    
    # Load the test results
    results = load_llama_test_results()
    if not results:
        print("❌ No test results available")
        return
    
    # Filter successful results with summaries and voice recommendations
    valid_results = [
        r for r in results 
        if (r.get('status') == 'success' and 
            r.get('summary') and 
            r.get('voice_recommendation', {}).get('voice_id'))
    ]
    
    if not valid_results:
        print("❌ No valid results with voice recommendations found")
        return
    
    print(f"✅ Found {len(valid_results)} valid results with voice recommendations")
    
    # Create output directory
    output_dir = Path("src/tts/data/voice_recommendations_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with first 5 results to keep test time reasonable
    test_results = valid_results[:5]
    
    generated_files = []
    
    for i, result in enumerate(test_results, 1):
        title = result.get('title', 'Unknown Title')
        category = result.get('category', 'unknown')
        summary = result.get('summary', '')
        voice_rec = result.get('voice_recommendation', {})
        voice_id = voice_rec.get('voice_id', 'af_heart')
        voice_name = voice_rec.get('voice_name', 'Unknown')
        reasoning = voice_rec.get('reasoning', 'No reasoning provided')
        
        print(f"\n📝 Test {i}/{len(test_results)}: {title[:50]}...")
        print(f"   🏷️  Category: {category}")
        print(f"   🎤 Recommended Voice: {voice_name} ({voice_id})")
        print(f"   💭 Reasoning: {reasoning}")
        print(f"   📊 Summary Length: {len(summary)} characters")
        
        try:
            # Configure TTS with recommended voice
            config = TTSConfig(
                voice=voice_id,
                speed=1.1,  # Slightly faster for TikTok energy
                normalize_audio=True,
                add_silence_padding=True
            )
            
            tts_engine = KokoroTTSEngine(config)
            await tts_engine.initialize()
            
            # Create safe filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')[:40]
            
            output_file = output_dir / f"test_{i:02d}_{voice_id}_{safe_title}.wav"
            
            # Generate audio
            audio_file = await tts_engine.generate_audio(summary, str(output_file))
            
            if audio_file:
                # Get audio info
                info = tts_engine.get_audio_info(audio_file)
                duration = info.get('duration', 0)
                file_size_mb = info.get('file_size', 0) / (1024 * 1024)
                
                print(f"   ✅ Audio generated successfully!")
                print(f"   ⏱️  Duration: {duration:.1f} seconds")
                print(f"   💾 File size: {file_size_mb:.1f} MB")
                print(f"   📁 File: {Path(audio_file).name}")
                
                # Check TikTok suitability
                if 20 <= duration <= 180:  # 20s to 3 minutes is good for TikTok
                    print(f"   🎯 Perfect for TikTok!")
                elif duration < 20:
                    print(f"   ⚠️  Might be too short for TikTok")
                else:
                    print(f"   ⚠️  Might be too long for TikTok")
                
                generated_files.append({
                    'title': title,
                    'category': category,
                    'voice_id': voice_id,
                    'voice_name': voice_name,
                    'reasoning': reasoning,
                    'audio_file': audio_file,
                    'duration': duration,
                    'file_size_mb': file_size_mb,
                    'success': True
                })
                
            else:
                print(f"   ❌ Failed to generate audio")
                generated_files.append({
                    'title': title,
                    'category': category,
                    'voice_id': voice_id,
                    'voice_name': voice_name,
                    'success': False
                })
            
            await tts_engine.cleanup()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            generated_files.append({
                'title': title,
                'category': category,
                'voice_id': voice_id,
                'voice_name': voice_name,
                'success': False,
                'error': str(e)
            })
    
    # Save test results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"voice_recommendations_test_{timestamp}.json"
    
    test_summary = {
        'metadata': {
            'timestamp': timestamp,
            'total_tests': len(test_results),
            'successful_generations': len([f for f in generated_files if f.get('success')]),
            'source_file': 'llama_test_results_20250613_122625.json'
        },
        'results': generated_files
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Final summary
    successful = [f for f in generated_files if f.get('success')]
    total_duration = sum(f.get('duration', 0) for f in successful)
    total_size = sum(f.get('file_size_mb', 0) for f in successful)
    
    print(f"\n🎉 Voice Recommendation Test Complete!")
    print("=" * 50)
    print(f"📊 Results:")
    print(f"   • Tests run: {len(test_results)}")
    print(f"   • Successful: {len(successful)}")
    print(f"   • Total audio: {total_duration:.1f} seconds")
    print(f"   • Total size: {total_size:.1f} MB")
    print(f"   • Average duration: {total_duration/len(successful):.1f}s" if successful else "   • No successful generations")
    
    print(f"\n🎵 Generated Audio Files:")
    for result in successful:
        filename = Path(result['audio_file']).name
        print(f"   • {filename} ({result['duration']:.1f}s) - {result['voice_name']}")
    
    print(f"\n💾 Test results saved to: {results_file}")
    print(f"📁 Audio files saved to: {output_dir}")


async def test_voice_profile_showcase():
    """Create a showcase of different voice profiles with the same content."""
    print("\n🎭 Voice Profile Showcase")
    print("=" * 40)
    
    # Sample TikTok content for voice comparison
    test_content = """
    🚨 BREAKING: This is absolutely mind-blowing! 
    Scientists just made a discovery that's going to change everything we know about technology. 
    We're talking about a breakthrough that could revolutionize how we live, work, and communicate. 
    The implications are staggering - this isn't just another incremental update, it's a complete game-changer! 
    And the best part? This technology could be in your hands within the next two years!
    """
    
    # Test different voice profiles mentioned in the results
    voices_to_test = [
        ('af_bella', 'Bella - High Energy'),
        ('af_nicole', 'Nicole - Tech Focus'),
        ('af_heart', 'Heart - Warm & Emotional'),
        ('am_adam', 'Adam - Strong Male'),
        ('bm_george', 'George - British Male')
    ]
    
    output_dir = Path("src/tts/data/voice_showcase")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    showcase_results = []
    
    for voice_id, description in voices_to_test:
        print(f"\n🎤 Testing {description}")
        
        try:
            config = TTSConfig(voice=voice_id, speed=1.0)
            tts_engine = KokoroTTSEngine(config)
            await tts_engine.initialize()
            
            output_file = output_dir / f"showcase_{voice_id}.wav"
            audio_file = await tts_engine.generate_audio(test_content, str(output_file))
            
            if audio_file:
                info = tts_engine.get_audio_info(audio_file)
                duration = info.get('duration', 0)
                print(f"   ✅ Generated: {duration:.1f}s")
                
                showcase_results.append({
                    'voice_id': voice_id,
                    'description': description,
                    'audio_file': audio_file,
                    'duration': duration,
                    'success': True
                })
            else:
                print(f"   ❌ Failed")
                showcase_results.append({
                    'voice_id': voice_id,
                    'description': description,
                    'success': False
                })
            
            await tts_engine.cleanup()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            showcase_results.append({
                'voice_id': voice_id,
                'description': description,
                'success': False,
                'error': str(e)
            })
    
    print(f"\n✅ Voice showcase complete! Check {output_dir}")
    return showcase_results


def test_voice_recommendations_sync():
    """Sync wrapper for voice recommendations test."""
    run_async_test(test_voice_recommendations_with_real_summaries())


def test_voice_showcase_sync():
    """Sync wrapper for voice showcase test."""
    run_async_test(test_voice_profile_showcase())


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
        test_quick_tts_sync,
        test_voice_recommendations_sync,
        test_voice_showcase_sync
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
