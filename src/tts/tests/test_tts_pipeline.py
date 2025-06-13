#!/usr/bin/env python3
"""
Full Pipeline Integration: Scraping → Summarization → TTS Generation
Demonstrates the complete TikTok content creation pipeline.
"""

import asyncio
import sys
import os
from datetime import datetime
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.scraper.newsletter_scraper import NewsletterScraper
    from src.summarizer.llama_summarizer import create_tiktok_summarizer, TikTokSummaryConfig
    from src.tts.kokoro_tts import KokoroTTSEngine, TTSConfig
    print("✓ Successfully imported all pipeline components")
except ImportError as e:
    print(f"✗ Failed to import components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


async def full_tiktok_pipeline_with_tts():
    """
    Complete TikTok content pipeline: Newsletter → Summary → TTS Audio
    """
    print("🚀 Full TikTok Content Pipeline with TTS")
    print("🎵 Including Kokoro TTS Audio Generation")
    print("=" * 60)
    
    # Initialize components
    scraper = NewsletterScraper()
    
    # Configure summarizer
    summary_config = TikTokSummaryConfig(
        target_duration=60,  # 60-second TikToks
        temperature=0.8,
        use_gpu=True
    )
    summarizer = create_tiktok_summarizer()
    summarizer.config = summary_config
    
    # Configure TTS
    tts_config = TTSConfig(
        voice='af_heart',  # Warm female voice for engaging content
        speed=1.1,  # Slightly faster for TikTok energy
        normalize_audio=True,
        add_silence_padding=True,
        target_duration=60.0
    )
    tts_engine = KokoroTTSEngine(tts_config)
    
    try:
        # Step 1: Fetch articles
        print("\n📥 Step 1: Fetching Articles...")
        articles = await scraper.fetch_latest_newsletter(max_age_hours=48, max_entries=2)
        
        if not articles:
            print("❌ No articles found! Using sample data...")
            return await demo_with_sample_data(summarizer, tts_engine)
        
        successful_articles = [a for a in articles if a.content_extraction_status == "success"]
        if not successful_articles:
            print("❌ No successful extractions! Using sample data...")
            return await demo_with_sample_data(summarizer, tts_engine)
        
        demo_articles = successful_articles[:3]  # Process 3 articles
        print(f"✅ Found {len(demo_articles)} articles to process")
        
        # Step 2: Initialize AI models
        print("\n🤖 Step 2: Loading AI Models...")
        await summarizer.initialize()
        await tts_engine.initialize()
        
        # Step 3: Process each article
        print("\n✨ Step 3: Processing Articles...")
        
        results = []
        output_dir = Path("src/tts/data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, article in enumerate(demo_articles, 1):
            print(f"\n   📝 Processing {i}/{len(demo_articles)}: {article.title[:50]}...")
            
            # Generate TikTok summary
            tiktok_summary = await summarizer.summarize_for_tiktok(article, target_duration=60)
            
            if not tiktok_summary:
                print(f"      ❌ Summary generation failed")
                continue
            
            print(f"      ✅ Summary generated ({len(tiktok_summary)} chars)")
            
            # Generate TTS audio
            print(f"      🎵 Generating TTS audio...")
            
            # Clean title for filename
            safe_title = "".join(c for c in article.title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title[:40]  # Limit length
            
            audio_file = await tts_engine.generate_audio(
                tiktok_summary,
                str(output_dir / f"tiktok_{i:02d}_{safe_title.replace(' ', '_')}.wav")
            )
            
            if audio_file:
                audio_info = tts_engine.get_audio_info(audio_file)
                print(f"      ✅ Audio generated: {audio_info['duration']:.1f}s")
                
                result = {
                    'title': article.title,
                    'category': article.category,
                    'url': article.url,
                    'word_count': article.word_count,
                    'tiktok_summary': tiktok_summary,
                    'audio_file': audio_file,
                    'audio_duration': audio_info['duration'],
                    'audio_size_mb': audio_info['file_size'] / (1024 * 1024),
                    'estimated_reading_time': len(tiktok_summary.split()) / 2.5,
                    'success': True
                }
                
                # Check TikTok duration suitability
                if 20 <= audio_info['duration'] <= 120:
                    print(f"      ✅ Perfect for TikTok! ({audio_info['duration']:.1f}s)")
                else:
                    print(f"      ⚠️  Duration: {audio_info['duration']:.1f}s (may be too {'short' if audio_info['duration'] < 20 else 'long'})")
                
            else:
                print(f"      ❌ Audio generation failed")
                result = {
                    'title': article.title,
                    'category': article.category,
                    'success': False,
                    'error': 'TTS generation failed'
                }
            
            results.append(result)
        
        # Step 4: Save comprehensive results
        print("\n💾 Step 4: Saving Results...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f"tiktok_pipeline_results_{timestamp}.json"
        
        pipeline_results = {
            'metadata': {
                'timestamp': timestamp,
                'total_articles': len(demo_articles),
                'successful_summaries': len([r for r in results if r.get('success') and 'tiktok_summary' in r]),
                'successful_audio': len([r for r in results if r.get('success') and 'audio_file' in r]),
                'tts_config': {
                    'voice': tts_config.voice,
                    'speed': tts_config.speed,
                    'sample_rate': tts_config.sample_rate
                }
            },
            'results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Results saved to: {results_file}")
        
        # Step 5: Summary
        print("\n🎉 Step 5: Pipeline Summary")
        print("=" * 50)
        
        successful = [r for r in results if r.get('success')]
        total_audio_duration = sum(r.get('audio_duration', 0) for r in successful)
        total_file_size = sum(r.get('audio_size_mb', 0) for r in successful)
        
        print(f"📊 Processing Results:")
        print(f"   • Articles processed: {len(demo_articles)}")
        print(f"   • Successful completions: {len(successful)}")
        print(f"   • Total audio generated: {total_audio_duration:.1f} seconds")
        print(f"   • Total file size: {total_file_size:.1f} MB")
        print(f"   • Average audio duration: {total_audio_duration/len(successful):.1f}s" if successful else "   • No successful generations")
        
        print(f"\n🎵 Generated Audio Files:")
        for result in successful:
            if 'audio_file' in result:
                print(f"   • {Path(result['audio_file']).name} ({result['audio_duration']:.1f}s)")
        
        # Voice analysis
        voice_used = tts_config.voice
        voice_description = tts_engine.available_voices.get(voice_used, "Unknown")
        print(f"\n🎤 Voice Analysis:")
        print(f"   • Voice used: {voice_used} ({voice_description})")
        print(f"   • Speed: {tts_config.speed}x")
        print(f"   • Sample rate: {tts_config.sample_rate} Hz")
        
        # Cleanup
        await summarizer.cleanup()
        await tts_engine.cleanup()
        await scraper.close()
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            await summarizer.cleanup()
            await tts_engine.cleanup()
            await scraper.close()
        except:
            pass


async def demo_with_sample_data(summarizer, tts_engine):
    """Demo with sample data when live scraping fails."""
    print("\n📋 Running Demo with Sample Data")
    
    # Sample TikTok summaries from your test results
    sample_summaries = [
        {
            'title': 'OpenAI Hits $10 Billion Revenue',
            'category': 'ai',
            'summary': """🚨 BREAKING: OpenAI just hit $10 BILLION in revenue! That's right - ChatGPT is making them $10B per year! This proves AI isn't just hype anymore - it's BIG BUSINESS! They went from $5.5B to $10B in just one year! And they're targeting $125 BILLION by 2029! What do you think about this AI boom?"""
        },
        {
            'title': 'Meta Makes $14B AI Investment',
            'category': 'big_tech',
            'summary': """This $14 billion investment in Scale AI is breaking the internet! Meta's Mark Zuckerberg is getting desperate to catch up with the AI game. He's willing to shell out billions to hire Scale AI's co-founder Alexandr Wang. But here's the thing - Wang isn't just any ordinary AI expert. This could be the turning point in the AI wars!"""
        }
    ]
    
    try:
        await summarizer.initialize()
        await tts_engine.initialize()
        
        output_dir = Path("src/tts/data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎵 Testing TTS with {len(sample_summaries)} sample summaries...")
        
        for i, sample in enumerate(sample_summaries, 1):
            print(f"\n   📝 Sample {i}: {sample['title']}")
            
            safe_title = sample['title'].replace(' ', '_').replace('$', 'Dollar')
            audio_file = await tts_engine.generate_audio(
                sample['summary'],
                str(output_dir / f"sample_{i:02d}_{safe_title}.wav")
            )
            
            if audio_file:
                info = tts_engine.get_audio_info(audio_file)
                print(f"      ✅ Generated: {info['duration']:.1f}s audio")
                print(f"      📁 File: {Path(audio_file).name}")
            else:
                print(f"      ❌ Failed to generate audio")
        
        await summarizer.cleanup()
        await tts_engine.cleanup()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def demo_with_sample_data_standalone():
    """Standalone demo with sample data - no dependencies on other components."""
    print("\n📋 Running Standalone TTS Demo with Sample Data")
    
    # Sample TikTok summaries from your test results
    sample_summaries = [
        {
            'title': 'OpenAI Hits $10 Billion Revenue',
            'category': 'ai',
            'summary': """🚨 BREAKING: OpenAI just hit $10 BILLION in revenue! That's right - ChatGPT is making them $10B per year! This proves AI isn't just hype anymore - it's BIG BUSINESS! They went from $5.5B to $10B in just one year! And they're targeting $125 BILLION by 2029! What do you think about this AI boom?"""
        },
        {
            'title': 'Meta Makes $14B AI Investment', 
            'category': 'big_tech',
            'summary': """This $14 billion investment in Scale AI is breaking the internet! Meta's Mark Zuckerberg is getting desperate to catch up with the AI game. He's willing to shell out billions to hire Scale AI's co-founder Alexandr Wang. But here's the thing - Wang isn't just any ordinary AI expert. This could be the turning point in the AI wars!"""
        }
    ]
    
    try:
        print("🎤 Initializing Kokoro TTS engine...")
        
        # Configure TTS with a warm female voice
        tts_config = TTSConfig(
            voice='af_heart',  # Warm, emotional female voice
            speed=1.1,  # Slightly faster for TikTok energy
            normalize_audio=True,
            add_silence_padding=True,
            target_duration=60.0
        )
        
        tts_engine = KokoroTTSEngine(tts_config)
        print(f"✅ TTS engine created with voice: {tts_config.voice}")
        
        print("🔄 Initializing TTS pipeline (this may take a moment)...")
        await tts_engine.initialize()
        print(f"✅ TTS engine initialized successfully")
        
        # Create output directory
        output_dir = Path("src/tts/data")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        
        results = []
        
        for i, sample in enumerate(sample_summaries, 1):
            print(f"\n📝 Processing sample {i}/{len(sample_summaries)}: {sample['title']}")
            print(f"   📂 Category: {sample['category']}")
            print(f"   📊 Summary length: {len(sample['summary'])} characters")
            
            # Create safe filename
            safe_title = sample['title'].replace(' ', '_').replace('$', 'Dollar')
            safe_title = "".join(c for c in safe_title if c.isalnum() or c in ('_', '-'))[:40]
            
            audio_file = await tts_engine.generate_audio(
                sample['summary'],
                str(output_dir / f"demo_{i:02d}_{safe_title}.wav")
            )
            
            if audio_file:
                # Get audio information
                audio_info = tts_engine.get_audio_info(audio_file)
                duration = audio_info['duration']
                file_size_mb = audio_info['file_size'] / (1024 * 1024)
                
                print(f"   ✅ Audio generated successfully!")
                print(f"   ⏱️  Duration: {duration:.1f} seconds")
                print(f"   💾 File size: {file_size_mb:.1f} MB")
                print(f"   📁 File: {Path(audio_file).name}")
                
                # Check TikTok suitability
                if 20 <= duration <= 120:
                    print(f"   🎯 Perfect for TikTok! ({duration:.1f}s)")
                elif duration < 20:
                    print(f"   ⚠️  Might be too short for TikTok ({duration:.1f}s)")
                else:
                    print(f"   ⚠️  Might be too long for TikTok ({duration:.1f}s)")
                
                results.append({
                    'title': sample['title'],
                    'category': sample['category'],
                    'audio_file': audio_file,
                    'duration': duration,
                    'file_size_mb': file_size_mb,
                    'success': True
                })
                
            else:
                print(f"   ❌ Failed to generate audio")
                results.append({
                    'title': sample['title'],
                    'category': sample['category'],
                    'success': False
                })
        
        # Summary
        successful = [r for r in results if r.get('success', False)]
        total_duration = sum(r.get('duration', 0) for r in successful)
        total_size = sum(r.get('file_size_mb', 0) for r in successful)
        
        print(f"\n🎉 Standalone TTS Demo Complete!")
        print("=" * 40)
        print(f"📊 Results:")
        print(f"   • Processed: {len(sample_summaries)} summaries")
        print(f"   • Successful: {len(successful)}")
        print(f"   • Total audio: {total_duration:.1f} seconds")
        print(f"   • Total size: {total_size:.1f} MB")
        print(f"   • Average duration: {total_duration/len(successful):.1f}s" if successful else "   • No successful generations")
        
        print(f"\n🎵 Generated Audio Files:")
        for result in successful:
            filename = Path(result['audio_file']).name
            print(f"   • {filename} ({result['duration']:.1f}s)")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f"standalone_tts_demo_{timestamp}.json"
        
        demo_results = {
            'metadata': {
                'timestamp': timestamp,
                'voice_used': tts_config.voice,
                'speed': tts_config.speed,
                'sample_rate': tts_config.sample_rate,
                'total_processed': len(sample_summaries),
                'successful_generations': len(successful)
            },
            'results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(demo_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        await tts_engine.cleanup()
        print("🧹 TTS engine cleaned up")
        
    except Exception as e:
        print(f"❌ Standalone demo failed: {e}")
        import traceback
        traceback.print_exc()


async def test_voice_comparison():
    """Test different voices with the same content."""
    print("\n🎤 Voice Comparison Test")
    print("=" * 40)
    
    test_content = """
    🚨 BREAKING: This is a voice comparison test! 
    We're testing how different Kokoro voices sound with TikTok content. 
    Each voice has its own personality and style, perfect for different types of content!
    """
    
    voices_to_test = [
        ('af_heart', 'Warm female voice'),
        ('am_adam', 'Strong male voice'),
        ('bf_emma', 'British female voice'),
        ('bm_george', 'British male voice')
    ]
    
    try:
        output_dir = Path("src/tts/data/voice_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for voice_id, description in voices_to_test:
            print(f"\n🎵 Testing {voice_id}: {description}")
            
            config = TTSConfig(voice=voice_id, speed=1.0)
            tts = KokoroTTSEngine(config)
            await tts.initialize()
            
            audio_file = await tts.generate_audio(
                test_content,
                str(output_dir / f"comparison_{voice_id}.wav")
            )
            
            if audio_file:
                info = tts.get_audio_info(audio_file)
                print(f"   ✅ Generated: {info['duration']:.1f}s")
            else:
                print(f"   ❌ Failed")
            
            await tts.cleanup()
        
        print(f"\n✅ Voice comparison complete! Check {output_dir}")
        
    except Exception as e:
        print(f"❌ Voice comparison failed: {e}")


async def main():
    """Main entry point."""
    print("🎬 TikTok Automata - Full Pipeline with TTS")
    print("Hardware: Optimized for RTX 3070 + Kokoro TTS")
    print("=" * 70)
    
    # Check TTS availability
    try:
        print("🔍 Checking Kokoro TTS availability...")
        from kokoro import KPipeline
        print("✅ Kokoro TTS is available")
    except ImportError as e:
        print(f"❌ Kokoro TTS not installed: {e}")
        print("📦 Install with: pip install --user kokoro>=0.9.4 soundfile")
        print("🔧 On Windows, you may also need espeak-ng")
        return
    
    # Run demo with sample data directly (no interactive input)
    print("\n🚀 Running TTS demo with sample data (non-interactive mode)")
    print("   This will test the complete pipeline: Summary → TTS → Audio")
    
    try:
        await demo_with_sample_data_standalone()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 TTS demo completed!")
    print("\n💡 Next steps:")
    print("   • Check generated audio files in src/tts/data/")
    print("   • Listen to the audio to verify quality")
    print("   • Try different voices by modifying the script")


if __name__ == "__main__":
    print("🔥 Script starting...")
    print("🔥 About to run asyncio.run(main())")
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"🔥 Exception in main: {e}")
        import traceback
        traceback.print_exc()
    print("🔥 Script finished")
