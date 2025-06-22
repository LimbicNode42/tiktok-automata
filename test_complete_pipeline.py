#!/usr/bin/env python3
"""
Complete TikTok Video Pipeline Test with Audio Integration

This test validates the entire pipeline from newsletter content to final video:
1. Newsletter scraping/loading 
2. TikTok summarization
3. TTS audio generation
4. Gaming video processing  
5. Audio integration (TTS + gaming audio)
6. Final video assembly and output

Follows coding guidelines: No interactive terminal commands, proper test scripts.
"""

import asyncio
import sys
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test if all modules can be imported
try:
    from src.scraper.newsletter_scraper import NewsletterScraper, Article
    from src.summarizer.llama_summarizer import create_tiktok_summarizer, TikTokSummaryConfig
    from src.tts.kokoro_tts import KokoroTTSEngine, TTSConfig
    from src.video.processors.video_processor import VideoProcessor, VideoConfig
    from src.video.audio.audio_mixer import AudioMixer, AudioConfig
    from src.video.audio.audio_synchronizer import AudioSynchronizer
    from src.utils.config import config
    print("✅ All pipeline components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import pipeline components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class CompletePipelineTest:
    """Complete pipeline test orchestrator."""
    
    def __init__(self):
        self.test_dir = Path("pipeline_test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.scraper = None
        self.summarizer = None
        self.tts = None
        self.video_processor = None
        self.audio_mixer = None
        
        logger.info("Pipeline test initialized")
    
    async def setup_components(self):
        """Initialize all pipeline components."""
        logger.info("🔧 Setting up pipeline components...")
        
        # Newsletter scraper
        self.scraper = NewsletterScraper()
        
        # TikTok summarizer (optimized for GTX 1060)
        self.summarizer = create_tiktok_summarizer()
        self.summarizer.config = TikTokSummaryConfig(
            target_duration=60,
            temperature=0.8,
            use_gpu=True
        )
          # TTS system
        self.tts = KokoroTTSEngine()
        
        # Video processor
        self.video_processor = VideoProcessor()
          # Audio mixer (new integration module)
        audio_config = AudioConfig(
            tts_volume=0.85,
            background_volume=0.25,
            master_volume=0.90,
            fade_duration=1.0,
            enable_dynamic_range=True,
            apply_eq=True
        )
        self.audio_mixer = AudioMixer(audio_config)
        
        logger.info("✅ All components initialized")
    
    async def load_sample_content(self) -> Optional[Article]:
        """Load sample newsletter content for testing."""
        logger.info("📰 Loading sample newsletter content...")
          # Try to load from existing scraped data
        scraper_data_dir = Path("src/scraper/data")
        data_files = list(scraper_data_dir.glob("tldr_articles_*.json"))
        
        if data_files:
            latest_file = max(data_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"📁 Loading from: {latest_file}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract articles from the data structure
            articles_data = data.get('articles', [])
            
            # Find a good article for testing
            for item in articles_data:
                if (item.get('content_extraction_status') == 'success' and 
                    len(item.get('content', '')) > 500 and
                    len(item.get('content', '')) < 2000):  # Good size for testing
                    
                    article = Article(
                        title=item['title'],
                        content=item['content'],
                        summary=item.get('summary', ''),
                        url=item['url'],
                        published_date=datetime.fromisoformat(item['published_date'].replace('Z', '+00:00')),
                        category=item.get('category', 'tech'),
                        word_count=item.get('word_count', 0),
                        content_extraction_status=item['content_extraction_status']
                    )
                    
                    logger.info(f"✅ Selected article: {article.title[:50]}...")
                    logger.info(f"   Content length: {len(article.content)} chars")
                    logger.info(f"   Category: {article.category}")
                    return article
        
        # Fallback: Create a sample article if no data available
        logger.info("📝 Creating sample article for testing...")
        return Article(
            title="AI Breakthrough: New Language Model Revolutionizes Tech",
            content="""
            In a groundbreaking development, researchers have unveiled a new AI language model that promises to transform how we interact with technology. This innovative system demonstrates unprecedented capabilities in understanding context, generating creative content, and solving complex problems.

            The model, trained on diverse datasets, shows remarkable improvements in accuracy and efficiency compared to previous generations. Early tests indicate it can handle multiple languages seamlessly while maintaining coherent reasoning across different domains.

            Industry experts are calling this a potential game-changer for applications ranging from customer service to scientific research. The technology could revolutionize everything from automated coding to creative writing assistance.

            Key features include enhanced safety measures, reduced computational requirements, and improved alignment with human values. This represents a significant step forward in making AI more accessible and beneficial for everyone.
            """,
            summary="New AI language model shows unprecedented capabilities and efficiency improvements.",
            url="https://example.com/ai-breakthrough",
            published_date=datetime.now(),
            category="ai",
            word_count=150,
            content_extraction_status="success"
        )
    
    async def generate_tiktok_script(self, article: Article) -> Optional[str]:
        """Generate TikTok script from article."""
        logger.info("✨ Generating TikTok script...")
          try:
            await self.summarizer.initialize()
            script = await self.summarizer.summarize_for_tiktok(article, target_duration=60)
            
            if script:
                logger.info(f"✅ Generated script ({len(script)} chars)")
                logger.info(f"   Preview: {script[:100]}...")
                return script
            else:
                logger.error("❌ Failed to generate TikTok script")
                return None
                
        except Exception as e:
            logger.error(f"❌ Script generation failed: {e}")
            return None
    
    async def generate_tts_audio(self, script: str) -> Optional[Path]:
        """Generate TTS audio from script."""
        logger.info("🎤 Generating TTS audio...")
        
        try:
            # Generate audio using individual parameters
            audio_path = await self.tts.generate_audio(
                text=script,
                output_path=str(self.test_dir / "tts_audio.wav"),
                voice='af_heart',  # Clear female voice
                speed=1.1         # Slightly faster for TikTok
            )
            
            if audio_path and audio_path.exists():
                logger.info(f"✅ TTS audio generated: {audio_path}")
                logger.info(f"   File size: {audio_path.stat().st_size / 1024:.1f} KB")
                return audio_path
            else:
                logger.error("❌ TTS audio generation failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ TTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def process_gaming_video(self) -> Optional[Path]:
        """Process gaming video footage."""
        logger.info("🎮 Processing gaming video...")
        
        try:
            # Check for gaming footage
            footage_dir = Path("src/video/data/footage/raw")
            if not footage_dir.exists():
                logger.warning("⚠️ No gaming footage directory found, creating sample...")
                footage_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a simple test video using moviepy
                from moviepy import ColorClip
                test_clip = ColorClip(
                    size=(1080, 1920), 
                    color=(50, 100, 150), 
                    duration=60
                ).with_fps(30)
                
                test_video_path = footage_dir / "test_gaming_footage.mp4"
                test_clip.write_videofile(
                    str(test_video_path),
                    codec='libx264',
                    audio=False,
                    verbose=False,
                    logger=None
                )
                test_clip.close()
                
                logger.info(f"✅ Created test gaming footage: {test_video_path}")
                return test_video_path
            
            # Use existing footage
            video_files = list(footage_dir.glob("*.mp4"))
            if video_files:
                selected_video = video_files[0]
                logger.info(f"✅ Using gaming footage: {selected_video}")
                return selected_video
            else:
                logger.error("❌ No gaming footage found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Gaming video processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def integrate_audio(self, tts_path: Path, video_path: Path) -> Optional[Path]:
        """Integrate TTS audio with gaming video audio using new audio module."""
        logger.info("🎵 Integrating audio streams...")
        
        try:
            # Output path for mixed audio
            mixed_audio_path = self.test_dir / "mixed_audio.wav"
            
            # Use the new audio integration module
            result = await self.audio_mixer.mix_audio_async(
                tts_audio_path=tts_path,
                background_video_path=video_path,
                output_path=mixed_audio_path
            )
            
            if result and mixed_audio_path.exists():
                logger.info(f"✅ Audio integration complete: {mixed_audio_path}")
                logger.info(f"   File size: {mixed_audio_path.stat().st_size / 1024:.1f} KB")
                return mixed_audio_path
            else:
                logger.error("❌ Audio integration failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Audio integration failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def create_final_video(self, video_path: Path, audio_path: Path, script: str) -> Optional[Path]:
        """Create final TikTok video with integrated audio."""
        logger.info("🎬 Creating final TikTok video...")
        
        try:
            from moviepy import VideoFileClip, TextClip, CompositeVideoClip
            
            # Load video and audio
            video_clip = VideoFileClip(str(video_path))
            
            # Set audio duration to match TTS audio duration  
            from moviepy import AudioFileClip
            audio_clip = AudioFileClip(str(audio_path))
            target_duration = audio_clip.duration
            audio_clip.close()
            
            # Trim/loop video to match audio duration
            if video_clip.duration < target_duration:
                # Loop video if too short
                loops_needed = int(target_duration / video_clip.duration) + 1
                video_clip = video_clip.with_repeat(loops_needed)
            
            video_clip = video_clip.with_subclipped(0, target_duration)
            
            # Replace audio with our mixed audio
            final_clip = video_clip.with_audio(str(audio_path))
            
            # Add text overlay (simple version)
            text_lines = script.split('.')[0][:50] + "..."  # First sentence
            text_clip = TextClip(
                text_lines,
                font_size=48,
                color='white',
                stroke_color='black',
                stroke_width=2,
                duration=target_duration,
                size=(1000, None)
            ).with_position(('center', 'center')).with_start(0)
            
            # Composite final video
            final_video = CompositeVideoClip([final_clip, text_clip])
            
            # Output path
            output_path = self.test_dir / f"tiktok_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            # Render final video
            logger.info("🔄 Rendering final video...")
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                fps=30,
                preset='medium',
                verbose=False,
                logger=None
            )
            
            # Cleanup
            video_clip.close()
            final_clip.close()
            text_clip.close()
            final_video.close()
            
            if output_path.exists():
                logger.info(f"✅ Final video created: {output_path}")
                logger.info(f"   File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
                logger.info(f"   Duration: {target_duration:.1f} seconds")
                return output_path
            else:
                logger.error("❌ Final video creation failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Final video creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run_complete_test(self):
        """Run the complete pipeline test."""
        logger.info("🚀 Starting Complete TikTok Pipeline Test")
        logger.info("=" * 60)
        
        try:
            # Setup
            await self.setup_components()
            
            # Step 1: Load content
            logger.info("\n📰 Step 1: Loading newsletter content...")
            article = await self.load_sample_content()
            if not article:
                logger.error("❌ Failed to load content")
                return False
            
            # Step 2: Generate script
            logger.info("\n✨ Step 2: Generating TikTok script...")
            script = await self.generate_tiktok_script(article)
            if not script:
                logger.error("❌ Failed to generate script")
                return False
            
            # Step 3: Generate TTS audio
            logger.info("\n🎤 Step 3: Generating TTS audio...")
            tts_audio_path = await self.generate_tts_audio(script)
            if not tts_audio_path:
                logger.error("❌ Failed to generate TTS audio")
                return False
            
            # Step 4: Process gaming video
            logger.info("\n🎮 Step 4: Processing gaming video...")
            video_path = await self.process_gaming_video()
            if not video_path:
                logger.error("❌ Failed to process gaming video")
                return False
            
            # Step 5: Integrate audio (NEW MODULE TEST)
            logger.info("\n🎵 Step 5: Integrating audio streams...")
            mixed_audio_path = await self.integrate_audio(tts_audio_path, video_path)
            if not mixed_audio_path:
                logger.error("❌ Failed to integrate audio")
                return False
            
            # Step 6: Create final video
            logger.info("\n🎬 Step 6: Creating final TikTok video...")
            final_video_path = await self.create_final_video(video_path, mixed_audio_path, script)
            if not final_video_path:
                logger.error("❌ Failed to create final video")
                return False
            
            # Success!
            logger.info("\n🎉 PIPELINE TEST COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"✅ Article: {article.title}")
            logger.info(f"✅ Script: {len(script)} characters")
            logger.info(f"✅ TTS Audio: {tts_audio_path}")
            logger.info(f"✅ Mixed Audio: {mixed_audio_path}")
            logger.info(f"✅ Final Video: {final_video_path}")
            logger.info(f"\n🎯 Test video ready for audio validation!")
            logger.info(f"   Play the video to hear the integrated audio:")
            logger.info(f"   {final_video_path.absolute()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup
            try:
                if self.summarizer:
                    await self.summarizer.cleanup()
                if self.scraper:
                    await self.scraper.close()
            except:
                pass
    
    async def cleanup(self):
        """Clean up test resources."""
        logger.info("🧹 Cleaning up test resources...")
        # Keep the output video for validation, but clean up intermediate files
        # You can manually delete pipeline_test_output/ when done testing


async def main():
    """Main test entry point."""
    print("🤖 TikTok Automata - Complete Pipeline Test")
    print("🎵 Testing new Audio Integration Module")
    print("=" * 60)
    
    test = CompletePipelineTest()
    
    try:
        success = await test.run_complete_test()
        
        if success:
            print("\n✅ PIPELINE TEST SUCCESSFUL!")
            print("🎧 The output video contains integrated audio:")
            print("   - TTS narration (foreground)")
            print("   - Gaming video audio (background)")
            print("   - Mobile-optimized mixing")
            print("\n🎯 Play the video to validate audio quality!")
        else:
            print("\n❌ PIPELINE TEST FAILED!")
            print("Check the logs above for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await test.cleanup()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # Run the test
    asyncio.run(main())
