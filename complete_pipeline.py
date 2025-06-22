#!/usr/bin/env python3
"""
Complete TikTok Automata Pipeline Integration

This script demonstrates the full end-to-end workflow:
Newsletter → AI Summary → TTS → Video Generation → Final TikTok Video

Optimized for real production use with all components integrated.
"""

import asyncio
import sys
import os
from datetime import datetime
import json
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Import all pipeline components
    from src.scraper.newsletter_scraper import NewsletterScraper, Article
    from src.summarizer.llama_summarizer import create_tiktok_summarizer, TikTokSummaryConfig, LlamaSummarizer
    from src.tts.kokoro_tts import KokoroTTSEngine
    from src.video.processors.video_processor import VideoProcessor, VideoConfig
    from src.video import FootageManager
    from src.utils.config import config
    
    logger.info("✅ All pipeline components imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import components: {e}")
    sys.exit(1)


class CompleteTikTokPipeline:
    """Complete automated pipeline for TikTok video generation."""
    
    def __init__(self):
        """Initialize all pipeline components."""
        self.scraper = None
        self.summarizer = None
        self.tts = None
        self.video_processor = None
        self.footage_manager = None
        
        # Output directories
        self.output_dir = Path("output")
        self.audio_dir = self.output_dir / "audio"
        self.video_dir = self.output_dir / "videos"
        self.reports_dir = self.output_dir / "reports"
        
        # Create directories
        for dir_path in [self.output_dir, self.audio_dir, self.video_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize all pipeline components (except GPU models which load on-demand)."""
        logger.info("🚀 Initializing Complete TikTok Pipeline...")
        
        try:
            # 1. Newsletter scraper
            logger.info("📰 Initializing newsletter scraper...")
            self.scraper = NewsletterScraper()            # 2. AI summarizer (Llama 3.2-3B) - Create with config
            logger.info("🤖 Creating Llama 3.2-3B summarizer...")
            config = TikTokSummaryConfig(
                target_duration=60,
                temperature=0.8,
                use_gpu=True
            )
            logger.info(f"Config created: {config}")
            self.summarizer = LlamaSummarizer(config)
            logger.info(f"Summarizer created: {self.summarizer}")
            logger.info(f"Summarizer type: {type(self.summarizer)}")
            if self.summarizer is None:
                raise ValueError("Failed to create summarizer")
            # Don't initialize yet - will load on-demand
              # 3. TTS (Kokoro) - Create but don't load yet
            logger.info("🎤 Creating Kokoro TTS config...")
            self.tts = KokoroTTSEngine()
            # Don't initialize yet - will load on-demand
            
            # 4. Video processor
            logger.info("🎬 Initializing video processor...")
            video_config = VideoConfig(
                duration=60,
                output_quality="high",
                use_gaming_footage=True,
                footage_intensity="medium"
            )
            self.video_processor = VideoProcessor(video_config)
            
            # 5. Footage manager
            logger.info("🎮 Initializing footage manager...")
            self.footage_manager = FootageManager()
            
            logger.success("✅ Non-GPU components initialized successfully!")
            logger.info("🔥 GPU models will load on-demand to optimize memory usage")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            raise
    
    async def fetch_latest_articles(self, max_articles: int = 5) -> List[Article]:
        """Fetch latest articles from TLDR newsletters."""
        logger.info(f"📥 Fetching latest {max_articles} articles...")
        
        try:
            articles = await self.scraper.fetch_latest_newsletter(
                max_age_hours=72,  # Last 3 days
                max_entries=3      # Recent newsletters
            )
            
            if not articles:
                logger.warning("No articles found")
                return []
            
            # Filter successful extractions only
            successful_articles = [
                a for a in articles 
                if a.content_extraction_status == "success" and len(a.content) > 500
            ]
            
            # Take top articles
            selected_articles = successful_articles[:max_articles]
            
            logger.info(f"✅ Found {len(articles)} total, {len(successful_articles)} successful, selected {len(selected_articles)}")
            return selected_articles
            
        except Exception as e:
            logger.error(f"❌ Article fetching failed: {e}")
            return []
    
    async def generate_tiktok_summary(self, article: Article) -> Optional[Dict]:
        """Generate TikTok summary with voice recommendation."""
        logger.info(f"✨ Generating TikTok summary for: {article.title[:50]}...")
        
        try:
            # Load Llama model on-demand
            logger.info("🔥 Loading Llama model on GPU...")
            logger.info(f"Summarizer object: {self.summarizer}")
            logger.info(f"Summarizer type: {type(self.summarizer)}")
            logger.info(f"Has initialize method: {hasattr(self.summarizer, 'initialize')}")
            await self.summarizer.initialize()
            
            summary_result = await self.summarizer.summarize_for_tiktok(
                article, 
                target_duration=60
            )
            
            # Cleanup Llama model immediately to free GPU memory
            logger.info("🧹 Cleaning up Llama model to free GPU memory...")
            await self.summarizer.cleanup()
            
            if summary_result:
                logger.success(f"✅ Summary generated ({len(summary_result)} chars)")
                return {
                    'article': article,
                    'summary': summary_result,
                    'estimated_duration': len(summary_result.split()) / 2.5  # ~150 WPM
                }
            else:
                logger.warning("❌ Summary generation failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Summary generation error: {e}")
            # Ensure cleanup even on error
            try:
                await self.summarizer.cleanup()
            except:
                pass
            return None
    
    async def generate_audio(self, summary_data: Dict) -> Optional[str]:
        """Generate TTS audio from summary."""
        article = summary_data['article']
        summary_text = summary_data['summary']
        
        logger.info(f"🎤 Generating TTS audio for: {article.title[:50]}...")
        
        try:
            # Ensure Kokoro uses GPU if available
            logger.info("🔥 Loading Kokoro TTS on GPU...")
            
            # Check GPU status for Kokoro
            import torch
            if torch.cuda.is_available():
                logger.info(f"🔥 CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
                
                # Clear any existing GPU cache
                torch.cuda.empty_cache()
                logger.info("🧹 Cleared GPU cache for TTS")
            
            # Use a good voice for the content type
            voice_id = "af_sarah"  # Default voice, could be enhanced with voice recommendation
            
            # Generate filename
            safe_title = "".join(c for c in article.title[:30] if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title.replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"tts_{safe_title}_{timestamp}.wav"
            audio_path = self.audio_dir / audio_filename
            
            # Generate TTS
            result_path = await self.tts.generate_audio(
                text=summary_text,
                voice=voice_id,
                output_path=str(audio_path)
            )
            
            # Clear GPU cache after TTS generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 Cleared GPU cache after TTS")
            
            if result_path and audio_path.exists():
                logger.success(f"✅ TTS audio generated: {audio_path.name}")
                return str(audio_path)
            else:
                logger.error("❌ TTS generation failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ TTS generation error: {e}")
            # Clear GPU cache on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            return None
    
    async def generate_video(self, summary_data: Dict, audio_path: str) -> Optional[str]:
        """Generate final TikTok video."""
        article = summary_data['article']
        summary_text = summary_data['summary']
        
        logger.info(f"🎬 Generating video for: {article.title[:50]}...")
        
        try:
            # Content analysis for footage selection
            content_analysis = {
                'content_type': article.category or 'tech',
                'urgency_level': 'medium',
                'is_breakthrough': 'breakthrough' in article.title.lower() or 'breakthrough' in summary_text.lower(),
                'category': article.category or 'tech'
            }
            
            # Generate filename
            safe_title = "".join(c for c in article.title[:30] if c.isalnum() or c in (' ', '-', '_'))
            safe_title = safe_title.replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"tiktok_{safe_title}_{timestamp}.mp4"
            video_path = self.video_dir / video_filename
            
            # Create video
            output_path = await self.video_processor.create_video(
                audio_file=audio_path,
                script_content=summary_text,
                content_analysis=content_analysis,
                output_path=str(video_path)
            )
            
            if output_path and Path(output_path).exists():
                file_size = Path(output_path).stat().st_size / (1024 * 1024)
                logger.success(f"✅ Video generated: {Path(output_path).name} ({file_size:.1f}MB)")
                return output_path
            else:
                logger.error("❌ Video generation failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Video generation error: {e}")
            return None
    
    async def process_single_article(self, article: Article) -> Optional[Dict]:
        """Process a single article through the complete pipeline."""
        logger.info(f"🔄 Processing: {article.title}")
        logger.info(f"   Category: {article.category} | Words: {article.word_count}")
        
        try:
            # Step 1: Generate summary
            summary_data = await self.generate_tiktok_summary(article)
            if not summary_data:
                return None
            
            # Step 2: Generate audio
            audio_path = await self.generate_audio(summary_data)
            if not audio_path:
                return None
            
            # Step 3: Generate video
            video_path = await self.generate_video(summary_data, audio_path)
            if not video_path:
                return None
            
            # Success!
            result = {
                'title': article.title,
                'category': article.category,
                'url': article.url,
                'word_count': article.word_count,
                'summary': summary_data['summary'],
                'estimated_duration': summary_data['estimated_duration'],
                'audio_path': audio_path,
                'video_path': video_path,
                'file_size_mb': Path(video_path).stat().st_size / (1024 * 1024),
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.success(f"✅ Complete pipeline success for: {article.title[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline processing failed for {article.title}: {e}")
            return {
                'title': article.title,
                'category': article.category,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_batch_processing(self, max_articles: int = 3) -> Dict:
        """Run the complete pipeline for multiple articles."""
        logger.info(f"🚀 Starting batch processing for {max_articles} articles")
        start_time = datetime.now()
        
        try:
            # Initialize pipeline
            await self.initialize()
            
            # Fetch articles
            articles = await self.fetch_latest_articles(max_articles)
            if not articles:
                logger.error("❌ No articles to process")
                return {'success': False, 'error': 'No articles found'}
            
            # Process each article
            results = []
            successful_videos = 0
            
            for i, article in enumerate(articles, 1):
                logger.info(f"\n🎬 Processing Video {i}/{len(articles)}")
                logger.info("=" * 60)
                
                result = await self.process_single_article(article)
                if result:
                    results.append(result)
                    if result.get('success'):
                        successful_videos += 1
            
            # Generate report
            processing_time = (datetime.now() - start_time).total_seconds()
            
            report = {
                'success': True,
                'total_articles': len(articles),
                'successful_videos': successful_videos,
                'success_rate': f"{successful_videos}/{len(articles)} ({successful_videos/len(articles)*100:.1f}%)",
                'processing_time_minutes': processing_time / 60,
                'results': results,
                'timestamp': start_time.isoformat()
            }
            
            # Save report
            report_file = self.reports_dir / f"pipeline_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.success(f"\n🎉 Batch processing complete!")
            logger.info(f"📊 Success rate: {successful_videos}/{len(articles)} videos generated")
            logger.info(f"⏱️ Total time: {processing_time/60:.1f} minutes")
            logger.info(f"📄 Report saved: {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return {'success': False, 'error': str(e)}
        
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up pipeline resources."""
        logger.info("🧹 Cleaning up pipeline resources...")
        
        try:
            if self.summarizer:
                await self.summarizer.cleanup()
            if self.scraper:
                await self.scraper.close()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


async def main():
    """Main entry point for complete pipeline."""
    logger.info("🤖 TikTok Automata: Complete Pipeline Integration")
    logger.info("🔧 Full End-to-End Video Generation")
    logger.info("=" * 70)
    
    try:
        pipeline = CompleteTikTokPipeline()
        
        # Run batch processing for 3 articles
        report = await pipeline.run_batch_processing(max_articles=3)
        
        if report.get('success'):
            logger.success("🎉 Pipeline completed successfully!")
            
            # Display results summary
            logger.info(f"\n📊 Final Results:")
            logger.info(f"   • Articles processed: {report['total_articles']}")
            logger.info(f"   • Videos generated: {report['successful_videos']}")
            logger.info(f"   • Success rate: {report['success_rate']}")
            logger.info(f"   • Total time: {report['processing_time_minutes']:.1f} minutes")
            
            # List generated videos
            successful_results = [r for r in report['results'] if r.get('success')]
            if successful_results:
                logger.info(f"\n🎬 Generated Videos:")
                for i, result in enumerate(successful_results, 1):
                    logger.info(f"   {i}. {result['title'][:40]}...")
                    logger.info(f"      📁 {Path(result['video_path']).name}")
                    logger.info(f"      📊 {result['file_size_mb']:.1f}MB | ⏱️ {result['estimated_duration']:.1f}s")
            
        else:
            logger.error(f"❌ Pipeline failed: {report.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
