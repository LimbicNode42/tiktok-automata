#!/usr/bin/env python3
"""
Full pipeline test for TikTok video generation with blurred background letterboxing.
This test creates a complete video with TTS audio and gaming footage.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video.processors.video_processor import VideoProcessor, VideoConfig
from tts.kokoro_tts import KokoroTTSEngine
from summarizer.llama_summarizer import LlamaSummarizer
from scraper.newsletter_scraper import Article
from datetime import datetime
from loguru import logger

async def test_full_pipeline_with_blurred_letterbox():
    """Test the complete TikTok video generation pipeline with blurred background letterboxing."""
    
    logger.info("🎬 Starting FULL PIPELINE test with blurred background letterboxing")
    
    # Sample content for TikTok video
    test_content = """
    Breaking: Revolutionary AI breakthrough changes everything! Scientists at DeepMind have developed 
    an AI system that can solve complex mathematical problems faster than any human. This incredible 
    technology could transform education, research, and scientific discovery. The AI demonstrated 
    remarkable capabilities in calculus, algebra, and even advanced theoretical mathematics. 
    Researchers are calling this a game-changing moment for artificial intelligence and mathematics.
    """
    
    try:        # Step 1: Generate TikTok script using summarizer
        logger.info("📝 Step 1: Generating TikTok script...")
        summarizer = LlamaSummarizer()
        
        # Create Article object
        article = Article(
            title="Revolutionary AI Breakthrough Changes Everything",
            content=test_content,
            summary="AI system solves complex mathematical problems faster than humans",
            url="https://example.com/ai-breakthrough",
            published_date=datetime.now(),
            category="tech",
            word_count=len(test_content.split())
        )
        
        # Generate script using the correct method
        script = await summarizer.summarize_for_tiktok(
            article=article,
            target_duration=45,  # 45 second video
            include_voice_recommendation=False
        )        
        if not script:
            logger.error("❌ Failed to generate script")
            return
        
        logger.success(f"✅ Generated script: {script[:100]}...")        # Step 2: Generate TTS audio
        logger.info("🎤 Step 2: Generating TTS audio...")
        tts = KokoroTTSEngine()
        
        # Use a professional voice for tech content
        audio_file = await tts.generate_audio(
            text=script,
            voice='af_sarah',  # Professional female voice
            speed=1.0,
            output_path="test_full_pipeline_audio.wav"
        )
        
        if not audio_file or not Path(audio_file).exists():
            logger.error("❌ Failed to generate TTS audio")
            return
        
        logger.success(f"✅ Generated TTS audio: {audio_file}")
        
        # Step 3: Create video with optimal blurred background settings
        logger.info("🎮 Step 3: Creating video with blurred background letterboxing...")
        
        config = VideoConfig(
            # Use our optimal blurred background settings
            letterbox_mode="blurred_background",
            letterbox_crop_percentage=0.60,  # 60% crop
            blur_strength=30.0,              # Heavy blur
            background_opacity=0.7,          # 70% opacity (pale)
            background_desaturation=0.3,     # 30% desaturation (70% color retention)
            
            # High quality output
            output_quality="high",
            width=1080,
            height=1920,            fps=30
        )
        
        processor = VideoProcessor(config)
        
        # Create content analysis for video processor
        content_analysis = {
            'category': 'tech',
            'is_breakthrough': True,
            'urgency_level': 'high',
            'controversy_score': 0
        }
        
        voice_info = {
            'voice_id': 'af_sarah',
            'speed': 1.0
        }
        
        # Create the complete video
        output_video = await processor.create_video(
            audio_file=audio_file,
            script_content=script,
            content_analysis=content_analysis,
            voice_info=voice_info,
            output_path="test_full_pipeline_blurred_letterbox.mp4"
        )
        
        if output_video:
            logger.success(f"✅ Complete video created: {output_video}")
            
            # Get file info
            video_path = Path(output_video)
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            
            logger.info("🎉 FULL PIPELINE TEST RESULTS:")
            logger.info(f"   📁 Video file: {output_video}")
            logger.info(f"   📏 File size: {file_size_mb:.1f}MB")
            logger.info(f"   🎨 Letterboxing: Blurred background with 60% crop")
            logger.info(f"   🌫️ Background: 30.0 blur, 70% opacity, 30% desaturation")
            logger.info(f"   🎤 Audio: TTS with {voice_info['voice_id']} voice")
            logger.info(f"   ⏱️ Expected features:")
            logger.info(f"      ✓ Sharp gaming footage in center (60% of source)")
            logger.info(f"      ✓ Blurred, pale, desaturated background")
            logger.info(f"      ✓ No black bars, modern TikTok appearance")
            logger.info(f"      ✓ Synchronized TTS audio")
            logger.info(f"      ✓ Professional quality output")
            
        else:
            logger.error("❌ Failed to create video")
            
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_comparison_videos():
    """Create comparison videos with different letterboxing modes."""
    
    logger.info("\n🔄 Creating comparison videos...")
    
    # Use a shorter script for quick comparison
    quick_script = "Revolutionary AI breakthrough! Scientists develop incredible new technology that solves complex problems faster than humans. This could change everything in education and research!"    # Generate TTS for comparison
    tts = KokoroTTSEngine()
    
    audio_file = await tts.generate_audio(
        text=quick_script,
        voice='af_sarah',
        speed=1.0,
        output_path="test_comparison_audio.wav"
    )
    
    if not audio_file:
        logger.error("❌ Failed to generate comparison audio")
        return
    
    # Test different letterboxing modes
    modes = [
        ("traditional", "Traditional black bars"),
        ("percentage", "Percentage crop (60%)"),
        ("blurred_background", "Blurred background (optimal)")
    ]
    
    content_analysis = {'category': 'tech', 'is_breakthrough': True}
    
    for mode, description in modes:
        logger.info(f"\n📹 Creating {description} video...")
        
        config = VideoConfig(
            letterbox_mode=mode,
            letterbox_crop_percentage=0.60,
            blur_strength=30.0,
            background_opacity=0.7,
            background_desaturation=0.3,
            output_quality="medium",  # Faster processing for comparison
            width=1080,
            height=1920
        )
        
        processor = VideoProcessor(config)
        
        output_path = f"test_comparison_{mode}.mp4"
        
        video_file = await processor.create_video(
            audio_file=audio_file,
            script_content=quick_script,
            content_analysis=content_analysis,
            voice_info={'voice_id': 'af_sarah', 'speed': 1.0},
            output_path=output_path
        )
        
        if video_file:
            logger.success(f"✅ {description} video: {output_path}")
        else:
            logger.error(f"❌ Failed to create {description} video")

if __name__ == "__main__":
    # Run the full pipeline test
    asyncio.run(test_full_pipeline_with_blurred_letterbox())
    
    # Create comparison videos to see the difference
    asyncio.run(test_comparison_videos())
    
    logger.success("🎉 All pipeline tests completed!")
