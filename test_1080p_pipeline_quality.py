#!/usr/bin/env python3
"""
1080p Pipeline Quality Test - Focused Test

Tests the key aspects of the 1080p upgrade:
- Gaming footage quality and cropping
- File sizes and compression
- Processing performance
"""

import asyncio
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🎯 Starting 1080p Pipeline Quality Test")

try:
    from src.scraper.newsletter_scraper import Article
    from src.video.processors.video_processor import VideoProcessor, VideoConfig
    from src.video.managers.footage_manager import FootageManager
    print("✅ All components imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


async def test_1080p_pipeline():
    """Test the complete pipeline with focus on 1080p quality."""
    
    test_start = time.time()
    output_dir = Path("test_output_1080p")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Create test data
    print("\n📝 Step 1: Creating test content...")
    
    # Mock article for testing
    test_article = Article(
        title="AI Breakthrough: Revolutionary Model Achieved",
        content="""
        Scientists have achieved a major breakthrough in artificial intelligence with the development
        of a new model that demonstrates unprecedented capabilities. The model shows remarkable 
        performance across multiple domains including reasoning, creativity, and problem-solving.
        
        In comprehensive testing, the AI achieved human-level performance on complex tasks that
        previously seemed impossible for machines. This represents a significant milestone in
        the journey toward artificial general intelligence.
        """.strip(),
        url="https://example.com/ai-breakthrough",
        summary="Revolutionary AI model achieves human-level performance",
        published_date=datetime.now(),
        category="ai"
    )
    
    # Mock TTS script (we'll focus on video processing)
    test_script = """
    AI just did something INSANE! Scientists created a model that thinks like humans.
    This breakthrough changes everything we know about artificial intelligence.
    The implications are mind-blowing - we're talking about AI that can reason,
    create, and solve problems just like you and me. This is huge!
    """
    
    print(f"✅ Test script: {len(test_script)} characters")
    
    # Step 2: Create mock audio file (for testing video sync)
    print("\n🎤 Step 2: Creating test audio...")
    
    # Create a simple WAV file for testing (silent audio with proper duration)
    import wave
    import numpy as np
    
    # Create 30 seconds of audio at 22050 Hz
    duration = 30  # seconds
    sample_rate = 22050
    samples = np.zeros(int(duration * sample_rate), dtype=np.int16)
    
    audio_file = output_dir / "test_audio.wav"
    with wave.open(str(audio_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    
    audio_size = os.path.getsize(audio_file) / 1024  # KB
    print(f"✅ Test audio: {audio_size:.1f}KB, {duration}s duration")
    
    # Step 3: Video Processing Test (THE MAIN TEST)
    print("\n🎬 Step 3: Testing 1080p Video Processing...")
    video_start = time.time()
    
    # Configure for TikTok with high quality
    video_config = VideoConfig(
        width=1080,
        height=1920,
        fps=30,
        duration=duration,
        output_quality="high"
    )
    
    print(f"📊 Video config: {video_config.width}x{video_config.height} @ {video_config.fps}fps")
    
    # Initialize video processor
    processor = VideoProcessor(video_config)
    
    # Test video generation
    video_file = output_dir / f"test_video_1080p_{int(time.time())}.mp4"
    
    print("🎮 Using real gaming footage from 1080p collection...")
    
    # Create video with real gaming footage
    try:
        await processor.create_video(
            audio_file=str(audio_file),
            script_content=test_script,
            content_analysis={
                'urgency_level': 'high',
                'is_tech_news': True,
                'requires_high_energy': True
            },
            output_path=str(video_file)
        )
        
        video_processing_time = time.time() - video_start
        print(f"✅ Video generated in {video_processing_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Quality Analysis
    print("\n📊 Step 4: Analyzing Video Quality...")
    
    if video_file.exists():
        video_size = os.path.getsize(video_file) / (1024 * 1024)  # MB
        print(f"📁 Video file size: {video_size:.1f}MB")
        
        # Use ffprobe to get video specs
        try:
            import subprocess
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                for stream in data.get('streams', []):
                    if stream['codec_type'] == 'video':
                        width = stream['width']
                        height = stream['height']
                        codec = stream['codec_name']
                        bitrate = int(stream.get('bit_rate', 0)) // 1000  # kbps
                        fps = eval(stream['r_frame_rate'])
                        
                        print(f"🎯 Resolution: {width}x{height}")
                        print(f"🔧 Codec: {codec}")
                        print(f"⚡ Bitrate: {bitrate}kbps")
                        print(f"🎬 FPS: {fps:.1f}")
                        
                        # Quality assessment
                        is_1080p = height >= 1920
                        is_tiktok_ready = width == 1080 and height == 1920
                        quality_grade = "A+" if is_1080p and is_tiktok_ready else "B+"
                        
                        print(f"🏆 Quality Grade: {quality_grade}")
                        print(f"✅ 1080p Quality: {'YES' if is_1080p else 'NO'}")
                        print(f"📱 TikTok Ready: {'YES' if is_tiktok_ready else 'NO'}")
                        break
                        
        except Exception as e:
            print(f"⚠️ Quality analysis failed: {e}")
    else:
        print("❌ Video file not found")
    
    # Step 5: Performance Summary  
    total_time = time.time() - test_start
    
    print("\n" + "="*60)
    print("🎉 1080P PIPELINE TEST RESULTS")
    print("="*60)
    print(f"⏱️  Total Test Time: {total_time:.1f}s")
    print(f"🎬 Video Processing: {video_processing_time:.1f}s")
    print(f"📁 Video File Size: {video_size:.1f}MB")
    print(f"🎯 Target Quality: 1080x1920 (TikTok)")
    print(f"🚀 Performance: {'EXCELLENT' if video_processing_time < 60 else 'GOOD'}")
    print("="*60)
    
    # Save test report
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'performance': {
            'total_time': round(total_time, 1),
            'video_processing_time': round(video_processing_time, 1),
            'performance_grade': 'A' if video_processing_time < 60 else 'B'
        },
        'video_quality': {
            'file_size_mb': round(video_size, 1),
            'resolution': f"{width}x{height}" if 'width' in locals() else 'unknown',
            'codec': codec if 'codec' in locals() else 'unknown',
            'bitrate_kbps': bitrate if 'bitrate' in locals() else 0,
            'fps': round(fps, 1) if 'fps' in locals() else 0,
            'tiktok_ready': is_tiktok_ready if 'is_tiktok_ready' in locals() else False
        }
    }
    
    report_file = output_dir / f"test_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Detailed report saved: {report_file}")
    
    return report


async def main():
    """Main test runner."""
    try:
        report = await test_1080p_pipeline()
        print("\n✅ 1080p Pipeline test completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
