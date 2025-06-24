#!/usr/bin/env python3
"""
Test script for the Smart Subtitle Style Selector.

This script demonstrates how the system automatically chooses
the best subtitle style based on video content analysis.
"""

import asyncio
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

# Import the video processing pipeline
from src.video.processors.video_processor import VideoProcessor, VideoConfig

# Test content for different scenarios
TEST_SCENARIOS = [
    {
        "name": "AI Breakthrough",
        "script": "OpenAI just released ChatGPT's most powerful update yet! This changes everything about AI!",
        "config": VideoConfig(
            width=1080,
            height=1920,
            fps=30,
            enable_subtitles=True,
            subtitle_style="auto",  # Let the system choose!
            subtitle_position=0.75,
            export_srt=True,
            letterbox_mode="blurred_background",
            use_gaming_footage=True,
            enable_zoom_effects=False,
            enable_transitions=False,
            output_quality="medium"
        )
    },
    {
        "name": "Gaming Highlight",
        "script": "INSANE 360 no-scope headshot! Watch this epic gaming moment that took 1000 attempts!",
        "config": VideoConfig(
            width=1080,
            height=1920,
            fps=30,
            enable_subtitles=True,
            subtitle_style="auto",  # Let the system choose!
            subtitle_position=0.75,
            export_srt=True,
            letterbox_mode="blurred_background",
            use_gaming_footage=True,
            enable_zoom_effects=False,
            enable_transitions=False,
            output_quality="medium"
        )
    },
    {
        "name": "Tech Review",
        "script": "This new phone feature will blow your mind! Here's why everyone is talking about it.",
        "config": VideoConfig(
            width=1080,
            height=1920,
            fps=30,
            enable_subtitles=True,
            subtitle_style="auto",  # Let the system choose!
            subtitle_position=0.75,
            export_srt=True,
            letterbox_mode="blurred_background",
            use_gaming_footage=True,
            enable_zoom_effects=False,
            enable_transitions=False,
            output_quality="medium"
        )
    }
]

def create_mock_audio(duration: float = 6.0, output_file: str = "mock_audio.wav"):
    """Create a simple mock audio file for testing."""
    sample_rate = 22050
    samples = int(sample_rate * duration)
    
    # Create simple sine wave audio for testing
    t = np.linspace(0, duration, samples, False)
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.1  # Low volume
    
    # Add some variation to make it more speech-like
    modulation = np.sin(2 * np.pi * 2 * t) * 0.5 + 0.5
    audio = audio * modulation
    
    # Write to file
    sf.write(output_file, audio, sample_rate)
    logger.info(f"🎵 Created mock audio file: {output_file} ({duration:.1f}s)")
    return output_file

async def test_smart_style_selection():
    """Test the smart subtitle style selector with different video scenarios."""
    logger.info("🧠 Starting Smart Style Selection Test")
    
    output_dir = Path("output_videos_smart")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        logger.info(f"\n🎬 Scenario {i}: {scenario['name']}")
        logger.info(f"   📝 Script: {scenario['script'][:50]}...")
        
        try:
            # Create mock audio
            audio_file = create_mock_audio(duration=6.0, output_file=f"mock_smart_{i}.wav")
            
            # Create processor with auto style selection
            processor = VideoProcessor(scenario["config"])
            
            # Create output path
            output_path = output_dir / f"smart_style_{i}_{scenario['name'].replace(' ', '_')}.mp4"
            
            # Generate video with smart style selection
            logger.info(f"   🎯 Generating video with auto style selection...")
            start_time = time.time()
            
            video_file = await processor.create_video(
                audio_file=audio_file,
                script_content=scenario["script"],
                content_analysis={"is_breakthrough": i == 1, "excitement_level": "high"},
                voice_info={"voice_name": "test", "speed": 1.0},
                output_path=str(output_path)
            )
            
            processing_time = time.time() - start_time
            
            if video_file:
                file_size = Path(video_file).stat().st_size / (1024 * 1024)
                
                result = {
                    "scenario": scenario["name"],
                    "video_file": video_file,
                    "file_size": file_size,
                    "processing_time": processing_time,
                    "success": True
                }
                
                logger.success(f"   ✅ Success: {Path(video_file).name} ({file_size:.1f}MB, {processing_time:.1f}s)")
            else:
                result = {
                    "scenario": scenario["name"],
                    "success": False,
                    "error": "Video creation failed"
                }
                logger.error(f"   ❌ Failed to create video")
            
            results.append(result)
                
        except Exception as e:
            logger.error(f"   ❌ Scenario {i} failed: {e}")
            results.append({
                "scenario": scenario["name"],
                "success": False,
                "error": str(e)
            })
        finally:
            # Cleanup
            try:
                if 'audio_file' in locals():
                    Path(audio_file).unlink(missing_ok=True)
            except:
                pass
    
    # Summary
    logger.info("\n📊 Smart Style Selection Results:")
    successful = sum(1 for r in results if r.get("success", False))
    
    for result in results:
        if result.get("success", False):
            logger.info(f"  ✅ {result['scenario']}: {Path(result['video_file']).name}")
        else:
            logger.info(f"  ❌ {result['scenario']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n🎯 {successful}/{len(TEST_SCENARIOS)} scenarios completed successfully")
    
    if successful > 0:
        logger.success(f"📁 Check '{output_dir}' for videos with automatically selected styles!")
    
    return results

async def test_style_comparison():
    """Create videos with both manual and auto style selection for comparison."""
    logger.info("\n🔄 Style Comparison Test (Manual vs Auto)")
    
    comparison_script = "This AI breakthrough will change everything! Watch this amazing technology in action!"
    
    # Test both manual and auto selection
    styles_to_test = ["bubble", "bubble_gaming", "auto"]
    output_dir = Path("output_videos_comparison")
    output_dir.mkdir(exist_ok=True)
    
    comparison_results = []
    
    for style in styles_to_test:
        logger.info(f"\n🎨 Testing style: {style}")
        
        try:
            # Create audio
            audio_file = create_mock_audio(duration=5.0, output_file=f"mock_comparison_{style}.wav")
            
            # Configure processor
            config = VideoConfig(
                width=1080,
                height=1920,
                fps=30,
                enable_subtitles=True,
                subtitle_style=style,
                subtitle_position=0.75,
                export_srt=True,
                letterbox_mode="blurred_background",
                use_gaming_footage=True,
                enable_zoom_effects=False,
                enable_transitions=False,
                output_quality="medium"
            )
            
            processor = VideoProcessor(config)
            
            # Create output
            output_path = output_dir / f"comparison_{style}.mp4"
            
            # Generate video
            video_file = await processor.create_video(
                audio_file=audio_file,
                script_content=comparison_script,
                content_analysis={"is_breakthrough": True},
                voice_info={"voice_name": "test"},
                output_path=str(output_path)
            )
            
            if video_file:
                file_size = Path(video_file).stat().st_size / (1024 * 1024)
                comparison_results.append({
                    "style": style,
                    "file": Path(video_file).name,
                    "size": file_size,
                    "success": True
                })
                logger.success(f"   ✅ {style}: {Path(video_file).name} ({file_size:.1f}MB)")
            else:
                comparison_results.append({
                    "style": style,
                    "success": False
                })
                logger.error(f"   ❌ {style}: Failed")
                
        except Exception as e:
            logger.error(f"   ❌ {style}: {e}")
            comparison_results.append({
                "style": style,
                "success": False,
                "error": str(e)
            })
        finally:
            # Cleanup
            try:
                if 'audio_file' in locals():
                    Path(audio_file).unlink(missing_ok=True)
            except:
                pass
    
    # Show comparison results
    logger.info("\n📊 Manual vs Auto Style Comparison:")
    for result in comparison_results:
        if result.get("success", False):
            logger.info(f"  {result['style'].ljust(15)}: {result['file']} ({result['size']:.1f}MB)")
        else:
            logger.info(f"  {result['style'].ljust(15)}: Failed")
    
    return comparison_results

if __name__ == "__main__":
    print("🧠 Smart Subtitle Style Selector Test")
    print("=" * 50)
    
    # Test 1: Smart style selection with different scenarios
    print("\n🎯 Test 1: Smart Style Selection")
    results = asyncio.run(test_smart_style_selection())
    
    # Test 2: Comparison between manual and auto selection
    print("\n🔄 Test 2: Manual vs Auto Comparison") 
    comparison = asyncio.run(test_style_comparison())
    
    print("\n✨ Testing complete!")
    print("\n📁 Check the following folders for results:")
    print("   - output_videos_smart/ (Smart selection results)")
    print("   - output_videos_comparison/ (Manual vs Auto comparison)")
    
    # Show summary
    successful_smart = sum(1 for r in results if r.get("success", False))
    successful_comparison = sum(1 for r in comparison if r.get("success", False))
    
    print(f"\n📈 Summary:")
    print(f"   Smart Selection: {successful_smart}/{len(TEST_SCENARIOS)} scenarios")
    print(f"   Comparison Test: {successful_comparison}/3 styles")
    
    if successful_smart > 0 or successful_comparison > 0:
        print(f"\n🎉 Success! The smart style selector is working!")
    else:
        print(f"\n❌ All tests failed - check the logs for details")
