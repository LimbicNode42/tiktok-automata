#!/usr/bin/env python3
"""
Test script to validate the improved duration control in the summarizer.
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from src.summarizer.llama_summarizer import LlamaSummarizer, TikTokSummaryConfig
from src.scraper.newsletter_scraper import Article
from src.utils.config import config

def create_test_article():
    """Create a test article for duration validation."""
    return Article(
        title="Revolutionary AI Breakthrough Changes Everything",
        content="""
        Scientists at MIT have developed a groundbreaking AI system that can predict 
        protein structures with unprecedented accuracy. This breakthrough could 
        revolutionize drug discovery and medical research, potentially saving millions 
        of lives and billions of dollars in research costs.
        
        The new AI model, called ProteinMaster, uses advanced neural networks to 
        analyze protein sequences and predict their three-dimensional structures 
        within minutes, compared to the months or years required by traditional methods.
        
        Initial tests show 95% accuracy rates, surpassing all previous methods. 
        The implications for treating diseases like cancer, Alzheimer's, and Parkinson's 
        are enormous. Pharmaceutical companies are already lining up to license the technology.
        
        The research team plans to open-source the model next year, making it available 
        to researchers worldwide and potentially accelerating medical breakthroughs 
        across the globe.
        """,
        summary="",
        url="https://example.com/ai-breakthrough",
        published_date=datetime.now(),
        category="ai"
    )

def estimate_tts_duration(text, tts_speed=1.35):
    """Estimate TTS duration for validation."""
    word_count = len(text.split())
    base_duration = (word_count / 150) * 60  # 150 words per minute base rate
    actual_duration = base_duration / tts_speed
    return actual_duration

async def test_duration_control():
    """Test the improved duration control."""
    print("🧪 Testing Strict Duration Control in Summarizer")
    print("=" * 60)
    
    # Test with different target durations
    test_durations = [30, 40, 45, 50, 55]
    
    # Create summarizer with strict config
    strict_config = TikTokSummaryConfig(
        target_duration=45,
        max_tokens=1200,
        max_attempts=3,
        duration_buffer=0.15
    )
    
    summarizer = LlamaSummarizer(strict_config)
    article = create_test_article()
    
    results = []
    
    for target_duration in test_durations:
        print(f"\n📊 Testing target duration: {target_duration}s")
        print("-" * 40)
        
        start_time = time.time()
        
        # Generate summary with voice recommendation to get detailed output
        result = await summarizer.summarize_for_tiktok(
            article, 
            target_duration=target_duration,
            include_voice_recommendation=True
        )
        
        generation_time = time.time() - start_time
        
        if result and isinstance(result, dict):
            summary = result['summary']
            estimated_duration = result.get('estimated_duration', 0)
            attempts = result.get('generation_attempts', 1)
            has_warning = result.get('duration_warning', False)
        elif result:
            summary = result
            estimated_duration = estimate_tts_duration(summary, config.get_tts_speed())
            attempts = 1
            has_warning = False
        else:
            print(f"❌ Failed to generate summary for {target_duration}s")
            continue
        
        # Validate duration
        word_count = len(summary.split())
        duration_ok = estimated_duration <= target_duration
        
        test_result = {
            'target_duration': target_duration,
            'estimated_duration': estimated_duration,
            'word_count': word_count,
            'generation_time': generation_time,
            'attempts': attempts,
            'duration_ok': duration_ok,
            'has_warning': has_warning,
            'summary_preview': summary[:100] + "..." if len(summary) > 100 else summary
        }
        
        results.append(test_result)
        
        # Print results
        status = "✅" if duration_ok else "⚠️"
        print(f"{status} Target: {target_duration}s | Estimated: {estimated_duration:.1f}s | Words: {word_count}")
        print(f"   Generation: {generation_time:.1f}s | Attempts: {attempts}")
        if has_warning:
            print(f"   ⚠️ Duration warning: exceeded target in all attempts")
        print(f"   Preview: {test_result['summary_preview']}")
    
    # Summary statistics
    print(f"\n📈 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['duration_ok'])
    total = len(results)
    
    print(f"Duration tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    avg_duration = sum(r['estimated_duration'] for r in results) / len(results)
    avg_words = sum(r['word_count'] for r in results) / len(results)
    avg_attempts = sum(r['attempts'] for r in results) / len(results)
    
    print(f"Average estimated duration: {avg_duration:.1f}s")
    print(f"Average word count: {avg_words:.1f}")
    print(f"Average attempts: {avg_attempts:.1f}")
    
    warnings = sum(1 for r in results if r['has_warning'])
    if warnings > 0:
        print(f"⚠️ {warnings} summaries had duration warnings")
    
    # Save detailed results
    output_file = Path("test_results") / f"duration_control_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_timestamp': datetime.now().isoformat(),
            'test_config': {
                'target_duration': strict_config.target_duration,
                'max_tokens': strict_config.max_tokens,
                'max_attempts': strict_config.max_attempts,
                'duration_buffer': strict_config.duration_buffer,
                'tts_speed': config.get_tts_speed()
            },
            'results': results,
            'summary_stats': {
                'passed': passed,
                'total': total,
                'pass_rate': passed/total*100,
                'avg_duration': avg_duration,
                'avg_words': avg_words,
                'avg_attempts': avg_attempts,
                'warnings': warnings
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Cleanup
    await summarizer.cleanup()
    
    return passed == total

if __name__ == "__main__":
    print("Starting strict duration control test...")
    success = asyncio.run(test_duration_control())
    
    if success:
        print("\n🎉 All duration control tests passed!")
        exit(0)
    else:
        print("\n⚠️ Some duration control tests failed - check results for details")
        exit(1)
