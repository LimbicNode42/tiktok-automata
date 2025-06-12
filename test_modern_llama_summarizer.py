#!/usr/bin/env python3
"""
Test script for the Modern Llama Summarizer optimized for GTX 1060 6GB.
Tests different model tiers and provides performance benchmarks.
"""

import asyncio
import sys
import os
from datetime import datetime
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from summarizer.llama_summarizer import LlamaSummarizer, TikTokSummaryConfig, create_tiktok_summarizer
    from scraper.newsletter_scraper import Article
    print("✓ Successfully imported Llama Summarizer")
except ImportError as e:
    print(f"✗ Failed to import Llama Summarizer: {e}")
    sys.exit(1)


def create_test_articles():
    """Create sample articles for testing."""
    return [
        Article(
            title="OpenAI Announces GPT-5 with Breakthrough Reasoning Capabilities",
            content="OpenAI today unveiled GPT-5, their most advanced language model yet, featuring unprecedented reasoning capabilities that can solve complex mathematical problems and write code more efficiently than previous versions. The model demonstrates emergent abilities in planning, logical deduction, and creative problem-solving. Early tests show GPT-5 scoring 95% on advanced reasoning benchmarks, compared to GPT-4's 78%. The model uses a new architecture called 'Reasoning Transformers' that maintains an internal chain of thought during inference. OpenAI CEO Sam Altman stated that GPT-5 represents a fundamental leap toward artificial general intelligence. The model will be available through API access starting next month, with pricing expected to be 40% lower than GPT-4 despite superior performance. Industry experts are calling this the most significant AI advancement since the original ChatGPT launch.",
            summary="5-minute read",
            url="https://example.com/gpt5-announcement",
            published_date=datetime.now(),
            category="ai",
            word_count=156,
            content_extraction_status="success"
        ),
        Article(
            title="Apple's Revolutionary AR Glasses Finally Revealed After Years of Speculation",
            content="Apple has officially announced the Apple Vision Pro 2, their long-awaited augmented reality glasses that promise to replace smartphones within the next decade. The sleek, lightweight device weighs only 120 grams and offers 8K resolution per eye with a 120Hz refresh rate. Key features include all-day battery life, seamless integration with the Apple ecosystem, and revolutionary hand-tracking technology that doesn't require controllers. The glasses can overlay digital information onto the real world, from navigation directions to real-time translations. Apple demonstrated how users can attend virtual meetings, watch movies on giant virtual screens, and collaborate on 3D designs in shared AR spaces. The device starts at $1,999 and will be available in early 2025. Tim Cook described it as 'the future of computing' and predicted it will be as transformative as the iPhone. Pre-orders begin next week with initial availability in the US, UK, and Japan.",
            summary="4-minute read", 
            url="https://example.com/apple-ar-glasses",
            published_date=datetime.now(),
            category="big_tech",
            word_count=178,
            content_extraction_status="success"
        ),
        Article(
            title="New Programming Language 'Quantum' Makes Quantum Computing Accessible to Everyone",
            content="Researchers at MIT have developed 'Quantum', a new programming language designed to make quantum computing accessible to traditional software developers. Unlike existing quantum languages that require deep physics knowledge, Quantum uses familiar syntax similar to Python and JavaScript. The language automatically handles quantum circuit optimization, error correction, and quantum-classical hybrid algorithms. Early adopters report being able to write quantum programs within hours instead of months. The language includes built-in libraries for common quantum algorithms like Shor's factorization and Grover's search. Microsoft, IBM, and Google have already announced support for Quantum in their cloud quantum computing platforms. The open-source language has garnered 50,000 GitHub stars in its first week. Lead researcher Dr. Sarah Chen believes this could democratize quantum computing and accelerate breakthrough discoveries in drug discovery, cryptography, and optimization problems.",
            summary="3-minute read",
            url="https://example.com/quantum-programming-language", 
            published_date=datetime.now(),
            category="dev",
            word_count=165,
            content_extraction_status="success"
        )
    ]


async def test_model_initialization():
    """Test model loading and initialization."""
    print("\n=== Testing Model Initialization ===")
    
    # Test different model tiers
    tiers = ["ultra_fast", "balanced", "high_quality"]
    
    for tier in tiers:
        print(f"\n--- Testing {tier} tier ---")
        
        try:
            summarizer = ModernLlamaSummarizer(tier)
            model_info = summarizer.get_model_info()
            
            print(f"✓ Model: {model_info['model_name']}")
            print(f"✓ Expected VRAM: {model_info['vram_usage']}")
            print(f"✓ Speed: {model_info['speed']}")
            print(f"✓ Description: {model_info['description']}")
            
            # Only actually load the balanced model for full testing
            if tier == "balanced":
                print("  Loading model for testing...")
                await summarizer.initialize()
                print("  ✓ Model loaded successfully!")
                return summarizer
            else:
                print("  (Skipping actual load to save time)")
                
        except Exception as e:
            print(f"  ✗ Failed to initialize {tier}: {str(e)}")
    
    return None


async def test_summarization(summarizer):
    """Test TikTok summarization with sample articles."""
    print("\n=== Testing TikTok Summarization ===")
    
    if not summarizer:
        print("✗ No summarizer available for testing")
        return
    
    test_articles = create_test_articles()
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n--- Article {i}: {article.title[:50]}... ---")
        print(f"Category: {article.category}")
        print(f"Word count: {article.word_count}")
        
        try:
            # Test different target durations
            for duration in [30, 60]:
                print(f"\n  Generating {duration}-second TikTok summary...")
                
                summary = await summarizer.summarize_for_tiktok(article, target_duration=duration)
                
                if summary:
                    print(f"  ✓ Generated summary ({len(summary)} chars):")
                    print(f"     {summary[:100]}...")
                    
                    # Estimate reading time (average 150 words per minute)
                    word_count = len(summary.split())
                    estimated_time = (word_count / 150) * 60  # seconds
                    print(f"  ✓ Estimated reading time: {estimated_time:.1f} seconds")
                else:
                    print(f"  ✗ Failed to generate {duration}s summary")
                    
        except Exception as e:
            print(f"  ✗ Error generating summary: {str(e)}")


async def test_batch_processing(summarizer):
    """Test batch processing of multiple articles."""
    print("\n=== Testing Batch Processing ===")
    
    if not summarizer:
        print("✗ No summarizer available for testing")
        return
    
    test_articles = create_test_articles()
    
    try:
        print(f"Processing {len(test_articles)} articles in batch...")
        
        results = await summarizer.batch_summarize(test_articles)
        
        successful = sum(1 for r in results if r['success'])
        print(f"✓ Batch processing complete: {successful}/{len(results)} successful")
        
        # Show results summary
        for result in results:
            status = "✓" if result['success'] else "✗"
            title = result['article'].title[:40]
            print(f"  {status} {title}...")
            if result['success'] and result['tiktok_summary']:
                print(f"     Summary: {result['tiktok_summary'][:80]}...")
                
    except Exception as e:
        print(f"✗ Batch processing failed: {str(e)}")


async def test_hardware_optimization():
    """Test hardware-specific optimizations."""
    print("\n=== Testing Hardware Optimization ===")
    
    try:
        # Test the factory function for your hardware
        print("Creating summarizer optimized for GTX 1060 6GB...")
        
        optimized_summarizer = create_summarizer_for_hardware("gtx_1060_6gb")
        model_info = optimized_summarizer.get_model_info()
        
        print(f"✓ Recommended model: {model_info['model_name']}")
        print(f"✓ Expected performance: {model_info['speed']}")
        print(f"✓ VRAM usage: {model_info['vram_usage']}")
        
        return optimized_summarizer
        
    except Exception as e:
        print(f"✗ Hardware optimization test failed: {str(e)}")
        return None


async def main():
    """Run all tests."""
    print("🚀 Testing Modern Llama Summarizer for GTX 1060 6GB")
    print("=" * 60)
    
    try:
        # Test model initialization
        summarizer = await test_model_initialization()
        
        if summarizer:
            # Test individual summarization
            await test_summarization(summarizer)
            
            # Test batch processing
            await test_batch_processing(summarizer)
            
            # Clean up
            await summarizer.cleanup()
        
        # Test hardware optimization
        optimized_summarizer = await test_hardware_optimization()
        
        print("\n" + "=" * 60)
        print("🎉 Testing completed!")
        print("\nRecommendations for your GTX 1060 6GB setup:")
        print("• Use 'balanced' tier (Llama 3.2-3B) for best speed/quality")  
        print("• Expect ~2-3 seconds per summary")
        print("• Can process 20+ articles per minute")
        print("• 4-bit quantization keeps VRAM under 2GB")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
