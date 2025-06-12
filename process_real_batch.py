#!/usr/bin/env python3
"""
Process real TLDR articles with Llama summarizer - Production Test
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from summarizer.llama_summarizer import LlamaSummarizer

def load_real_articles(limit=None):
    """Load real TLDR articles"""
    input_file = "data/tldr_articles_20250610_231259.json"
    
    print(f"📂 Loading articles from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for successfully extracted articles
    successful_articles = []
    for article in data.get('articles', []):
        if (article.get('content_extraction_status') == 'success' and 
            article.get('content') and 
            len(article.get('content', '')) > 200):  # Minimum content length
            successful_articles.append(article)
    
    if limit:
        successful_articles = successful_articles[:limit]
    
    print(f"✅ Selected {len(successful_articles)} articles for processing")
    
    # Show categories
    categories = {}
    for article in successful_articles:
        cat = article.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"📋 Categories: {dict(sorted(categories.items()))}")
    
    return successful_articles, data.get('metadata', {})

def process_articles_batch():
    """Process real articles with Llama summarizer"""
    print("="*60)
    print("LLAMA SUMMARIZER - REAL TLDR ARTICLES BATCH PROCESSING")
    print("="*60)
    
    # Load articles (start with 5 for initial test)
    articles, metadata = load_real_articles(limit=5)
    
    if not articles:
        print("❌ No articles to process")
        return
    
    # Initialize summarizer
    print(f"\n🔄 Initializing Llama summarizer...")
    start_time = datetime.now()
    
    try:
        summarizer = LlamaSummarizer()
        init_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Summarizer initialized in {init_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Failed to initialize summarizer: {e}")
        return
    
    # Process articles
    print(f"\n🚀 Processing {len(articles)} articles...")
    results = []
    
    for i, article in enumerate(articles, 1):
        print(f"\n--- Article {i}/{len(articles)} ---")
        print(f"Title: {article['title'][:60]}...")
        print(f"Category: {article['category']}")
        print(f"Word count: {article['word_count']}")
        
        try:
            # Generate TikTok summary
            start_summary = datetime.now()
            
            tiktok_summary = summarizer.summarize_for_tiktok(
                title=article['title'],
                content=article['content'],
                category=article['category']
            )
            
            summary_time = (datetime.now() - start_summary).total_seconds()
            
            # Create result
            result = {
                'processing_order': i,
                'article': {
                    'title': article['title'],
                    'url': article['url'],
                    'category': article['category'],
                    'word_count': article['word_count'],
                    'published_date': article.get('published_date', '')
                },
                'tiktok_summary': tiktok_summary,
                'performance': {
                    'generation_time_seconds': round(summary_time, 2),
                    'characters_generated': len(tiktok_summary),
                    'generation_speed_chars_per_sec': round(len(tiktok_summary) / summary_time, 2)
                },
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            print(f"✅ Generated in {summary_time:.1f}s ({len(tiktok_summary)} chars)")
            print(f"📝 Preview: {tiktok_summary[:100]}...")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            
            result = {
                'processing_order': i,
                'article': {
                    'title': article['title'],
                    'url': article['url'],
                    'category': article['category'],
                    'word_count': article['word_count'],
                    'published_date': article.get('published_date', '')
                },
                'tiktok_summary': None,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
            results.append(result)
    
    # Calculate statistics
    total_time = (datetime.now() - start_time).total_seconds()
    successful_results = [r for r in results if r['status'] == 'success']
    
    # Save results
    output_file = f"data/real_articles_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_data = {
        'metadata': {
            'processing_timestamp': datetime.now().isoformat(),
            'llama_model': 'meta-llama/Llama-3.2-3B-Instruct',
            'source_file': 'tldr_articles_20250610_231259.json',
            'total_processing_time_seconds': round(total_time, 2),
            'initialization_time_seconds': round(init_time, 2),
            'statistics': {
                'total_articles': len(results),
                'successful': len(successful_results),
                'failed': len(results) - len(successful_results),
                'success_rate_percent': round(len(successful_results) / len(results) * 100, 1) if results else 0,
                'average_generation_time_seconds': round(
                    sum(r['performance']['generation_time_seconds'] for r in successful_results) / len(successful_results), 2
                ) if successful_results else 0,
                'total_characters_generated': sum(r['performance']['characters_generated'] for r in successful_results),
                'average_summary_length': round(
                    sum(r['performance']['characters_generated'] for r in successful_results) / len(successful_results), 1
                ) if successful_results else 0
            }
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Total articles: {len(results)}")
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Failed: {len(results) - len(successful_results)}")
    print(f"📈 Success rate: {output_data['metadata']['statistics']['success_rate_percent']:.1f}%")
    print(f"⚡ Total time: {total_time:.1f}s (init: {init_time:.1f}s)")
    if successful_results:
        print(f"📝 Avg generation time: {output_data['metadata']['statistics']['average_generation_time_seconds']:.2f}s")
        print(f"📏 Avg summary length: {output_data['metadata']['statistics']['average_summary_length']:.0f} chars")
    print(f"💾 Results saved to: {output_file}")
    
    # Show sample summaries
    if successful_results:
        print(f"\n📋 SAMPLE SUMMARIES:")
        for result in successful_results[:2]:
            print(f"\n🎬 {result['article']['title'][:50]}...")
            print(f"   Category: {result['article']['category']}")
            print(f"   Time: {result['performance']['generation_time_seconds']:.1f}s")
            print(f"   Summary: {result['tiktok_summary'][:120]}...")

if __name__ == "__main__":
    process_articles_batch()
