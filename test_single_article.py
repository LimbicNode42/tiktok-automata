#!/usr/bin/env python3
"""
Test Llama summarizer on a single real TLDR article
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_single_article():
    """Load one article for testing"""
    input_file = "data/tldr_articles_20250610_231259.json"
    
    print(f"📂 Loading article from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find first successful article
    for article in data.get('articles', []):
        if (article.get('content_extraction_status') == 'success' and 
            article.get('content') and 
            len(article.get('content', '')) > 200):
            return article
    
    return None

def test_single_article():
    """Test summarizer on single article"""
    print("="*60)
    print("LLAMA SUMMARIZER - SINGLE ARTICLE TEST")
    print("="*60)
    
    # Load one article
    article = load_single_article()
    if not article:
        print("❌ No suitable article found")
        return
    
    print(f"📰 Selected article:")
    print(f"   Title: {article['title']}")
    print(f"   Category: {article['category']}")
    print(f"   Word count: {article['word_count']}")
    print(f"   URL: {article['url']}")
    print(f"   Content preview: {article['content'][:150]}...")
    
    # Initialize summarizer
    print(f"\n🔄 Initializing Llama summarizer...")
    try:
        from summarizer.llama_summarizer import LlamaSummarizer
        
        print("📦 LlamaSummarizer imported successfully")
        print("🚀 Creating summarizer instance (this may take a few minutes)...")
        
        start_init = datetime.now()
        summarizer = LlamaSummarizer()
        init_time = (datetime.now() - start_init).total_seconds()
        
        print(f"✅ Summarizer initialized in {init_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Failed to initialize summarizer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate summary
    print(f"\n🤖 Generating TikTok summary...")
    try:
        start_summary = datetime.now()
        
        tiktok_summary = summarizer.summarize_for_tiktok(
            title=article['title'],
            content=article['content'],
            category=article['category']
        )
        
        summary_time = (datetime.now() - start_summary).total_seconds()
        
        print(f"✅ Summary generated in {summary_time:.1f} seconds")
        print(f"📝 Summary length: {len(tiktok_summary)} characters")
        print(f"\n{'='*60}")
        print("GENERATED TIKTOK SUMMARY:")
        print(f"{'='*60}")
        print(tiktok_summary)
        print(f"{'='*60}")
        
        # Save result
        result = {
            'test_timestamp': datetime.now().isoformat(),
            'article': {
                'title': article['title'],
                'url': article['url'],
                'category': article['category'],
                'word_count': article['word_count']
            },
            'performance': {
                'initialization_time_seconds': init_time,
                'summary_generation_time_seconds': summary_time,
                'total_time_seconds': init_time + summary_time
            },
            'tiktok_summary': tiktok_summary,
            'summary_stats': {
                'character_count': len(tiktok_summary),
                'word_count': len(tiktok_summary.split()),
                'estimated_duration_seconds': len(tiktok_summary) / 15  # ~15 chars per second reading
            }
        }
        
        output_file = f"data/single_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Test result saved to: {output_file}")
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Initialization: {init_time:.1f}s")
        print(f"   Summary generation: {summary_time:.1f}s")
        print(f"   Total time: {init_time + summary_time:.1f}s")
        print(f"   Summary: {len(tiktok_summary)} chars, ~{len(tiktok_summary)/15:.1f}s reading time")
        
    except Exception as e:
        print(f"❌ Failed to generate summary: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_article()
