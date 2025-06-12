#!/usr/bin/env python3
"""
Final Demo: Generate TikTok summary from real TLDR article
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scraper.newsletter_scraper import Article

async def generate_summary():
    """Generate and save a real TikTok summary"""
    
    print("🎬 TIKTOK AUTOMATA - FINAL VALIDATION")
    print("="*60)
    
    # Load article data
    print("📂 Loading TLDR article...")
    with open("data/tldr_articles_20250610_231259.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find OpenAI article
    article_data = None
    for a in data['articles']:
        if (a.get('content_extraction_status') == 'success' and 
            'OpenAI' in a.get('title', '') and 
            len(a.get('content', '')) > 300):
            article_data = a
            break
    
    if not article_data:
        print("❌ OpenAI article not found, using first available article")
        for a in data['articles']:
            if (a.get('content_extraction_status') == 'success' and 
                len(a.get('content', '')) > 200):
                article_data = a
                break
    
    if not article_data:
        print("❌ No suitable articles found")
        return None
    
    # Create Article object
    try:
        published_date = datetime.fromisoformat(article_data.get('published_date', '2025-06-10T00:00:00'))
    except:
        published_date = datetime(2025, 6, 10)
    
    article = Article(
        title=article_data['title'],
        content=article_data['content'],
        summary=article_data.get('summary', ''),
        url=article_data['url'],
        published_date=published_date,
        category=article_data.get('category', 'ai'),
        word_count=article_data.get('word_count', 0)
    )
    
    print(f"✅ Selected: {article.title[:60]}...")
    print(f"📊 Category: {article.category} | Words: {article.word_count}")
    print(f"🔗 URL: {article.url}")
    
    # Initialize summarizer
    print(f"\n🤖 Initializing Llama 3.2-3B...")
    start_time = datetime.now()
    
    try:
        from summarizer.llama_summarizer import LlamaSummarizer
        summarizer = LlamaSummarizer()
        await summarizer.initialize()
        
        init_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Model loaded in {init_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None
    
    # Generate summary
    print(f"\n🎬 Generating TikTok summary...")
    try:
        gen_start = datetime.now()
        summary = await summarizer.summarize_for_tiktok(article)
        gen_time = (datetime.now() - gen_start).total_seconds()
        
        if not summary:
            print("❌ No summary generated")
            return None
        
        print(f"✅ Summary generated in {gen_time:.1f} seconds")
        print(f"📏 Length: {len(summary)} characters")
        
        # Show the summary
        print(f"\n{'='*60}")
        print("🎯 GENERATED TIKTOK SUMMARY:")
        print(f"{'='*60}")
        print(summary)
        print(f"{'='*60}")
        
        # Save result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data/tiktok_summary_{timestamp}.json"
        
        result = {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'model': 'meta-llama/Llama-3.2-3B-Instruct',
                'source': 'tldr_articles_20250610_231259.json',
                'performance': {
                    'initialization_time_seconds': round(init_time, 2),
                    'generation_time_seconds': round(gen_time, 2),
                    'total_time_seconds': round(init_time + gen_time, 2)
                }
            },
            'article': {
                'title': article.title,
                'url': article.url,
                'category': article.category,
                'word_count': article.word_count,
                'content_preview': article.content[:300] + "..."
            },
            'tiktok_summary': summary,
            'summary_analysis': {
                'character_count': len(summary),
                'word_count': len(summary.split()),
                'estimated_reading_time_seconds': round(len(summary) / 15, 1),
                'has_timestamps': '[' in summary and ']' in summary,
                'has_emojis': any(ord(char) > 127 for char in summary)
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 RESULT SAVED TO: {output_file}")
        print(f"\n📊 FINAL STATS:")
        print(f"   ⚡ Total time: {init_time + gen_time:.1f}s")
        print(f"   📝 Summary: {len(summary)} chars, {len(summary.split())} words")
        print(f"   🕒 Reading time: ~{len(summary)/15:.1f} seconds")
        print(f"   🎯 Format: {'✅ TikTok format' if '[' in summary else '❌ No timestamps'}")
        
        print(f"\n🎉 SUCCESS! TikTok summary generated and saved!")
        return output_file
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result_file = asyncio.run(generate_summary())
    if result_file:
        print(f"\n✅ VALIDATION COMPLETE!")
        print(f"📄 Summary file created: {result_file}")
    else:
        print(f"\n❌ VALIDATION FAILED!")
