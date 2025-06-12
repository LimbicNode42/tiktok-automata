#!/usr/bin/env python3
"""
Manual Demo: Process one real TLDR article with Llama
Run this to see the system working on real data!
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_real_article():
    """Demonstrate processing a real TLDR article"""
    
    print("🎬 TIKTOK AUTOMATA - LIVE DEMO")
    print("="*50)
    print("Processing real TLDR article with Llama 3.2-3B...")
    print()
      # Load a real article
    print("📂 Loading real TLDR article...")
    with open("data/tldr_articles_20250610_231259.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find the OpenAI article (most interesting for demo)
    article = None
    for a in data['articles']:
        if (a.get('content_extraction_status') == 'success' and 
            'OpenAI' in a['title'] and 
            len(a.get('content', '')) > 300):
            article = a
            break
    
    if not article:
        # Fallback to first good article
        for a in data['articles']:
            if (a.get('content_extraction_status') == 'success' and 
                len(a.get('content', '')) > 200):
                article = a
                break
    
    if not article:
        print("❌ No suitable article found")
        return
    
    # Show article info
    print(f"📰 Selected Article:")
    print(f"   Title: {article['title']}")
    print(f"   Category: {article['category']}")
    print(f"   Word Count: {article['word_count']}")
    print(f"   URL: {article['url']}")
    print()
    print(f"📄 Content Preview:")
    print(f"   {article['content'][:200]}...")
    print()
    
    # Initialize Llama
    print("🤖 Initializing Llama 3.2-3B (this may take a moment)...")
    try:
        from summarizer.llama_summarizer import LlamaSummarizer
        
        start_time = datetime.now()
        summarizer = LlamaSummarizer()
        init_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Llama loaded in {init_time:.1f} seconds")
        print()
        
    except Exception as e:
        print(f"❌ Error loading Llama: {e}")
        return
    
    # Generate TikTok summary
    print("🎬 Generating TikTok summary...")
    try:
        start_gen = datetime.now()
        
        summary = summarizer.summarize_for_tiktok(
            title=article['title'],
            content=article['content'],
            category=article['category']
        )
        
        gen_time = (datetime.now() - start_gen).total_seconds()
        
        print(f"✅ Generated in {gen_time:.1f} seconds")
        print()
        
        # Show result
        print("🎯 GENERATED TIKTOK SCRIPT:")
        print("="*50)
        print(summary)
        print("="*50)
        print()
        
        # Show stats
        print("📊 PERFORMANCE STATS:")
        print(f"   Initialization: {init_time:.1f}s")
        print(f"   Generation: {gen_time:.1f}s")
        print(f"   Total time: {init_time + gen_time:.1f}s")
        print(f"   Output length: {len(summary)} characters")
        print(f"   Reading time: ~{len(summary)/15:.1f} seconds")
        print()
        
        # Save result
        result = {
            'demo_timestamp': datetime.now().isoformat(),
            'article': {
                'title': article['title'],
                'category': article['category'],
                'url': article['url'],
                'word_count': article['word_count']
            },
            'tiktok_summary': summary,
            'performance': {
                'init_time_seconds': init_time,
                'generation_time_seconds': gen_time,
                'total_time_seconds': init_time + gen_time,
                'output_characters': len(summary)
            }
        }
        
        output_file = f"data/demo_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Demo result saved to: {output_file}")
        print()
        print("🎉 SUCCESS! TikTok Automata is working perfectly!")
        print("   Ready for production use with real TLDR articles.")
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")

if __name__ == "__main__":
    demo_real_article()
