#!/usr/bin/env python3
"""
Quick test of one real TLDR article (non-blocking)
"""

import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def quick_real_test():
    """Quick test with immediate output"""
    print("🚀 QUICK REAL ARTICLE TEST")
    print("="*40)
    
    # Load one article
    with open("data/tldr_articles_20250610_231259.json", 'r') as f:
        data = json.load(f)
    
    # Find first good article
    article = None
    for a in data['articles']:
        if (a.get('content_extraction_status') == 'success' and 
            len(a.get('content', '')) > 200):
            article = a
            break
    
    if not article:
        print("❌ No suitable article found")
        return
    
    print(f"📰 Article: {article['title'][:50]}...")
    print(f"🏷️ Category: {article['category']}")
    print(f"📊 Words: {article['word_count']}")
    print(f"🔗 URL: {article['url']}")
    
    print(f"\n📄 Content preview:")
    print(f"{article['content'][:200]}...")
    
    print(f"\n✅ Article loaded successfully!")
    print(f"📋 Ready for Llama processing")
    
    # Save article for easy access
    test_article = {
        'title': article['title'],
        'content': article['content'],
        'category': article['category'],
        'url': article['url'],
        'word_count': article['word_count']
    }
    
    with open('data/test_article_ready.json', 'w') as f:
        json.dump(test_article, f, indent=2)
    
    print(f"💾 Test article saved to: data/test_article_ready.json")

if __name__ == "__main__":
    quick_real_test()
