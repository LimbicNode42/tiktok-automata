#!/usr/bin/env python3
"""
Quick test to load and check TLDR articles
"""

import json
import os

def main():
    print("="*60)
    print("LOADING TLDR ARTICLES - QUICK TEST")
    print("="*60)
    
    input_file = "data/tldr_articles_20250610_231259.json"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📂 Loading articles from: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data.get('articles', [])
        print(f"📊 Total articles in file: {len(articles)}")
        
        # Filter for successfully extracted articles
        successful_articles = []
        for article in articles:
            if (article.get('content_extraction_status') == 'success' and 
                article.get('content') and 
                len(article.get('content', '')) > 100):
                successful_articles.append(article)
        
        print(f"✅ Successfully extracted articles: {len(successful_articles)}")
        
        # Show categories
        categories = {}
        for article in successful_articles:
            cat = article.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n📋 Article categories:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} articles")
        
        # Show sample articles
        print(f"\n📝 Sample articles (first 3):")
        for i, article in enumerate(successful_articles[:3], 1):
            print(f"\n{i}. Title: {article['title'][:60]}...")
            print(f"   URL: {article['url']}")
            print(f"   Category: {article['category']}")
            print(f"   Word count: {article['word_count']}")
            print(f"   Content preview: {article['content'][:100]}...")
        
        print(f"\n✅ Articles loaded successfully! Ready for summarization.")
        
    except Exception as e:
        print(f"❌ Error loading articles: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
