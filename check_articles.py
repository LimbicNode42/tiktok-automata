#!/usr/bin/env python3
"""
Quick test to check the TLDR articles data
"""

import json
import sys
import os

def check_articles():
    """Check what articles we have available."""
    print("📰 Checking TLDR Articles Data")
    print("=" * 50)
    
    try:
        with open("data/tldr_articles_20250610_231259.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total articles in file: {len(data['articles'])}")
        
        # Count by status
        successful = [a for a in data['articles'] if a.get('content_extraction_status') == 'success']
        failed = [a for a in data['articles'] if a.get('content_extraction_status') == 'failed']
        partial = [a for a in data['articles'] if a.get('content_extraction_status') == 'partial']
        
        print(f"✅ Successful extractions: {len(successful)}")
        print(f"❌ Failed extractions: {len(failed)}")
        print(f"⚠️  Partial extractions: {len(partial)}")
        
        # Show successful articles with good content
        good_articles = [
            a for a in successful 
            if a.get('word_count', 0) > 100 and 
               len(a.get('content', '')) > 500
        ]
        
        print(f"\n🎯 Articles ready for processing: {len(good_articles)}")
        
        if good_articles:
            print("\nTop articles to process:")
            for i, article in enumerate(good_articles[:5], 1):
                title = article['title'][:60] + "..." if len(article['title']) > 60 else article['title']
                print(f"{i}. {title}")
                print(f"   Category: {article.get('category', 'unknown')}")
                print(f"   Words: {article.get('word_count', 0)}")
                print()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_articles()
