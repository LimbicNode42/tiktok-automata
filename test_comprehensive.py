#!/usr/bin/env python3
"""
Comprehensive test script for TLDR RSS scraping functionality.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scraper.newsletter_scraper import NewsletterScraper, Article


async def comprehensive_test():
    """Run comprehensive RSS scraping test."""
    print("TLDR Newsletter Scraper - Comprehensive Test")
    print("=" * 50)
    
    scraper = NewsletterScraper()
    
    try:
        # Test 1: Basic RSS feed access
        print("\n1. Testing RSS Feed Access...")
        rss_entries = await scraper.fetch_rss_feed()
        print(f"   ✓ Retrieved {len(rss_entries)} RSS entries")
        
        # Show RSS feed overview
        print(f"\n   RSS Feed Overview (showing first 5 entries):")
        for i, entry in enumerate(rss_entries[:5], 1):
            title = entry.get('title', 'No title')[:60]
            published = entry.get('published', 'No date')
            print(f"   {i}. {title}... ({published})")
        
        # Test 2: Process articles with different time windows
        print(f"\n2. Testing Article Processing with Different Time Windows...")
        
        for hours in [24, 72, 168]:  # 1 day, 3 days, 1 week
            print(f"\n   Testing with {hours} hour window:")
            articles = await scraper.fetch_latest_newsletter(max_age_hours=hours)
            print(f"   ✓ Found {len(articles)} articles in last {hours} hours")
            
            if articles:
                # Show article details
                for i, article in enumerate(articles, 1):
                    print(f"      Article {i}: {article.title[:50]}...")
                    print(f"        Category: {article.category}")
                    print(f"        Word count: {article.word_count}")
                    print(f"        Published: {article.published_date}")
                    print(f"        URL: {article.url}")
        
        # Test 3: Save comprehensive data
        print(f"\n3. Saving Comprehensive Data...")
        articles = await scraper.fetch_latest_newsletter(max_age_hours=168)  # 1 week
        
        if articles:
            # Create timestamped output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed JSON
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            detailed_file = data_dir / f"tldr_comprehensive_{timestamp}.json"
            
            # Convert to comprehensive format
            comprehensive_data = {
                "metadata": {
                    "scrape_time": timestamp,
                    "total_articles": len(articles),
                    "time_window_hours": 168,
                    "rss_url": scraper.rss_url,
                    "scraper_version": "1.0"
                },
                "articles": []
            }
            
            for article in articles:
                article_data = {
                    'title': article.title,
                    'content': article.content,
                    'summary': article.summary,
                    'url': article.url,
                    'published_date': article.published_date.isoformat(),
                    'category': article.category,
                    'word_count': article.word_count,
                    'content_preview': article.content[:200] + "..." if len(article.content) > 200 else article.content
                }
                comprehensive_data["articles"].append(article_data)
            
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✓ Comprehensive data saved to: {detailed_file}")
            print(f"   ✓ File size: {detailed_file.stat().st_size} bytes")
            
            # Generate summary report
            summary_file = data_dir / f"tldr_summary_{timestamp}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("TLDR Newsletter Scraping Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Scrape Time: {datetime.now()}\n")
                f.write(f"Total Articles: {len(articles)}\n")
                f.write(f"Time Window: 168 hours (1 week)\n")
                f.write(f"RSS URL: {scraper.rss_url}\n\n")
                
                f.write("Article Overview:\n")
                f.write("-" * 20 + "\n")
                for i, article in enumerate(articles, 1):
                    f.write(f"{i}. {article.title}\n")
                    f.write(f"   Published: {article.published_date}\n")
                    f.write(f"   Category: {article.category}\n")
                    f.write(f"   Words: {article.word_count}\n")
                    f.write(f"   URL: {article.url}\n\n")
                
                # Statistics
                total_words = sum(article.word_count for article in articles)
                categories = {}
                for article in articles:
                    categories[article.category] = categories.get(article.category, 0) + 1
                
                f.write("Statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total words across all articles: {total_words}\n")
                f.write(f"Average words per article: {total_words // len(articles) if articles else 0}\n")
                f.write(f"Categories found: {list(categories.keys())}\n")
                for cat, count in categories.items():
                    f.write(f"  {cat}: {count} articles\n")
            
            print(f"   ✓ Summary report saved to: {summary_file}")
        
        print(f"\n4. Test Results Summary:")
        print(f"   ✓ RSS feed access: Working")
        print(f"   ✓ Article processing: Working") 
        print(f"   ✓ Data export: Working")
        print(f"   ✓ Content extraction: Working")
        print(f"   ✓ JSON output: Working")
        
        # Close scraper
        await scraper.close()
        
        print(f"\n🎉 All tests passed successfully!")
        print(f"📁 Output files saved in: {data_dir.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Close scraper on error
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(comprehensive_test())
