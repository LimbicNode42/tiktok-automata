#!/usr/bin/env python3
"""
Test script for RSS scraping functionality.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from scraper.newsletter_scraper import NewsletterScraper, Article
    print("✓ Successfully imported NewsletterScraper and Article")
except ImportError as e:
    print(f"✗ Failed to import NewsletterScraper: {e}")
    sys.exit(1)


async def test_rss_scraping():
    """Test RSS scraping functionality."""
    print("\n=== Testing RSS Scraping ===")
    
    # Initialize scraper
    scraper = NewsletterScraper()
    print("✓ NewsletterScraper initialized")
    
    try:
        # Test RSS feed parsing
        print("\n1. Testing RSS feed parsing...")
        rss_entries = await scraper.fetch_rss_feed()
        
        if rss_entries:
            print(f"✓ Found {len(rss_entries)} RSS entries")
            
            # Display first RSS entry details
            first_entry = rss_entries[0]
            print(f"\nFirst RSS entry preview:")
            print(f"  Title: {first_entry.get('title', 'N/A')[:80]}...")
            print(f"  Link: {first_entry.get('link', 'N/A')}")
            print(f"  Published: {first_entry.get('published', 'N/A')}")
            
        else:
            print("✗ No RSS entries found")
            
        # Close the scraper session
        await scraper.close()
            
    except Exception as e:
        print(f"✗ RSS scraping failed: {e}")
        import traceback
        traceback.print_exc()


async def test_article_processing():
    """Test article content processing."""
    print("\n=== Testing Article Processing ===")
    
    scraper = NewsletterScraper()
    
    try:
        # Get latest newsletter articles
        print("Fetching latest newsletter articles...")
        articles = await scraper.fetch_latest_newsletter(max_age_hours=48)  # 48 hours for testing
        
        if not articles:
            print("✗ No articles to process")
            return
            
        print(f"✓ Found {len(articles)} recent articles")
        
        # Display details of first article
        if articles:
            article = articles[0]
            print(f"\nFirst article details:")
            print(f"  Title: {article.title}")
            print(f"  Category: {article.category}")
            print(f"  Word Count: {article.word_count}")
            print(f"  Published: {article.published_date}")
            print(f"  URL: {article.url}")
            
            if article.summary:
                print(f"  Summary: {article.summary[:200]}...")
            if article.content:
                print(f"  Content: {article.content[:300]}...")
                
        # Close the scraper session
        await scraper.close()
                
    except Exception as e:
        print(f"✗ Article processing failed: {e}")
        import traceback
        traceback.print_exc()


async def test_data_output():
    """Test data output functionality."""
    print("\n=== Testing Data Output ===")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    scraper = NewsletterScraper()
    
    try:
        # Test full scraping and save
        print("Fetching and processing articles...")
        articles = await scraper.fetch_latest_newsletter(max_age_hours=48)
        
        if articles:
            print(f"✓ Processed {len(articles)} articles")
            
            # Save to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tldr_articles_{timestamp}.json"
            filepath = data_dir / filename
            
            # Convert articles to dict format for JSON serialization
            articles_data = []
            for article in articles:
                articles_data.append({
                    'title': article.title,
                    'content': article.content,
                    'summary': article.summary,
                    'url': article.url,
                    'published_date': article.published_date.isoformat(),
                    'category': article.category,
                    'word_count': article.word_count
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(articles_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Output file created: {filepath}")
            
            # Check file size
            file_size = filepath.stat().st_size
            print(f"✓ File size: {file_size} bytes")
            
        else:
            print("✗ No articles to save")
            
        # Close the scraper session
        await scraper.close()
            
    except Exception as e:
        print(f"✗ Data output test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("TLDR Newsletter Scraper Test")
    print("=" * 40)
    
    await test_rss_scraping()
    await test_article_processing() 
    await test_data_output()
    
    print("\n" + "=" * 40)
    print("Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())
