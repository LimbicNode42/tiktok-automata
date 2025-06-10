#!/usr/bin/env python3
"""
Comprehensive test script for TLDR Newsletter RSS scraping functionality.
Includes detailed analytics, success rate analysis, and failure reporting.
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


def analyze_extraction_results(articles):
    """Analyze extraction success rates and provide detailed statistics."""
    total_articles = len(articles)
    if total_articles == 0:
        return {}
    
    # Count status types
    success_count = 0
    partial_count = 0
    failed_count = 0
    
    # Track detailed failure/partial info
    failed_articles = []
    partial_articles = []
    
    for article in articles:
        status = article.content_extraction_status
        if status == 'success':
            success_count += 1
        elif status == 'partial':
            partial_count += 1
            partial_articles.append({
                'title': article.title,
                'url': article.url,
                'reason': article.failure_reason or 'Unknown partial extraction issue',
                'word_count': article.word_count
            })
        elif status == 'failed':
            failed_count += 1
            failed_articles.append({
                'title': article.title,
                'url': article.url,
                'reason': article.failure_reason or 'Unknown extraction failure'
            })
    
    # Calculate percentages
    success_rate = (success_count / total_articles) * 100
    partial_rate = (partial_count / total_articles) * 100
    failure_rate = (failed_count / total_articles) * 100
    
    return {
        'total_articles': total_articles,
        'success_count': success_count,
        'partial_count': partial_count,
        'failed_count': failed_count,
        'success_rate': round(success_rate, 1),
        'partial_rate': round(partial_rate, 1),
        'failure_rate': round(failure_rate, 1),
        'failed_articles': failed_articles,
        'partial_articles': partial_articles
    }


async def test_data_output():
    """Test data output functionality with enhanced analytics."""
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
            
            # Analyze extraction results
            analysis = analyze_extraction_results(articles)
            
            # Print summary statistics
            print(f"\n=== Extraction Success Analysis ===")
            print(f"Total Articles: {analysis['total_articles']}")
            print(f"Success Rate: {analysis['success_rate']}% ({analysis['success_count']} articles)")
            print(f"Partial Rate: {analysis['partial_rate']}% ({analysis['partial_count']} articles)")
            print(f"Failure Rate: {analysis['failure_rate']}% ({analysis['failed_count']} articles)")
            
            # Print detailed failure information
            if analysis['failed_articles']:
                print(f"\n=== Failed Extractions ({len(analysis['failed_articles'])}) ===")
                for i, failed in enumerate(analysis['failed_articles'], 1):
                    print(f"{i}. {failed['title']}")
                    print(f"   URL: {failed['url']}")
                    print(f"   Reason: {failed['reason']}")
                    print()
            
            # Print detailed partial extraction information
            if analysis['partial_articles']:
                print(f"=== Partial Extractions ({len(analysis['partial_articles'])}) ===")
                for i, partial in enumerate(analysis['partial_articles'], 1):
                    print(f"{i}. {partial['title']}")
                    print(f"   URL: {partial['url']}")
                    print(f"   Words Extracted: {partial['word_count']}")
                    print(f"   Reason: {partial['reason']}")
                    print()
            
            # Save to JSON file with enhanced metadata
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
                    'word_count': article.word_count,
                    'content_extraction_status': article.content_extraction_status,
                    'failure_reason': article.failure_reason
                })
            
            # Create final output with metadata
            output_data = {
                'metadata': {
                    'extraction_timestamp': datetime.now().isoformat(),
                    'scraper_version': '1.0',
                    'extraction_statistics': analysis
                },
                'articles': articles_data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
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
    """Main comprehensive test function."""
    print("TLDR Newsletter Scraper - Comprehensive Test")
    print("=" * 50)
    
    await test_rss_scraping()
    await test_article_processing() 
    await test_data_output()    
    print("\n" + "=" * 50)
    print("Comprehensive testing complete!")
    print("📊 Features tested: RSS parsing, content extraction, success analytics, failure reporting")


if __name__ == "__main__":
    asyncio.run(main())
