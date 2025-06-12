#!/usr/bin/env python3
"""
TLDR Newsletter Scraper - Main Entry Point

Simple scraper for testing RSS and email approaches.
"""

import asyncio
import sys
from pathlib import Path
from typing import List
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger

from src.scraper.newsletter_scraper import NewsletterScraper, Article
from src.utils.config import config

# Load environment variables
load_dotenv()


def setup_logging():
    """Configure logging for the application."""
    log_level = "INFO"
    
    # Create logs directory
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        config.logs_dir / "scraper.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="7 days"
    )


def save_articles_to_json(articles: List[Article], filename: str = "articles.json"):
    """Save articles to JSON file for inspection."""
    articles_data = []
    for article in articles:
        articles_data.append({
            "title": article.title,
            "content": article.content[:500] + "..." if len(article.content) > 500 else article.content,
            "summary": article.summary,
            "url": article.url,
            "published_date": article.published_date.isoformat(),
            "category": article.category,
            "word_count": article.word_count
        })
    
    # Save to scraper data directory
    scraper_data_dir = Path("src/scraper/data")
    scraper_data_dir.mkdir(parents=True, exist_ok=True)
    output_file = scraper_data_dir / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved articles to {output_file}")


async def test_rss_scraping():
    """Test RSS-based scraping approach."""
    logger.info("=== Testing RSS Scraping Approach ===")
    
    try:
        scraper = NewsletterScraper()
        articles = await scraper.fetch_latest_newsletter()
        
        if not articles:
            logger.warning("No articles found via RSS")
            return []
        
        logger.success(f"RSS: Found {len(articles)} articles")
        
        # Show details of first few articles
        for i, article in enumerate(articles[:3]):
            logger.info(f"Article {i+1}: {article.title}")
            logger.info(f"  Category: {article.category}")
            logger.info(f"  Word count: {article.word_count}")
            logger.info(f"  Published: {article.published_date}")
            logger.info(f"  URL: {article.url}")
        
        save_articles_to_json(articles, "rss_articles.json")
        return articles
        
    except Exception as e:
        logger.error(f"RSS scraping failed: {str(e)}")
        return []


async def test_email_scraping():
    """Test email-based scraping approach."""
    logger.info("=== Testing Email Scraping Approach ===")
    
    # Check if email is configured
    if not config.email.username or not config.email.password:
        logger.warning("Email credentials not configured. Skipping email test.")
        logger.info("To test email scraping, set EMAIL_USERNAME and EMAIL_PASSWORD in .env")
        return []
    
    try:
        # This would be implemented later
        logger.info("Email scraping not yet implemented")
        return []
        
    except Exception as e:
        logger.error(f"Email scraping failed: {str(e)}")
        return []


async def compare_approaches():
    """Compare RSS vs Email approaches."""
    logger.info("=== Comparing RSS vs Email Approaches ===")
    
    # Test RSS
    rss_articles = await test_rss_scraping()
    
    # Test Email (placeholder for now)
    email_articles = await test_email_scraping()
    
    # Analysis
    logger.info("\n=== Analysis ===")
    logger.info(f"RSS approach: {len(rss_articles)} articles")
    logger.info(f"Email approach: {len(email_articles)} articles")
    
    if rss_articles:
        logger.info("\nRSS Pros:")
        logger.info("  ✓ Public access, no credentials needed")
        logger.info("  ✓ Standard format, reliable parsing")
        logger.info("  ✓ Real-time updates")
        logger.info("  ✓ Can access historical data")
        
        logger.info("\nRSS Cons:")
        logger.info("  - May have limited content in feed")
        logger.info("  - Dependent on TLDR maintaining RSS feed")
        logger.info("  - May require additional web scraping for full content")
    
    logger.info("\nEmail Pros:")
    logger.info("  ✓ Full newsletter content")
    logger.info("  ✓ Rich formatting preserved")
    logger.info("  ✓ Exactly what subscribers receive")
    
    logger.info("\nEmail Cons:")
    logger.info("  - Requires email subscription and credentials")
    logger.info("  - More complex authentication")
    logger.info("  - Dependent on email delivery timing")
    logger.info("  - May need email parsing logic")


async def main():
    """Main entry point."""
    setup_logging()
    
    logger.info("TLDR Newsletter Scraper Starting...")
    logger.info(f"Config: {config.tldr_rss_url}")
    
    # Run comparison
    await compare_approaches()


if __name__ == "__main__":
    asyncio.run(main())
