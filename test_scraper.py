#!/usr/bin/env python3
"""
Test script for TLDR Newsletter RSS scraping functionality.
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

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
    import os
    
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Create logs directory if it doesn't exist
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
    )
    logger.add(
        config.logs_dir / "scraper.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        rotation="1 day",
        retention="30 days"
    )


def save_articles_to_json(articles: list[Article], filename: str = None):
    """Save articles to JSON file for analysis."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tldr_articles_{timestamp}.json"
    
    filepath = config.output_dir / filename
    
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
    
    logger.info(f"Saved {len(articles)} articles to {filepath}")
    return filepath


async def test_rss_scraping():
    """Test RSS scraping functionality."""
    logger.info("Starting RSS scraping test...")
    
    try:
        # Initialize scraper
        scraper = NewsletterScraper()
        
        logger.info("Fetching RSS feed...")
        rss_entries = await scraper.fetch_rss_feed()
        
        if not rss_entries:
            logger.error("No RSS entries found!")
            return
        
        logger.success(f"Found {len(rss_entries)} RSS entries")
        
        # Show some basic info about the RSS entries
        logger.info("RSS Feed Overview:")
        for i, entry in enumerate(rss_entries[:5], 1):
            title = entry.get('title', 'No title')
            link = entry.get('link', 'No link')
            logger.info(f"  {i}. {title}")
            logger.info(f"     URL: {link}")
        
        logger.info("Fetching latest newsletter articles...")
        articles = await scraper.fetch_latest_newsletter(max_age_hours=config.scraping.max_age_hours)
        
        if not articles:
            logger.warning("No recent articles found")
            return
        
        logger.success(f"Successfully processed {len(articles)} articles")
        
        # Display article details
        logger.info("Article Details:")
        for i, article in enumerate(articles, 1):
            logger.info(f"  Article {i}:")
            logger.info(f"    Title: {article.title}")
            logger.info(f"    Category: {article.category}")
            logger.info(f"    Word Count: {article.word_count}")
            logger.info(f"    Published: {article.published_date}")
            logger.info(f"    URL: {article.url}")
            if article.summary:
                logger.info(f"    Summary: {article.summary[:100]}...")
            if article.content:
                logger.info(f"    Content: {article.content[:200]}...")
            logger.info("")
        
        # Save to JSON file
        json_file = save_articles_to_json(articles)
        logger.success(f"Test completed! Data saved to {json_file}")
        
        # Close the scraper
        await scraper.close()
        
    except Exception as e:
        logger.error(f"Error during RSS scraping test: {str(e)}")
        raise


async def test_configuration():
    """Test configuration loading."""
    logger.info("Testing configuration...")
    
    logger.info("Configuration settings:")
    config_dict = config.to_dict()
    for section, settings in config_dict.items():
        logger.info(f"  {section}:")
        for key, value in settings.items():
            logger.info(f"    {key}: {value}")
    
    logger.success("Configuration loaded successfully!")


async def main():
    """Main test function."""
    setup_logging()
    
    logger.info("=== TLDR Newsletter Scraper Test ===")
    
    # Test 1: Configuration
    await test_configuration()
    logger.info("")
    
    # Test 2: RSS Scraping
    await test_rss_scraping()
    
    logger.success("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
