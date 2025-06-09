"""
Configuration utilities for TLDR Newsletter Scraper.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ScrapingConfig:
    """Scraping configuration settings."""
    max_age_hours: int = 24
    max_articles_per_run: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class Config:
    """Central configuration manager."""
    
    def __init__(self):
        self.scraping = ScrapingConfig(
            max_age_hours=int(os.getenv("MAX_AGE_HOURS", 24)),
            max_articles_per_run=int(os.getenv("MAX_ARTICLES_PER_RUN", 10)),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", 30)),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", 3)),
            user_agent=os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        )
        
        # API endpoints
        self.tldr_rss_url = os.getenv("TLDR_RSS_FEED", "https://tldr.tech/rss")
        self.tldr_base_url = os.getenv("TLDR_NEWSLETTER_URL", "https://tldr.tech/")
        
        # Paths
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./data"))
        self.logs_dir = Path(os.getenv("LOGS_DIR", "./logs"))
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for scraping requests."""
        return {
            'User-Agent': self.scraping.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'scraping': {
                'max_age_hours': self.scraping.max_age_hours,
                'max_articles_per_run': self.scraping.max_articles_per_run,
                'request_timeout': self.scraping.request_timeout,
                'retry_attempts': self.scraping.retry_attempts,
                'user_agent': self.scraping.user_agent
            },
            'urls': {
                'tldr_rss': self.tldr_rss_url,
                'tldr_base': self.tldr_base_url
            },
            'paths': {
                'output_dir': str(self.output_dir),
                'logs_dir': str(self.logs_dir)
            }
        }


# Global config instance
config = Config()

if __name__ == "__main__":
    print("Config loaded successfully!")
    print(f"RSS URL: {config.tldr_rss_url}")
    print(f"Output dir: {config.output_dir}")
