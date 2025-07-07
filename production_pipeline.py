#!/usr/bin/env python3
"""
TikTok Automata Production Pipeline
===================================

A complete production script that runs the entire TikTok content creation pipeline:
1. Fetch latest articles (avoiding duplicates)
2. Summarize new articles with AI
3. Generate TTS audio for summaries
4. Download latest gaming videos (avoiding duplicates)
5. Analyze videos for best action segments
6. Generate final TikTok videos with all effects

Features:
- Randomized video/segment selection (no duplicates per day)
- Randomized subtitle styles
- TTS-length-aware video segments with buffer
- Duplicate prevention for articles and videos
- Initial batch setup for new installations
- Production logging and error handling

Usage:
    python production_pipeline.py                    # Daily production run
    python production_pipeline.py --initial-setup   # First-time batch setup
    python production_pipeline.py --dry-run         # Test run without output
"""

import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scraper.newsletter_scraper import NewsletterScraper
from summarizer.llama_summarizer import LlamaSummarizer
from tts.kokoro_tts import KokoroTTSEngine, TTSConfig
from video.managers.footage_manager import FootageManager
from video.processors.video_processor import VideoConfig
from video import VideoActionAnalyzer
from utils.config import config

# Configure production logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TikTokAutomata")

def safe_log_message(message: str) -> str:
    """Sanitize log messages for Windows console compatibility."""
    try:
        # Try to encode to ASCII to check for Unicode issues
        message.encode('ascii')
        return message
    except UnicodeEncodeError:
        # Replace problematic characters with ASCII equivalents
        replacements = {
            '✅': '[OK]',
            '⚠️': '[WARN]',
            '❌': '[ERROR]',
            '📅': '[DATE]',
            '📹': '[VIDEO]',
            '📄': '[FILE]',
            '🚨': '[ALERT]',
            '→': '->'
        }
        sanitized = message
        for emoji, replacement in replacements.items():
            sanitized = sanitized.replace(emoji, replacement)
        return sanitized

def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS by removing/replacing problematic characters."""
    if not text:
        return text
    
    # Remove or replace Unicode characters that cause issues
    replacements = {
        # Smart quotes
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        
        # Special characters
        '–': '-',  # en dash
        '—': '-',  # em dash
        '…': '...',  # ellipsis
        
        # Accented characters (if they cause TTS issues)
        'é': 'e',
        'ñ': 'n',
        'ü': 'u',
        'ä': 'a',
        'ö': 'o',
        
        # Remove emojis for TTS clarity
        '🚨': '',
        '💰': '',
        '🤖': '',
        '📝': '',
        '🎯': '',
        '⚡': '',
        '🔥': '',
        '💻': '',
        '🎮': '',
        '📱': '',
        '🌟': '',
        '🚀': '',
        '💡': '',
        
        # Common problematic Unicode sequences
        '\u00e9': 'e',  # é
        '\u00f1': 'n',  # ñ
        '\ud83d\udea8': '',  # 🚨 emoji
    }
    
    cleaned = text
    for char, replacement in replacements.items():
        cleaned = cleaned.replace(char, replacement)
    
    # Remove any remaining non-ASCII characters that might cause issues
    cleaned = cleaned.encode('ascii', errors='ignore').decode('ascii')
    
    # Clean up multiple spaces
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

@dataclass
class ProductionConfig:
    """Configuration for production pipeline."""
      # Storage paths
    storage_dir: str = "./storage"
    state_file: str = "./storage/production_state.json"
    # Processing limits (optional - if None, no limits applied)
    max_articles_per_run: Optional[int] = None
    max_videos_per_run: Optional[int] = None
    
    # Fallback options
    fallback_enabled: bool = True  # Enable fallback to older articles when no new ones found
    fallback_max_age_hours: int = 168  # 1 week for fallback articles
    fallback_max_articles: int = 10  # Maximum fallback articles to process
    
    # Video configuration
    video_buffer_seconds: float = 2.0  # Buffer on each side of TTS length
    min_video_duration: int = 15  # Minimum segment duration
    max_video_duration: int = 60  # Maximum segment duration
    
    # RSS feeds for article sources
    rss_feeds: List[Dict[str, str]] = None
    
    # Gaming video sources for downloading (channels or individual videos)
    gaming_video_sources: List[Dict[str, str]] = None
    
    # Subtitle styles to randomize
    subtitle_styles: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.rss_feeds is None:
            self.rss_feeds = [
                {"url": "https://tldr.tech/rss", "name": "TLDR Tech", "category": "tech"}
            ]
        
        if self.gaming_video_sources is None:
            self.gaming_video_sources = [
                {"url": "https://www.youtube.com/@GameGrumps/videos", "name": "Game Grumps", "content_type": "high_action", "max_videos": 20}
            ]
        
        if self.subtitle_styles is None:
            self.subtitle_styles = ["default", "modern", "bold", "minimal", "highlight"]

@dataclass
class ProductionState:
    """Tracks what has been processed and points to output files."""
    processed_article_urls: Set[str]
    downloaded_video_urls: Set[str]
    last_run_date: str
    daily_video_assignments: Dict[str, str]  # article_id -> video_id mapping for today
      # Output file tracking - maps URLs to file paths
    article_content_files: Dict[str, str]  # url -> content.json path
    article_summary_files: Dict[str, str]  # url -> summary.json path  
    article_tts_files: Dict[str, str]  # url -> audio.wav path
    article_final_videos: Dict[str, str]  # url -> final_video.mp4 path
    
    @classmethod
    def load(cls, state_file: str) -> "ProductionState":
        """Load production state from file."""
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            return cls(
                processed_article_urls=set(data.get('processed_article_urls', [])),
                downloaded_video_urls=set(data.get('downloaded_video_urls', [])),
                last_run_date=data.get('last_run_date', ''),
                daily_video_assignments=data.get('daily_video_assignments', {}),
                article_content_files=data.get('article_content_files', {}),
                article_summary_files=data.get('article_summary_files', {}),
                article_tts_files=data.get('article_tts_files', {}),
                article_final_videos=data.get('article_final_videos', {})
            )
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing state found, starting fresh")
            return cls(
                processed_article_urls=set(),
                downloaded_video_urls=set(),
                last_run_date='',
                daily_video_assignments={},
                article_content_files={},
                article_summary_files={},
                article_tts_files={},
                article_final_videos={}
            )
    
    def save(self, state_file: str):
        """Save production state to file."""
        data = {
            'processed_article_urls': list(self.processed_article_urls),
            'downloaded_video_urls': list(self.downloaded_video_urls),
            'last_run_date': self.last_run_date,
            'daily_video_assignments': self.daily_video_assignments,
            'article_content_files': self.article_content_files,
            'article_summary_files': self.article_summary_files,
            'article_tts_files': self.article_tts_files,
            'article_final_videos': self.article_final_videos
        }
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Production state saved to {state_file}")    
    def save_article_content(self, article: Dict, storage_dir: str) -> str:
        """Save article content to file and track in state."""
        import hashlib
        import os
        
        url = article.get('url')
        if not url:
            return None
            
        # Create content filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        safe_title = "".join(c for c in article.get('title', 'unknown')[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"content_{url_hash}_{safe_title.replace(' ', '_')}.json"
        
        # Ensure content directory exists
        content_dir = Path(storage_dir) / "content"
        content_dir.mkdir(exist_ok=True)
        
        filepath = content_dir / filename        # Save content with metadata
        content_data = {
            'url': url,
            'title': article.get('title'),
            'content': article.get('content'),
            'published_date': str(article.get('published_date')) if article.get('published_date') else None,
            'category': article.get('category'),
            'word_count': article.get('word_count'),
            'content_extraction_status': article.get('content_extraction_status'),
            'failure_reason': article.get('failure_reason'),
            'feed_name': article.get('feed_name'),
            'scraped_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, indent=2)
        
        # Track in state
        self.article_content_files[url] = str(filepath)
        self.processed_article_urls.add(url)
        
        return str(filepath)
    
    def save_article_summary(self, article: Dict, summary: str, storage_dir: str) -> str:
        """Save article summary to file and track in state."""
        import hashlib
        
        url = article.get('url')
        if not url or not summary:
            return None
            
        # Create summary filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        safe_title = "".join(c for c in article.get('title', 'unknown')[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"summary_{url_hash}_{safe_title.replace(' ', '_')}.json"
        
        # Ensure summaries directory exists
        summary_dir = Path(storage_dir) / "summaries"
        summary_dir.mkdir(exist_ok=True)
        
        filepath = summary_dir / filename
        
        # Save summary with metadata
        summary_data = {
            'url': url,
            'title': article.get('title'),
            'original_content': article.get('content'),
            'tiktok_summary': summary,
            'summary_length': len(summary),
            'summary_words': len(summary.split()),
            'summarized_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)
        
        # Track in state
        self.article_summary_files[url] = str(filepath)
        
        return str(filepath)
    
    def save_article_tts(self, article: Dict, tts_path: str, duration: float, voice: str, storage_dir: str) -> str:
        """Move TTS file to organized location and track in state."""
        import hashlib
        import shutil
        
        url = article.get('url')
        if not url or not tts_path or not Path(tts_path).exists():
            return None
            
        # Create TTS filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        safe_title = "".join(c for c in article.get('title', 'unknown')[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"tts_{url_hash}_{safe_title.replace(' ', '_')}.wav"
        
        # Ensure TTS directory exists
        tts_dir = Path(storage_dir) / "tts"
        tts_dir.mkdir(exist_ok=True)
        
        filepath = tts_dir / filename
        
        # Move the TTS file to organized location
        shutil.move(tts_path, filepath)
        
        # Save metadata alongside
        metadata_file = filepath.with_suffix('.json')
        metadata = {
            'url': url,
            'title': article.get('title'),
            'tts_path': str(filepath),
            'duration_seconds': duration,
            'voice': voice,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Track in state
        self.article_tts_files[url] = str(filepath)
        
        return str(filepath)
    
    def save_article_final_video(self, article: Dict, video_path: str, storage_dir: str) -> str:
        """Move final video to organized location and track in state."""
        import hashlib
        import shutil
        
        url = article.get('url')
        if not url or not video_path or not Path(video_path).exists():
            return None
              # Create video filename from URL hash, article title, and source video
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        safe_title = "".join(c for c in article.get('title', 'unknown')[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        
        # Include source video name for traceability
        source_video = article.get('assigned_video_id', 'unknown_video')
        safe_video_name = "".join(c for c in source_video[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
        
        filename = f"final_{url_hash}_{safe_title.replace(' ', '_')}_from_{safe_video_name.replace(' ', '_')}.mp4"
        
        # Ensure videos directory exists
        videos_dir = Path(storage_dir) / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        filepath = videos_dir / filename
        
        # Move the video file to organized location
        shutil.move(video_path, filepath)
        
        # Save metadata alongside
        metadata_file = filepath.with_suffix('.json')
        metadata = {
            'url': url,
            'title': article.get('title'),
            'video_path': str(filepath),
            'tts_audio_path': article.get('tts_audio_path'),
            'tts_duration': article.get('tts_duration'),
            'assigned_video_id': article.get('assigned_video_id'),
            'subtitle_style': article.get('subtitle_style'),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Track in state
        self.article_final_videos[url] = str(filepath)
        
        return str(filepath)
    
    def has_article_content(self, url: str) -> bool:
        """Check if article content exists."""
        if url not in self.article_content_files:
            return False
        return Path(self.article_content_files[url]).exists()
    
    def has_article_summary(self, url: str) -> bool:
        """Check if article summary exists."""
        if url not in self.article_summary_files:
            return False
        return Path(self.article_summary_files[url]).exists()
    
    def has_article_tts(self, url: str) -> bool:
        """Check if article TTS exists."""
        if url not in self.article_tts_files:
            return False
        return Path(self.article_tts_files[url]).exists()
    
    def has_article_final_video(self, url: str) -> bool:
        """Check if article final video exists."""
        if url not in self.article_final_videos:
            return False
        return Path(self.article_final_videos[url]).exists()
    
    def load_article_content(self, url: str) -> Dict:
        """Load article content from file."""
        if not self.has_article_content(url):
            return None
        
        with open(self.article_content_files[url], 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_article_summary(self, url: str) -> Dict:
        """Load article summary from file."""
        if not self.has_article_summary(url):
            return None
        
        with open(self.article_summary_files[url], 'r', encoding='utf-8') as f:
            return json.load(f)    
    def get_processing_report(self) -> Dict:
        """Generate a detailed processing report."""
        total_content = len(self.article_content_files)
        total_summaries = len(self.article_summary_files)
        total_tts = len(self.article_tts_files)
        total_videos = len(self.article_final_videos)
        
        # Find articles that need processing
        content_urls = set(self.article_content_files.keys())
        summary_urls = set(self.article_summary_files.keys())
        tts_urls = set(self.article_tts_files.keys())
        video_urls = set(self.article_final_videos.keys())
        
        need_summary = content_urls - summary_urls
        need_tts = summary_urls - tts_urls
        need_video = tts_urls - video_urls
        
        # Check for successful vs failed content extractions
        successful_extractions = 0
        failed_extractions = 0
        
        for url, filepath in self.article_content_files.items():
            try:
                content_data = self.load_article_content(url)
                if content_data and content_data.get('content_extraction_status') == 'success':
                    successful_extractions += 1
                else:
                    failed_extractions += 1
            except:
                failed_extractions += 1
        
        return {
            'total_articles_scraped': total_content,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'total_articles_summarized': total_summaries,
            'total_articles_with_tts': total_tts,
            'total_final_videos': total_videos,
            'articles_needing_summary': len(need_summary),
            'articles_needing_tts': len(need_tts),
            'articles_needing_video': len(need_video),
            'fully_processed_articles': len(content_urls & summary_urls & tts_urls & video_urls),
            'last_run': self.last_run_date,
            'content_files': dict(list(self.article_content_files.items())[:3]),
            'summary_files': dict(list(self.article_summary_files.items())[:3]),
            'tts_files': dict(list(self.article_tts_files.items())[:3])
        }

class TikTokProductionPipeline:
    """Complete TikTok production pipeline."""
    
    def __init__(self, config: ProductionConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.state = ProductionState.load(config.state_file)
        
        # Initialize components
        self.scraper = None
        self.summarizer = None
        self.tts_engine = None
        self.footage_manager = None
          # Create directories
        Path(config.storage_dir).mkdir(exist_ok=True)
        # Create subdirectories for organized storage
        for subdir in ["content", "summaries", "tts", "videos"]:
            Path(config.storage_dir, subdir).mkdir(exist_ok=True)
        
        logger.info(f"TikTok Production Pipeline initialized (dry_run={dry_run})")
    
    async def initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # Initialize scraper
        self.scraper = NewsletterScraper()
          # Initialize summarizer
        self.summarizer = LlamaSummarizer()
        await self.summarizer.initialize()
        
        # Initialize TTS with optimized settings
        tts_config = TTSConfig(
            voice='af_heart',  # Default voice, will be randomized
            speed=config.get_tts_speed(),  # Use config TTS speed (1.35x)
            use_gpu=True
        )
        self.tts_engine = KokoroTTSEngine(tts_config)
        await self.tts_engine.initialize()          # Initialize footage manager
        self.footage_manager = FootageManager(storage_dir=Path(self.config.storage_dir))
        
        logger.info("All components initialized successfully")
    
    def get_already_downloaded_video_ids(self) -> set:
        """Get set of YouTube IDs that have already been downloaded."""
        storage_path = Path(self.config.storage_dir) / "raw"
        downloaded_ids = set()
        
        if storage_path.exists():
            for file_path in storage_path.glob("*.mp4"):
                # Extract YouTube ID from filename (format: {youtube_id}_{title}.mp4)
                filename = file_path.name
                if '_' in filename:
                    youtube_id = filename.split('_')[0]
                    downloaded_ids.add(youtube_id)
        
        logger.info(f"Found {len(downloaded_ids)} already downloaded videos")
        return downloaded_ids
    
    async def fetch_new_articles(self, initial_setup: bool = False) -> List[Dict]:
        """Fetch new articles from all configured RSS feeds."""
        # Set lookback period based on whether this is initial setup
        if initial_setup:
            max_age_hours = 24 * 7  # 1 week for initial setup
            max_entries = 20  # More entries for initial setup
            logger.info(f"INITIAL SETUP: Fetching articles from last {max_age_hours//24} days from {len(self.config.rss_feeds)} RSS feeds...")
        else:
            max_age_hours = 24  # 1 day for daily runs
            max_entries = 3  # Fewer entries for daily runs
            logger.info(f"DAILY RUN: Fetching articles from last {max_age_hours} hours from {len(self.config.rss_feeds)} RSS feeds...")
        
        all_articles = []
        
        for feed_config in self.config.rss_feeds:
            feed_url = feed_config["url"]
            feed_name = feed_config["name"]
            feed_category = feed_config.get("category", "general")
            
            try:
                logger.info(f"Fetching from {feed_name}: {feed_url} (looking back {max_age_hours//24} days)")
                
                # Temporarily set the scraper's RSS URL to this feed
                original_rss_url = self.scraper.rss_url
                self.scraper.rss_url = feed_url
                
                # Get articles from this feed with appropriate time range
                feed_articles = await self.scraper.fetch_latest_newsletter(
                    max_age_hours=max_age_hours,
                    max_entries=max_entries
                )
                  # Restore original RSS URL
                self.scraper.rss_url = original_rss_url
                
                # Convert Article objects to dictionaries and add feed metadata
                for article in feed_articles:
                    # Convert Article dataclass to dictionary
                    if hasattr(article, '__dict__'):
                        article_dict = {
                            'title': article.title,
                            'content': article.content,
                            'summary': article.summary,
                            'url': article.url,
                            'published_date': article.published_date,
                            'category': article.category,
                            'word_count': article.word_count,
                            'content_extraction_status': article.content_extraction_status,
                            'failure_reason': article.failure_reason
                        }
                    else:
                        article_dict = article  # Already a dict
                    
                    # Add feed metadata
                    article_dict["feed_name"] = feed_name
                    article_dict["feed_category"] = feed_category
                    
                    all_articles.append(article_dict)
                logger.info(f"Found {len(feed_articles)} articles from {feed_name}")
                
            except Exception as e:
                logger.error(f"Error fetching from {feed_name}: {e}")
                continue        # Filter articles and track processing state
        new_articles = []
        already_scraped = []
        
        for article in all_articles:
            url = article.get('url')
            if not self.state.has_article_content(url):
                # NEW ARTICLE: Save content and track regardless of extraction success
                content_file = self.state.save_article_content(article, self.config.storage_dir)
                
                # Only add to processing queue if content extraction was successful
                extraction_status = article.get('content_extraction_status', 'unknown')
                if extraction_status == 'success':
                    new_articles.append(article)
                    logger.info(safe_log_message(f"📄 New article (successful extraction): {article.get('title', 'Unknown')[:50]}..."))
                else:
                    # Content saved but marked as failed - don't process further
                    failure_reason = article.get('failure_reason', 'Unknown extraction failure')
                    logger.info(safe_log_message(f"⚠️ Article saved but extraction failed ({failure_reason}): {article.get('title', 'Unknown')[:50]}..."))
            else:
                # ALREADY ATTEMPTED: Skip this article entirely
                already_scraped.append(article)
                # Get the stored extraction status to provide better logging
                stored_content = self.state.load_article_content(url)
                stored_status = stored_content.get('content_extraction_status', 'unknown') if stored_content else 'unknown'
                if stored_status == 'success':
                    logger.debug(safe_log_message(f"⏭️ Already scraped (successful): {article.get('title', 'Unknown')[:40]}..."))
                else:
                    logger.debug(safe_log_message(f"⏭️ Already attempted (failed): {article.get('title', 'Unknown')[:40]}..."))        # Report article processing status
        if already_scraped:
            successful_count = 0
            failed_count = 0
            for article in already_scraped:
                stored_content = self.state.load_article_content(article.get('url'))
                if stored_content and stored_content.get('content_extraction_status') == 'success':
                    successful_count += 1
                else:
                    failed_count += 1
            
            logger.info(f"📋 Skipped {len(already_scraped)} already attempted articles ({successful_count} successful, {failed_count} failed)")
            for article in already_scraped[:3]:  # Show first 3 as examples
                stored_content = self.state.load_article_content(article.get('url'))
                status = stored_content.get('content_extraction_status', 'unknown') if stored_content else 'unknown'
                status_emoji = "✅" if status == 'success' else "❌"
                logger.info(safe_log_message(f"  {status_emoji} Already attempted ({status}): {article.get('title', 'Unknown')[:40]}..."))
            if len(already_scraped) > 3:
                logger.info(f"  ... and {len(already_scraped) - 3} more")
        
        logger.info(f"📊 Article Summary: {len(new_articles)} new for processing, {len(already_scraped)} already attempted, {len(all_articles)} total found")
        
        # Fallback: if no new articles and fallback is enabled, get some older articles
        if len(new_articles) == 0 and self.config.fallback_enabled and not initial_setup:
            logger.info(f"No new articles found, attempting fallback to articles from last {self.config.fallback_max_age_hours//24} days...")
            
            fallback_articles = []
            for feed_config in self.config.rss_feeds:
                try:
                    original_rss_url = self.scraper.rss_url
                    self.scraper.rss_url = feed_config["url"]
                    
                    feed_articles = await self.scraper.fetch_latest_newsletter(
                        max_age_hours=self.config.fallback_max_age_hours,
                        max_entries=self.config.fallback_max_articles
                    )
                    
                    self.scraper.rss_url = original_rss_url
                    
                    # Convert and filter articles not processed today
                    for article in feed_articles:
                        if hasattr(article, '__dict__'):
                            article_dict = {
                                'title': article.title,
                                'content': article.content,
                                'summary': article.summary,
                                'url': article.url,
                                'published_date': article.published_date,
                                'category': article.category,
                                'word_count': article.word_count,
                                'content_extraction_status': article.content_extraction_status,
                                'failure_reason': article.failure_reason
                            }
                        else:
                            article_dict = article
                        
                        article_dict["feed_name"] = feed_config["name"]
                        article_dict["feed_category"] = feed_config.get("category", "general")
                        article_dict["is_fallback"] = True  # Mark as fallback content
                        
                        # Only include if not processed today (allow re-processing older articles)
                        if article_dict.get('url') not in self.state.daily_video_assignments:
                            fallback_articles.append(article_dict)
                            
                except Exception as e:
                    logger.error(f"Error during fallback fetch from {feed_config['name']}: {e}")
                    continue
            
            if fallback_articles:
                # Limit fallback articles
                fallback_articles = fallback_articles[:self.config.fallback_max_articles]
                new_articles.extend(fallback_articles)
                logger.info(f"Added {len(fallback_articles)} fallback articles for processing")
            else:
                logger.warning("No fallback articles found either")
          # Apply article limit if configured
        if self.config.max_articles_per_run is not None and len(new_articles) > self.config.max_articles_per_run:
            new_articles = new_articles[:self.config.max_articles_per_run]
            logger.info(f"Limited to {self.config.max_articles_per_run} articles for this run")
        else:
            logger.info(f"Processing all {len(new_articles)} new articles (no limit set)")
        
        return new_articles
    
    async def summarize_articles(self, articles: List[Dict]) -> List[Dict]:
        """Generate TikTok summaries for articles."""
        logger.info(f"📝 Processing {len(articles)} articles for summarization...")
        
        summarized_articles = []
        skipped_articles = []
        
        for i, article in enumerate(articles):
            try:
                url = article.get('url')
                title = article.get('title', 'Unknown')
                
                # Skip articles with failed content extraction
                extraction_status = article.get('content_extraction_status', 'unknown')
                if extraction_status != 'success':
                    logger.info(safe_log_message(f"⏭️ Article {i+1}/{len(articles)} skipped - extraction failed ({extraction_status}): {title[:40]}..."))
                    continue
                
                # Check if already summarized
                if self.state.has_article_summary(url):
                    logger.info(safe_log_message(f"⏭️ Article {i+1}/{len(articles)} already summarized: {title[:40]}..."))
                    # Load existing summary into article for next pipeline step
                    summary_data = self.state.load_article_summary(url)
                    if summary_data:
                        article['tiktok_summary'] = summary_data.get('tiktok_summary')
                        article['summary_generated_at'] = summary_data.get('summarized_at')
                    skipped_articles.append(article)
                    continue
                
                logger.info(safe_log_message(f"📝 Summarizing article {i+1}/{len(articles)}: {title[:50]}..."))
                
                if self.dry_run:
                    # Mock summary for dry run
                    summary = f"BREAKING: {title}! This is a test summary for dry run mode. #TechNews #AI #TikTok"
                else:
                    # Get target duration from config for optimal video length
                    target_duration = int(config.tts.target_duration)
                    
                    # Generate real summary with duration constraint
                    summary = await self.summarizer.generate_tiktok_summary(
                        content=article.get('content', ''),
                        title=title,
                        url=url,
                        target_duration=target_duration  # Control summary length for target video duration
                    )
                
                if summary:
                    # Clean summary for TTS compatibility
                    cleaned_summary = clean_text_for_tts(summary)
                    article['tiktok_summary'] = cleaned_summary
                    article['summary_generated_at'] = datetime.now().isoformat()
                    
                    # Save cleaned summary to file and track in state
                    summary_file = self.state.save_article_summary(article, cleaned_summary, self.config.storage_dir)
                    
                    summarized_articles.append(article)
                    logger.info(safe_log_message(f"✅ Summary generated ({len(cleaned_summary)} chars): {title[:40]}..."))
                    if len(cleaned_summary) != len(summary):
                        logger.info(f"🧹 Text cleaned: {len(summary)} -> {len(cleaned_summary)} chars")
                else:
                    logger.warning(safe_log_message(f"⚠️ Failed to generate summary for: {title[:40]}..."))
                    
            except Exception as e:
                logger.error(safe_log_message(f"❌ Error summarizing article {i+1}: {e}"))
                continue        # Report results
        logger.info(f"📊 Summary Results: {len(summarized_articles)} new summaries, {len(skipped_articles)} already done, {len(articles)} total")
        
        if skipped_articles:
            logger.info(f"⏭️ Skipped {len(skipped_articles)} already summarized articles")
        
        # Return both new and already-processed articles for next pipeline step
        all_processed_articles = summarized_articles + skipped_articles
        return all_processed_articles
    
    async def generate_tts_audio(self, articles: List[Dict]) -> List[Dict]:
        """Generate TTS audio for article summaries."""
        logger.info(f"🔊 Processing {len(articles)} articles for TTS generation...")
        
        # Available voices for randomization
        available_voices = list(self.tts_engine.available_voices.keys())
        
        tts_articles = []
        skipped_articles = []
        
        for i, article in enumerate(articles):
            try:
                url = article.get('url')
                title = article.get('title', 'Unknown')
                summary = article.get('tiktok_summary')
                
                # Skip articles with failed content extraction
                extraction_status = article.get('content_extraction_status', 'unknown')
                if extraction_status != 'success':
                    logger.info(safe_log_message(f"⏭️ Article {i+1}/{len(articles)} skipped - extraction failed ({extraction_status}): {title[:40]}..."))
                    continue
                
                # Check if already has TTS
                if self.state.has_article_tts(url):
                    logger.info(safe_log_message(f"⏭️ Article {i+1}/{len(articles)} already has TTS: {title[:40]}..."))
                    # Load existing TTS info into article for next pipeline step
                    tts_file = self.state.article_tts_files.get(url)
                    if tts_file and Path(tts_file).exists():
                        # Load TTS metadata
                        metadata_file = Path(tts_file).with_suffix('.json')
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                tts_metadata = json.load(f)
                            article['tts_audio_path'] = tts_file
                            article['tts_duration'] = tts_metadata.get('duration_seconds')
                            article['tts_voice'] = tts_metadata.get('voice')
                            article['tts_generated_at'] = tts_metadata.get('generated_at')
                    skipped_articles.append(article)
                    continue
                
                if not summary:
                    logger.warning(safe_log_message(f"⚠️ No summary found for article {i+1}: {title[:40]}..."))
                    continue
                
                # Randomize voice for variety
                voice = random.choice(available_voices)
                
                logger.info(safe_log_message(f"🔊 Generating TTS {i+1}/{len(articles)} with voice '{voice}': {title[:30]}..."))
                
                if self.dry_run:
                    # Mock TTS for dry run with realistic duration
                    audio_path = f"mock_audio_{int(time.time())}_{i+1}.wav"
                    
                    # Calculate realistic mock duration based on target
                    summary_words = len(article.get('tiktok_summary', '').split())
                    if summary_words > 0:
                        tts_speed = config.get_tts_speed()
                        duration = (summary_words / 150) * 60 / tts_speed  # Same calculation as real TTS
                    else:
                        duration = config.tts.target_duration  # Use target if no summary
                    
                    article['tts_audio_path'] = audio_path
                    article['tts_duration'] = duration
                    article['tts_voice'] = voice
                    
                    # Save mock TTS to organized location
                    organized_path = self.state.save_article_tts(
                        article, audio_path, duration, voice, self.config.storage_dir
                    )
                    if organized_path:
                        article['tts_audio_path'] = organized_path
                    
                    logger.info(safe_log_message(f"✅ Mock TTS generated ({duration:.1f}s): {title[:40]}..."))
                    tts_articles.append(article)
                else:
                    # Generate real TTS
                    audio_filename = f"tts_{int(time.time())}_{i+1}.wav"
                    temp_audio_path = Path("temp") / audio_filename
                    temp_audio_path.parent.mkdir(exist_ok=True)
                    
                    generated_path = await self.tts_engine.generate_audio(
                        text=summary,
                        voice=voice,
                        output_path=str(temp_audio_path)
                    )
                    
                    if generated_path and Path(generated_path).exists():
                        # Calculate more accurate duration based on word count and TTS speed
                        word_count = len(summary.split())
                        tts_speed = config.get_tts_speed()
                        
                        # More accurate estimation: average speaking rate is ~150 words/min
                        # Adjusted for TTS speed: (word_count / 150) * 60 / tts_speed
                        base_duration = (word_count / 150) * 60  # seconds at normal speed
                        duration = base_duration / tts_speed  # adjust for TTS speed
                        
                        # Ensure duration stays within target range
                        target_duration = config.tts.target_duration
                        if duration > target_duration * 1.2:  # Allow 20% variance
                            logger.warning(f"TTS duration ({duration:.1f}s) exceeds target ({target_duration}s) for {title[:40]}...")
                            duration = min(duration, target_duration * 1.2)
                        
                        # Save TTS to organized location and track in state
                        organized_path = self.state.save_article_tts(
                            article, generated_path, duration, voice, self.config.storage_dir
                        )
                        
                        if organized_path:
                            article['tts_audio_path'] = organized_path
                            article['tts_duration'] = duration
                            article['tts_voice'] = voice
                            article['tts_generated_at'] = datetime.now().isoformat()
                        
                            logger.info(safe_log_message(f"✅ TTS generated ({duration:.1f}s): {title[:40]}..."))
                            tts_articles.append(article)
                        else:
                            logger.warning(safe_log_message(f"⚠️ Failed to organize TTS file for: {title[:40]}..."))
                    else:
                        logger.warning(safe_log_message(f"⚠️ TTS generation failed for: {title[:40]}..."))
                        continue
                        
            except Exception as e:
                logger.error(safe_log_message(f"❌ Error generating TTS for article {i+1}: {e}"))
                continue
          # Report results
        logger.info(f"📊 TTS Results: {len(tts_articles)} new TTS files, {len(skipped_articles)} already done, {len(articles)} total")
        
        if skipped_articles:
            logger.info(f"⏭️ Skipped {len(skipped_articles)} articles that already have TTS")
        
        # Return both new and already-processed articles for next pipeline step
        all_processed_articles = tts_articles + skipped_articles
        return all_processed_articles
    async def download_new_videos(self, initial_setup: bool = False) -> List[str]:
        """Download new gaming videos from configured sources."""
        # Set lookback period based on mode
        max_age_days = 30 * 6 if initial_setup else 1  # 6 months for initial setup, 1 day for daily
        
        logger.info(f"Downloading new gaming videos from {len(self.config.gaming_video_sources)} sources...")
        logger.info(f"📅 Video lookback period: {max_age_days} days ({max_age_days//30} months)" if initial_setup else f"📅 Video lookback period: {max_age_days} day")
        
        new_video_ids = []
        total_videos_downloaded = 0
        
        for i, source_config in enumerate(self.config.gaming_video_sources):
            source_url = source_config["url"]
            source_name = source_config["name"]
            max_videos_from_source = source_config.get("max_videos", 50)  # Default reasonable limit per source
            
            # Check if we've reached our overall limit (if configured)
            if (self.config.max_videos_per_run is not None and 
                total_videos_downloaded >= self.config.max_videos_per_run):
                logger.info(f"Reached max videos per run ({self.config.max_videos_per_run})")
                break
            
            try:
                logger.info(f"Processing source {i+1}/{len(self.config.gaming_video_sources)}: {source_name}")
                
                if self.dry_run:
                    # Mock download for dry run
                    mock_count = min(5, max_videos_from_source)  # Mock 5 videos per source
                    for j in range(mock_count):
                        video_id = f"mock_video_{source_name.replace(' ', '_')}_{j+1}"
                        new_video_ids.append(video_id)
                        self.state.downloaded_video_urls.add(f"{source_url}/video_{j+1}")
                        total_videos_downloaded += 1
                          # Break if we hit the global limit
                        if (self.config.max_videos_per_run is not None and 
                            total_videos_downloaded >= self.config.max_videos_per_run):
                            break
                else:
                    # Real download using FootageManager
                    # Calculate remaining videos we can download
                    remaining_quota = None
                    if self.config.max_videos_per_run is not None:
                        remaining_quota = self.config.max_videos_per_run - total_videos_downloaded
                        max_videos_from_source = min(max_videos_from_source, remaining_quota)
                    
                    # Check for already downloaded videos to prevent duplicates
                    already_downloaded = self.get_already_downloaded_video_ids()
                    if already_downloaded:
                        logger.info(f"Found {len(already_downloaded)} previously downloaded videos to skip")
                    
                    # Create source info for the downloader
                    source_info = {
                        "channel_url": source_url,
                        "channel_name": source_name,
                        "content_type": source_config.get("content_type", "medium_action"),
                        "max_videos": max_videos_from_source,
                        "min_duration": source_config.get("min_duration", self.config.min_video_duration),
                        "max_duration": source_config.get("max_duration", self.config.max_video_duration),
                        "skip_video_ids": already_downloaded  # Pass to downloader to skip
                    }
                    
                    # Download videos from this source with date filtering
                    downloaded_ids = await self.footage_manager.download_from_source(
                        source_info, 
                        max_new_videos=source_info["max_videos"],
                        max_age_days=max_age_days                    )
                    
                    if downloaded_ids:
                        new_video_ids.extend(downloaded_ids)
                        total_videos_downloaded += len(downloaded_ids)
                        
                        # Track downloaded videos in state
                        for video_id in downloaded_ids:
                            self.state.downloaded_video_urls.add(f"{source_url}/video_{video_id}")
                        
                        logger.info(f"✅ Downloaded {len(downloaded_ids)} videos from {source_name}")
                    else:
                        logger.warning(f"⚠️ No new videos downloaded from: {source_name}")
                        
            except Exception as e:
                logger.error(f"Error downloading from source {source_name}: {e}")
                continue
        
        if self.config.max_videos_per_run is None:
            logger.info(f"Downloaded {len(new_video_ids)} new videos (no limit set)")
        else:
            logger.info(f"Downloaded {len(new_video_ids)} new videos from {total_videos_downloaded} total downloads")
        
        return new_video_ids
    
    async def analyze_videos(self) -> List[str]:
        """Get list of all available analyzed videos."""
        logger.info("Getting available analyzed videos...")
        
        try:
            if self.dry_run:
                # Mock video IDs for dry run
                return [f"mock_video_{i}" for i in range(1, 6)]
            else:
                # Get all downloaded videos from the raw storage directory
                raw_storage_path = Path(self.config.storage_dir) / "raw"
                video_files = list(raw_storage_path.glob("*.mp4")) + list(raw_storage_path.glob("*.mkv"))
                video_ids = [f.stem for f in video_files]
                
                logger.info(f"Found {len(video_ids)} videos available for analysis")
                return video_ids
                
        except Exception as e:
            logger.error(f"Error getting video list: {e}")
            return []
    
    def reset_daily_assignments(self):
        """Reset daily video assignments if it's a new day."""
        today = datetime.now().strftime('%Y-%m-%d')
        if self.state.last_run_date != today:
            logger.info(f"New day detected, resetting video assignments (was {self.state.last_run_date}, now {today})")
            self.state.daily_video_assignments = {}
            self.state.last_run_date = today
    
    def assign_videos_to_articles(self, articles: List[Dict], available_videos: List[str]) -> List[Dict]:
        """Assign unique videos to articles with randomization."""
        logger.info(f"Assigning videos to {len(articles)} articles from {len(available_videos)} available videos...")
        
        if not available_videos:
            logger.error("No videos available for assignment")
            return []
        
        # Reset assignments if new day
        self.reset_daily_assignments()
        
        # Get videos that haven't been used today
        used_videos = set(self.state.daily_video_assignments.values())
        available_for_today = [v for v in available_videos if v not in used_videos]
        
        # If we've used all videos, reset for this batch
        if not available_for_today:
            logger.info("All videos used today, resetting for new batch")
            available_for_today = available_videos.copy()
            self.state.daily_video_assignments = {}
        
        # Randomly shuffle videos for assignment
        random.shuffle(available_for_today)
        
        assigned_articles = []
        
        for i, article in enumerate(articles):
            if i >= len(available_for_today):
                logger.warning(f"Not enough unique videos for all articles (need {len(articles)}, have {len(available_for_today)})")
                break
            
            video_id = available_for_today[i]
            article_id = article.get('url', f'article_{i}')
            
            article['assigned_video_id'] = video_id
            self.state.daily_video_assignments[article_id] = video_id
            assigned_articles.append(article)
            
            logger.info(f"Assigned video '{video_id}' to article: {article.get('title', 'Unknown')[:30]}...")        
        logger.info(f"Successfully assigned videos to {len(assigned_articles)} articles")
        return assigned_articles
    
    async def generate_final_videos(self, articles: List[Dict]) -> List[str]:
        """Generate final TikTok videos with all effects."""
        logger.info(f"Generating final TikTok videos for {len(articles)} articles...")
        
        generated_videos = []
        skipped_videos = []
        subtitle_styles = self.config.subtitle_styles.copy()
        
        for i, article in enumerate(articles):
            try:
                url = article.get('url')
                title = article.get('title', 'Unknown')
                
                # Check if final video already exists
                if self.state.has_article_final_video(url):
                    logger.info(safe_log_message(f"⏭️ Article {i+1}/{len(articles)} already has final video: {title[:40]}..."))
                    existing_video_path = self.state.article_final_videos.get(url)
                    if existing_video_path:
                        generated_videos.append(existing_video_path)
                        skipped_videos.append(article)
                    continue
                
                # Get TTS duration and calculate video segment length
                tts_duration = article.get('tts_duration', 25.0)
                segment_duration = int(tts_duration + (self.config.video_buffer_seconds * 2))
                
                # Ensure duration is within bounds
                segment_duration = max(self.config.min_video_duration, segment_duration)
                segment_duration = min(self.config.max_video_duration, segment_duration)
                
                # Randomize subtitle style
                subtitle_style = random.choice(subtitle_styles)
                subtitle_styles.remove(subtitle_style)  # Don't repeat until all used
                if not subtitle_styles:  # Reset when all styles used
                    subtitle_styles = self.config.subtitle_styles.copy()
                
                article['subtitle_style'] = subtitle_style  # Track for metadata
                
                logger.info(f"Generating video {i+1}/{len(articles)}: {segment_duration}s segment, '{subtitle_style}' subtitles")
                logger.info(f"  Article: {title[:40]}...")
                logger.info(f"  Video: {article.get('assigned_video_id')}")
                logger.info(f"  TTS: {tts_duration:.1f}s audio")
                
                if self.dry_run:
                    # Mock video generation
                    output_filename = f"tiktok_video_{i+1}_{int(time.time())}.mp4"
                    temp_output_path = Path("temp") / output_filename
                    temp_output_path.parent.mkdir(exist_ok=True)
                    temp_output_path.write_text("mock video content")  # Create mock file
                      # Save to organized location
                    organized_path = self.state.save_article_final_video(
                        article, str(temp_output_path), self.config.storage_dir
                    )
                    if organized_path:
                        generated_videos.append(organized_path)
                        logger.info(f"✅ Mock video generated: {Path(organized_path).name}")
                    
                else:
                    # Real video generation using VideoProcessor directly
                    video_config = VideoConfig(
                        enable_subtitles=True,
                        subtitle_style=subtitle_style,
                        output_quality="medium",
                        export_srt=True,
                        duration=segment_duration
                    )
                    
                    # Generate video segment to temporary location first
                    temp_output_dir = Path("temp") / "videos"
                    temp_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create unique output filename
                    temp_filename = f"temp_tiktok_{int(time.time())}_{i+1}.mp4"
                    temp_output_path = temp_output_dir / temp_filename
                    
                    # Import and use VideoProcessor directly
                    from video.processors.video_processor import VideoProcessor
                    processor = VideoProcessor(video_config)
                    
                    # Get TTS summary for overlay text
                    script_content = article.get('tiktok_summary', '')
                    tts_audio_path = article.get('tts_audio_path')
                    
                    if tts_audio_path and Path(tts_audio_path).exists():
                        generated_path = await processor.create_video(
                            audio_file=tts_audio_path,
                            script_content=script_content,
                            output_path=str(temp_output_path)
                        )
                        
                        if generated_path and Path(generated_path).exists():
                            # Move to organized location and track in state
                            organized_path = self.state.save_article_final_video(
                                article, generated_path, self.config.storage_dir
                            )
                            if organized_path:
                                generated_videos.append(organized_path)
                                logger.info(f"✅ Video generated: {Path(organized_path).name}")
                            else:
                                logger.warning(f"⚠️ Failed to organize video for article {i+1}")
                        else:
                            logger.warning(f"⚠️ Failed to generate video for article {i+1}")
                    else:
                        logger.warning(f"⚠️ TTS audio file not found for article {i+1}: {tts_audio_path}")
                        
            except Exception as e:
                logger.error(f"Error generating video for article {i+1}: {e}")
                continue
        
        logger.info(f"📊 Video Results: {len(generated_videos)} total videos ({len(generated_videos) - len(skipped_videos)} new, {len(skipped_videos)} already existed)")
        return generated_videos
    
    async def run_production_pipeline(self, initial_setup: bool = False) -> Dict:
        """Run the complete production pipeline."""
        start_time = time.time()
        logger.info(f"Starting TikTok Production Pipeline (initial_setup={initial_setup})")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'articles_processed': 0,
            'videos_generated': 0,
            'errors': [],
            'generated_files': []        }
        
        try:
            # Initialize all components
            await self.initialize_components()
            
            # Output current processing status
            self.output_processing_report()
            
            # Step 1: Fetch new articles
            logger.info("=== STEP 1: Fetching Articles ===")
            articles = await self.fetch_new_articles(initial_setup)
            
            if not articles:
                if initial_setup:
                    logger.info("No new articles to process, but continuing with video download for initial setup")
                    articles = []  # Continue with empty articles list
                else:
                    logger.info("No new articles to process")
                    return results            # Step 2: Generate summaries (check both new articles and existing articles needing summaries)
            articles_for_summary = articles.copy() if articles else []
            
            # Add existing articles that need summaries (deduplicate by URL)
            existing_articles_needing_summaries = self.get_articles_needing_summaries()
            if existing_articles_needing_summaries:
                logger.info(f"📝 Found {len(existing_articles_needing_summaries)} existing articles needing summaries")
                
                # Deduplicate by URL
                existing_urls = {article.get('url') for article in articles_for_summary if article.get('url')}
                for article in existing_articles_needing_summaries:
                    if article.get('url') not in existing_urls:
                        articles_for_summary.append(article)
                    
                logger.info(f"📝 Total articles for summarization: {len(articles_for_summary)} (including {len(existing_articles_needing_summaries)} existing)")
            
            if articles_for_summary:
                logger.info("=== STEP 2: Generating Summaries ===")
                summarized_articles = await self.summarize_articles(articles_for_summary)
                
                if not summarized_articles:
                    if initial_setup:
                        logger.error("No articles were successfully summarized, but continuing with video download for initial setup")
                        summarized_articles = []
                    else:
                        logger.error("No articles were successfully summarized")
                        return results
            else:
                logger.info("=== STEP 2: No articles need summarization ===")
                summarized_articles = []
                
            # Step 3: Generate TTS (check both summarized articles and existing articles needing TTS)
            articles_for_tts = summarized_articles.copy() if summarized_articles else []
            
            # Add existing articles that need TTS (deduplicate by URL)
            existing_articles_needing_tts = self.get_articles_needing_tts()
            if existing_articles_needing_tts:
                logger.info(f"🔊 Found {len(existing_articles_needing_tts)} existing articles needing TTS")
                
                # Deduplicate by URL
                existing_urls = {article.get('url') for article in articles_for_tts if article.get('url')}
                for article in existing_articles_needing_tts:
                    if article.get('url') not in existing_urls:
                        articles_for_tts.append(article)
                        
                logger.info(f"🔊 Total articles for TTS: {len(articles_for_tts)} (including {len(existing_articles_needing_tts)} existing)")
            
            if articles_for_tts:
                logger.info("=== STEP 3: Generating TTS Audio ===")
                articles = await self.generate_tts_audio(articles_for_tts)
                
                if not articles:
                    if initial_setup:
                        logger.error("No TTS audio was successfully generated, but continuing with video download for initial setup")
                        articles = []
                    else:
                        logger.error("No TTS audio was successfully generated")
                        return results
            else:
                logger.info("=== STEP 3: No articles need TTS generation ===")
                articles = []

            # Check for articles ready for video generation (have summaries and TTS)
            articles_ready_for_video = self.get_articles_needing_video()
            if articles_ready_for_video:
                logger.info(f"🎬 Found {len(articles_ready_for_video)} articles ready for video generation")
                # Merge with articles from current processing (deduplicate by URL)
                existing_urls = {article.get('url') for article in articles if article.get('url')}
                for article in articles_ready_for_video:
                    if article.get('url') not in existing_urls:
                        articles.append(article)
                logger.info(f"🎬 Total articles for video generation: {len(articles)}")

            # Step 4: Download new videos (if needed)
            logger.info("=== STEP 4: Downloading Gaming Videos ===")
            # For initial setup or when we need more videos, download from sources
            new_videos = await self.download_new_videos(initial_setup)
            if new_videos:
                logger.info(f"Downloaded {len(new_videos)} new videos")
            else:
                logger.info("No new videos to download at this time")
            
            # Step 5: Get available videos for analysis
            logger.info("=== STEP 5: Preparing Video Analysis ===")
            available_videos = await self.analyze_videos()            
            if not available_videos:
                if initial_setup:
                    logger.warning("No videos available for processing, but initial setup video download completed")
                    # Update results to show video download was attempted
                    results['videos_downloaded'] = len(new_videos) if new_videos else 0
                    return results
                else:
                    logger.error("No videos available for processing")
                    return results
            
            # Step 6: Assign videos to articles (only if we have articles)
            if articles:
                logger.info("=== STEP 6: Assigning Videos to Articles ===")
                articles = self.assign_videos_to_articles(articles, available_videos)
                
                if not articles:
                    logger.error("Failed to assign videos to articles")
                    return results
                
                # Step 7: Generate final TikTok videos
                logger.info("=== STEP 7: Generating Final TikTok Videos ===")
                generated_videos = await self.generate_final_videos(articles)
            else:
                logger.info("=== STEP 6-7: Skipping Video Assignment and Generation (no articles) ===")
                generated_videos = []
            
            # Update results
            results['articles_processed'] = len(articles)
            results['videos_generated'] = len(generated_videos)
            results['generated_files'] = generated_videos
            
            # Save state
            self.state.save(self.config.state_file)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Production pipeline completed in {elapsed_time:.1f}s")
            logger.info(safe_log_message(f"Results: {results['articles_processed']} articles → {results['videos_generated']} videos"))
            
        except Exception as e:
            logger.error(f"Production pipeline failed: {e}")
            results['errors'].append(str(e))
        
        finally:
            # Cleanup components
            if self.scraper:
                await self.scraper.close()
            if self.summarizer:
                await self.summarizer.cleanup()
            if self.tts_engine:
                await self.tts_engine.cleanup()
        
        results['end_time'] = datetime.now().isoformat()
        results['duration_seconds'] = time.time() - start_time
        
        return results

    def output_processing_report(self):
        """Output a detailed report of what articles have been processed."""
        report = self.state.get_processing_report()
        
        print("\n" + "="*60)
        print("📊 ARTICLE PROCESSING STATUS REPORT")
        print("="*60)
        print(f"📄 Total Articles Scraped: {report['total_articles_scraped']}")
        print(f"✍️  Total Articles Summarized: {report['total_articles_summarized']}")
        print(f"🔊 Total Articles with TTS: {report['total_articles_with_tts']}")
        print(f"🎬 Total Final Videos: {report['total_final_videos']}")
        print(f"✅ Fully Processed Articles: {report['fully_processed_articles']}")
        print(f"📅 Last Run: {report['last_run'] or 'Never'}")
        
        if report['articles_needing_summary'] > 0:
            print(f"⏳ Articles Needing Summary: {report['articles_needing_summary']}")
        
        if report['articles_needing_tts'] > 0:
            print(f"⏳ Articles Needing TTS: {report['articles_needing_tts']}")
        
        if report['articles_needing_video'] > 0:
            print(f"⏳ Articles Needing Video: {report['articles_needing_video']}")
        
        # Show samples of processed files
        if report['content_files']:
            print(f"\n📋 Recent Content Files:")
            for url, filepath in list(report['content_files'].items())[:3]:
                print(f"  • {Path(filepath).name}")
        
        if report['summary_files']:
            print(f"\n✍️  Recent Summary Files:")
            for url, filepath in list(report['summary_files'].items())[:3]:
                print(f"  • {Path(filepath).name}")
        
        if report['tts_files']:
            print(f"\n🔊 Recent TTS Files:")
            for url, filepath in list(report['tts_files'].items())[:3]:
                print(f"  • {Path(filepath).name}")
        
        print("="*60 + "\n")        
        # Save detailed report to file
        report_file = Path(self.config.storage_dir) / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            json_report = {k: (list(v) if isinstance(v, set) else v) for k, v in report.items()}
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Detailed processing report saved to: {report_file}")

    def _serialize_for_json(self, obj):
        """Helper method to serialize objects for JSON storage."""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return obj

    def get_articles_needing_summaries(self) -> List[Dict]:
        """Get articles that have content but need summaries."""
        articles_needing_summaries = []
        
        # Check all articles with content
        for url, content_file in self.state.article_content_files.items():
            if not Path(content_file).exists():
                continue
                
            # Skip if already has summary
            if self.state.has_article_summary(url):
                continue
                
            # Load article content
            content = self.state.load_article_content(url)
            if not content or content.get('content_extraction_status') != 'success':
                continue                
            articles_needing_summaries.append(content)
                
        return articles_needing_summaries
        
    def get_articles_needing_tts(self) -> List[Dict]:
        """Get articles that have summaries but need TTS."""
        articles_needing_tts = []
        
        # Check all articles with summaries
        for url, summary_file in self.state.article_summary_files.items():
            if not Path(summary_file).exists():
                continue
                
            # Skip if already has TTS
            if self.state.has_article_tts(url):
                continue
                
            # Load article summary
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                
            if summary_data.get('tiktok_summary'):
                # Add content extraction status from content file (if exists)
                content_data = self.state.load_article_content(url)
                if content_data:
                    summary_data['content_extraction_status'] = content_data.get('content_extraction_status', 'success')
                else:
                    # If no content file exists but we have a summary, assume success
                    summary_data['content_extraction_status'] = 'success'
                    
                articles_needing_tts.append(summary_data)
                
        return articles_needing_tts

    def get_articles_needing_video(self) -> List[Dict]:
        """Get articles that have summaries and TTS but need final video generation."""
        articles_needing_video = []
        
        # Check all articles with TTS
        for url, tts_file in self.state.article_tts_files.items():
            if not Path(tts_file).exists():
                continue
                
            # Skip if already has final video
            if self.state.has_article_final_video(url):
                continue
                
            # Load article TTS metadata
            tts_json_file = tts_file.replace('.wav', '.json')
            if Path(tts_json_file).exists():
                with open(tts_json_file, 'r', encoding='utf-8') as f:
                    tts_data = json.load(f)
                    
                # Also load summary data for complete article info
                summary_file = self.state.article_summary_files.get(url)
                if summary_file and Path(summary_file).exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                    
                    # Merge TTS and summary data
                    article_data = summary_data.copy()
                    article_data.update({
                        'tts_audio_path': tts_file,
                        'tts_duration': tts_data.get('duration_seconds', 30.0),
                        'tts_voice': tts_data.get('voice', 'default'),
                        'tts_generated_at': tts_data.get('generated_at')
                    })
                    
                    articles_needing_video.append(article_data)
                
        return articles_needing_video

async def main():
    """Main entry point for production pipeline."""
    parser = argparse.ArgumentParser(description="TikTok Automata Production Pipeline")
    parser.add_argument('--initial-setup', action='store_true', 
                       help='Run initial setup to download all videos and process backlog')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode without actually generating content')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom production config JSON file')
    
    args = parser.parse_args()
      # Load configuration
    config_file = args.config or 'production_config.json'
    
    if Path(config_file).exists():
        logger.info(f"Loading configuration from: {config_file}")
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        production_config = ProductionConfig(**config_data)
    else:
        logger.warning(f"Config file not found: {config_file}, using defaults")
        production_config = ProductionConfig()
    
    # Create and run pipeline
    pipeline = TikTokProductionPipeline(production_config, dry_run=args.dry_run)
    
    try:
        results = await pipeline.run_production_pipeline(initial_setup=args.initial_setup)
        
        # Print summary
        print("\n" + "="*60)
        print("TikTok Automata Production Pipeline - COMPLETE")
        print("="*60)
        print(f"Articles Processed: {results['articles_processed']}")
        print(f"Videos Generated: {results['videos_generated']}")
        print(f"Duration: {results.get('duration_seconds', 0):.1f} seconds")
        
        if results['generated_files']:
            print(f"\nGenerated Files:")
            for file_path in results['generated_files']:
                print(f"  📹 {Path(file_path).name}")
        
        if results['errors']:
            print(f"\nErrors Encountered:")
            for error in results['errors']:
                print(f"  ❌ {error}")
        
        print("="*60)        
        # Save results summary
        results_file = Path(production_config.storage_dir) / f"production_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure we're using the correct event loop policy on Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
