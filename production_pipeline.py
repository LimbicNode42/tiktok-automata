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

@dataclass
class ProductionConfig:
    """Configuration for production pipeline."""
    
    # Storage paths
    storage_dir: str = "./storage"
    output_dir: str = "./production_output"
    state_file: str = "./production_state.json"    
    # Processing limits (optional - if None, no limits applied)
    max_articles_per_run: Optional[int] = None
    max_videos_per_run: Optional[int] = None
    
    # Fallback options
    fallback_enabled: bool = True  # Enable fallback to older articles when no new ones found
    fallback_max_age_hours: int = 168  # 1 week for fallback articles
    fallback_max_articles: int = 3  # Maximum fallback articles to process
    
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
    """Tracks what has been processed to avoid duplicates."""
    processed_article_urls: Set[str]
    downloaded_video_urls: Set[str]
    last_run_date: str
    daily_video_assignments: Dict[str, str]  # article_id -> video_id mapping for today
    
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
                daily_video_assignments=data.get('daily_video_assignments', {})
            )
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing state found, starting fresh")
            return cls(
                processed_article_urls=set(),
                downloaded_video_urls=set(),
                last_run_date='',
                daily_video_assignments={}
            )
    
    def save(self, state_file: str):
        """Save production state to file."""
        data = {
            'processed_article_urls': list(self.processed_article_urls),
            'downloaded_video_urls': list(self.downloaded_video_urls),
            'last_run_date': self.last_run_date,
            'daily_video_assignments': self.daily_video_assignments
        }
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Production state saved to {state_file}")

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
        Path(config.output_dir).mkdir(exist_ok=True)
        
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
            speed=1.55,        # Optimized TikTok speed
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
                continue
          # Filter out already processed articles
        new_articles = []
        for article in all_articles:
            if article.get('url') not in self.state.processed_article_urls:
                new_articles.append(article)
                self.state.processed_article_urls.add(article.get('url'))
        
        logger.info(f"Found {len(new_articles)} new articles (total scraped: {len(all_articles)})")
        
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
        logger.info(f"Generating summaries for {len(articles)} articles...")
        
        summarized_articles = []
        
        for i, article in enumerate(articles):
            try:
                logger.info(f"Summarizing article {i+1}/{len(articles)}: {article.get('title', 'Unknown')[:50]}...")
                
                if self.dry_run:
                    # Mock summary for dry run
                    summary = f"**[0s-3s]** 🚨 BREAKING: {article.get('title', 'Test article')}!\n**[3s-8s]** This is a test summary for dry run mode.\n**[8s-12s]** #TechNews #AI #TikTok"
                else:
                    # Generate real summary                    
                    summary = await self.summarizer.generate_tiktok_summary(
                        content=article.get('content', ''),
                        title=article.get('title', ''),
                        url=article.get('url', '')
                    )
                
                if summary:
                    article['tiktok_summary'] = summary
                    article['summary_generated_at'] = datetime.now().isoformat()
                    summarized_articles.append(article)
                    logger.info(safe_log_message(f"✅ Summary generated for: {article.get('title', 'Unknown')[:50]}"))
                else:
                    logger.warning(safe_log_message(f"⚠️ Failed to generate summary for: {article.get('title', 'Unknown')[:50]}"))
                    
            except Exception as e:
                logger.error(f"Error summarizing article {i+1}: {e}")
                continue
        
        logger.info(f"Successfully summarized {len(summarized_articles)}/{len(articles)} articles")
        return summarized_articles
    
    async def generate_tts_audio(self, articles: List[Dict]) -> List[Dict]:
        """Generate TTS audio for article summaries."""
        logger.info(f"Generating TTS audio for {len(articles)} summaries...")
        
        # Available voices for randomization
        available_voices = list(self.tts_engine.available_voices.keys())
        
        for i, article in enumerate(articles):
            try:
                summary = article.get('tiktok_summary')
                if not summary:
                    logger.warning(f"No summary found for article {i+1}")
                    continue
                
                # Randomize voice for variety
                voice = random.choice(available_voices)
                
                logger.info(f"Generating TTS {i+1}/{len(articles)} with voice '{voice}': {article.get('title', 'Unknown')[:30]}...")
                
                if self.dry_run:
                    # Mock TTS for dry run
                    article['tts_audio_path'] = f"mock_audio_{i+1}.wav"
                    article['tts_duration'] = 25.0  # Mock duration
                    article['tts_voice'] = voice
                else:
                    # Generate real TTS
                    audio_filename = f"tts_{int(time.time())}_{i+1}.wav"
                    audio_path = Path(self.config.output_dir) / audio_filename
                    
                    generated_path = await self.tts_engine.generate_audio(
                        text=summary,
                        output_path=str(audio_path),
                        voice=voice,
                        speed=1.55  # Optimized speed
                    )
                    
                    if generated_path:
                        # Get audio duration for video segment calculation
                        audio_info = self.tts_engine.get_audio_info(generated_path)
                        duration = audio_info.get('duration', 25.0)
                        
                        article['tts_audio_path'] = generated_path
                        article['tts_duration'] = duration
                        article['tts_voice'] = voice
                        
                        logger.info(safe_log_message(f"✅ TTS generated: {duration:.1f}s audio with voice '{voice}'"))
                    else:
                        logger.warning(f"⚠️ Failed to generate TTS for article {i+1}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error generating TTS for article {i+1}: {e}")
                continue
        
        # Filter articles that have TTS
        articles_with_tts = [a for a in articles if 'tts_audio_path' in a]
        logger.info(f"Successfully generated TTS for {len(articles_with_tts)}/{len(articles)} articles")
        
        return articles_with_tts
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
                        max_age_days=max_age_days
                    )
                    
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
                # Get all downloaded videos
                storage_path = Path(self.config.storage_dir)
                video_files = list(storage_path.glob("*.mp4")) + list(storage_path.glob("*.mkv"))
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
        subtitle_styles = self.config.subtitle_styles.copy()
        
        for i, article in enumerate(articles):
            try:
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
                
                logger.info(f"Generating video {i+1}/{len(articles)}: {segment_duration}s segment, '{subtitle_style}' subtitles")
                logger.info(f"  Article: {article.get('title', 'Unknown')[:40]}...")
                logger.info(f"  Video: {article.get('assigned_video_id')}")
                logger.info(f"  TTS: {tts_duration:.1f}s audio")
                
                if self.dry_run:
                    # Mock video generation
                    output_filename = f"tiktok_video_{i+1}_{int(time.time())}.mp4"
                    output_path = Path(self.config.output_dir) / output_filename
                    generated_videos.append(str(output_path))
                    logger.info(f"✅ Mock video generated: {output_filename}")
                else:
                    # Real video generation
                    video_config = VideoConfig(
                        enable_subtitles=True,
                        subtitle_style=subtitle_style,
                        enable_letterboxing=True,
                        quality="medium",
                        enable_audio_mixing=True,
                        tts_audio_path=article.get('tts_audio_path'),
                        export_srt=True
                    )
                    
                    # Generate video segment
                    result = await self.footage_manager.create_tiktok_segment(
                        video_id=article.get('assigned_video_id'),
                        duration=segment_duration,
                        config=video_config,
                        output_dir=self.config.output_dir
                    )
                    
                    if result and result.get('output_path'):
                        generated_videos.append(result['output_path'])
                        logger.info(f"✅ Video generated: {Path(result['output_path']).name}")
                    else:
                        logger.warning(f"⚠️ Failed to generate video for article {i+1}")
                        
            except Exception as e:
                logger.error(f"Error generating video for article {i+1}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(generated_videos)}/{len(articles)} final videos")
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
            'generated_files': []
        }
        
        try:
            # Initialize all components
            await self.initialize_components()            # Step 1: Fetch new articles
            logger.info("=== STEP 1: Fetching Articles ===")
            articles = await self.fetch_new_articles(initial_setup)
            
            if not articles:
                if initial_setup:
                    logger.info("No new articles to process, but continuing with video download for initial setup")
                    articles = []  # Continue with empty articles list
                else:
                    logger.info("No new articles to process")
                    return results
            
            # Step 2: Summarize articles (only if we have articles)
            if articles:
                logger.info("=== STEP 2: Generating Summaries ===")
                articles = await self.summarize_articles(articles)
                
                if not articles:
                    if initial_setup:
                        logger.error("No articles were successfully summarized, but continuing with video download for initial setup")
                        articles = []  # Continue with empty articles list
                    else:
                        logger.error("No articles were successfully summarized")
                        return results
                
                # Step 3: Generate TTS audio (only if we have articles)
                if articles:
                    logger.info("=== STEP 3: Generating TTS Audio ===")
                    articles = await self.generate_tts_audio(articles)
                    
                    if not articles:
                        if initial_setup:
                            logger.error("No TTS audio was successfully generated, but continuing with video download for initial setup")
                            articles = []  # Continue with empty articles list
                        else:
                            logger.error("No TTS audio was successfully generated")
                            return results
            else:
                logger.info("=== STEP 2-3: Skipping Summaries and TTS (no articles) ===")            # Step 4: Download new videos (if needed)
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
        results_file = Path(production_config.output_dir) / f"production_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
