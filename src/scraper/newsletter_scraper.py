"""
Newsletter scraper for TLDR newsletter content.
"""

import asyncio
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
from dataclasses import dataclass
import re


@dataclass
class Article:
    """Represents a newsletter article."""
    title: str
    content: str
    summary: str
    url: str
    published_date: datetime
    category: str = "tech"
    word_count: int = 0
    content_extraction_status: str = "unknown"  # "success", "failed", "partial", "paywall"
    failure_reason: Optional[str] = None  # Detailed reason for failed/partial extractions


class NewsletterScraper:
    """Scraper for TLDR newsletter content."""
    
    def __init__(self):
        self.base_url = "https://tldr.tech/"
        self.rss_url = "https://tldr.tech/rss"
        self.session = None
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Session timeout configuration - increased for better reliability
        self.timeout = aiohttp.ClientTimeout(total=45, connect=15)
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout,
                connector=connector
            )
        return self.session
    
    async def fetch_rss_feed(self) -> List[Dict]:
        """
        Fetch the latest entries from TLDR RSS feed.
        
        Returns:
            List of RSS feed entries
        """
        try:
            logger.info(f"Fetching RSS feed from {self.rss_url}")
            
            session = await self._get_session()
            async with session.get(self.rss_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch RSS feed: HTTP {response.status}")
                    return []
                
                rss_content = await response.text()
            
            # Parse RSS feed
            feed = feedparser.parse(rss_content)
            
            if not feed.entries:
                logger.warning("No entries found in RSS feed")
                return []
            
            logger.info(f"Found {len(feed.entries)} entries in RSS feed")
            return feed.entries
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed: {str(e)}")
            return []
    
    async def fetch_latest_newsletter(self, max_age_hours: int = 24) -> List[Article]:
        """
        Fetch articles from the latest TLDR newsletter.
        
        Args:
            max_age_hours: Maximum age of articles to fetch (in hours)
            
        Returns:
            List of Article objects
        """
        try:
            logger.info("Fetching latest newsletter content...")
            
            # Get RSS entries
            rss_entries = await self.fetch_rss_feed()
            if not rss_entries:
                return []
            
            # Filter recent entries
            cutoff_date = datetime.now() - timedelta(hours=max_age_hours)
            recent_entries = []
            
            for entry in rss_entries:
                # Parse published date
                try:
                    published = datetime(*entry.published_parsed[:6])
                    if published >= cutoff_date:
                        recent_entries.append(entry)
                except (AttributeError, TypeError):
                    # If we can't parse the date, include it anyway
                    recent_entries.append(entry)
            
            if not recent_entries:
                logger.info("No recent newsletter entries found")
                return []
            
            logger.info(f"Processing {len(recent_entries)} recent entries")
            
            # Process each entry
            articles = []
            for entry in recent_entries:
                try:
                    entry_articles = await self._process_rss_entry(entry)
                    if entry_articles:
                        articles.extend(entry_articles)
                except Exception as e:
                    logger.error(f"Error processing entry '{entry.get('title', 'Unknown')}': {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching latest newsletter: {str(e)}")
            return []
    
    async def _process_rss_entry(self, entry: Dict) -> List[Article]:
        """
        Process a single RSS entry (newsletter page) into multiple Article objects.
        
        Args:
            entry: RSS feed entry pointing to a newsletter page
            
        Returns:
            List of Article objects extracted from the newsletter page
        """
        try:
            title = entry.get('title', 'Untitled')
            url = entry.get('link', '')
            
            # Get published date
            try:
                published_date = datetime(*entry.published_parsed[:6])
            except (AttributeError, TypeError):
                published_date = datetime.now()
            
            logger.info(f"Processing newsletter page: {url}")
            
            # Fetch and parse the newsletter page to extract individual articles
            articles = await self._extract_articles_from_newsletter(url, published_date)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error processing RSS entry: {str(e)}")
            return []
    
    async def _extract_articles_from_newsletter(self, newsletter_url: str, published_date: datetime) -> List[Article]:
        """
        Extract individual articles from a TLDR newsletter page.
        
        Args:
            newsletter_url: URL of the newsletter page
            published_date: Date the newsletter was published
            
        Returns:
            List of Article objects
        """
        try:
            session = await self._get_session()
            async with session.get(newsletter_url, timeout=15) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch newsletter page: HTTP {response.status}")
                    return []
                
                html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            articles = []
            
            # Find all article links that have read time indicators
            # Pattern: "Title (X minute read)"
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                link_text = link.get_text(strip=True)
                
                # Check if this looks like an article link with read time
                if '(' in link_text and 'minute read' in link_text:
                    try:
                        # Extract title and read time
                        if link_text.endswith(')'):
                            # Split on last occurrence of '('
                            title_part, read_time_part = link_text.rsplit('(', 1)
                            title = title_part.strip()
                            read_time_text = read_time_part.rstrip(')')
                            
                            # Extract read time in minutes
                            read_time_match = re.search(r'(\d+)\s*minute', read_time_text)
                            read_time_minutes = int(read_time_match.group(1)) if read_time_match else 0
                            
                            # Get the article URL
                            article_url = link.get('href')
                            
                            # Make URL absolute if needed
                            if article_url.startswith('/'):
                                article_url = f"https://tldr.tech{article_url}"
                            elif not article_url.startswith('http'):
                                continue  # Skip invalid URLs
                            
                            # Determine category based on the section
                            category = self._determine_category_from_context(link, soup)
                            
                            # Fetch article content from the external URL
                            article_content, failure_reason = await self._fetch_external_article_content(article_url)
                            
                            # Analyze content extraction results and expected length
                            expected_words = read_time_minutes * 200  # ~200 words per minute
                            
                            if article_content and len(article_content) > 100:
                                actual_words = len(article_content.split())
                                content = article_content
                                word_count = actual_words
                                
                                # Determine extraction status based on content analysis
                                if actual_words >= expected_words * 0.7:  # Got at least 70% of expected content
                                    extraction_status = "success"
                                    failure_reason = None  # Clear failure reason for successful extractions
                                elif actual_words >= expected_words * 0.3:  # Got 30-70% of expected content
                                    extraction_status = "partial"
                                    failure_reason = f"Partial extraction: got {actual_words}/{expected_words} words ({actual_words/expected_words*100:.1f}%)"
                                    logger.warning(f"Partial content extraction for {title}: {actual_words}/{expected_words} words")
                                else:  # Got less than 30% of expected content
                                    extraction_status = "failed"
                                    content = f"[CONTENT EXTRACTION FAILED] Expected ~{expected_words} words, got {actual_words}. URL: {article_url}"
                                    failure_reason = f"Insufficient content: got {actual_words}/{expected_words} words ({actual_words/expected_words*100:.1f}%)"
                                    logger.warning(f"Content extraction failed for {title}: {actual_words}/{expected_words} words")
                                    
                            else:
                                # No content extracted from external URL
                                extraction_status = "failed"
                                content = f"[CONTENT EXTRACTION FAILED] Unable to extract content from: {article_url}"
                                word_count = 0
                                if not failure_reason:
                                    failure_reason = "No content extracted - empty or very short response"
                                logger.warning(f"Complete content extraction failure for {title}")

                            article = Article(
                                title=title,
                                content=content,
                                summary=f"{read_time_minutes}-minute read",
                                url=article_url,
                                published_date=published_date,
                                category=category,
                                word_count=word_count,
                                content_extraction_status=extraction_status,
                                failure_reason=failure_reason
                            )
                            
                            articles.append(article)
                            logger.info(f"Extracted article: {title} ({read_time_minutes} min read)")
                            
                    except Exception as e:
                        logger.warning(f"Error processing article link '{link_text}': {str(e)}")
                        continue
            
            logger.info(f"Extracted {len(articles)} articles from newsletter page")
            return articles
            
        except Exception as e:
            logger.error(f"Error extracting articles from newsletter page {newsletter_url}: {str(e)}")
            return []
    
    def _determine_category_from_context(self, link_element, soup) -> str:
        """
        Determine article category based on its position in the newsletter structure.
        
        Args:
            link_element: The link element containing the article
            soup: BeautifulSoup object of the newsletter page
            
        Returns:
            Category string based on newsletter sections
        """
        try:
            # Look for section headers or category indicators near the link
            parent = link_element.parent
            
            # Search up the DOM tree to find section headers
            for i in range(5):  # Check up to 5 levels up
                if parent is None:
                    break
                
                # Look for text content that indicates section
                text_content = parent.get_text().lower()
                
                if any(keyword in text_content for keyword in ['big tech', 'startup']):
                    return 'big_tech'
                elif any(keyword in text_content for keyword in ['science', 'futuristic']):
                    return 'science'
                elif any(keyword in text_content for keyword in ['programming', 'design', 'data science']):
                    return 'dev'
                elif any(keyword in text_content for keyword in ['ai', 'artificial intelligence']):
                    return 'ai'
                elif any(keyword in text_content for keyword in ['crypto', 'blockchain']):
                    return 'crypto'
                
                parent = parent.parent
            
            # Fallback: analyze the link text itself
            link_text = link_element.get_text().lower()
            if any(keyword in link_text for keyword in ['apple', 'google', 'microsoft', 'amazon', 'meta']):
                return 'big_tech'
            elif any(keyword in link_text for keyword in ['ai', 'gpt', 'llm']):
                return 'ai'
            elif any(keyword in link_text for keyword in ['programming', 'code', 'developer']):
                return 'dev'
            elif any(keyword in link_text for keyword in ['research', 'study', 'science']):
                return 'science'
            
            return 'tech'  # Default category
            
        except Exception as e:
            logger.warning(f"Error determining category from context: {str(e)}")
            return 'tech'

    async def _fetch_external_article_content(self, article_url: str) -> tuple[Optional[str], Optional[str]]:
        """
        Fetch content from external article URL with improved extraction logic and retry handling.
        
        Args:
            article_url: URL of the external article
            
        Returns:
            Tuple of (Article content or None if failed, failure reason or None)
        """
        max_retries = 2
        base_timeout = 25  # Increased base timeout
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Add delay between retries
                    await asyncio.sleep(2 * attempt)
                    logger.debug(f"Retry {attempt} for: {article_url}")
                
                logger.debug(f"Fetching external content from: {article_url} (attempt {attempt + 1})")
                
                session = await self._get_session()
                
                # Use different headers for different sites to avoid blocking
                site_headers = self.headers.copy()
                if 'techcrunch.com' in article_url:
                    site_headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                elif 'bloomberg.com' in article_url:
                    site_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
                elif 'spectrum.ieee.org' in article_url:
                    site_headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
                
                # Progressive timeout increase with retries
                timeout = base_timeout + (attempt * 10)
                
                # Shorter timeout for known problematic sites only on first attempt
                if attempt == 0 and any(domain in article_url for domain in ['bloomberg.com', 'nytimes.com']):
                    timeout = 18
                
                try:
                    async with session.get(article_url, timeout=timeout, headers=site_headers) as response:
                        if response.status == 403:
                            if attempt < max_retries:
                                continue  # Retry with different timeout/headers
                            return None, "HTTP 403 - Access denied (likely paywall or bot protection)"
                        elif response.status == 404:
                            return None, "HTTP 404 - Article not found"
                        elif response.status == 502:
                            if attempt < max_retries:
                                continue  # Retry for server errors
                            return None, "HTTP 502 - Request failed"
                        elif response.status not in [200, 301, 302]:
                            if attempt < max_retries:
                                continue  # Retry
                            return None, f"HTTP {response.status} - Request failed"
                        
                        html = await response.text()
                        
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        logger.debug(f"Timeout on attempt {attempt + 1} for {article_url}, retrying with longer timeout...")
                        continue
                    return None, f"Request timeout after {timeout}s (tried {max_retries + 1} times)"
                except Exception as e:
                    if attempt < max_retries:
                        logger.debug(f"Network error on attempt {attempt + 1} for {article_url}: {str(e)}, retrying...")
                        continue
                    return None, f"Network error: {str(e)}"
                
                # If we got here, we have valid HTML - process it
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove unwanted elements that clutter content
                unwanted_selectors = [
                    'script', 'style', 'nav', 'header', 'footer', 'aside',
                    '.advertisement', '.ad', '.ads', '.social-share', '.share',
                    '.newsletter-signup', '.subscription', '.paywall',
                    '.comments', '.comment', '.related', '.sidebar',
                    '.navigation', '.breadcrumb', '.menu', '.toolbar',
                    '[class*="ad-"]', '[id*="ad-"]', '[class*="social"]',
                    '.cookie-banner', '.gdpr', '.privacy-notice'
                ]
                
                for selector in unwanted_selectors:
                    for element in soup.select(selector):
                        element.decompose()
                
                # Site-specific content extraction
                content = self._extract_content_by_site(soup, article_url)
                failure_reason = None
                
                if not content:
                    # Generic content extraction with multiple strategies
                    content = self._extract_content_generic(soup)
                    if not content:
                        failure_reason = "Content extraction failed - no content found using site-specific or generic strategies"
                
                if content:
                    # Clean and validate content
                    cleaned_content = self._clean_and_validate_content(content, article_url)
                    if cleaned_content:
                        return cleaned_content, None
                    else:
                        failure_reason = "Content validation failed - content too short, paywall detected, or formatting issues"
                
                if not failure_reason:
                    failure_reason = "Unknown content extraction failure"
                    
                logger.warning(f"No content extracted from: {article_url} - {failure_reason}")
                return None, failure_reason
                
            except Exception as e:
                if attempt < max_retries:
                    logger.debug(f"Unexpected error on attempt {attempt + 1} for {article_url}: {str(e)}, retrying...")
                    continue
                failure_reason = f"Exception during content extraction: {str(e)}"
                logger.warning(f"Error fetching external article content from {article_url}: {failure_reason}")
                return None, failure_reason
        
        # If we get here, all retries failed
        return None, f"All {max_retries + 1} attempts failed"
    
    def _extract_content_by_site(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract content using site-specific selectors."""
        try:
            domain = url.split('/')[2].lower()
            
            # Site-specific selectors for better content extraction
            site_selectors = {
                'techcrunch.com': [
                    '.article-content',
                    '.entry-content',
                    '[data-module="ArticleBody"]',
                    '.post-block'
                ],
                'reuters.com': [
                    '[data-testid="paragraph"]',
                    '.StandardArticleBody_body',
                    '.ArticleBodyWrapper'
                ],
                'bloomberg.com': [
                    '.body-content',
                    '[data-module="ArticleBody"]',
                    '.story-body',
                    '.fence-body'
                ],
                'theverge.com': [
                    '.duet--article--article-body-component',
                    '.c-entry-content',
                    '.entry-content',
                    '.c-entry-summary'
                ],
                'arstechnica.com': [
                    '.post-content',
                    '.entry-content',
                    'section.article-guts'
                ],
                'wired.com': [
                    '.ArticleBodyComponent',
                    '.post-content',
                    '.content'
                ],
                'venturebeat.com': [
                    '.article-content',
                    '.post-content',
                    '.entry-content'
                ],
                'theinformation.com': [
                    '.article-content',
                    '.story-body'
                ],
                'axios.com': [
                    '.story-body',
                    '.prose'
                ],
                'spectrum.ieee.org': [
                    '.article-body',
                    '.entry-content',
                    '.post-content',
                    '.article-content'
                ],
                'theguardian.com': [
                    '.article-body-commercial-selector',
                    '.content__article-body',
                    '[data-gu-name="body"]'
                ],
                'qz.com': [
                    '.article-body',
                    '.content-body',
                    '.post-content'
                ],
                'diwank.space': [
                    '.post-content',
                    '.entry-content',
                    '.article-content',
                    'main article',
                    '.content',
                    '.single-content',
                    '.blog-content',
                    '.post-body',
                    '[role="main"]',
                    'main',
                    '.prose',  # Common in modern blogs
                    '.markdown-content',
                    '.post-wrapper',
                    '.article-wrapper',
                    '.main-content',
                    '.page-content',
                    # Try broader selectors for content
                    '#content',
                    '#main-content',
                    '[class*="content"]',
                    '[class*="post"]',
                    '[class*="article"]'
                ],
                'tensorzero.com': [
                    '.post-content',
                    '.blog-content',
                    '.article-content',
                    'main',
                    '.prose',  # Tailwind CSS prose class
                    '.markdown-body',
                    '.blog-post-content',
                    '.post-body',
                    '[class*="content"]',
                    'article',
                    '.entry-content'
                ],
                'joshuapurtell.com': [
                    '.post-content',
                    '.entry-content',
                    'article',
                    'main'
                ],
                'ashishb.net': [
                    '.post-content',
                    '.entry-content',
                    '.content'
                ],
                'glama.ai': [
                    '.post-content',
                    '.blog-content',
                    'article',
                    'main'
                ],
                'github.com': [
                    '.markdown-body',
                    '.gist-content',
                    '.file-content'
                ]
            }
            
            # Try site-specific selectors first
            for site_domain, selectors in site_selectors.items():
                if site_domain in domain:
                    logger.debug(f"Trying site-specific selectors for {site_domain}: {len(selectors)} selectors")
                    # Try each selector for this site
                    for i, selector in enumerate(selectors):
                        elements = soup.select(selector)
                        if elements:
                            content = elements[0].get_text(separator=' ', strip=True)
                            logger.debug(f"Selector {i+1}/{len(selectors)} '{selector}' found {len(elements)} elements, content length: {len(content)}")
                            if len(content) > 200:  # Ensure we got substantial content
                                logger.info(f"Successfully extracted {len(content)} chars using selector '{selector}' for {site_domain}")
                                return content
                        else:
                            logger.debug(f"Selector '{selector}' found no elements")
                    
                    # If site-specific selectors didn't work, try enhanced extraction for known partial sites
                    logger.info(f"Site-specific selectors failed for {site_domain}, trying enhanced extraction")
                    if site_domain in ['tensorzero.com', 'diwank.space']:
                        enhanced_content = self._enhanced_extraction_for_partial_sites(soup, site_domain)
                        if enhanced_content:
                            logger.info(f"Enhanced extraction successful for {site_domain}: {len(enhanced_content)} chars")
                            return enhanced_content
                        else:
                            logger.warning(f"Enhanced extraction also failed for {site_domain}")
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in site-specific extraction: {str(e)}")
            return None
    
    def _extract_content_generic(self, soup: BeautifulSoup) -> Optional[str]:
        """Generic content extraction using common patterns."""
        try:
            # Try multiple content extraction strategies
            content_strategies = [
                # Strategy 1: Common article containers
                lambda: self._try_selectors(soup, [
                    'article',
                    '[role="main"]',
                    '.post-content',
                    '.entry-content',
                    '.article-content',
                    '.story-body',
                    '.content-body'
                ]),
                
                # Strategy 2: Enhanced paragraph extraction with better content detection
                lambda: self._extract_from_paragraphs(soup),
                
                # Strategy 3: Main content areas
                lambda: self._try_selectors(soup, [
                    'main',
                    '.main',
                    '#main',
                    '.main-content',
                    '#content',
                    '.content'
                ]),
                
                # Strategy 4: Structured data
                lambda: self._extract_from_structured_data(soup),
                
                # Strategy 5: Enhanced paragraph aggregation
                lambda: self._extract_by_paragraph_density(soup)
            ]
            
            for strategy in content_strategies:
                content = strategy()
                if content and len(content) > 100:  # Reduced threshold
                    return content
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in generic extraction: {str(e)}")
            return None
    
    def _try_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
        """Try a list of CSS selectors to find content."""
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                content = elements[0].get_text(separator=' ', strip=True)
                if len(content) > 100:  # Reduced threshold for better results
                    return content
        return None
    
    def _extract_from_paragraphs(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract content by finding the area with the most relevant paragraphs."""
        try:
            # Find containers that likely contain article content
            content_containers = []
            
            # Look for common article containers
            main_containers = soup.find_all(['article', 'main', 'div'], recursive=True)
            for container in main_containers:
                # Skip if container has obvious non-content classes
                container_classes = ' '.join(container.get('class', [])).lower()
                if any(skip_word in container_classes for skip_word in ['nav', 'menu', 'header', 'footer', 'sidebar', 'comment', 'ad']):
                    continue
                
                # Find all paragraphs within this container
                paragraphs = container.find_all('p', recursive=True)
                if len(paragraphs) < 3:  # Need at least 3 paragraphs for article content
                    continue
                
                # Calculate content quality metrics
                total_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                if len(total_text) < 500:  # Skip short content
                    continue
                
                # Check for article-like indicators
                article_indicators = ['published', 'written', 'reported', 'according to', 'research', 'study', 'analysis']
                indicator_count = sum(1 for indicator in article_indicators if indicator.lower() in total_text.lower())
                
                content_containers.append({
                    'container': container,
                    'paragraph_count': len(paragraphs),
                    'text_length': len(total_text),
                    'text': total_text,
                    'article_indicators': indicator_count
                })
            
            if not content_containers:
                return None
                
            # Sort by quality metrics: prioritize more paragraphs, longer text, and article indicators
            best_container = max(content_containers, key=lambda x: (
                x['paragraph_count'] * 0.3 +  # Paragraph count weight
                (x['text_length'] / 1000) * 0.5 +  # Text length weight (normalized)
                x['article_indicators'] * 0.2  # Article indicators weight
            ))
            
            return best_container['text'] if best_container['text'] else None
            
        except Exception as e:
            logger.debug(f"Error extracting from paragraphs: {str(e)}")
            return None
    
    def _extract_from_structured_data(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract content from structured data (JSON-LD, etc.)."""
        try:
            # Look for JSON-LD structured data
            scripts = soup.find_all('script', {'type': 'application/ld+json'})
            for script in scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'articleBody' in data:
                        return data['articleBody']
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'articleBody' in item:
                                return item['articleBody']
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting from structured data: {str(e)}")
            return None
    
    def _extract_by_paragraph_density(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract content by analyzing paragraph density across the page."""
        try:
            # Find all divs/sections that might contain articles
            containers = soup.find_all(['div', 'section', 'article', 'main'], recursive=True)
            
            content_candidates = []
            
            for container in containers:
                # Skip containers with navigation-like classes
                container_classes = ' '.join(container.get('class', [])).lower()
                container_id = (container.get('id') or '').lower()
                
                skip_indicators = ['nav', 'menu', 'header', 'footer', 'sidebar', 'comment', 'ad', 'related', 'share', 'social']
                if any(indicator in container_classes + ' ' + container_id for indicator in skip_indicators):
                    continue
                
                # Count paragraphs and calculate density
                paragraphs = container.find_all('p', recursive=True)
                if len(paragraphs) < 2:
                    continue
                
                # Get all text content
                text_content = container.get_text(separator=' ', strip=True)
                if len(text_content) < 300:  # Too short to be main content
                    continue
                
                # Calculate quality score
                paragraph_density = len(paragraphs) / max(1, len(text_content) / 1000)  # paragraphs per 1000 chars
                word_count = len(text_content.split())
                
                # Look for article-like patterns
                article_patterns = [
                    r'\b(according to|research shows|study finds|analysis reveals)\b',
                    r'\b(however|moreover|furthermore|additionally|meanwhile)\b',
                    r'\b(first|second|third|finally|in conclusion)\b',
                    r'\b(published|written|reported|says|explains|notes)\b'
                ]
                pattern_count = sum(len(re.findall(pattern, text_content, re.IGNORECASE)) for pattern in article_patterns)
                
                quality_score = (
                    word_count * 0.4 +  # Word count (primary factor)
                    paragraph_density * 100 +  # Paragraph density
                    pattern_count * 50 +  # Article-like patterns
                    (100 if 'article' in container_classes else 0)  # Article container bonus
                )
                
                content_candidates.append({
                    'content': text_content,
                    'score': quality_score,
                    'word_count': word_count,
                    'paragraphs': len(paragraphs)
                })
            
            if not content_candidates:
                return None
            
            # Return the highest scoring content
            best_candidate = max(content_candidates, key=lambda x: x['score'])
            return best_candidate['content'] if best_candidate['word_count'] > 200 else None
            
        except Exception as e:
            logger.debug(f"Error in paragraph density extraction: {str(e)}")
            return None
    
    def _enhanced_extraction_for_partial_sites(self, soup: BeautifulSoup, domain: str) -> Optional[str]:
        """Enhanced extraction specifically for sites showing partial results."""
        try:
            content_parts = []
            
            if domain == 'tensorzero.com':
                # TensorZero blog specific extraction
                selectors_to_try = [
                    '.prose',  # Common in modern blogs
                    '.blog-post-content',
                    '.post-body',
                    '[class*="content"]',
                    'article',
                    'main',
                    '[class*="post"]',
                    '[class*="article"]',
                    '.markdown-body'  # If using markdown rendering
                ]
                
                for selector in selectors_to_try:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(separator=' ', strip=True)
                        if len(text) > 500 and text not in content_parts:  # Avoid duplicates
                            content_parts.append(text)
                
                # Also try extracting all paragraphs in sequence
                paragraphs = soup.find_all('p')
                if len(paragraphs) > 10:  # Likely article content
                    paragraph_text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                    if len(paragraph_text) > 1000:
                        content_parts.append(paragraph_text)
            
            elif domain == 'diwank.space':
                # Diwank.space specific extraction - enhanced for long-form articles
                selectors_to_try = [
                    '.post-content',
                    '.entry-content', 
                    '.article-content',
                    'main article',
                    '.content',
                    '[role="main"]',
                    '.blog-content',
                    '.single-content',
                    '.post-body',
                    '.prose',
                    '.markdown-content',
                    '.post-wrapper',
                    '.article-wrapper',
                    '.main-content'
                ]
                
                for selector in selectors_to_try:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(separator=' ', strip=True)
                        if len(text) > 500 and text not in content_parts:
                            content_parts.append(text)
                
                # Aggressive fallback: try to extract ALL paragraphs from the page
                all_paragraphs = soup.find_all('p')
                if len(all_paragraphs) > 20:  # Likely a long-form article
                    # Filter out navigation/header/footer paragraphs
                    article_paragraphs = []
                    for p in all_paragraphs:
                        p_text = p.get_text(strip=True)
                        # Skip very short paragraphs and obvious navigation text
                        if (len(p_text) > 30 and 
                            not any(nav_word in p_text.lower() for nav_word in [
                                'menu', 'navigation', 'subscribe', 'follow', 'contact',
                                'home', 'about', 'privacy', 'terms', 'copyright'
                            ])):
                            # Check if paragraph is within a reasonable content area
                            parent_classes = ' '.join(p.parent.get('class', [])).lower() if p.parent else ''
                            parent_id = (p.parent.get('id') or '').lower() if p.parent else ''
                            
                            # Skip if clearly in navigation/header/footer area
                            if not any(skip_area in parent_classes + ' ' + parent_id for skip_area in [
                                'nav', 'menu', 'header', 'footer', 'sidebar', 'aside', 'advertisement'
                            ]):
                                article_paragraphs.append(p_text)
                    
                    if len(article_paragraphs) > 10:
                        combined_paragraphs = ' '.join(article_paragraphs)
                        if len(combined_paragraphs) > 2000:  # Substantial content
                            content_parts.append(combined_paragraphs)
                            logger.info(f"Extracted {len(article_paragraphs)} paragraphs totaling {len(combined_paragraphs)} chars for {domain}")
            
            # Combine and return the longest/best content
            if content_parts:
                # Return the longest content part (likely the most complete)
                best_content = max(content_parts, key=len)
                logger.info(f"Enhanced extraction for {domain}: found {len(content_parts)} content parts, using best with {len(best_content)} chars")
                return best_content if len(best_content) > 1000 else None
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in enhanced extraction for {domain}: {str(e)}")
            return None
    
    def _clean_and_validate_content(self, content: str, url: str) -> Optional[str]:
        """Clean and validate extracted content."""
        if not content:
            return None
        
        # Clean up whitespace and formatting
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove common prefixes/suffixes that aren't part of article content
        prefixes_to_remove = [
            r'^(home|news|tech|business|sports|entertainment|lifestyle|opinion|about|contact|search|menu|subscribe|newsletter)[\s\-–—]*',
            r'^(sign in|log in|register|login|create account)[\s\-–—]*',
            r'^(advertisement|sponsored|promoted)[\s\-–—]*'
        ]
        
        for prefix_pattern in prefixes_to_remove:
            content = re.sub(prefix_pattern, '', content, flags=re.IGNORECASE)
        
        # Remove common suffixes
        suffixes_to_remove = [
            r'[\s\-–—]*(subscribe|newsletter|follow us|contact us|about us|privacy policy|terms of service|copyright).*$',
            r'[\s\-–—]*(share this|tweet this|facebook|twitter|linkedin|instagram).*$'
        ]
        
        for suffix_pattern in suffixes_to_remove:
            content = re.sub(suffix_pattern, '', content, flags=re.IGNORECASE)
        
        content = content.strip()
        
        # Validate content quality
        if len(content) < 50:  # Very short content
            logger.debug(f"Content too short ({len(content)} chars) for {url}")
            return None
        
        # Check for paywall indicators
        paywall_indicators = [
            'subscribe to continue reading',
            'sign up to read more',
            'become a member',
            'premium content',
            'subscription required',
            'paywall',
            'upgrade to premium',
            'sign in to read',
            'create a free account',
            'this article is for subscribers',
            'unlock this article',
            'continue reading with',
            'get unlimited access'
        ]
        
        content_lower = content.lower()
        paywall_detected = any(indicator in content_lower for indicator in paywall_indicators)
        
        if paywall_detected and len(content) < 500:
            logger.debug(f"Paywall content detected for {url}")
            return None
        
        # Check for minimal content that's just headers/navigation
        words = content.split()
        if len(words) < 30:  # Very few words
            logger.debug(f"Too few words ({len(words)}) for {url}")
            return None
        
        # Advanced content quality check
        if not self._has_article_like_structure(content):
            logger.debug(f"Content lacks article-like structure for {url}")
            return None
        
        return content
    
    def _has_article_like_structure(self, content: str) -> bool:
        """Check if content has article-like structure patterns."""
        try:
            # Count sentences (rough approximation)
            sentence_count = len(re.findall(r'[.!?]+', content))
            
            # Count paragraphs (double newlines or substantial breaks)
            paragraph_count = len(re.split(r'\n\s*\n|\. {2,}', content))
            
            # Look for article-like patterns
            article_patterns = [
                r'\b(according to|research shows|study finds|analysis reveals)\b',
                r'\b(however|moreover|furthermore|additionally|meanwhile)\b',
                r'\b(first|second|third|finally|in conclusion)\b',
                r'\b(published|written|reported|says|explains|notes)\b'
            ]
            
            pattern_matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in article_patterns)
            
            # Calculate structure score
            word_count = len(content.split())
            
            # Article-like content should have:
            # - Multiple sentences
            # - Reasonable sentence length
            # - Some article patterns
            # - Not just a list of short phrases
            
            avg_sentence_length = word_count / max(1, sentence_count)
            
            structure_score = (
                (sentence_count > 5) * 2 +  # Has multiple sentences
                (paragraph_count > 2) * 2 +  # Has multiple paragraphs  
                (pattern_matches > 0) * 3 +  # Has article-like patterns
                (avg_sentence_length > 8) * 2 +  # Reasonable sentence length
                (word_count > 100) * 1  # Substantial content
            )
            
            return structure_score >= 5  # Threshold for article-like content
            
        except Exception as e:
            logger.debug(f"Error checking article structure: {str(e)}")
            return True  # Default to accepting content if check fails

    async def close(self):
        """Close the scraper session."""
        if self.session:
            await self.session.close()


# Convenience function for one-off usage
async def fetch_latest_articles(max_age_hours: int = 24) -> List[Article]:
    """
    Convenience function to fetch latest articles.
    
    Args:
        max_age_hours: Maximum age of articles to fetch
        
    Returns:
        List of articles
    """
    async with NewsletterScraper() as scraper:
        return await scraper.fetch_latest_newsletter(max_age_hours)
