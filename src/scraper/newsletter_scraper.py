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
          # Session timeout configuration
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
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
                            article_content = await self._fetch_external_article_content(article_url)
                            
                # Create Article object
                            if article_content and len(article_content) > 100:
                                content = article_content
                                word_count = len(article_content.split())
                            else:
                                # Fallback: try to get content from newsletter page itself
                                content = self._extract_article_snippet_from_newsletter(link, soup)
                                if not content or len(content) < 100:
                                    content = f"Article content from {article_url}"
                                word_count = len(content.split()) if content != f"Article content from {article_url}" else read_time_minutes * 200

                            article = Article(
                                title=title,
                                content=content,
                                summary=f"{read_time_minutes}-minute read",
                                url=article_url,
                                published_date=published_date,
                                category=category,
                                word_count=word_count
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
                
                parent = parent.parent            # Fallback: analyze the link text itself
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
    
    async def _fetch_external_article_content(self, article_url: str) -> Optional[str]:
        """
        Fetch content from external article URL with improved extraction logic.
        
        Args:
            article_url: URL of the external article
            
        Returns:
            Article content or None if failed
        """
        try:
            logger.debug(f"Fetching external content from: {article_url}")
            
            session = await self._get_session()
            
            # Use different headers for different sites to avoid blocking
            site_headers = self.headers.copy()
            if 'techcrunch.com' in article_url:
                site_headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            elif 'bloomberg.com' in article_url:
                site_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
            elif 'spectrum.ieee.org' in article_url:
                site_headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
            
            # Shorter timeout for external sites to avoid hanging
            timeout = 10 if any(domain in article_url for domain in ['bloomberg.com', 'nytimes.com']) else 15
            
            try:
                async with session.get(article_url, timeout=timeout, headers=site_headers) as response:
                    if response.status == 403:
                        logger.warning(f"Access denied (403) for: {article_url}")
                        return None
                    elif response.status == 404:
                        logger.warning(f"Article not found (404) for: {article_url}")
                        return None
                    elif response.status not in [200, 301, 302]:
                        logger.warning(f"Failed to fetch external article: HTTP {response.status}")
                        return None
                    
                    html = await response.text()
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching article: {article_url}")
                return None
            except Exception as e:
                logger.warning(f"Network error fetching article {article_url}: {str(e)}")
                return None
            
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
            
            if not content:
                # Generic content extraction with multiple strategies
                content = self._extract_content_generic(soup)
            
            if content:
                # Clean and validate content
                content = self._clean_and_validate_content(content, article_url)
                return content
            
            logger.warning(f"No content extracted from: {article_url}")
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching external article content from {article_url}: {str(e)}")
            return None
    
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
                    'main article'
                ],
                'tensorzero.com': [
                    '.post-content',
                    '.blog-content',
                    '.article-content',
                    'main'
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
                    for selector in selectors:
                        elements = soup.select(selector)
                        if elements:
                            content = elements[0].get_text(separator=' ', strip=True)
                            if len(content) > 200:  # Ensure we got substantial content
                                return content
            
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
                
                # Strategy 2: Look for paragraphs in main content areas
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
                lambda: self._extract_from_structured_data(soup)
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
        """Extract content by finding the area with the most paragraphs."""
        try:
            # Find containers with multiple paragraphs
            containers = soup.find_all(['article', 'main', 'div'], recursive=True)
            
            best_content = ""
            max_paragraph_count = 0
            
            for container in containers:
                paragraphs = container.find_all('p', recursive=True)
                if len(paragraphs) > max_paragraph_count:
                    content = container.get_text(separator=' ', strip=True)
                    # Filter out navigation and other non-content areas
                    if len(content) > 500 and not self._is_navigation_content(content):
                        best_content = content
                        max_paragraph_count = len(paragraphs)
            
            return best_content if best_content else None
            
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
    
    def _is_navigation_content(self, content: str) -> bool:
        """Check if content appears to be navigation/header/footer rather than article content."""
        nav_indicators = [
            'subscribe', 'newsletter', 'follow us', 'about us', 'contact',
            'privacy policy', 'terms of service', 'home', 'menu',
            'sign in', 'log in', 'register', 'search', 'trending'
        ]
        
        content_lower = content.lower()
        nav_count = sum(1 for indicator in nav_indicators if indicator in content_lower)
        
        # If more than 3 navigation indicators and short content, likely navigation
        return nav_count > 3 and len(content) < 1000
    
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
        if len(content) < 50:  # Reduced from 100 to 50 for better handling
            logger.debug(f"Content too short ({len(content)} chars) for {url}")
            return None
        
        # Check for paywall indicators
        paywall_indicators = [
            'subscribe to continue reading',
            'this article is for subscribers only',
            'please subscribe',
            'upgrade to premium',
            'sign up for free',
            'create a free account',
            'become a member',
            'unlock this article'
        ]
        
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in paywall_indicators):
            logger.debug(f"Paywall detected for {url}")
            return None
        
        # Limit content length
        if len(content) > 8000:
            content = content[:8000] + "..."
        
        return content
    
    def _clean_html(self, html_content: str) -> str:
        """
        Clean HTML tags from content.
        
        Args:
            html_content: HTML content
            
        Returns:
            Clean text content
        """
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _determine_category(self, title: str, content: str) -> str:
        """
        Determine article category based on title and content.
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            Category string
        """
        text = f"{title} {content}".lower()
        
        categories = {
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'neural', 'gpt', 'llm'],
            'crypto': ['crypto', 'bitcoin', 'ethereum', 'blockchain', 'nft', 'defi'],
            'startup': ['startup', 'funding', 'venture', 'seed', 'series a', 'ipo'],
            'big_tech': ['google', 'apple', 'microsoft', 'amazon', 'meta', 'facebook'],
            'dev': ['developer', 'programming', 'code', 'api', 'framework', 'library'],
            'science': ['research', 'study', 'discovery', 'breakthrough', 'science']        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return 'tech'  # Default category
    
    def _extract_article_snippet_from_newsletter(self, link_element, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract article snippet/preview from the newsletter page itself when external fetch fails.
        
        Args:
            link_element: The link element for the article
            soup: BeautifulSoup object of the newsletter page
            
        Returns:
            Article snippet or None
        """
        try:
            # Look for content near the link (siblings, parent content, etc.)
            parent = link_element.parent
            
            # Check for description paragraphs near the link
            for _ in range(3):  # Check up to 3 levels
                if parent is None:
                    break
                    
                # Look for descriptive text in siblings
                siblings = parent.find_all(['p', 'div', 'span'], recursive=False)
                for sibling in siblings:
                    text = sibling.get_text(strip=True)
                    if len(text) > 50 and text.lower() not in link_element.get_text().lower():
                        # Filter out navigation/header text
                        if not self._is_navigation_content(text):
                            return text[:500]  # Limit to reasonable snippet size
                
                # Look for content within the parent
                text_content = parent.get_text(strip=True)
                if len(text_content) > 100:
                    # Remove the link text itself from the content
                    link_text = link_element.get_text()
                    content = text_content.replace(link_text, '').strip()
                    if len(content) > 50 and not self._is_navigation_content(content):
                        return content[:500]
                
                parent = parent.parent
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting snippet from newsletter: {str(e)}")
            return None
    
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
