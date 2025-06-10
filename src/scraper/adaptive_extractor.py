#!/usr/bin/env python3
"""
Adaptive Content Extractor with AI-powered learning for new websites.
Automatically discovers and learns extraction patterns for unseen sites.
"""

import json
import asyncio
import re
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import pickle

from bs4 import BeautifulSoup, Tag
from loguru import logger
import aiohttp


@dataclass
class ExtractionPattern:
    """Represents a learned extraction pattern for a website."""
    domain: str
    selectors: List[str]
    success_rate: float
    last_used: datetime
    usage_count: int
    confidence_score: float
    pattern_type: str  # 'css_selector', 'semantic', 'ai_discovered'
    
    def to_dict(self) -> dict:
        return {
            'domain': self.domain,
            'selectors': self.selectors,
            'success_rate': self.success_rate,
            'last_used': self.last_used.isoformat(),
            'usage_count': self.usage_count,
            'confidence_score': self.confidence_score,
            'pattern_type': self.pattern_type
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExtractionPattern':
        return cls(
            domain=data['domain'],
            selectors=data['selectors'],
            success_rate=data['success_rate'],
            last_used=datetime.fromisoformat(data['last_used']),
            usage_count=data['usage_count'],
            confidence_score=data['confidence_score'],
            pattern_type=data['pattern_type']
        )


@dataclass
class ContentCandidate:
    """Represents a potential content extraction candidate."""
    selector: str
    content: str
    score: float
    word_count: int
    paragraph_count: int
    semantic_indicators: List[str]
    structure_quality: float


class AdaptiveContentExtractor:
    """
    AI-powered content extractor that learns and adapts to new websites.
    """
    
    def __init__(self, patterns_file: str = "data/extraction_patterns.json"):
        self.patterns_file = Path(patterns_file)
        self.patterns: Dict[str, ExtractionPattern] = {}
        self.load_patterns()
        
        # Common article indicators for pattern discovery
        self.article_indicators = [
            'published', 'written', 'author', 'byline', 'date',
            'according to', 'research shows', 'study finds',
            'however', 'moreover', 'furthermore', 'meanwhile',
            'first', 'second', 'third', 'finally', 'conclusion'
        ]
        
        # High-quality content selectors for AI discovery
        self.discovery_selectors = [
            'article', 'main', '[role="main"]',
            '.article', '.post', '.content', '.story',
            '.entry-content', '.post-content', '.article-content',
            '.blog-content', '.story-body', '.article-body',
            '.prose', '.markdown-body', '.rich-text',
            '[class*="content"]', '[class*="article"]', '[class*="post"]',
            '[id*="content"]', '[id*="article"]', '[id*="post"]'
        ]
    
    def load_patterns(self):
        """Load learned extraction patterns from file."""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.patterns = {
                        domain: ExtractionPattern.from_dict(pattern_data)
                        for domain, pattern_data in data.items()
                    }
                logger.info(f"Loaded {len(self.patterns)} extraction patterns")
        except Exception as e:
            logger.warning(f"Could not load patterns: {e}")
            self.patterns = {}
    
    def save_patterns(self):
        """Save learned extraction patterns to file."""
        try:
            self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
            data = {domain: pattern.to_dict() for domain, pattern in self.patterns.items()}
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {len(self.patterns)} extraction patterns")
        except Exception as e:
            logger.error(f"Could not save patterns: {e}")
    
    async def extract_content(self, soup: BeautifulSoup, url: str) -> Tuple[Optional[str], str]:
        """
        Extract content using adaptive learning approach.
        Returns (content, extraction_method)
        """
        domain = self._get_domain(url)
        
        # Try learned patterns first
        if domain in self.patterns:
            content = await self._extract_with_learned_pattern(soup, domain)
            if content:
                self._update_pattern_success(domain, True)
                return content, f"learned_pattern_{self.patterns[domain].pattern_type}"
            else:
                self._update_pattern_success(domain, False)
        
        # Discover new patterns
        content, new_pattern = await self._discover_extraction_pattern(soup, url)
        if content and new_pattern:
            self._learn_new_pattern(domain, new_pattern)
            return content, "ai_discovered"
        
        # Fallback to basic semantic extraction
        content = self._semantic_extraction_fallback(soup)
        if content:
            return content, "semantic_fallback"
        
        return None, "extraction_failed"
    
    async def _extract_with_learned_pattern(self, soup: BeautifulSoup, domain: str) -> Optional[str]:
        """Extract content using a learned pattern."""
        pattern = self.patterns[domain]
        
        for selector in pattern.selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(separator=' ', strip=True)
                    if self._validate_content_quality(content):
                        pattern.last_used = datetime.now()
                        pattern.usage_count += 1
                        return content
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed for {domain}: {e}")
                continue
        
        return None
    
    async def _discover_extraction_pattern(self, soup: BeautifulSoup, url: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Use AI-powered pattern discovery to find content extraction selectors.
        """
        candidates = []
        
        # Phase 1: Score all potential content containers
        for selector in self.discovery_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    candidate = self._analyze_content_candidate(element, selector)
                    if candidate and candidate.score > 0.3:  # Minimum quality threshold
                        candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Error analyzing selector '{selector}': {e}")
                continue
        
        if not candidates:
            return None, None
        
        # Phase 2: Rank candidates by multi-factor scoring
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Phase 3: Validate top candidates
        for candidate in candidates[:5]:  # Test top 5 candidates
            if self._validate_content_quality(candidate.content):
                # Build selector hierarchy for this successful pattern
                selectors = self._build_selector_hierarchy(candidate, soup)
                logger.info(f"Discovered new extraction pattern for {self._get_domain(url)}: {selectors[0]} (score: {candidate.score:.2f})")
                return candidate.content, selectors
        
        return None, None
    
    def _analyze_content_candidate(self, element: Tag, selector: str) -> Optional[ContentCandidate]:
        """Analyze an element to determine its content extraction potential."""
        try:
            content = element.get_text(separator=' ', strip=True)
            
            if len(content) < 100:  # Too short
                return None
            
            # Basic metrics
            word_count = len(content.split())
            paragraph_count = len(element.find_all('p'))
            
            # Semantic indicators
            semantic_indicators = []
            content_lower = content.lower()
            for indicator in self.article_indicators:
                if indicator in content_lower:
                    semantic_indicators.append(indicator)
            
            # Structure quality analysis
            structure_quality = self._analyze_structure_quality(element)
            
            # Calculate composite score
            score = self._calculate_content_score(
                word_count, paragraph_count, semantic_indicators, 
                structure_quality, element, selector
            )
            
            return ContentCandidate(
                selector=selector,
                content=content,
                score=score,
                word_count=word_count,
                paragraph_count=paragraph_count,
                semantic_indicators=semantic_indicators,
                structure_quality=structure_quality
            )
            
        except Exception as e:
            logger.debug(f"Error analyzing element with selector '{selector}': {e}")
            return None
    
    def _calculate_content_score(self, word_count: int, paragraph_count: int, 
                               semantic_indicators: List[str], structure_quality: float,
                               element: Tag, selector: str) -> float:
        """Calculate a composite score for content extraction potential."""
        
        # Base score from content length (normalized)
        length_score = min(word_count / 1000, 1.0) * 0.3
        
        # Paragraph density score
        para_score = min(paragraph_count / 10, 1.0) * 0.2
        
        # Semantic indicators score
        semantic_score = min(len(semantic_indicators) / 5, 1.0) * 0.2
        
        # Structure quality score
        structure_score = structure_quality * 0.2
        
        # Selector quality score (prefer specific article selectors)
        selector_score = 0.1
        if 'article' in selector:
            selector_score = 0.1
        elif any(x in selector for x in ['content', 'post', 'story']):
            selector_score = 0.08
        elif selector in ['main', '[role="main"]']:
            selector_score = 0.06
        
        # Penalize navigation/footer content
        element_classes = ' '.join(element.get('class', [])).lower()
        element_id = (element.get('id') or '').lower()
        
        penalty = 0
        if any(x in element_classes + ' ' + element_id for x in ['nav', 'menu', 'header', 'footer', 'sidebar']):
            penalty = 0.3
        
        final_score = length_score + para_score + semantic_score + structure_score + selector_score - penalty
        return max(0, min(1.0, final_score))
    
    def _analyze_structure_quality(self, element: Tag) -> float:
        """Analyze the structural quality of content (headings, lists, etc.)."""
        try:
            # Count structural elements
            headings = len(element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
            lists = len(element.find_all(['ul', 'ol']))
            links = len(element.find_all('a'))
            emphasis = len(element.find_all(['strong', 'em', 'b', 'i']))
            
            # Calculate structure score
            structure_elements = headings * 0.3 + lists * 0.2 + emphasis * 0.1
            
            # Penalize too many links (likely navigation)
            text_length = len(element.get_text())
            if text_length > 0:
                link_density = links / (text_length / 100)  # Links per 100 chars
                if link_density > 0.1:  # High link density
                    structure_elements *= 0.5
            
            return min(structure_elements / 10, 1.0)
            
        except Exception as e:
            logger.debug(f"Error analyzing structure: {e}")
            return 0.5
    
    def _build_selector_hierarchy(self, candidate: ContentCandidate, soup: BeautifulSoup) -> List[str]:
        """Build a hierarchy of selectors for robust extraction."""
        selectors = [candidate.selector]
        
        # Try to find more specific selectors for the same content
        element = soup.select(candidate.selector)[0]
        
        # Add ID-based selector if available
        if element.get('id'):
            selectors.append(f"#{element.get('id')}")
        
        # Add class-based selectors
        if element.get('class'):
            for class_name in element.get('class'):
                if class_name and len(class_name) > 2:  # Avoid single-letter classes
                    selectors.append(f".{class_name}")
        
        # Add tag-based selectors
        if element.name in ['article', 'main', 'section']:
            selectors.append(element.name)
        
        return selectors[:5]  # Limit to top 5 selectors
    
    def _validate_content_quality(self, content: str) -> bool:
        """Validate if extracted content meets quality standards."""
        if not content or len(content) < 200:
            return False
        
        words = content.split()
        if len(words) < 50:
            return False
        
        # Check for article-like patterns
        sentence_count = len(re.findall(r'[.!?]+', content))
        if sentence_count < 3:
            return False
        
        # Check for paywall indicators
        paywall_indicators = [
            'subscribe to continue', 'sign up to read', 'premium content',
            'paywall', 'subscription required', 'upgrade to premium'
        ]
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in paywall_indicators) and len(content) < 500:
            return False
        
        return True
    
    def _semantic_extraction_fallback(self, soup: BeautifulSoup) -> Optional[str]:
        """Fallback semantic extraction using content density analysis."""
        try:
            # Find all text containers
            containers = soup.find_all(['div', 'section', 'article', 'main'])
            
            best_content = None
            best_score = 0
            
            for container in containers:
                text = container.get_text(separator=' ', strip=True)
                if len(text) < 200:
                    continue
                
                # Calculate content density score
                paragraphs = container.find_all('p')
                words = len(text.split())
                
                # Prefer containers with good paragraph structure
                para_density = len(paragraphs) / max(1, len(text) / 1000)
                word_score = min(words / 500, 1.0)
                
                score = para_density * 0.7 + word_score * 0.3
                
                if score > best_score and self._validate_content_quality(text):
                    best_score = score
                    best_content = text
            
            return best_content
            
        except Exception as e:
            logger.debug(f"Error in semantic fallback: {e}")
            return None
    
    def _learn_new_pattern(self, domain: str, selectors: List[str]):
        """Learn a new extraction pattern for a domain."""
        pattern = ExtractionPattern(
            domain=domain,
            selectors=selectors,
            success_rate=1.0,  # Start optimistic
            last_used=datetime.now(),
            usage_count=1,
            confidence_score=0.8,  # High confidence for discovered patterns
            pattern_type='ai_discovered'
        )
        
        self.patterns[domain] = pattern
        self.save_patterns()
        logger.info(f"Learned new extraction pattern for {domain}: {selectors}")
    
    def _update_pattern_success(self, domain: str, success: bool):
        """Update pattern success rate based on extraction results."""
        if domain in self.patterns:
            pattern = self.patterns[domain]
            
            # Update success rate with exponential moving average
            alpha = 0.1  # Learning rate
            if success:
                pattern.success_rate = pattern.success_rate * (1 - alpha) + alpha
            else:
                pattern.success_rate = pattern.success_rate * (1 - alpha)
            
            # Update confidence based on success rate and usage
            pattern.confidence_score = (
                pattern.success_rate * 0.7 + 
                min(pattern.usage_count / 10, 1.0) * 0.3
            )
            
            # Remove patterns with consistently low success
            if pattern.success_rate < 0.2 and pattern.usage_count > 5:
                logger.warning(f"Removing low-performing pattern for {domain} (success rate: {pattern.success_rate:.2f})")
                del self.patterns[domain]
                self.save_patterns()
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            return urlparse(url).netloc.lower()
        except:
            return url.split('/')[2].lower() if '/' in url else url
    
    def get_pattern_stats(self) -> Dict:
        """Get statistics about learned patterns."""
        if not self.patterns:
            return {"total_patterns": 0}
        
        success_rates = [p.success_rate for p in self.patterns.values()]
        usage_counts = [p.usage_count for p in self.patterns.values()]
        
        return {
            "total_patterns": len(self.patterns),
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "total_usage": sum(usage_counts),
            "pattern_types": list(set(p.pattern_type for p in self.patterns.values())),
            "top_domains": sorted(
                [(domain, p.success_rate, p.usage_count) for domain, p in self.patterns.items()],
                key=lambda x: x[2], reverse=True
            )[:10]
        }
    
    def cleanup_old_patterns(self, days: int = 30):
        """Remove patterns that haven't been used recently."""
        cutoff = datetime.now() - timedelta(days=days)
        removed = []
        
        for domain, pattern in list(self.patterns.items()):
            if pattern.last_used < cutoff and pattern.usage_count < 3:
                removed.append(domain)
                del self.patterns[domain]
        
        if removed:
            logger.info(f"Cleaned up {len(removed)} old patterns: {removed}")
            self.save_patterns()
        
        return removed
