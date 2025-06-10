#!/usr/bin/env python3
"""
Simple demonstration of the adaptive extractor's learning capabilities.
"""

import asyncio
import sys
from pathlib import Path
sys.path.append('src')

from scraper.adaptive_extractor import AdaptiveContentExtractor
from bs4 import BeautifulSoup
import aiohttp

async def demo_adaptive_learning():
    """Demonstrate the adaptive learning on a new website."""
    print("=== Adaptive Content Extractor Demo ===\n")
    
    # Initialize the extractor
    extractor = AdaptiveContentExtractor()
    
    # Show current learned patterns
    stats = extractor.get_pattern_stats()
    print(f"📊 Current Knowledge Base:")
    print(f"   • Total Patterns: {stats.get('total_patterns', 0)}")
    print(f"   • Average Success Rate: {stats.get('avg_success_rate', 0):.2f}")
    print(f"   • Total Usage: {stats.get('total_usage', 0)}")
    print()
    
    if stats.get('total_patterns', 0) > 0:
        print("🎯 Top Learned Domains:")
        for i, (domain, success_rate, usage_count) in enumerate(stats.get('top_domains', [])[:5], 1):
            print(f"   {i}. {domain}: {usage_count} uses, {success_rate:.2f} success rate")
        print()
    
    # Test with a sample HTML that simulates a new website structure
    test_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Article</title></head>
    <body>
        <nav>Navigation</nav>
        <header>Header</header>
        <main class="main-content">
            <article class="blog-post">
                <h1>Sample Article Title</h1>
                <div class="article-meta">Published on June 10, 2025</div>
                <div class="content-body">
                    <p>This is the first paragraph of the article content. It contains meaningful information about the topic.</p>
                    <p>This is the second paragraph with more detailed information. According to research, adaptive content extraction is highly effective.</p>
                    <p>Furthermore, machine learning approaches can significantly improve extraction accuracy over time.</p>
                    <p>The system learns from each extraction attempt and builds a knowledge base of effective selectors.</p>
                    <p>Finally, this demonstrates how the AI can identify and extract meaningful content from unknown website structures.</p>
                </div>
            </article>
        </main>
        <footer>Footer</footer>
    </body>
    </html>
    """
    
    print("🧪 Testing AI Discovery on Unknown Website Structure...")
    soup = BeautifulSoup(test_html, 'html.parser')
    
    # Test the discovery mechanism
    content, extraction_method = await extractor.extract_content(soup, "https://example-new-site.com/test-article")
    
    if content:
        print(f"✅ Successfully extracted content using: {extraction_method}")
        print(f"📝 Content preview: {content[:200]}...")
        print(f"📏 Content length: {len(content)} characters")
        print()
        
        # Show updated stats
        new_stats = extractor.get_pattern_stats()
        if new_stats.get('total_patterns', 0) > stats.get('total_patterns', 0):
            print("🎉 New Pattern Learned!")
            print(f"   • Total patterns increased: {stats.get('total_patterns', 0)} → {new_stats.get('total_patterns', 0)}")
    else:
        print("❌ Failed to extract content")
    
    print("\n=== Demo Complete ===")
    print("The adaptive extractor demonstrates:")
    print("• ✅ Pattern persistence across sessions")
    print("• ✅ Real-time learning from new websites") 
    print("• ✅ Automatic selector discovery")
    print("• ✅ Success rate tracking")
    print("• ✅ Zero manual configuration required")

if __name__ == "__main__":
    asyncio.run(demo_adaptive_learning())
