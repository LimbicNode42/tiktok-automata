#!/usr/bin/env python3
"""
Complete integration example: Newsletter scraping + Modern Llama summarization for TikTok.
Optimized for GTX 1060 6GB hardware.
"""

import asyncio
import sys
import os
from datetime import datetime
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.scraper.newsletter_scraper import NewsletterScraper
    from src.summarizer.llama_summarizer import create_tiktok_summarizer, TikTokSummaryConfig
    print("✓ Successfully imported all components")
except ImportError as e:
    print(f"✗ Failed to import components: {e}")
    sys.exit(1)


async def full_tiktok_pipeline_demo():
    """
    Complete demo: Scrape TLDR newsletters → Generate TikTok summaries with Modern Llama.
    """
    print("🚀 Full TikTok Content Pipeline Demo")
    print("🔧 Hardware: GTX 1060 6GB optimized")
    print("=" * 60)
    
    # Step 1: Initialize the newsletter scraper
    print("\n📰 Step 1: Initializing Newsletter Scraper...")
    scraper = NewsletterScraper()
      # Step 2: Initialize the Llama summarizer for your hardware
    print("🦙 Step 2: Initializing Llama Summarizer...")
    print("   • Model: Llama 3.2-3B (optimized for GTX 1060 6GB)")
    print("   • Expected VRAM: ~1.8GB")
    print("   • Performance: Fast & High Quality")
    
    config = TikTokSummaryConfig(
        target_duration=60,  # 60-second TikToks
        temperature=0.8,     # More creative for engaging content
        use_gpu=True
    )
    
    summarizer = create_tiktok_summarizer()
    summarizer.config = config
    
    try:
        # Step 3: Fetch latest newsletter articles
        print("\n📥 Step 3: Fetching Latest Newsletter Articles...")
        articles = await scraper.fetch_latest_newsletter(
            max_age_hours=72,  # Last 3 days
            max_entries=3      # 3 recent newsletters
        )
        
        if not articles:
            print("❌ No articles found!")
            return
        
        print(f"✅ Found {len(articles)} articles")
        
        # Filter successful extractions only
        successful_articles = [a for a in articles if a.content_extraction_status == "success"]
        print(f"📊 {len(successful_articles)} articles with successful content extraction")
        
        if not successful_articles:
            print("⚠️  No articles with successful content extraction found")
            return
        
        # Take top 5 for demo to save time
        demo_articles = successful_articles[:5]
        print(f"🎯 Processing top {len(demo_articles)} articles for TikTok content")
        
        # Step 4: Initialize Llama model
        print("\n🤖 Step 4: Loading Llama Model...")
        await summarizer.initialize()
        
        # Step 5: Generate TikTok summaries
        print("\n✨ Step 5: Generating TikTok Summaries...")
        
        results = []
        for i, article in enumerate(demo_articles, 1):
            print(f"\n   📝 Processing {i}/{len(demo_articles)}: {article.title[:50]}...")
            print(f"      Category: {article.category} | Words: {article.word_count}")
            
            # Generate TikTok summary
            tiktok_summary = await summarizer.summarize_for_tiktok(article, target_duration=60)
            
            if tiktok_summary:
                print(f"      ✅ Generated TikTok summary ({len(tiktok_summary)} chars)")
                
                result = {
                    'title': article.title,
                    'category': article.category,
                    'url': article.url,
                    'word_count': article.word_count,
                    'tiktok_summary': tiktok_summary,
                    'estimated_reading_time': len(tiktok_summary.split()) / 2.5,  # ~150 WPM = 2.5 WPS
                    'success': True
                }
            else:
                print(f"      ❌ Failed to generate summary")
                result = {
                    'title': article.title,
                    'category': article.category,
                    'success': False
                }
            
            results.append(result)
        
        # Step 6: Display results and save
        print("\n🎉 Step 6: Results Summary")
        print("=" * 60)
        
        successful_summaries = [r for r in results if r.get('success', False)]
        print(f"✅ Successfully generated {len(successful_summaries)}/{len(results)} TikTok summaries")
        
        for i, result in enumerate(successful_summaries, 1):
            print(f"\n📱 TikTok Video {i}: {result['title'][:40]}...")
            print(f"   🏷️  Category: {result['category']}")
            print(f"   ⏱️  Reading time: {result['estimated_reading_time']:.1f} seconds")
            print(f"   📝 Script preview:")
            print(f"      {result['tiktok_summary'][:120]}...")
            print(f"   🔗 Source: {result['url']}")
          # Save results to file
        summarizer_data_dir = Path("src/summarizer/data")
        summarizer_data_dir.mkdir(parents=True, exist_ok=True)
        output_file = summarizer_data_dir / f"tiktok_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        print(f"   • Newsletter extraction: {len(articles)} articles found")
        print(f"   • Content extraction success rate: {len(successful_articles)}/{len(articles)} ({len(successful_articles)/len(articles)*100:.1f}%)")
        print(f"   • TikTok summarization success rate: {len(successful_summaries)}/{len(demo_articles)} ({len(successful_summaries)/len(demo_articles)*100:.1f}%)")
        
        # Cleanup
        await summarizer.cleanup()
        await scraper.close()
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            await summarizer.cleanup()
            await scraper.close()
        except:
            pass


async def quick_demo_with_sample_data():
    """Quick demo using sample data when live scraping isn't available."""
    print("\n🔄 Running Quick Demo with Sample Data...")
      # Load existing scraped data if available
    scraper_data_dir = Path("src/scraper/data")
    data_files = list(scraper_data_dir.glob("tldr_articles_*.json"))
    
    if data_files:
        latest_file = max(data_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Loading articles from: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            articles_data = json.load(f)
        
        # Convert to Article objects (simplified)
        from scraper.newsletter_scraper import Article
        
        sample_articles = []
        for item in articles_data[:3]:  # Take first 3
            if item.get('content_extraction_status') == 'success' and len(item.get('content', '')) > 500:
                article = Article(
                    title=item['title'],
                    content=item['content'],
                    summary=item.get('summary', ''),
                    url=item['url'],
                    published_date=datetime.fromisoformat(item['published_date'].replace('Z', '+00:00')),
                    category=item.get('category', 'tech'),
                    word_count=item.get('word_count', 0),
                    content_extraction_status=item['content_extraction_status']
                )
                sample_articles.append(article)
        
        if sample_articles:
            print(f"✅ Loaded {len(sample_articles)} articles for demo")
              # Initialize summarizer
            summarizer = create_tiktok_summarizer()
            await summarizer.initialize()
            
            # Generate summaries
            for i, article in enumerate(sample_articles, 1):
                print(f"\n🎬 Generating TikTok script {i}: {article.title[:40]}...")
                summary = await summarizer.summarize_for_tiktok(article)
                
                if summary:
                    print(f"✅ Generated! Preview:")
                    print(f"   {summary[:150]}...")
                else:
                    print("❌ Failed to generate")
            
            await summarizer.cleanup()
            
        else:
            print("❌ No suitable articles found in data file")
    else:
        print("❌ No previous scraping data found. Run newsletter scraper first.")


async def main():
    """Main entry point."""
    print("🤖 TikTok Automata: Modern Llama Integration")
    print("🔧 Optimized for GTX 1060 6GB")
    print("=" * 60)
    
    try:
        # Try full pipeline first
        await full_tiktok_pipeline_demo()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Full pipeline failed: {str(e)}")
        print("🔄 Falling back to quick demo...")
        
        try:
            await quick_demo_with_sample_data()
        except Exception as e2:
            print(f"❌ Quick demo also failed: {str(e2)}")


if __name__ == "__main__":
    asyncio.run(main())
