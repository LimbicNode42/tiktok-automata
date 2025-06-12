#!/usr/bin/env python3
"""
Test Llama summarizer on real TLDR articles and save results to JSON
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from summarizer.llama_summarizer import create_tiktok_summarizer
    from scraper.newsletter_scraper import Article
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def load_articles_from_json(file_path: str) -> list:
    """Load successfully extracted articles from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = []
        successful_articles = [
            article for article in data['articles'] 
            if article.get('content_extraction_status') == 'success' and 
               article.get('word_count', 0) > 100  # Only articles with substantial content
        ]
        
        print(f"Found {len(successful_articles)} successfully extracted articles")
        
        for article_data in successful_articles:
            article = Article(
                title=article_data['title'],
                url=article_data['url'],
                content=article_data['content'],
                summary=article_data.get('summary', ''),
                published_date=article_data.get('published_date', ''),
                category=article_data.get('category', 'tech')
            )
            article.content_extraction_status = 'success'
            articles.append(article)
        
        return articles
    
    except Exception as e:
        print(f"❌ Error loading articles: {e}")
        return []

def save_results_to_json(results: list, output_file: str):
    """Save summarization results to JSON file."""
    output_data = {
        "metadata": {
            "processing_timestamp": datetime.now().isoformat(),
            "model_used": "meta-llama/Llama-3.2-3B-Instruct",
            "total_articles_processed": len(results),
            "successful_summaries": sum(1 for r in results if r['success']),
            "failed_summaries": sum(1 for r in results if not r['success'])
        },
        "summaries": []
    }
    
    for result in results:
        article = result['article']
        summary_data = {
            "title": article.title,
            "url": article.url,
            "category": article.category,
            "word_count": len(article.content.split()),
            "success": result['success'],
            "tiktok_summary_30s": result.get('tiktok_summary_30s', ''),
            "tiktok_summary_60s": result.get('tiktok_summary_60s', ''),
            "processing_time_30s": result.get('processing_time_30s', 0),
            "processing_time_60s": result.get('processing_time_60s', 0),
            "error_message": result.get('error_message', '')
        }
        output_data["summaries"].append(summary_data)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

async def test_real_articles():
    """Test Llama summarizer on real TLDR articles."""
    print("🦙 Testing Llama Summarizer on Real TLDR Articles")
    print("=" * 60)
    
    # Load articles
    input_file = "data/tldr_articles_20250610_231259.json"
    articles = load_articles_from_json(input_file)
    
    if not articles:
        print("❌ No articles found to process")
        return
    
    print(f"📰 Loaded {len(articles)} articles for processing")
    
    # Limit to first 5 articles for testing (to avoid long processing times)
    test_articles = articles[:5]
    print(f"🎯 Processing first {len(test_articles)} articles for testing\n")
    
    # Initialize summarizer
    print("🔄 Initializing Llama summarizer...")
    summarizer = create_tiktok_summarizer()
    
    try:
        await summarizer.initialize()
        print("✅ Summarizer initialized successfully!\n")
        
        results = []
        
        for i, article in enumerate(test_articles, 1):
            print(f"--- Article {i}/{len(test_articles)}: {article.title[:50]}... ---")
            print(f"Category: {article.category}")
            print(f"Word count: {len(article.content.split())}")
            
            result = {
                'article': article,
                'success': False,
                'processing_order': i
            }
            
            try:
                # Generate 30-second summary
                print("  🎬 Generating 30-second TikTok summary...")
                start_time = datetime.now()
                summary_30s = await summarizer.summarize_for_tiktok(article, target_duration=30)
                processing_time_30s = (datetime.now() - start_time).total_seconds()
                
                if summary_30s:
                    result['tiktok_summary_30s'] = summary_30s
                    result['processing_time_30s'] = processing_time_30s
                    print(f"  ✅ 30s summary: {len(summary_30s)} chars in {processing_time_30s:.1f}s")
                    print(f"     Preview: {summary_30s[:100]}...")
                else:
                    print("  ❌ Failed to generate 30s summary")
                
                # Generate 60-second summary
                print("  🎬 Generating 60-second TikTok summary...")
                start_time = datetime.now()
                summary_60s = await summarizer.summarize_for_tiktok(article, target_duration=60)
                processing_time_60s = (datetime.now() - start_time).total_seconds()
                
                if summary_60s:
                    result['tiktok_summary_60s'] = summary_60s
                    result['processing_time_60s'] = processing_time_60s
                    print(f"  ✅ 60s summary: {len(summary_60s)} chars in {processing_time_60s:.1f}s")
                    print(f"     Preview: {summary_60s[:100]}...")
                    result['success'] = True
                else:
                    print("  ❌ Failed to generate 60s summary")
                
            except Exception as e:
                print(f"  ❌ Error processing article: {e}")
                result['error_message'] = str(e)
            
            results.append(result)
            print()
        
        # Clean up
        await summarizer.cleanup()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/tiktok_summaries_{timestamp}.json"
        save_results_to_json(results, output_file)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print("=" * 60)
        print("🎉 PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {successful}/{len(results)} articles")
        print(f"📁 Results saved to: {output_file}")
        
        if successful > 0:
            avg_time_30s = sum(r.get('processing_time_30s', 0) for r in results if r['success']) / successful
            avg_time_60s = sum(r.get('processing_time_60s', 0) for r in results if r['success']) / successful
            print(f"⚡ Average processing time: {avg_time_30s:.1f}s (30s), {avg_time_60s:.1f}s (60s)")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
if __name__ == "__main__":
    asyncio.run(test_real_articles())
