#!/usr/bin/env python3
"""
Test Llama summarizer on real TLDR articles - Simple Synchronous Version
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from summarizer.llama_summarizer import LlamaSummarizer

def load_articles(json_path):
    """Load articles from JSON file"""
    print(f"Loading articles from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for successfully extracted articles only
    successful_articles = []
    for article in data.get('articles', []):
        if (article.get('content_extraction_status') == 'success' and 
            article.get('content') and 
            len(article.get('content', '')) > 100):  # Minimum content length
            successful_articles.append(article)
    
    print(f"Found {len(successful_articles)} successfully extracted articles")
    return successful_articles, data.get('metadata', {})

def summarize_articles(articles, max_articles=5):
    """Summarize articles using Llama"""
    print("\n" + "="*60)
    print("INITIALIZING LLAMA SUMMARIZER")
    print("="*60)
    
    try:
        summarizer = LlamaSummarizer()
        print("✅ Llama summarizer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize summarizer: {e}")
        return []
    
    results = []
    
    # Limit articles for testing
    test_articles = articles[:max_articles]
    print(f"\nProcessing {len(test_articles)} articles...")
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING ARTICLE {i}/{len(test_articles)}")
        print(f"{'='*60}")
        print(f"Title: {article['title'][:80]}...")
        print(f"Category: {article['category']}")
        print(f"Word Count: {article['word_count']}")
        print(f"URL: {article['url']}")
        
        try:
            # Generate TikTok summary
            print("\n🤖 Generating TikTok summary...")
            start_time = datetime.now()
            
            tiktok_summary = summarizer.summarize_for_tiktok(
                title=article['title'],
                content=article['content'],
                category=article['category']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result entry
            result = {
                'original_article': {
                    'title': article['title'],
                    'url': article['url'],
                    'category': article['category'],
                    'word_count': article['word_count'],
                    'published_date': article['published_date']
                },
                'tiktok_summary': tiktok_summary,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'processing_status': 'success'
            }
            
            print(f"✅ Successfully generated summary in {processing_time:.1f}s")
            print(f"📝 Summary ({len(tiktok_summary)} chars): {tiktok_summary[:100]}...")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed to process article: {e}")
            
            # Add failed result
            result = {
                'original_article': {
                    'title': article['title'],
                    'url': article['url'],
                    'category': article['category'],
                    'word_count': article['word_count'],
                    'published_date': article['published_date']
                },
                'tiktok_summary': None,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': 0,
                'processing_status': 'failed',
                'error_message': str(e)
            }
            results.append(result)
    
    return results

def save_results(results, original_metadata, output_path):
    """Save summarization results to JSON"""
    
    # Calculate statistics
    successful_count = len([r for r in results if r['processing_status'] == 'success'])
    failed_count = len([r for r in results if r['processing_status'] == 'failed'])
    success_rate = (successful_count / len(results) * 100) if results else 0
    
    # Calculate average processing time for successful summaries
    successful_times = [r['processing_time_seconds'] for r in results if r['processing_status'] == 'success']
    avg_processing_time = sum(successful_times) / len(successful_times) if successful_times else 0
    
    # Create output data structure
    output_data = {
        'metadata': {
            'generation_timestamp': datetime.now().isoformat(),
            'llama_model': 'meta-llama/Llama-3.2-3B-Instruct',
            'summarizer_version': '1.0',
            'original_data_source': 'tldr_articles_20250610_231259.json',
            'original_extraction_timestamp': original_metadata.get('extraction_timestamp'),
            'processing_statistics': {
                'total_processed': len(results),
                'successful_summaries': successful_count,
                'failed_summaries': failed_count,
                'success_rate_percent': round(success_rate, 1),
                'average_processing_time_seconds': round(avg_processing_time, 2)
            }
        },
        'tiktok_summaries': results
    }
    
    # Save to file
    print(f"\n💾 Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved successfully!")
    return output_data

def main():
    print("="*60)
    print("LLAMA SUMMARIZER - REAL ARTICLES TEST")
    print("="*60)
    
    # File paths
    input_file = "data/tldr_articles_20250610_231259.json"
    output_file = f"data/tiktok_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    try:
        # Load articles
        articles, metadata = load_articles(input_file)
        
        if not articles:
            print("❌ No successfully extracted articles found")
            return
        
        # Show article categories
        categories = {}
        for article in articles:
            cat = article['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nArticle categories:")
        for cat, count in categories.items():
            print(f"  {cat}: {count} articles")
        
        # Process articles (start with 3 for testing)
        print(f"\nAvailable articles: {len(articles)}")
        print("Starting with 3 articles for initial testing...")
        max_articles = 3
        
        # Summarize articles
        results = summarize_articles(articles, max_articles)
        
        if not results:
            print("❌ No results generated")
            return
        
        # Save results
        output_data = save_results(results, metadata, output_file)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Total articles processed: {len(results)}")
        print(f"✅ Successful summaries: {output_data['metadata']['processing_statistics']['successful_summaries']}")
        print(f"❌ Failed summaries: {output_data['metadata']['processing_statistics']['failed_summaries']}")
        print(f"📈 Success rate: {output_data['metadata']['processing_statistics']['success_rate_percent']:.1f}%")
        print(f"⚡ Average processing time: {output_data['metadata']['processing_statistics']['average_processing_time_seconds']:.2f}s")
        print(f"💾 Results saved to: {output_file}")
        
        # Show sample summaries
        successful_results = [r for r in results if r['processing_status'] == 'success']
        if successful_results:
            print(f"\n📝 SAMPLE SUMMARIES:")
            for i, result in enumerate(successful_results[:2], 1):
                print(f"\n{i}. {result['original_article']['title'][:60]}...")
                print(f"   Category: {result['original_article']['category']}")
                print(f"   Processing time: {result['processing_time_seconds']:.1f}s")
                print(f"   Summary: {result['tiktok_summary'][:150]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
