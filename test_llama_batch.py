#!/usr/bin/env python3
"""
Test the Llama summarizer against successfully extracted articles from the JSON data.
This script processes the 41 successful articles and evaluates the summarizer performance.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(json_file: str) -> List[Dict[str, Any]]:
    """Load successful articles from the JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter only successful articles
        successful_articles = [
            article for article in data.get('articles', [])
            if article.get('content_extraction_status') == 'success'
            and not article.get('content', '').startswith('[CONTENT EXTRACTION FAILED]')
            and article.get('word_count', 0) > 100  # Ensure meaningful content
        ]
        
        logger.info(f"Loaded {len(successful_articles)} successful articles from {json_file}")
        return successful_articles
        
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        return []

def test_summarizer_batch(articles: List[Dict[str, Any]], max_articles: int = 10):
    """Test the Llama summarizer on a batch of articles."""
    try:
        # Import the summarizer
        from src.summarizer.llama_summarizer import create_tiktok_summarizer
        
        # Create summarizer
        logger.info("Initializing Llama summarizer...")
        summarizer = create_tiktok_summarizer()
        
        # Test articles
        results = []
        successful_summaries = 0
        total_time = 0
        
        for i, article in enumerate(articles[:max_articles]):
            logger.info(f"Processing article {i+1}/{min(len(articles), max_articles)}: {article['title'][:60]}...")
            
            try:
                start_time = time.time()
                
                # Generate TikTok summary
                summary = summarizer.generate_tiktok_summary(
                    title=article['title'],
                    content=article['content'],
                    category=article.get('category', 'tech')
                )
                
                processing_time = time.time() - start_time
                total_time += processing_time
                
                if summary and summary.strip():
                    successful_summaries += 1
                    logger.info(f"✓ Generated summary in {processing_time:.2f}s")
                    
                    results.append({
                        'title': article['title'],
                        'category': article.get('category', 'tech'),
                        'url': article['url'],
                        'word_count': article.get('word_count', 0),
                        'summary': summary,
                        'processing_time': processing_time,
                        'status': 'success'
                    })
                else:
                    logger.warning(f"✗ Empty summary generated")
                    results.append({
                        'title': article['title'],
                        'category': article.get('category', 'tech'),
                        'status': 'failed',
                        'error': 'Empty summary'
                    })
                    
            except Exception as e:
                logger.error(f"✗ Failed to process article: {e}")
                results.append({
                    'title': article['title'],
                    'category': article.get('category', 'tech'),
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Performance summary
        avg_time = total_time / max(1, successful_summaries)
        success_rate = (successful_summaries / len(results)) * 100
        
        logger.info(f"\n=== BATCH TEST RESULTS ===")
        logger.info(f"Articles processed: {len(results)}")
        logger.info(f"Successful summaries: {successful_summaries}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Average processing time: {avg_time:.2f}s")
        logger.info(f"Total time: {total_time:.2f}s")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"data/llama_test_results_{timestamp}.json"
        
        test_results = {
            'metadata': {
                'test_timestamp': timestamp,
                'articles_processed': len(results),
                'successful_summaries': successful_summaries,
                'success_rate': success_rate,
                'average_processing_time': avg_time,
                'total_time': total_time
            },
            'results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Display some sample summaries
        logger.info(f"\n=== SAMPLE SUMMARIES ===")
        for result in results[:3]:
            if result.get('status') == 'success':
                logger.info(f"\nTitle: {result['title']}")
                logger.info(f"Category: {result['category']}")
                logger.info(f"Summary:\n{result['summary']}")
                logger.info("-" * 80)
        
        return results
        
    except ImportError as e:
        logger.error(f"Failed to import summarizer: {e}")
        logger.error("Make sure the Llama summarizer is properly installed")
        return []
    except Exception as e:
        logger.error(f"Batch test failed: {e}")
        return []

def main():
    """Main test function."""
    logger.info("Starting Llama summarizer batch test...")
    
    # Load test data
    json_file = "data/tldr_articles_20250610_231259.json"
    articles = load_test_data(json_file)
    
    if not articles:
        logger.error("No articles loaded - cannot proceed with test")
        return
    
    # Display available articles
    logger.info(f"\nAvailable articles by category:")
    categories = {}
    for article in articles:
        cat = article.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        logger.info(f"  {cat}: {count} articles")
    
    # Test with a subset first (10 articles)
    logger.info(f"\nTesting with first 10 articles...")
    results = test_summarizer_batch(articles, max_articles=10)
    
    if results and any(r.get('status') == 'success' for r in results):
        logger.info("\n✓ Initial test successful!")
        
        # Ask user if they want to process all articles
        response = input(f"\nProcess all {len(articles)} articles? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            logger.info(f"Processing all {len(articles)} articles...")
            full_results = test_summarizer_batch(articles, max_articles=len(articles))
            logger.info("✓ Full batch test completed!")
        else:
            logger.info("Stopping after initial test.")
    else:
        logger.error("Initial test failed - check configuration and dependencies")

if __name__ == "__main__":
    main()
