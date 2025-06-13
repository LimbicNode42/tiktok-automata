#!/usr/bin/env python3
"""
Test the Llama summarizer against successfully extracted articles from the JSON data.
This script processes the 41 successful articles and evaluates the summarizer performance.
"""

import json
import time
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

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

def test_voice_selection_only(articles: List[Dict[str, Any]], max_articles: int = 20):
    """Test only the voice selection functionality without running the full summarizer."""
    import asyncio
    
    async def run_voice_test():
        try:
            from src.summarizer.llama_summarizer import create_tiktok_summarizer
            from src.scraper.newsletter_scraper import Article
            from datetime import datetime
            
            logger.info("Testing voice selection functionality...")
            summarizer = create_tiktok_summarizer()
            
            # Test voice selection on articles
            voice_results = []
            
            for i, article_data in enumerate(articles[:max_articles]):
                logger.info(f"Testing voice selection {i+1}/{min(len(articles), max_articles)}: {article_data['title'][:60]}...")
                
                try:
                    # Create Article object
                    article = Article(
                        title=article_data['title'],
                        content=article_data['content'],
                        summary="",
                        url=article_data.get('url', ''),
                        published_date=datetime.now(),
                        category=article_data.get('category', 'tech'),
                        word_count=article_data.get('word_count', 0),
                        content_extraction_status='success'
                    )
                    
                    # Analyze content and select voice
                    content_analysis = summarizer._analyze_article_content(article)
                    voice_recommendation = summarizer.select_voice_for_content(article, content_analysis)
                    
                    # Get alternative voices for category
                    alternative_voices = summarizer.get_available_voices_for_category(article.category)
                    
                    voice_results.append({
                        'title': article_data['title'],
                        'category': article_data.get('category', 'tech'),
                        'content_analysis': content_analysis,
                        'voice_recommendation': voice_recommendation,
                        'alternative_voices': alternative_voices,
                        'status': 'success'
                    })
                    
                    logger.info(f"  Content type: {content_analysis.get('content_type', 'standard')}")
                    logger.info(f"  Selected voice: {voice_recommendation['voice_name']} ({voice_recommendation['voice_id']})")
                    logger.info(f"  Reasoning: {voice_recommendation['reasoning']}")
                    
                except Exception as e:
                    logger.error(f"✗ Failed to test voice selection: {e}")
                    voice_results.append({
                        'title': article_data['title'],
                        'category': article_data.get('category', 'tech'),
                        'status': 'failed',
                        'error': str(e)
                    })
            
            return voice_results
            
        except Exception as e:
            logger.error(f"Voice selection test failed: {e}")
            return []
    
    # Run the async test
    results = asyncio.run(run_voice_test())
    
    # Analyze results
    successful_results = [r for r in results if r.get('status') == 'success']
    success_rate = (len(successful_results) / len(results)) * 100 if results else 0
    
    logger.info(f"\n=== VOICE SELECTION TEST RESULTS ===")
    logger.info(f"Articles tested: {len(results)}")
    logger.info(f"Successful tests: {len(successful_results)}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    if successful_results:
        # Voice distribution analysis
        voice_usage = {}
        content_type_distribution = {}
        category_voice_mapping = {}
        
        for result in successful_results:
            voice_rec = result['voice_recommendation']
            content_analysis = result['content_analysis']
            category = result['category']
            
            voice_id = voice_rec['voice_id']
            content_type = content_analysis.get('content_type', 'standard')
            
            # Count voice usage
            voice_usage[voice_id] = voice_usage.get(voice_id, 0) + 1
            
            # Count content types
            content_type_distribution[content_type] = content_type_distribution.get(content_type, 0) + 1
            
            # Track category to voice mapping
            if category not in category_voice_mapping:
                category_voice_mapping[category] = {}
            category_voice_mapping[category][voice_id] = category_voice_mapping[category].get(voice_id, 0) + 1
        
        logger.info(f"\n=== VOICE USAGE ANALYSIS ===")
        for voice_id, count in sorted(voice_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful_results)) * 100
            logger.info(f"{voice_id}: {count} selections ({percentage:.1f}%)")
        
        logger.info(f"\n=== CONTENT TYPE ANALYSIS ===")
        for content_type, count in sorted(content_type_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful_results)) * 100
            logger.info(f"{content_type}: {count} articles ({percentage:.1f}%)")
        
        logger.info(f"\n=== CATEGORY TO VOICE MAPPING ===")
        for category, voices in category_voice_mapping.items():
            logger.info(f"{category}:")
            for voice_id, count in sorted(voices.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / sum(voices.values())) * 100
                logger.info(f"  {voice_id}: {count} selections ({percentage:.1f}%)")
    
    return results

def test_summarizer_batch(articles: List[Dict[str, Any]], max_articles: int = 10):
    """Test the Llama summarizer on a batch of articles."""
    import asyncio
    
    async def run_batch_test():
        try:
            # Import the summarizer
            from src.summarizer.llama_summarizer import create_tiktok_summarizer
            from src.scraper.newsletter_scraper import Article
            from datetime import datetime
            
            # Create summarizer
            logger.info("Initializing Llama summarizer...")
            summarizer = create_tiktok_summarizer()
            
            # Initialize the model
            await summarizer.initialize()
            
            # Test articles
            results = []
            successful_summaries = 0
            total_time = 0
            
            for i, article_data in enumerate(articles[:max_articles]):
                logger.info(f"Processing article {i+1}/{min(len(articles), max_articles)}: {article_data['title'][:60]}...")
                
                try:
                    start_time = time.time()
                      # Create Article object
                    article = Article(
                        title=article_data['title'],
                        content=article_data['content'],
                        summary="",  # Empty summary since we'll generate one
                        url=article_data.get('url', ''),
                        published_date=datetime.now(),
                        category=article_data.get('category', 'tech'),
                        word_count=article_data.get('word_count', 0),
                        content_extraction_status='success'
                    )
                      # Generate TikTok summary (using new 120s default)
                    summary = await summarizer.summarize_for_tiktok(article)
                    
                    # Test voice selection functionality
                    content_analysis = summarizer._analyze_article_content(article)
                    voice_recommendation = summarizer.select_voice_for_content(article, content_analysis)
                    
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    if summary and summary.strip():
                        successful_summaries += 1
                        logger.info(f"✓ Generated summary in {processing_time:.2f}s")
                        logger.info(f"  Voice: {voice_recommendation['voice_name']} ({voice_recommendation['voice_id']})")
                        logger.info(f"  Reasoning: {voice_recommendation['reasoning']}")
                        
                        # Check for timestamps and emojis in the output
                        has_timestamps = any(pattern in summary for pattern in ['[', ']', '**[', ']**', 's -', 's-'])
                        import re
                        has_emojis = bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000027BF]', summary))
                        
                        results.append({
                            'title': article_data['title'],
                            'category': article_data.get('category', 'tech'),
                            'url': article_data.get('url', ''),
                            'word_count': article_data.get('word_count', 0),
                            'summary': summary,
                            'voice_recommendation': voice_recommendation,
                            'content_analysis': content_analysis,
                            'processing_time': processing_time,
                            'has_timestamps': has_timestamps,
                            'has_emojis': has_emojis,
                            'summary_length': len(summary),
                            'estimated_reading_time': (len(summary.split()) / 150) * 60,  # seconds
                            'status': 'success'
                        })
                    else:
                        logger.warning(f"✗ Empty summary generated")
                        results.append({
                            'title': article_data['title'],
                            'category': article_data.get('category', 'tech'),
                            'status': 'failed',
                            'error': 'Empty summary'
                        })
                        
                except Exception as e:
                    logger.error(f"✗ Failed to process article: {e}")
                    results.append({
                        'title': article_data['title'],
                        'category': article_data.get('category', 'tech'),
                        'status': 'failed',
                        'error': str(e)
                    })
              # Clean up
            await summarizer.cleanup()
            
            return results, successful_summaries, total_time
            
        except Exception as e:
            logger.error(f"Batch test failed: {e}")
            return [], 0, 0
    
    # Run the async test
    results, successful_summaries, total_time = asyncio.run(run_batch_test())
    
    # Performance summary
    avg_time = total_time / max(1, successful_summaries)
    success_rate = (successful_summaries / len(results)) * 100 if results else 0
    
    logger.info(f"\n=== BATCH TEST RESULTS ===")
    logger.info(f"Articles processed: {len(results)}")
    logger.info(f"Successful summaries: {successful_summaries}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Average processing time: {avg_time:.2f}s")
    logger.info(f"Total time: {total_time:.2f}s")
      # Check configuration compliance and voice selection analysis
    if results:
        successful_results = [r for r in results if r.get('status') == 'success']
        if successful_results:
            timestamp_violations = sum(1 for r in successful_results if r.get('has_timestamps', False))
            emoji_violations = sum(1 for r in successful_results if r.get('has_emojis', False))
            avg_reading_time = sum(r.get('estimated_reading_time', 0) for r in successful_results) / len(successful_results)
            
            logger.info(f"\n=== CONFIGURATION COMPLIANCE ===")
            logger.info(f"Timestamp violations: {timestamp_violations}/{len(successful_results)} summaries")
            logger.info(f"Emoji violations: {emoji_violations}/{len(successful_results)} summaries")
            logger.info(f"Average estimated reading time: {avg_reading_time:.1f}s (target: 120s)")
            
            # Voice selection analysis
            logger.info(f"\n=== VOICE SELECTION ANALYSIS ===")
            voice_usage = {}
            voice_reasoning = {}
            content_types = {}
            
            for r in successful_results:
                voice_rec = r.get('voice_recommendation', {})
                voice_id = voice_rec.get('voice_id', 'unknown')
                reasoning = voice_rec.get('reasoning', 'No reasoning provided')
                content_analysis = r.get('content_analysis', {})
                content_type = content_analysis.get('content_type', 'standard')
                
                # Count voice usage
                voice_usage[voice_id] = voice_usage.get(voice_id, 0) + 1
                
                # Track reasoning patterns
                if voice_id not in voice_reasoning:
                    voice_reasoning[voice_id] = reasoning
                
                # Count content types
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            logger.info(f"Voice selection distribution:")
            for voice_id, count in sorted(voice_usage.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(successful_results)) * 100
                logger.info(f"  {voice_id}: {count} articles ({percentage:.1f}%) - {voice_reasoning.get(voice_id, 'N/A')}")
            
            logger.info(f"\nContent type distribution:")
            for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(successful_results)) * 100
                logger.info(f"  {content_type}: {count} articles ({percentage:.1f}%)")
            
            # Check for voice profile diversity
            unique_voices = len(voice_usage)
            logger.info(f"\nVoice diversity: {unique_voices} different voices used out of {len(successful_results)} articles")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save to summarizer data directory
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_file = data_dir / f"llama_test_results_{timestamp}.json"
    
    test_results = {
        'metadata': {
            'test_timestamp': timestamp,
            'articles_processed': len(results),
            'successful_summaries': successful_summaries,
            'success_rate': success_rate,
            'average_processing_time': avg_time,
            'total_time': total_time,
            'configuration': {
                'target_duration': 120,
                'no_timestamps': True,
                'no_emojis': True
            }
        },
        'results': results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {results_file}")
      # Display some sample summaries with voice recommendations
    logger.info(f"\n=== SAMPLE SUMMARIES WITH VOICE RECOMMENDATIONS ===")
    for result in results[:3]:
        if result.get('status') == 'success':
            voice_rec = result.get('voice_recommendation', {})
            content_analysis = result.get('content_analysis', {})
            
            logger.info(f"\nTitle: {result['title']}")
            logger.info(f"Category: {result['category']}")
            logger.info(f"Reading time: {result.get('estimated_reading_time', 0):.1f}s")
            logger.info(f"Content type: {content_analysis.get('content_type', 'standard')}")
            logger.info(f"Selected voice: {voice_rec.get('voice_name', 'Unknown')} ({voice_rec.get('voice_id', 'unknown')})")
            logger.info(f"Voice reasoning: {voice_rec.get('reasoning', 'No reasoning provided')}")
            logger.info(f"Has timestamps: {result.get('has_timestamps', False)}")
            logger.info(f"Has emojis: {result.get('has_emojis', False)}")
            logger.info(f"Summary:\n{result['summary']}")
            logger.info("-" * 80)
    
    return results

def main():
    """Main test function."""
    logger.info("Starting Llama summarizer batch test...")
    
    # Load test data from scraper data directory
    scraper_data_dir = Path(__file__).parent.parent.parent / "scraper" / "data"
    json_files = list(scraper_data_dir.glob("tldr_articles_*.json"))
    
    if not json_files:
        logger.error("No article data files found in scraper data directory")
        return
    
    # Use the most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading articles from: {json_file}")
    
    articles = load_test_data(str(json_file))
    
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
    
    # Test voice selection first (faster test)
    logger.info(f"\nTesting voice selection on first 20 articles...")
    voice_results = test_voice_selection_only(articles, max_articles=20)
    
    if voice_results and any(r.get('status') == 'success' for r in voice_results):
        logger.info("\n✓ Voice selection test successful!")
        
        # Ask user if they want to run full summarization test
        response = input(f"\nRun full summarization test with 10 articles? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            # Test with a subset first (10 articles)
            logger.info(f"\nTesting full summarization with first 10 articles...")
            results = test_summarizer_batch(articles, max_articles=10)
            
            if results and any(r.get('status') == 'success' for r in results):
                logger.info("\n✓ Initial summarization test successful!")
                
                # Ask user if they want to process all articles
                response = input(f"\nProcess all {len(articles)} articles? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    logger.info(f"Processing all {len(articles)} articles...")
                    full_results = test_summarizer_batch(articles, max_articles=len(articles))
                    logger.info("✓ Full batch test completed!")
                else:
                    logger.info("Stopping after initial test.")
            else:
                logger.error("Initial summarization test failed - check configuration and dependencies")
        else:
            logger.info("Skipping full summarization test.")
    else:
        logger.error("Voice selection test failed - check configuration and dependencies")

if __name__ == "__main__":
    main()
