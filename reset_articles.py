#!/usr/bin/env python3
"""
Reset Articles Script - Clear processed article state to force reprocessing

This script removes article URLs from the production state so they can be 
reprocessed even if they were previously scraped.
"""

import json
from pathlib import Path

def reset_articles():
    """Reset article processing state to allow reprocessing."""
    state_file = Path("storage/production_state.json")
    
    if not state_file.exists():
        print("❌ Production state file not found: storage/production_state.json")
        return False
    
    # Load current state
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    original_count = len(state.get('processed_article_urls', []))
    
    # Clear processed article URLs to force reprocessing
    state['processed_article_urls'] = []
    
    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"✅ Reset {original_count} processed article URLs")
    print("📄 Articles will now be reprocessed for summarization and TTS")
    print("🔄 Run the pipeline again to process all articles")
    
    return True

if __name__ == "__main__":
    reset_articles()
