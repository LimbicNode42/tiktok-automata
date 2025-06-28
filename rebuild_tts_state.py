#!/usr/bin/env python3
"""
Rebuild TTS State Script - Rebuild TTS file tracking in production state

This script scans the storage/tts/ directory and rebuilds the article_tts_files
mapping in production_state.json based on existing TTS files.
"""

import json
from pathlib import Path
import re

def rebuild_tts_state():
    """Rebuild TTS file tracking in production state."""
    state_file = Path("storage/production_state.json")
    tts_dir = Path("storage/tts")
    
    if not state_file.exists():
        print("❌ Production state file not found: storage/production_state.json")
        return False
    
    if not tts_dir.exists():
        print("❌ TTS directory not found: storage/tts/")
        return False
    
    # Load current state
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    # Get current article content files mapping to find URLs
    content_files = state.get('article_content_files', {})
    summary_files = state.get('article_summary_files', {})
    
    # Create reverse mapping from hash to URL
    hash_to_url = {}
    
    # Extract hashes from content file paths
    for url, content_path in content_files.items():
        # Extract hash from filename like: content_a633b3f53438_AI_is_doing_up_to_50_of_the_w.json
        match = re.search(r'content_([a-f0-9]+)_', content_path)
        if match:
            hash_to_url[match.group(1)] = url
    
    # Also check summary files for additional hashes
    for url, summary_path in summary_files.items():
        # Extract hash from filename like: summary_a633b3f53438_AI_is_doing_up_to_50_of_the_w.json
        match = re.search(r'summary_([a-f0-9]+)_', summary_path)
        if match:
            hash_to_url[match.group(1)] = url
    
    # Scan TTS directory for existing files
    tts_files = {}
    wav_files = list(tts_dir.glob("*.wav"))
    
    print(f"📁 Found {len(wav_files)} TTS WAV files")
    
    for wav_file in wav_files:
        # Extract hash from TTS filename like: tts_a633b3f53438_AI_is_doing_up_to_50_of_the_w.wav
        match = re.search(r'tts_([a-f0-9]+)_', wav_file.name)
        if match:
            file_hash = match.group(1)
            if file_hash in hash_to_url:
                url = hash_to_url[file_hash]
                tts_files[url] = str(wav_file)
                print(f"✅ Mapped TTS: {wav_file.name} -> {url[:50]}...")
            else:
                print(f"⚠️ No URL found for hash: {file_hash} ({wav_file.name})")
        else:
            print(f"⚠️ Could not extract hash from: {wav_file.name}")
    
    # Update state
    original_count = len(state.get('article_tts_files', {}))
    state['article_tts_files'] = tts_files
    
    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\n🔄 Updated TTS file tracking:")
    print(f"   Before: {original_count} tracked TTS files")
    print(f"   After:  {len(tts_files)} tracked TTS files")
    print(f"   Net change: +{len(tts_files) - original_count}")
    
    return True

if __name__ == "__main__":
    rebuild_tts_state()
