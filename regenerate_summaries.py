#!/usr/bin/env python3
"""
Regenerate Summaries Script - Replace dry-run mock summaries with real AI summaries

This script identifies summary files that contain mock/test content and regenerates them
with proper AI-generated summaries using the Llama summarizer.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from summarizer.llama_summarizer import LlamaSummarizer
from utils.config import config

def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS compatibility - same function as in production pipeline."""
    replacements = {
        '"': "'",
        '"': "'", 
        '"': "'",
        ''': "'",
        ''': "'",
        '—': '-',
        '–': '-',
        '…': '...',
        '°': ' degrees',
        '×': 'x',
        '÷': ' divided by ',
        '±': ' plus or minus ',
        '€': ' euros',
        '£': ' pounds',
        '¥': ' yen',
        '©': ' copyright ',
        '®': ' registered ',
        '™': ' trademark '
    }
    
    cleaned = text
    for char, replacement in replacements.items():
        cleaned = cleaned.replace(char, replacement)
    
    # Remove any remaining non-ASCII characters that might cause issues
    cleaned = cleaned.encode('ascii', errors='ignore').decode('ascii')
    
    # Clean up multiple spaces
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

async def regenerate_summaries():
    """Find and regenerate mock summaries with real AI summaries."""
    summaries_dir = Path("storage/summaries")
    
    if not summaries_dir.exists():
        print("❌ Summaries directory not found: storage/summaries/")
        return False
    
    # Initialize summarizer
    print("🤖 Initializing Llama summarizer...")
    summarizer = LlamaSummarizer()
    await summarizer.initialize()
    
    try:
        # Find all summary files
        summary_files = list(summaries_dir.glob("summary_*.json"))
        print(f"📄 Found {len(summary_files)} summary files to check")
        
        mock_summaries = []
        
        # Check each summary file for mock content
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                tiktok_summary = summary_data.get('tiktok_summary', '')
                
                # Check if this is a mock summary
                if ('test summary for dry run mode' in tiktok_summary.lower() or
                    'this is a test summary' in tiktok_summary.lower() or
                    '#TechNews #AI #TikTok' in tiktok_summary):
                    
                    mock_summaries.append((summary_file, summary_data))
                    print(f"🔍 Found mock summary: {summary_file.name}")
                
            except Exception as e:
                print(f"❌ Error reading {summary_file.name}: {e}")
                continue
        
        if not mock_summaries:
            print("✅ No mock summaries found - all summaries appear to be real")
            return True
        
        print(f"🔄 Found {len(mock_summaries)} mock summaries to regenerate")
        
        # Regenerate each mock summary
        for i, (summary_file, summary_data) in enumerate(mock_summaries):
            try:
                print(f"\n📝 Regenerating {i+1}/{len(mock_summaries)}: {summary_file.name}")
                
                title = summary_data.get('title', 'Unknown')
                content = summary_data.get('original_content', '')
                url = summary_data.get('url', '')
                
                if not content:
                    print(f"⚠️ No original content found for {title[:40]}...")
                    continue
                
                print(f"   Title: {title[:60]}...")
                print(f"   Content length: {len(content)} chars")
                
                # Generate real summary
                new_summary = await summarizer.generate_tiktok_summary(
                    content=content,
                    title=title,
                    url=url
                )
                
                if new_summary:
                    # Clean summary for TTS compatibility
                    cleaned_summary = clean_text_for_tts(new_summary)
                    
                    # Update summary data
                    summary_data['tiktok_summary'] = cleaned_summary
                    summary_data['summary_length'] = len(cleaned_summary)
                    summary_data['summary_words'] = len(cleaned_summary.split())
                    summary_data['summarized_at'] = f"{summary_data.get('summarized_at', '')}_regenerated"
                    
                    # Save updated summary
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(summary_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ Regenerated ({len(cleaned_summary)} chars): {title[:40]}...")
                    if len(cleaned_summary) != len(new_summary):
                        print(f"   🧹 Text cleaned: {len(new_summary)} -> {len(cleaned_summary)} chars")
                else:
                    print(f"❌ Failed to generate summary for: {title[:40]}...")
                
            except Exception as e:
                print(f"❌ Error regenerating summary for {summary_file.name}: {e}")
                continue
        
        print(f"\n🎉 Summary regeneration complete!")
        print(f"   Processed: {len(mock_summaries)} mock summaries")
        
    finally:
        # Cleanup
        await summarizer.cleanup()
    
    return True

if __name__ == "__main__":
    asyncio.run(regenerate_summaries())
