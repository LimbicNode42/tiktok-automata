#!/usr/bin/env python3
"""
Regenerate TTS Script - Replace TTS audio generated from mock summaries

This script identifies TTS files that were generated from mock summaries and regenerates them
with audio from the new real AI-generated summaries.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tts.kokoro_tts import KokoroTTSEngine, TTSConfig

async def regenerate_tts():
    """Regenerate TTS files from updated summaries."""
    tts_dir = Path("storage/tts")
    summaries_dir = Path("storage/summaries")
    
    if not tts_dir.exists() or not summaries_dir.exists():
        print("❌ Required directories not found")
        return False
    
    # Initialize TTS engine
    print("🔊 Initializing Kokoro TTS engine...")
    tts_config = TTSConfig(
        voice='af_heart',  # Default voice
        speed=1.55,        # Optimized TikTok speed
        use_gpu=True
    )
    tts_engine = KokoroTTSEngine(tts_config)
    await tts_engine.initialize()
    
    try:
        # Find all TTS JSON metadata files
        tts_json_files = list(tts_dir.glob("tts_*.json"))
        print(f"🔊 Found {len(tts_json_files)} TTS metadata files to check")
        
        regenerated_count = 0
        
        for tts_json_file in tts_json_files:
            try:
                # Load TTS metadata
                with open(tts_json_file, 'r', encoding='utf-8') as f:
                    tts_metadata = json.load(f)
                
                url = tts_metadata.get('url')
                if not url:
                    continue
                
                # Find corresponding summary file by looking for the same hash in filename
                tts_filename = tts_json_file.name  # e.g., "tts_f1ccaabf022a_SpaceX_launches_4_people_into.json"
                hash_part = tts_filename.split('_')[1]  # Extract hash: "f1ccaabf022a"
                
                # Find matching summary file
                summary_file = None
                for summary_path in summaries_dir.glob(f"summary_{hash_part}_*.json"):
                    summary_file = summary_path
                    break
                
                if not summary_file or not summary_file.exists():
                    print(f"⚠️ No matching summary file found for {tts_json_file.name}")
                    continue
                
                # Load summary to get new content
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                new_summary = summary_data.get('tiktok_summary', '')
                if not new_summary:
                    print(f"⚠️ No summary content found in {summary_file.name}")
                    continue
                
                # Check if this summary was regenerated (ends with "_regenerated")
                summarized_at = summary_data.get('summarized_at', '')
                if not summarized_at.endswith('_regenerated'):
                    print(f"⏭️ Skipping {tts_json_file.name} - summary was not regenerated")
                    continue
                
                title = summary_data.get('title', 'Unknown')
                voice = tts_metadata.get('voice', 'af_heart')
                
                print(f"\n🔊 Regenerating TTS: {title[:50]}...")
                print(f"   Voice: {voice}")
                print(f"   Summary length: {len(new_summary)} chars")
                
                # Find corresponding WAV file
                wav_file = tts_json_file.with_suffix('.wav')
                if not wav_file.exists():
                    print(f"⚠️ WAV file not found: {wav_file.name}")
                    continue
                
                # Generate new TTS audio
                temp_audio_filename = f"temp_regenerated_tts_{int(time.time())}.wav"
                temp_audio_path = Path("temp") / temp_audio_filename
                temp_audio_path.parent.mkdir(exist_ok=True)
                
                generated_path = await tts_engine.generate_audio(
                    text=new_summary,
                    voice=voice,
                    output_path=str(temp_audio_path)
                )
                
                if generated_path and Path(generated_path).exists():
                    # Replace old WAV file with new one
                    import shutil
                    shutil.move(generated_path, wav_file)
                    
                    # Update metadata
                    duration = len(new_summary.split()) * 0.5  # Rough estimate: 0.5s per word
                    tts_metadata['duration_seconds'] = duration
                    tts_metadata['generated_at'] = f"{tts_metadata.get('generated_at', '')}_regenerated"
                    tts_metadata['summary_length'] = len(new_summary)
                    
                    # Save updated metadata
                    with open(tts_json_file, 'w', encoding='utf-8') as f:
                        json.dump(tts_metadata, f, indent=2)
                    
                    regenerated_count += 1
                    print(f"✅ TTS regenerated ({duration:.1f}s): {title[:40]}...")
                else:
                    print(f"❌ Failed to generate TTS for: {title[:40]}...")
                
            except Exception as e:
                print(f"❌ Error processing {tts_json_file.name}: {e}")
                continue
        
        print(f"\n🎉 TTS regeneration complete!")
        print(f"   Regenerated: {regenerated_count} TTS files")
        
    finally:
        # Cleanup
        await tts_engine.cleanup()
    
    return True

if __name__ == "__main__":
    asyncio.run(regenerate_tts())
