# Video Module Data Directory

This directory contains:

## Footage
- `raw/` - Downloaded gaming footage from YouTube
- `processed/` - TikTok-ready video segments (1080x1920, 30-120s)
- `footage_metadata.json` - Metadata about downloaded videos

## Output  
- Generated TikTok videos
- Temporary processing files

## Temp
- Temporary files during video processing
- Automatically cleaned up after processing

## Usage

1. Download footage: Use FootageManager to download from copyright-free sources
2. Process footage: Convert raw videos to TikTok-ready segments  
3. Generate videos: Combine with TTS audio and text overlays
4. Export: Create final MP4 files optimized for TikTok

## Storage Recommendations

- Keep 10-20GB of processed footage for variety
- Regularly clean old raw footage to save space
- Monitor output directory size (videos can be 10-50MB each)
