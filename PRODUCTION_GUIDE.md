# TikTok Automata Production System

## 🎯 Overview

The TikTok Automata Production System is a complete, automated pipeline that runs daily to create TikTok content from TLDR newsletter articles. It handles everything from article extraction to final video generation with randomized content selection.

## 🚀 Quick Start

### First-Time Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   # Copy and edit environment file
   cp .env.example .env
   # Add your HF_TOKEN and other settings
   ```

3. **Update gaming video URLs:**
   Edit `production_config.json` and replace the example URLs with actual gaming videos:
   ```json
   {
     "gaming_video_urls": [
       "https://www.youtube.com/watch?v=REAL_VIDEO_ID_1",
       "https://www.youtube.com/watch?v=REAL_VIDEO_ID_2",
       "https://www.youtube.com/watch?v=REAL_VIDEO_ID_3"
     ]
   }
   ```

4. **Run initial setup:**
   ```bash
   # Windows
   run_production.bat initial-setup
   
   # Linux/Mac
   ./run_production.sh initial-setup
   
   # Or directly
   python production_pipeline.py --initial-setup
   ```

### Daily Production

```bash
# Windows
run_production.bat daily

# Linux/Mac
./run_production.sh daily

# Or directly
python production_pipeline.py
```

## 📋 Pipeline Steps

The production pipeline executes these steps automatically:

### 1. **Article Fetching**
- Connects to TLDR RSS feed
- Extracts new articles not previously processed
- Filters by content quality and length
- Avoids duplicate processing

### 2. **AI Summarization**
- Processes articles with Llama 3.2-3B model
- Generates TikTok-optimized scripts with timestamps
- Includes engagement hooks, emojis, and hashtags
- Target duration: 25-30 seconds

### 3. **TTS Audio Generation**
- Converts summaries to speech using Kokoro TTS
- Uses optimized 1.55x speed for natural pacing
- Randomizes voice selection from 9 available voices
- Measures audio duration for video segment calculation

### 4. **Gaming Video Management**
- Downloads new gaming videos from configured URLs
- Analyzes videos for high-action segments
- Caches analysis results for performance
- Maintains library of available footage

### 5. **Video Assignment & Generation**
- Assigns unique videos to each article (no daily duplicates)
- Calculates segment duration: TTS length + 2s buffer on each side
- Randomizes subtitle styles for variety
- Generates final TikTok videos with:
  - 9:16 letterboxing for mobile
  - Synchronized subtitles
  - Mixed audio (TTS foreground + gaming background)
  - Professional mobile optimization

## ⚙️ Configuration

### Production Config (`production_config.json`)

```json
{
  "storage_dir": "./storage",               // Video storage location
  "output_dir": "./production_output",      // Final video output
  "state_file": "./production_state.json", // Tracks processed content
  "max_articles_per_run": 10,              // Limit articles per day
  "max_videos_per_run": 5,                 // Limit video downloads per day
  "video_buffer_seconds": 2.0,             // Buffer around TTS duration
  "min_video_duration": 15,                // Minimum segment length
  "max_video_duration": 60,                // Maximum segment length
  "gaming_video_urls": [...],              // YouTube URLs to download
  "subtitle_styles": [...]                 // Available subtitle styles
}
```

### Environment Variables (`.env`)

```bash
# Required
HF_TOKEN=your_hugging_face_token

# Optional
MAX_AGE_HOURS=24
MAX_ARTICLES_PER_RUN=10
REQUEST_TIMEOUT=30
TTS_DEFAULT_VOICE=af_heart
TTS_DEFAULT_SPEED=1.55
YOUTUBE_DOWNLOAD_QUALITY=720p
ENABLE_ACTION_ANALYSIS=true
```

## 🤖 Automation Options

### Option 1: Manual Daily Runs
```bash
# Run daily at your preferred time
python production_pipeline.py
```

### Option 2: Built-in Scheduler
```bash
# Run at 9:00 AM daily
python daily_scheduler.py

# Run at custom time
python daily_scheduler.py --time 14:30

# Run once now
python daily_scheduler.py --once
```

### Option 3: Cron Job (Linux/Mac)
```bash
# Add to crontab (run at 9:00 AM daily)
0 9 * * * cd /path/to/tiktok-automata && python production_pipeline.py

# Edit crontab
crontab -e
```

### Option 4: Windows Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at preferred time
4. Set action: Start program
5. Program: `python`
6. Arguments: `production_pipeline.py`
7. Start in: `C:\path\to\tiktok-automata`

## 📊 State Management

The system tracks processed content to avoid duplicates:

### State File (`production_state.json`)
```json
{
  "processed_article_urls": [...],        // URLs already processed
  "downloaded_video_urls": [...],         // Videos already downloaded
  "last_run_date": "2025-06-25",        // Last execution date
  "daily_video_assignments": {           // Today's video assignments
    "article_url": "video_id"
  }
}
```

### Daily Reset Behavior
- Video assignments reset each day
- Ensures no duplicate video usage per day
- Allows video reuse across different days
- Maintains article processing history

## 🎨 Randomization Features

### Video Selection
- Randomizes which gaming video is used for each article
- Ensures no video is used twice in the same day
- Resets assignments daily for fresh combinations

### Voice Selection
- Randomly selects from 9 available TTS voices
- Provides variety across daily content
- Voices: af_heart, af_sky, af_sunset, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis

### Subtitle Styles
- Cycles through 5 different subtitle styles per day
- Styles: default, modern, bold, minimal, highlight
- Ensures visual variety in final videos

### Segment Selection
- Uses AI action analysis to find best segments
- Randomizes within high-action periods
- Adapts segment length to TTS duration + buffer

## 📁 Output Structure

```
production_output/
├── tiktok_video_1_20250625_091234.mp4  // Final TikTok videos
├── tiktok_video_2_20250625_091456.mp4
├── tts_audio_1_20250625_091123.wav     // TTS audio files
├── tts_audio_2_20250625_091345.wav
├── production_results_20250625_091600.json  // Execution summary
└── *.srt                               // Subtitle files (if enabled)

storage/
├── downloaded_video_1.mp4              // Gaming footage library
├── downloaded_video_2.mp4
├── video_1_action_cache.json           // Action analysis cache
├── video_2_action_cache.json
└── footage_metadata.json               // Video metadata
```

## 🔍 Monitoring & Debugging

### Log Files
- `production_pipeline.log` - Main pipeline execution
- `daily_scheduler.log` - Scheduler activity (if using)

### Dry Run Mode
```bash
# Test without generating content
python production_pipeline.py --dry-run
```

### Manual Component Testing
```bash
# Test individual components
python src/scraper/newsletter_scraper.py    # Article extraction
python src/summarizer/llama_summarizer.py   # AI summarization
python src/tts/kokoro_tts.py                # TTS generation
python src/video/managers/footage_manager.py # Video processing
```

### Common Issues

1. **No articles processed**: Check HF_TOKEN and internet connection
2. **Video download fails**: Verify YouTube URLs are valid and accessible
3. **TTS generation fails**: Ensure Kokoro dependencies are installed
4. **Video processing errors**: Check FFmpeg installation and storage space
5. **GPU errors**: Verify CUDA installation and available VRAM

### Performance Optimization

1. **Enable caching**: First run will be slow, subsequent runs 10x+ faster
2. **Batch processing**: Use `--initial-setup` for first-time bulk processing
3. **Storage management**: Clean old videos/cache files periodically
4. **GPU memory**: Close other GPU applications during processing

## 🎯 Production Tips

### Content Quality
- Review generated videos before publishing
- Adjust `gaming_video_urls` for better footage variety
- Monitor TTS quality and adjust voice selection
- Check subtitle synchronization

### Scaling Up
- Increase `max_articles_per_run` for more daily content
- Add more gaming video sources for variety
- Use multiple subtitle styles for brand consistency
- Consider multiple daily runs for higher volume

### Customization
- Edit `production_config.json` for different settings
- Modify subtitle styles in video processor
- Adjust TTS speed/voice preferences
- Customize video effects and filters

## 📈 Expected Performance

### First Run (Initial Setup)
- **Duration**: 2-4 hours (depending on video count)
- **Downloads**: All configured gaming videos
- **Processing**: Full analysis and caching
- **Output**: 5-10 TikTok videos

### Daily Production Runs
- **Duration**: 15-30 minutes
- **Processing**: 5-10 new articles
- **Output**: 5-10 TikTok videos
- **Cache hits**: 90%+ for video analysis

### Hardware Requirements
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) or equivalent
- **RAM**: 16GB recommended
- **Storage**: 20GB free space (for videos and cache)
- **Internet**: Stable connection for downloads

---

**Ready for Production**: This system is designed to run autonomously and produce high-quality TikTok content daily with minimal manual intervention.
