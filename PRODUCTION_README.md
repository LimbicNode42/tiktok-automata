# TikTok Automata Production Setup

This document provides comprehensive instructions for setting up and running the TikTok Automata pipeline in production.

## 🚀 Quick Start

### Windows (Recommended)
```powershell
# Run daily pipeline
.\run_production.ps1

# Test run (no actual content generated)
.\run_production.ps1 -DryRun

# Initial setup (process backlog)
.\run_production.ps1 -InitialSetup
```

### Cross-Platform
```bash
# Daily run
python production_pipeline.py

# Test run
python production_pipeline.py --dry-run

# Initial setup
python production_pipeline.py --initial-setup
```

## 📋 Prerequisites

1. **Python 3.9+** with all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **GPU Support** (recommended for faster processing):
   - NVIDIA GPU with CUDA support
   - At least 4GB VRAM for optimal performance

3. **Storage Requirements**:
   - At least 10GB free space for videos and generated content
   - SSD recommended for better performance

## ⚙️ Configuration

### Production Config (`production_config.json`)

The main configuration file controls all aspects of the pipeline:

```json
{
  "storage_dir": "./storage",              # Local storage for downloads
  "output_dir": "./production_output",     # Generated content output
  "state_file": "./production_state.json", # Pipeline state tracking
  
  "fallback_enabled": true,                # Enable fallback to older articles
  "fallback_max_age_hours": 168,          # 1 week for fallback articles
  "fallback_max_articles": 3,             # Max fallback articles per run
  
  "video_buffer_seconds": 0.5,            # Buffer around TTS duration
  "min_video_duration": 60,               # Minimum video segment length
  "max_video_duration": 120,              # Maximum video segment length
  
  "rss_feeds": [                           # Article sources
    {
      "url": "https://tldr.tech/api/rss/tech",
      "name": "TLDR Tech",
      "category": "tech"
    }
  ],
  
  "gaming_video_sources": [                # Gaming video sources
    {
      "url": "https://www.youtube.com/@NoCopyrightGameplays/videos",
      "name": "No Copyright Gameplays",
      "content_type": "high_action"
    }
  ],
  
  "subtitle_styles": [                     # Randomized subtitle styles
    "default", "modern", "bold", "minimal", "highlight"
  ]
}
```

### Key Configuration Options

- **Fallback System**: When no new articles are found, the pipeline can fall back to processing older articles from the past week
- **Multiple RSS Feeds**: Add multiple news sources for diverse content
- **Video Sources**: Configure YouTube channels for gaming footage
- **Processing Limits**: Set `max_articles_per_run` and `max_videos_per_run` to limit resource usage

## 📅 Scheduling

### Daily Automation

Use the included scheduler for daily runs:

```bash
# Run at 9:00 AM daily
python daily_scheduler.py

# Run at custom time
python daily_scheduler.py --time 14:30

# Run once and exit
python daily_scheduler.py --once
```

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger to "Daily"
4. Set action to start program: `powershell.exe`
5. Add arguments: `-File "C:\path\to\tiktok-automata\run_production.ps1"`

### Linux/macOS (cron)

Add to crontab (`crontab -e`):
```bash
# Run daily at 9:00 AM
0 9 * * * cd /path/to/tiktok-automata && python production_pipeline.py
```

## 🔄 Operation Modes

### 1. Daily Mode (Default)
- Processes articles from the last 24 hours
- Downloads videos from the last 24 hours
- Ideal for regular content creation

### 2. Initial Setup Mode
- Processes articles from the last week
- Downloads videos from the last 6 months
- Use when first setting up or after extended downtime
- Command: `--initial-setup`

### 3. Dry Run Mode
- Tests the entire pipeline without generating actual content
- Useful for testing configuration changes
- Command: `--dry-run`

## 📊 Monitoring

### Log Files
- **`production_pipeline.log`**: Main pipeline execution log
- **`daily_scheduler.log`**: Scheduler activity log

### State Tracking
- **`production_state.json`**: Tracks processed articles and downloaded videos
- Prevents duplicate processing
- Maintains daily video assignments

### Output Files
- **`production_output/`**: Generated TikTok videos (MP4 format)
- **`production_results_*.json`**: Detailed run results with timestamps

## 🛠️ Troubleshooting

### Common Issues

1. **Unicode Errors in Logs**
   - Fixed with UTF-8 encoding in logging configuration
   - Use PowerShell script on Windows for better Unicode support

2. **No New Articles Found**
   - Enable fallback system in configuration
   - Check RSS feed URLs are accessible
   - Verify network connectivity

3. **GPU Memory Issues**
   - Reduce batch sizes in TTS/summarization
   - Close other GPU-intensive applications
   - Monitor VRAM usage

4. **Video Download Failures**
   - Check YouTube channel URLs are valid
   - Verify internet connectivity
   - Some channels may have restrictions

### Performance Optimization

1. **GPU Utilization**
   - Ensure CUDA is properly installed
   - Monitor GPU usage during processing
   - Consider upgrading GPU for faster processing

2. **Storage Management**
   - Regularly clean old videos from storage
   - Use SSD for better I/O performance
   - Monitor disk space usage

3. **Network Optimization**
   - Use stable internet connection
   - Consider CDN or local mirrors for RSS feeds
   - Implement retry mechanisms for downloads

## 🔒 Best Practices

### Security
- Keep API keys and credentials secure
- Use environment variables for sensitive data
- Regularly update dependencies

### Content Management
- Review generated content before publishing
- Implement content moderation filters
- Maintain backup of important videos

### Maintenance
- Regular log rotation and cleanup
- Monitor system resources
- Update dependencies monthly
- Test pipeline changes in dry-run mode first

## 📈 Scaling

### Horizontal Scaling
- Run multiple instances with different RSS feeds
- Use separate storage directories per instance
- Coordinate through shared state management

### Resource Scaling
- Increase GPU memory for larger batches
- Use faster storage for improved I/O
- Optimize network bandwidth for downloads

## 🆘 Support

For issues and questions:
1. Check the logs for error details
2. Run in dry-run mode to isolate issues
3. Verify configuration file syntax
4. Test individual components separately

## 📋 Changelog

### Version 2.0 (Current)
- Added fallback system for when no new articles are found
- Improved Unicode handling for Windows
- Enhanced error handling and cleanup
- Added production runner scripts
- Better configuration management
- Comprehensive logging improvements

### Version 1.0
- Initial production pipeline
- Basic article fetching and video generation
- Simple scheduling support
