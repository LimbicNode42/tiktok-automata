# Video Module - TikTok Automation

## 🎬 Overview

The video module handles the complete video generation pipeline for TikTok content creation, combining gaming footage with AI-generated summaries and text-to-speech audio.

## ✅ Current Status (Working)

### Core Features
- **Video Processor**: Main orchestrator for video creation ✅
- **Gaming Footage Integration**: Placeholder system ready for YouTube footage ✅
- **Audio Synchronization**: TTS audio integration ✅
- **TikTok Optimization**: Proper 9:16 aspect ratio, MP4 format ✅
- **Automated Export**: File size and quality optimization ✅

### Successfully Tested
- Basic video creation with colored backgrounds
- Audio-video synchronization
- TikTok format compliance (1080x1920, MP4)
- File size management (under 30GB limit)

## 🚀 Next Steps

### 1. Gaming Footage Implementation
```python
# Use the FootageManager to download gaming content
from video.footage_manager import FootageManager

footage_manager = FootageManager()
await footage_manager.download_channel_content(
    "https://www.youtube.com/@NoCopyrightGameplays/videos",
    max_videos=20
)
```

### 2. Text Overlays (Requires ImageMagick)
```bash
# Install ImageMagick for text overlays
# Windows: Download from https://imagemagick.org/script/download.php#windows
# Or use alternative text overlay methods
```

### 3. Enhanced Effects
- Zoom and pan effects synchronized with content beats
- Color grading for different content types
- Smooth transitions between footage segments

## 📁 Module Structure

```
src/video/
├── video_processor.py      # Main orchestrator ✅
├── footage_manager.py      # YouTube download & management
├── overlay_generator.py    # Text overlay creation
├── video_effects.py        # Effects and transitions
└── data/
    ├── footage/           # Downloaded gaming videos
    ├── temp/              # Temporary processing files
    └── output/            # Generated TikTok videos ✅
```

## 🎯 Usage Example

```python
from video.video_processor import VideoProcessor, VideoConfig

# Configure for TikTok
config = VideoConfig(
    duration=120,  # 2 minutes
    output_quality="high",
    enable_zoom_effects=True
)

processor = VideoProcessor(config)

# Create video from TTS audio and script
video_path = await processor.create_video(
    audio_file="path/to/tts_audio.wav",
    script_content="AI just did something INSANE...",
    content_analysis={'is_breakthrough': True}
)
```

## ⚙️ Configuration Options

### VideoConfig
- **Resolution**: 1080x1920 (TikTok vertical format)
- **Duration**: 30-120 seconds (configurable)
- **Quality**: low/medium/high (affects bitrate and file size)
- **Effects**: Zoom, transitions, beat synchronization

### Content-Based Automation
- **High Energy**: Fast-paced gaming footage for breakthrough news
- **Medium Energy**: Balanced footage for standard tech news
- **Low Energy**: Calmer footage for educational content

## 🎮 Gaming Footage Strategy

### Source Channels
- NoCopyrightGameplays (copyright-free content)
- Additional curated channels with proper licensing

### Footage Categories
- **High Action**: Fast-paced gameplay for exciting news
- **Medium Action**: Balanced gameplay for standard content
- **Ambient**: Calmer gameplay for educational topics

### Processing Pipeline
1. Download YouTube videos in 1080p
2. Extract 30-60 second segments
3. Categorize by action intensity
4. Crop/resize for TikTok format
5. Cache for reuse across multiple videos

## 🔧 Dependencies

### Core (Currently Installed)
- `moviepy>=1.0.3` - Video processing ✅
- `yt-dlp` - YouTube downloading ✅
- `scipy` - Audio processing ✅

### Optional (For Enhanced Features)
- ImageMagick - Text overlays (requires manual installation)
- FFmpeg - Advanced video processing (auto-installed with moviepy)

## 📊 Performance Metrics

### Current Test Results
- **Video Generation**: ~10 seconds for 10-second video
- **File Size**: 0.19MB for 10-second placeholder video
- **Format Compliance**: ✅ TikTok-ready MP4
- **Resolution**: ✅ 1080x1920 (9:16 aspect ratio)

### Optimization Targets
- Target processing time: 2-3x video duration
- File size: Under 25MB for 120-second videos
- Quality: High enough for mobile viewing

## 🎯 Integration with Pipeline

```python
# Full pipeline integration
from summarizer.llama_summarizer import LlamaSummarizer
from tts.kokoro_tts import KokoroTTS
from video.video_processor import VideoProcessor

# 1. Generate summary with voice recommendation
summarizer = LlamaSummarizer()
result = await summarizer.summarize_for_tiktok(article, include_voice_recommendation=True)

# 2. Generate TTS audio
tts = KokoroTTS()
audio_path = await tts.generate_speech(
    text=result['summary'],
    voice_id=result['voice_recommendation']['voice_id']
)

# 3. Create video
video_processor = VideoProcessor()
video_path = await video_processor.create_video(
    audio_file=audio_path,
    script_content=result['summary'],
    content_analysis=result['content_analysis'],
    voice_info=result['voice_recommendation']
)
```

## 🔄 Development Roadmap

### Phase 1: Basic Functionality ✅
- Video creation pipeline
- Audio synchronization
- TikTok format compliance

### Phase 2: Gaming Footage (In Progress)
- YouTube content downloading
- Footage categorization and management
- Content-aware footage selection

### Phase 3: Advanced Features
- Text overlay system (with ImageMagick)
- Dynamic effects and transitions
- Beat-synchronized editing

### Phase 4: Optimization
- Batch processing capabilities
- Caching and storage management
- Quality vs. speed optimization

## 🐛 Known Issues

1. **Text Overlays**: Require ImageMagick installation
2. **Effects**: Some effects disabled due to PIL compatibility
3. **Footage**: Currently using placeholder backgrounds

## 💡 Next Development Tasks

1. **Install ImageMagick** for text overlays
2. **Implement FootageManager** for YouTube downloading
3. **Test with real gaming footage** from NoCopyrightGameplays
4. **Add beat detection** for audio-synchronized effects
5. **Optimize processing speed** for batch video generation
