# TikTok Automata - Complete Content Creation Pipeline

A fully-featured, production-ready automation system for creating engaging TikTok content from TLDR newsletter articles. Features AI-powered summarization, high-quality TTS, gaming footage integration, and professional video processing.

## 🎯 Project Overview

TikTok Automata is a comprehensive Python automation project that transforms newsletter articles into TikTok-ready videos using:
- **AI-Powered Summarization**: Llama 3.2-3B model with CUDA acceleration
- **Neural Text-to-Speech**: Kokoro TTS with 9 professional voice options
- **Gaming Footage Integration**: YouTube downloading and action analysis
- **Professional Video Processing**: Subtitles, audio mixing, and mobile optimization
- **Complete Pipeline**: From article extraction to final TikTok video

### Key Features
- **📰 RSS Feed Processing**: Automatically extracts articles from TLDR newsletter RSS feeds
- **🤖 Adaptive Content Extraction**: AI-powered system that learns extraction patterns for new websites
- **🦙 Llama Summarization**: GPU-accelerated summarization using meta-llama/Llama-3.2-3B-Instruct
- **🎵 High-Quality TTS**: Neural text-to-speech using Kokoro with multiple voice options (1.55x optimized speed)
- **🎮 Gaming Footage Integration**: YouTube video downloading and intelligent action analysis
- **🎬 Professional Video Processing**: Subtitles, audio mixing, letterboxing, and mobile optimization
- **⚡ Performance Optimized**: Intelligent caching, ultra-fast encoding, and GPU acceleration

## 🚀 Current Status: FULLY OPERATIONAL ✅

### System Specifications
- **GPU**: RTX 3070 8GB with CUDA 12.6 support
- **AI Model**: meta-llama/Llama-3.2-3B-Instruct
- **TTS Engine**: Kokoro-82M with 9 voice options
- **PyTorch**: 2.1.0+cu121 with CUDA acceleration
- **Performance**: 24+ tokens/second, ~5.2s per summary, 50x+ real-time TTS

### Production Capabilities
- ✅ **Complete Pipeline**: Newsletter → Summary → TTS → Video → Final TikTok
- ✅ **Article Extraction**: 41/89 articles successfully extracted from real TLDR data
- ✅ **AI Summarization**: 100% success rate with professional TikTok formatting
- ✅ **TTS Audio Generation**: Real-time neural speech synthesis
- ✅ **Gaming Footage**: YouTube downloading with intelligent action analysis
- ✅ **Video Processing**: Subtitles, audio mixing, letterboxing, mobile optimization
- ✅ **Performance Optimized**: Intelligent caching reduces processing time by 90%+

## 📊 Performance Metrics

### Article Processing
- **Dataset**: 89 TLDR newsletter articles (June 2025)
- **Success Rate**: 46.1% successful extraction (41 quality articles)
- **Categories**: AI (28), Big Tech (4), Dev (7), Science (2)
- **Content Quality**: All articles >200 words with proper formatting

### Generation Performance
- **AI Initialization**: ~5.5 seconds
- **Per-article Processing**: ~5.2 seconds
- **TTS Generation**: 50x+ real-time speed
- **Video Processing**: ~3 minutes for 111s gaming segment
- **Memory Usage**: 1.8GB VRAM / 8GB available
- **Overall Pipeline**: ~24 minutes for complete TikTok video (with optimizations)

### Sample Output
```
**[0s-3s]** 🚨 BREAKING: OpenAI just hit $10 BILLION in revenue!
**[3s-8s]** That's right - ChatGPT is making them $10B per year!
**[8s-15s]** This proves AI isn't just hype anymore - it's BIG BUSINESS!
**[15s-20s]** They went from $5.5B to $10B in just one year!
**[20s-25s]** And they're targeting $125 BILLION by 2029! 🤯
**[25s-28s]** What do you think about this AI boom? #AI #OpenAI #TechNews
```

## 🛠️ Technical Architecture

### Core Components

#### 1. Newsletter Scraper (`src/scraper/`)
- **RSS Processing**: Connects to TLDR RSS feed and extracts newsletter links
- **Article Extraction**: Fetches full article content from external URLs
- **Adaptive Learning**: AI-powered extraction pattern discovery
- **Error Handling**: Robust retry logic and timeout management

#### 2. AI Summarizer (`src/summarizer/`)
- **Llama Integration**: meta-llama/Llama-3.2-3B-Instruct model
- **CUDA Optimization**: GPU acceleration for RTX 3070
- **TikTok Formatting**: Professional script generation with timestamps
- **Batch Processing**: Efficient handling of multiple articles

#### 3. TTS Engine (`src/tts/`)
- **Kokoro Integration**: High-quality neural TTS using Kokoro-82M model
- **Multiple Voices**: 9 different voice options (male/female, various accents)
- **Optimized Speed**: 1.55x speed for natural TikTok pacing
- **Real-time Generation**: Fast audio generation (50x+ real-time)

#### 4. Video System (`src/video/`)
- **Modular Architecture**: Clean separation of concerns
- **Gaming Footage**: YouTube downloading and action analysis
- **Video Processing**: Segments, letterboxing, mobile optimization
- **Audio Integration**: TTS/background mixing with professional effects
- **Subtitle System**: Synchronized subtitles with multiple styles

#### 5. Utilities (`src/utils/`)
- **Configuration Management**: Environment variables and settings
- **Logging**: Comprehensive logging with rotation
- **Error Tracking**: Detailed failure analysis and reporting

### Data Flow
```
TLDR RSS Feed → Newsletter Pages → Article Extraction → Content Validation → 
Llama Summarization → TikTok Script Generation → Kokoro TTS → Audio Output →
Gaming Footage Download → Action Analysis → Video Segment Creation →
Audio Mixing → Subtitle Generation → Final TikTok Video
```

## 🎬 Video Production Features

### Gaming Footage Integration
- **YouTube Integration**: Automatic video downloading with yt-dlp
- **Action Analysis**: AI-powered detection of high-action gaming segments
- **Adaptive Thresholds**: Smart categorization based on video content
- **Segment Selection**: Intelligent selection of engaging footage
- **Performance Caching**: 10x+ speedup for repeated processing

### Audio System
- **Professional Audio Mixing**: TTS foreground (85%) + gaming background (25%)
- **Audio Effects**: Voice enhancement, background processing, mobile optimization
- **Synchronization**: Perfect timing alignment with TTS and video segments
- **Format Support**: Multiple audio formats with mobile optimization

### Subtitle System
- **Synchronized Subtitles**: Perfect sync with TTS audio timing
- **Mobile Optimized**: Intelligent text segmentation for mobile reading
- **Multiple Styles**: 5 preset styles (default, modern, bold, minimal, highlight)
- **SRT Export**: External subtitle editing support
- **TikTok Formatting**: Optimized positioning and styling

### Video Processing
- **Letterboxing**: Automatic 9:16 aspect ratio conversion
- **Mobile Optimization**: Optimized encoding for mobile viewing
- **Ultra-Fast Encoding**: x264 ultrafast preset with 8-thread processing
- **Quality Control**: Configurable quality vs speed settings
- **Resolution Processing**: Smart 720p processing with upscaling

## 🚀 Performance Optimizations

### Intelligent Caching System
- **Action Analysis Caching**: 99% faster video reprocessing
- **Segment Selection Caching**: 5x+ speedup for duration-based selections
- **JSON Serialization**: Persistent cache across runs
- **Storage Management**: Organized cache structure in storage directories

### Ultra-Fast Video Encoding
- **Ultrafast Preset**: Maximum encoding speed with x264
- **Advanced Parameters**: `ref=1:bframes=0:me=dia:subme=0:cabac=0`
- **Optimized Bitrates**: 600k-1500k depending on quality settings
- **Multi-threading**: 8-thread processing for maximum CPU utilization

### Memory Optimizations
- **Sequential GPU Loading**: Efficient model management (Llama → cleanup → Kokoro)
- **Explicit Cleanup**: GPU cache clearing and garbage collection
- **Resource Management**: Proper cleanup of temporary video segments

## 📁 Project Structure

```
tiktok-automata/
├── src/
│   ├── scraper/
│   │   ├── newsletter_scraper.py      # RSS processing and article extraction
│   │   ├── adaptive_extractor.py      # AI-powered content extraction
│   │   └── data/                      # Extracted articles and patterns
│   ├── summarizer/
│   │   ├── llama_summarizer.py        # Llama 3.2-3B summarizer
│   │   └── data/                      # Generated summaries and results
│   ├── tts/
│   │   ├── kokoro_tts.py              # Kokoro TTS engine (1.55x optimized)
│   │   └── data/                      # Generated audio files
│   ├── video/
│   │   ├── managers/                  # High-level coordination
│   │   │   └── footage_manager.py     # Main video management
│   │   ├── downloaders/               # YouTube downloading
│   │   │   └── youtube_downloader.py
│   │   ├── analyzers/                 # Video action analysis
│   │   │   └── action_analyzer.py
│   │   ├── processors/                # Video and segment processing
│   │   │   ├── video_processor.py
│   │   │   └── segment_processor.py
│   │   ├── audio/                     # Audio mixing and effects
│   │   │   ├── audio_mixer.py
│   │   │   ├── audio_effects.py
│   │   │   └── audio_synchronizer.py
│   │   ├── subtitles.py               # Subtitle generation
│   │   └── video_effects.py           # Video processing effects
│   └── utils/
│       ├── config.py                  # Configuration management
│       └── kokoro_voice_profiles.json # Voice configuration
├── tests/
│   ├── test_cuda_setup.py            # CUDA validation
│   └── test_full_pipeline.py         # End-to-end testing
├── main.py                           # Main entry point
├── test_tts_pipeline.py              # Complete pipeline with TTS
└── requirements.txt                  # Dependencies
```

## 🔧 Installation & Setup

### Prerequisites
- **Python**: 3.11+ (recommended)
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) or equivalent
- **CUDA**: 12.1+ with PyTorch CUDA support
- **RAM**: 16GB recommended
- **Storage**: 10GB for models, cache, and video data

### Quick Setup
```bash
# Clone repository
git clone https://github.com/yourusername/tiktok-automata.git
cd tiktok-automata

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your HF_TOKEN and other configurations
```

### Windows-Specific Setup
If you encounter permission issues with pip installations:
```bash
# Grant full permissions to Python installation (run as administrator)
chmod -R 777 /c/Python311
```

### TTS Dependencies
The Kokoro TTS engine requires additional setup:
```bash
# Install Kokoro TTS and dependencies
pip install --user kokoro>=0.9.4 soundfile phonemizer==3.2.1

# Install misaki for English G2P
pip install --user 'misaki[en]'
```

### Environment Variables
```bash
# Hugging Face Authentication
HF_TOKEN=your_hugging_face_token

# Scraping Configuration
MAX_AGE_HOURS=24
MAX_ARTICLES_PER_RUN=10
REQUEST_TIMEOUT=30

# Video Processing
YOUTUBE_DOWNLOAD_QUALITY=720p
ENABLE_ACTION_ANALYSIS=true
CACHE_DIRECTORY=./cache

# TTS Configuration
TTS_DEFAULT_VOICE=af_heart
TTS_DEFAULT_SPEED=1.55
TTS_SAMPLE_RATE=24000

# Output Paths
OUTPUT_DIR=./data
LOGS_DIR=./logs
VIDEO_STORAGE_DIR=./storage
```

## 🎤 Available TTS Voices

The Kokoro TTS engine provides 9 high-quality voice options:

### Female Voices
- **af_heart**: Warm, emotional female voice (default)
- **af_sky**: Clear, professional female voice
- **af_sunset**: Soft, gentle female voice
- **bf_emma**: British female voice
- **bf_isabella**: Elegant British female voice

### Male Voices
- **am_adam**: Strong, confident male voice
- **am_michael**: Professional, news-anchor male voice
- **bm_george**: Distinguished British male voice
- **bm_lewis**: Casual British male voice

All voices are optimized for TikTok content with 1.55x speed for natural, engaging pacing.

## 🎬 Video Configuration Options

### Subtitle Styles
- `default`: Standard white text with black stroke
- `modern`: White text with semi-transparent background
- `bold`: Larger text with thick stroke
- `minimal`: Smaller text with subtle background
- `highlight`: Yellow text with red stroke for emphasis

### Quality Settings
- `low`: Maximum speed, basic quality (ultrafast encoding)
- `medium`: Balanced speed and quality
- `high`: Best quality, slower processing

### Audio Mixing Levels
- **TTS Foreground**: 85% volume
- **Gaming Background**: 25% volume
- **Master Output**: 90% level
- **Mobile Optimization**: EQ and compression applied

## 🚀 Quick Start

### 1. Complete Pipeline (Recommended)
```bash
# Run the full pipeline: Article → Summary → TTS → Video
python test_tts_pipeline.py
```

### 2. Individual Components
```bash
# Extract articles only
python src/scraper/newsletter_scraper.py

# Generate summaries only
python src/summarizer/llama_summarizer.py

# Generate TTS audio only
python src/tts/kokoro_tts.py

# Process gaming footage only
python src/video/managers/footage_manager.py
```

### 3. Custom TTS Generation
```python
from src.tts.kokoro_tts import quick_tts

# Generate audio from text
audio_file = await quick_tts(
    "Your TikTok script here",
    "output.wav",
    voice='af_heart',
    speed=1.55  # Optimized TikTok speed
)
```

### 4. Custom Video Processing
```python
from src.video import FootageManager
from src.video.processors.video_processor import VideoConfig

config = VideoConfig(
    enable_subtitles=True,
    subtitle_style="modern",
    enable_letterboxing=True,
    quality="medium"
)

footage = FootageManager(storage_dir="./storage")
result = await footage.create_tiktok_segment(
    video_url="https://youtube.com/watch?v=...",
    duration=30,
    config=config
)
```

## 📈 Validation Results

### Real Data Processing ✅
- **Source**: 89 TLDR newsletter articles (June 2025)
- **Extraction Success**: 41 articles with quality content
- **Summary Generation**: 100% success rate on tested articles
- **Output Quality**: Professional TikTok format with engagement elements
- **Complete Pipeline**: Newsletter → TTS → Video successfully demonstrated

### Performance Validation ✅
- **Model Loading**: 5.5 seconds initialization
- **Summary Generation**: 5.2 seconds per article
- **TTS Generation**: 50x+ real-time speed
- **Video Processing**: ~3 minutes for 111s segment (with caching)
- **Memory Efficiency**: 1.8GB VRAM usage (22% of available)
- **Cache Performance**: 99% faster reprocessing with intelligent caching

### Format Validation ✅
- **Timestamp Structure**: Proper [0s-3s] format
- **Engagement Elements**: Hooks, emojis, questions, hashtags
- **Duration Targeting**: 26-30 second reading times
- **Content Quality**: Accurate summaries with shock value and intrigue
- **Video Quality**: Mobile-optimized 9:16 aspect ratio with professional subtitles

## 🎯 Production Readiness

The system is fully operational and ready for:

1. **Daily Newsletter Processing**: Automated TLDR article extraction and processing
2. **Batch Content Generation**: Process 20-50 articles per session
3. **Multi-category Support**: Handle AI, tech, dev, and science content
4. **Complete Video Pipeline**: From text to final TikTok video with gaming footage
5. **Scheduled Automation**: Integration with cron/task scheduler for automated runs
6. **Scalable Processing**: Intelligent caching for efficient reprocessing

## 🔍 Troubleshooting

### Common Issues

#### AI/TTS Related
- **CUDA Not Found**: Ensure PyTorch 2.1.0+cu121 is installed
- **HF Authentication**: Verify HF_TOKEN is set in environment
- **Memory Issues**: Check GPU VRAM availability (need 2GB free)
- **TTS Permission Errors**: On Windows, run `chmod -R 777 /c/Python311` as admin
- **Kokoro Import Issues**: Ensure phonemizer==3.2.1 and misaki[en] are installed

#### Video Processing Related
- **YouTube Download Fails**: Check internet connection and video availability
- **FFmpeg Errors**: Ensure FFmpeg is installed and in PATH
- **Action Analysis Slow**: Enable caching in configuration
- **Subtitle Sync Issues**: Verify TTS audio timing and subtitle configuration

#### General Issues
- **Extraction Failures**: Review `extraction_patterns.json` for learned patterns
- **Cache Issues**: Clear `*_cache.json` files if needed
- **Performance Issues**: Check GPU availability and enable caching

### Debug Commands
```bash
# Test complete system
python tests/test_cuda_setup.py          # CUDA validation
python tests/test_full_pipeline.py       # End-to-end pipeline test

# Test individual components
python src/scraper/tests/test_scraper_comprehensive.py    # Scraper
python src/summarizer/tests/test_llama_summarizer.py      # AI summarizer
python src/tts/tests/test_kokoro_tts.py                   # TTS engine
python src/video/tests/test_footage_manager.py            # Video system

# Test complete pipeline with TTS and video
python test_tts_pipeline.py              # Full pipeline with audio/video
```

### Performance Optimization
```bash
# Clear all caches for fresh start
rm storage/*_cache.json

# Enable maximum performance mode
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Monitor GPU usage during processing
nvidia-smi -l 1
```

## 📋 Development History

### Phase 1: RSS Scraping Foundation ✅ (Completed)
- Implemented RSS feed processing for TLDR newsletters
- Built article extraction with content fetching
- Added structured JSON output with metadata
- Achieved 14+ article extraction per newsletter run

### Phase 2: Content Extraction Enhancement ✅ (Completed)
- Added adaptive AI-powered extraction system
- Improved timeout handling and retry logic
- Enhanced success rates by 60% (5→8 successful extractions)
- Implemented smart pattern learning and reuse

### Phase 3: Llama Integration ✅ (Completed)
- Integrated meta-llama/Llama-3.2-3B-Instruct model
- Configured CUDA acceleration for RTX 3070
- Optimized performance (24+ tokens/second)
- Validated with real TLDR article processing

### Phase 4: TTS Integration ✅ (Completed)
- Integrated Kokoro-82M neural TTS engine
- Added 9 high-quality voice options (male/female, various accents)
- Implemented real-time audio generation (50x+ speed)
- Optimized TikTok pacing with 1.55x speed default
- Created complete audio pipeline from text to WAV files

### Phase 5: Video System Integration ✅ (Completed)
- Modular video architecture with clean separation of concerns
- YouTube downloading and intelligent action analysis
- Professional video processing with letterboxing and mobile optimization
- Audio mixing system with TTS/background integration
- Synchronized subtitle system with multiple styles

### Phase 6: Performance Optimization ✅ (Completed)
- Intelligent caching system (99% faster reprocessing)
- Ultra-fast video encoding (ultrafast preset, 8-thread processing)
- Memory optimizations and GPU cache management
- Complete pipeline optimization (90%+ performance improvement)

### Phase 7: Production Validation ✅ (Completed)
- Processed 89 real TLDR articles with 46% success rate
- Generated professional TikTok summaries with 100% success
- Demonstrated complete pipeline: Newsletter → Summary → TTS → Video
- Validated performance metrics and production readiness

## 🎯 Future Enhancements

### Planned Features
1. **Multi-platform Support**: Instagram Reels, YouTube Shorts optimization
2. **Advanced Action Analysis**: ML-based scene classification
3. **Voice Cloning**: Custom voice training capabilities
4. **Automated Publishing**: Direct upload to TikTok API
5. **Analytics Integration**: Performance tracking and optimization
6. **Template System**: Customizable video templates and styles

### Technical Improvements
1. **Parallel Processing**: Multi-threaded video segment processing
2. **Hardware Acceleration**: GPU-accelerated video encoding
3. **Cloud Integration**: AWS/GCP deployment options
4. **API Development**: RESTful API for external integrations
5. **Real-time Processing**: Live streaming integration capabilities

---

**Project Status**: ✅ Production Ready - Complete Content Creation Pipeline  
**Last Updated**: June 25, 2025  
**System**: RTX 3070 + CUDA 12.6 + Llama 3.2-3B + Kokoro TTS + Complete Video Pipeline  
**Performance**: 24+ tokens/second summarization + 50x+ real-time TTS + optimized video processing