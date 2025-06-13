# TikTok Automata

An automated system for converting TLDR newsletter articles into engaging TikTok content using AI-powered summarization.

## 🎯 Project Overview

TikTok Automata is a Python automation project that extracts articles from TLDR newsletters and transforms them into TikTok-ready video scripts using the Llama 3.2-3B model with CUDA acceleration.

### Key Features
- **RSS Feed Processing**: Automatically extracts articles from TLDR newsletter RSS feeds
- **Adaptive Content Extraction**: AI-powered system that learns extraction patterns for new websites
- **Llama Summarization**: GPU-accelerated summarization using meta-llama/Llama-3.2-3B-Instruct
- **TikTok Optimization**: Generates professionally formatted scripts with timestamps and engagement elements
- **High-Quality TTS**: Neural text-to-speech using Kokoro with multiple voice options
- **Complete Audio Pipeline**: End-to-end generation from articles to ready-to-use audio files

## 🚀 Current Status: FULLY OPERATIONAL ✅

### System Specifications
- **GPU**: RTX 3070 8GB with CUDA 12.6 support
- **Model**: meta-llama/Llama-3.2-3B-Instruct
- **PyTorch**: 2.1.0+cu121 with CUDA acceleration
- **Performance**: 24+ tokens/second, ~5.2s per summary
- **TTS**: Kokoro-82M neural TTS with 9 voice options

### Validated Capabilities
- ✅ **Article Extraction**: 41/89 articles successfully extracted from real TLDR data
- ✅ **AI Summarization**: 100% success rate on tested articles
- ✅ **TikTok Formatting**: Professional scripts with timestamps, emojis, and hashtags
- ✅ **GPU Acceleration**: Full CUDA support with 1.8GB VRAM usage
- ✅ **TTS Audio Generation**: High-quality neural speech synthesis with Kokoro
- ✅ **Voice Options**: 9 different voices (male/female, American/British accents)
- ✅ **Real-time TTS**: Fast audio generation (1x real-time or faster)
- ✅ **Production Ready**: Complete pipeline from newsletter → TikTok audio content

## 📊 Performance Metrics

### Article Processing
- **Dataset**: 89 TLDR newsletter articles (June 10, 2025)
- **Success Rate**: 46.1% successful extraction (41 articles)
- **Categories**: AI (28), Big Tech (4), Dev (7), Science (2)
- **Content Quality**: All articles >200 words with proper formatting

### Generation Performance
- **Initialization**: ~5.5 seconds
- **Per-article Processing**: ~5.2 seconds
- **Throughput**: ~11 articles per minute
- **Memory Usage**: 1.8GB VRAM / 8GB available

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

#### 2. Summarizer (`src/summarizer/`)
- **Llama Integration**: meta-llama/Llama-3.2-3B-Instruct model
- **CUDA Optimization**: GPU acceleration for RTX 3070
- **TikTok Formatting**: Professional script generation with timestamps
- **Batch Processing**: Efficient handling of multiple articles

#### 3. TTS Engine (`src/tts/`)
- **Kokoro Integration**: High-quality neural TTS using Kokoro-82M model
- **Multiple Voices**: 9 different voice options (male/female, various accents)
- **Real-time Generation**: Fast audio generation (1x real-time or faster)
- **TikTok Optimization**: Automatic text cleaning and duration targeting

#### 4. Utilities (`src/utils/`)
- **Configuration Management**: Environment variables and settings
- **Logging**: Comprehensive logging with rotation
- **Error Tracking**: Detailed failure analysis and reporting

### Data Flow
```
TLDR RSS Feed → Newsletter Pages → Article Extraction → Content Validation → 
Llama Summarization → TikTok Script Generation → Kokoro TTS → Audio Output
```

## 🎬 Adaptive Content Extraction

The system includes an AI-powered adaptive extraction system that automatically learns extraction patterns for new websites:

### Features
- **Pattern Discovery**: Automatically identifies optimal content selectors
- **Success Tracking**: Monitors extraction performance and adapts
- **Pattern Persistence**: Saves learned patterns for reuse
- **Zero Configuration**: Works out-of-the-box without manual setup

### Performance
- **Pattern Discovery**: 9 new patterns learned during testing
- **Success Rate**: 100% for AI-discovered patterns
- **Pattern Reuse**: Automatic optimization for repeat sites
- **Learning Speed**: Real-time pattern discovery and application

## 📁 Project Structure

```
tiktok-automata/
├── data/                             # Legacy data directory (empty)
├── src/
│   ├── scraper/
│   │   ├── newsletter_scraper.py      # Main scraper with RSS processing
│   │   ├── adaptive_extractor.py      # AI-powered content extraction
│   │   ├── data/                      # Scraper output files
│   │   │   ├── extraction_patterns.json    # Learned extraction patterns
│   │   │   └── tldr_articles_*.json        # Extracted articles
│   │   └── tests/
│   │       └── test_scraper_comprehensive.py  # Scraper tests
│   ├── summarizer/
│   │   ├── llama_summarizer.py        # Llama 3.2-3B summarizer
│   │   ├── data/                      # Summarizer output files
│   │   │   ├── tiktok_hooks.json           # Dynamic engagement hooks
│   │   │   ├── llama_test_results_*.json   # Batch test results
│   │   │   ├── tiktok_summaries_*.json     # Batch TikTok summaries
│   │   │   └── tiktok_summary_*.json       # Single TikTok summaries
│   │   └── tests/
│   │       ├── test_llama_summarizer.py       # Basic summarizer tests
│   │       └── test_llama_batch.py            # Batch processing tests
│   ├── tts/
│   │   ├── kokoro_tts.py              # Kokoro TTS engine
│   │   ├── data/                      # Generated audio files
│   │   │   ├── *.wav                       # TTS-generated audio
│   │   │   ├── voice_comparison/           # Voice comparison samples
│   │   │   └── tiktok_pipeline_results_*.json  # Pipeline results with audio
│   │   └── tests/
│   │       ├── test_kokoro_tts.py          # TTS unit tests
│   │       └── __init__.py                 # TTS integration tests
│   │   ├── data/                      # Generated audio files
│   │   │   ├── *.wav                       # TTS-generated audio
│   │   │   ├── voice_comparison/           # Voice comparison samples
│   │   │   └── tiktok_pipeline_results_*.json  # Pipeline results with audio
│   │   └── tests/
│   │       ├── test_kokoro_tts.py          # TTS unit tests
│   │       └── __init__.py                 # TTS integration tests
│   └── utils/
│       └── config.py                  # Configuration management
├── tests/
│   ├── test_cuda_setup.py            # CUDA validation utility
│   └── test_full_pipeline.py         # End-to-end pipeline tests
├── test_tts_pipeline.py              # Complete pipeline with TTS
│   └── test_full_pipeline.py         # End-to-end pipeline tests
├── main.py                           # Main entry point
└── requirements.txt                  # Dependencies
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Configure Hugging Face authentication
# Add HF_TOKEN to .env file for Llama model access
```

### 2. Test Article Extraction
```bash
python main.py  # Extract articles from TLDR RSS
```

### 3. Test Summarization
```bash
# Run all tests with the test runner
python run_tests.py

# Or run individual tests:
# Test basic summarizer functionality
python src/summarizer/tests/test_llama_summarizer.py

# Test batch processing
python src/summarizer/tests/test_llama_batch.py

# Test full pipeline
python tests/test_full_pipeline.py
```

### 4. View Results
Generated files:
- `src/scraper/data/tldr_articles_*.json` - Extracted articles
- `src/summarizer/data/tiktok_summary_*.json` - Generated TikTok summaries

## 📋 Development History

### Phase 1: RSS Scraping Foundation ✅
- Implemented RSS feed processing for TLDR newsletters
- Built article extraction with content fetching
- Added structured JSON output with metadata
- Achieved 14+ article extraction per newsletter run

### Phase 2: Content Extraction Enhancement ✅
- Added adaptive AI-powered extraction system
- Improved timeout handling and retry logic
- Enhanced success rates by 60% (5→8 successful extractions)
- Implemented smart pattern learning and reuse

### Phase 3: Llama Integration ✅
- Integrated meta-llama/Llama-3.2-3B-Instruct model
- Configured CUDA acceleration for RTX 3070
- Optimized performance (24+ tokens/second)
- Validated with real TLDR article processing

### Phase 4: TTS Integration ✅
- Integrated Kokoro-82M neural TTS engine
- Added 9 high-quality voice options (male/female, various accents)
- Implemented real-time audio generation (1x speed or faster)
- Created complete audio pipeline from text to WAV files
- Added TikTok-optimized audio processing with normalization

### Phase 5: Production Validation ✅
- Processed 89 real TLDR articles
- Generated professional TikTok summaries
- Achieved 100% success rate on quality content
- Demonstrated full pipeline functionality
- Validated complete pipeline: Newsletter → Summary → TTS → Audio

## 🔧 Installation & Setup

### Prerequisites
- **Python**: 3.11+ (recommended)
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) or equivalent
- **CUDA**: 12.1+ with PyTorch CUDA support
- **RAM**: 16GB recommended
- **Storage**: 5GB for models and data

### Quick Setup
```bash
# Clone repository
git clone https://github.com/yourusername/tiktok-automata.git
cd tiktok-automata

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your HF_TOKEN
```

### Windows-Specific Setup
If you encounter permission issues with pip installations (especially for TTS dependencies):
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

# Output Paths
OUTPUT_DIR=./data
LOGS_DIR=./logs
```

### Hardware Requirements
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) or equivalent
- **CUDA**: 12.1+ with PyTorch CUDA support
- **RAM**: 16GB recommended
- **Storage**: 5GB for models and data

### Kokoro TTS Voices
Available voice options:
- **af_heart**: Warm, emotional female voice
- **af_sky**: Clear, professional female voice
- **af_sunset**: Soft, gentle female voice
- **am_adam**: Strong, confident male voice
- **am_michael**: Professional, news-anchor male voice
- **bf_emma**: British female voice
- **bf_isabella**: Elegant British female voice
- **bm_george**: Distinguished British male voice
- **bm_lewis**: Casual British male voice

## 📈 Validation Results

### Real Data Processing ✅
- **Source**: 89 TLDR newsletter articles (June 10, 2025)
- **Extraction Success**: 41 articles with quality content
- **Summary Generation**: 100% success rate on tested articles
- **Output Quality**: Professional TikTok format with engagement elements

### Performance Validation ✅
- **Model Loading**: 5.5 seconds initialization
- **Summary Generation**: 5.2 seconds per article
- **Memory Efficiency**: 1.8GB VRAM usage (22% of available)
- **Throughput**: 11 articles per minute processing speed

### Format Validation ✅
- **Timestamp Structure**: Proper [0s-3s] format
- **Engagement Elements**: Hooks, emojis, questions, hashtags
- **Duration Targeting**: 26-30 second reading times
- **Content Quality**: Accurate summaries with shock value and intrigue

## 🎯 Usage

### Quick Start
```bash
# Run the complete pipeline with TTS
python test_tts_pipeline.py

# Run individual components
python src/scraper/newsletter_scraper.py    # Extract articles
python src/summarizer/llama_summarizer.py   # Generate summaries
python src/tts/kokoro_tts.py                # Generate audio
```

### TTS-Only Usage
```python
from src.tts.kokoro_tts import quick_tts

# Generate audio from text
audio_file = await quick_tts(
    "Your TikTok script here",
    "output.wav",
    voice='af_heart',
    speed=1.1
)
```

## 🎯 Production Readiness

The system is fully operational and ready for:

1. **Daily Newsletter Processing**: Automated TLDR article extraction
2. **Batch Content Generation**: Process 20-50 articles per session
3. **Multi-category Support**: Handle AI, tech, dev, and science content
4. **Complete Audio Pipeline**: Text-to-speech with 9 voice options
5. **Scheduled Automation**: Integration with cron/task scheduler
6. **Video Pipeline Integration**: Ready for video assembly and upload

## 🔍 Troubleshooting

### Common Issues
- **CUDA Not Found**: Ensure PyTorch 2.1.0+cu121 is installed
- **HF Authentication**: Verify HF_TOKEN is set in environment
- **Memory Issues**: Check GPU VRAM availability (need 2GB free)
- **Extraction Failures**: Review `extraction_patterns.json` for learned patterns
- **TTS Permission Errors**: On Windows, run `chmod -R 777 /c/Python311` as admin
- **Kokoro Import Issues**: Ensure phonemizer==3.2.1 and misaki[en] are installed

### Debug Commands
```bash
# Test CUDA setup
python tests/test_cuda_setup.py

# Test scraper functionality
python src/scraper/tests/test_scraper_comprehensive.py

# Test summarizer only
python src/summarizer/tests/test_llama_summarizer.py

# Test TTS functionality
python src/tts/tests/test_kokoro_tts.py

# Full pipeline test with TTS
python test_tts_pipeline.py

# Test individual TTS voices
python src/tts/kokoro_tts.py
```

---

**Project Status**: ✅ Production Ready with Complete TTS Pipeline  
**Last Updated**: June 13, 2025  
**System**: RTX 3070 + CUDA 12.6 + Llama 3.2-3B + Kokoro TTS  
**Performance**: 24+ tokens/second summarization + real-time TTS generation