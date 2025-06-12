<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# TikTok Automata Project Instructions

This is a Python automation project for creating TikTok videos from TLDR newsletter articles.

## Project Guidelines

- **Video Specifications**: All videos must be MP4 format, 1080p resolution, 9:16 vertical aspect ratio, max 30GB size, max 60 minutes length
- **Content Duration**: Target 30 seconds to 2 minutes per video, preferably around 1 minute
- **Architecture**: Use modular design with separate components for scraping, summarization, TTS, video processing, and automation
- **Dependencies**: Use established libraries like requests, BeautifulSoup, moviepy, pyttsx3/gTTS, and openai/transformers
- **Error Handling**: Implement robust error handling and logging throughout the system
- **Configuration**: Use environment variables and config files for API keys and settings
- **Data Flow**: Newsletter → Scraping → Summarization → TTS → Video Assembly → Output

## Code Style

- Follow PEP 8 standards
- Use type hints where appropriate
- Include comprehensive docstrings
- Implement proper exception handling
- Use logging instead of print statements

## Terminal Usage Guidelines

- **AVOID**: Interactive Python commands in terminal (e.g., `python -c "..."`) as they can cause the assistant to get stuck
- **PREFER**: Create temporary test scripts and run them with `python script_name.py`
- **USE**: Non-interactive commands like `pip install`, `git status`, `ls`, etc.
- **TESTING**: Write test scripts to validate functionality instead of inline Python execution
