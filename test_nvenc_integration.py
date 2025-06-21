#!/usr/bin/env python3
"""
Test NVENC integration in the video processor.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.video.processors.video_processor import VideoProcessor

def test_nvenc_detection():
    """Test NVENC detection and settings."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize config
        config = Config()
        
        # Add video config attributes for testing
        config.width = 1080
        config.height = 1920
        config.fps = 30
        config.output_quality = "medium"
        config.max_file_size_mb = 100
        
        # Initialize video processor
        logger.info("Initializing video processor...")
        processor = VideoProcessor(config)
        
        # Test NVENC detection
        logger.info("Testing NVENC detection...")
        export_settings = processor._get_export_settings()
        
        logger.info(f"Export settings: {export_settings}")
        
        if export_settings.get("codec") == "h264_nvenc":
            logger.info("✅ NVENC detected and will be used!")
            logger.info(f"Bitrate: {export_settings.get('bitrate')}")
            logger.info(f"Preset: {export_settings.get('preset')}")
            logger.info(f"NVENC params: {export_settings.get('nvenc_params')}")
        else:
            logger.info("💻 CPU encoding will be used")
            logger.info(f"Codec: {export_settings.get('codec')}")
            logger.info(f"Preset: {export_settings.get('preset')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nvenc_detection()
    sys.exit(0 if success else 1)
