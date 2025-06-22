#!/usr/bin/env python3
"""
Quick script to check video dimensions and create a letterboxing demonstration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from moviepy import VideoFileClip
    
    # Check original source video dimensions
    source_video = "src/video/data/footage/raw/6nQtYwUqRxM_Squirrel Girl ｜ Marvel Rivals Free To Use Gameplay.mp4"
    
    if Path(source_video).exists():
        clip = VideoFileClip(source_video)
        print(f"Source video: {source_video}")
        print(f"Dimensions: {clip.w}x{clip.h}")
        print(f"Aspect ratio: {clip.h/clip.w:.3f}")
        print(f"Duration: {clip.duration:.2f}s")
        clip.close()
    else:
        print(f"Source video not found: {source_video}")
        
except Exception as e:
    print(f"Error: {e}")
