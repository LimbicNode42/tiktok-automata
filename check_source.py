#!/usr/bin/env python3
"""
Check source footage dimensions.
"""

try:
    from moviepy import VideoFileClip
    
    # Check source footage
    source_path = "src/video/data/footage/raw/ne_T-GNwObk_COD Black Ops 6 Gameplay - Free To Use.mp4"
    clip = VideoFileClip(source_path)
    
    print(f"Source video dimensions: {clip.w}x{clip.h}")
    print(f"Source aspect ratio: {clip.h/clip.w:.3f}")
    print(f"Duration: {clip.duration:.2f}s")
    
    # Calculate what letterboxing should do
    target_aspect = 1920 / 1080  # 1.778
    source_aspect = clip.h / clip.w
    
    print(f"\nTarget aspect ratio: {target_aspect:.3f}")
    print(f"Source aspect ratio: {source_aspect:.3f}")
    
    if source_aspect > target_aspect:
        print("Source is taller - should add horizontal bars")
    elif source_aspect < target_aspect:
        print("Source is wider - should add vertical bars (letterbox)")
    else:
        print("Aspect ratios match - no bars needed")
    
    clip.close()
    
except Exception as e:
    print(f"Error: {e}")
