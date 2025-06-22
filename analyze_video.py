#!/usr/bin/env python3
"""
Quick video analysis script to check letterboxing.
"""

try:
    from moviepy import VideoFileClip
    
    video_path = "letterbox_test/letterboxed_video.mp4"
    clip = VideoFileClip(video_path)
    
    print(f"Video dimensions: {clip.w}x{clip.h}")
    print(f"Aspect ratio: {clip.h/clip.w:.3f}")
    print(f"Duration: {clip.duration:.2f}s")
    print(f"FPS: {clip.fps}")
    
    # Check if it's exactly TikTok format
    if clip.w == 1080 and clip.h == 1920:
        print("✅ Video is in TikTok format (1080x1920)")
    else:
        print(f"❌ Video is not in TikTok format")
    
    clip.close()
    
except Exception as e:
    print(f"Error analyzing video: {e}")
