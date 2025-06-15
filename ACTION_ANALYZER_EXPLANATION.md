# Action Analyzer Variables & Thresholds Explanation

## 🔢 Core Variables Used for Action Categorization

### 1. **Motion Intensity** (`motion_intensity`)
- **What it measures**: How much movement occurs between consecutive frames
- **How it's calculated**: 
  - Frame difference: `np.mean(np.abs(frame_current - frame_previous))`
  - Enhanced with optical flow approximation for better accuracy
  - Combined: `(basic_motion * 0.7) + (optical_flow * 0.3)`
- **Typical ranges**: 
  - Low action: 0-10
  - High action gaming: 15-40+
- **Weight in overall score**: 40% (most important factor)

### 2. **Edge Density** (`edge_density`)
- **What it measures**: Amount of visual detail and complexity in the scene
- **How it's calculated**: 
  - Sobel-like edge detection on grayscale frames
  - `edges_x = np.abs(np.diff(gray, axis=1))` (horizontal edges)
  - `edges_y = np.abs(np.diff(gray, axis=0))` (vertical edges)
  - `edge_density = np.mean(edges_x) + np.mean(edges_y)`
- **Typical ranges**:
  - Simple scenes: 2-5
  - Complex gaming: 8-20+
- **Weight in overall score**: 35%

### 3. **Scene Complexity** (`scene_complexity`)
- **What it measures**: Texture and local variance within the scene
- **How it's calculated**:
  - Analyzes 3x3 patches across the frame
  - Calculates local variance in each patch
  - `np.var(patch)` for texture measurement
- **Typical ranges**:
  - Static scenes: 0-5
  - Complex scenes: 5-15+
- **Weight in overall score**: 25%

### 4. **Color Variance** (`color_variance`)
- **What it measures**: How much colors change across the entire frame
- **How it's calculated**: `np.var(frame.flatten())` across all RGB channels
- **Typical ranges**: 500-5000+ (very wide range)
- **Weight in overall score**: 0.01% (scaled down heavily due to large values)

### 5. **Overall Score** (`overall_score`)
- **Formula**: 
  ```python
  overall_score = (
      motion_intensity * 0.4 +      # 40% weight
      color_variance * 0.0001 +     # 0.01% weight (scaled)
      edge_density * 0.35 +         # 35% weight  
      scene_complexity * 0.25       # 25% weight
  )
  ```
- **This is the PRIMARY variable used for categorization**

## 🎚️ Threshold Systems

### **Fixed Thresholds** (Default/Simple)
- **High Action**: `overall_score > 20`
- **Medium Action**: `10 < overall_score <= 20`
- **Low Action**: `overall_score <= 10`

**Pros**: Simple, consistent across all videos
**Cons**: Doesn't adapt to video content type

### **Adaptive Thresholds** (Smart/Statistical)
**Purpose**: Dynamically adjust thresholds based on the specific video's action distribution

**How it works**:
1. **Analyze all segments** in the video first
2. **Calculate percentiles** of overall_score distribution:
   - P90 = 90th percentile (top 10% of action)
   - P70 = 70th percentile (top 30% of action)
   - P50 = 50th percentile (median)

3. **Set adaptive thresholds**:
   ```python
   high_threshold = max(P90, median + 5.0)
   medium_threshold = max(P70, median + 2.0)
   ```

4. **Ensure minimum separation** to avoid overlap

**Example from our gaming video**:
- Original: 99.6% high action (fixed thresholds)
- Adaptive: 10.1% high action (much more selective)
- High threshold: 162.4 (vs fixed 20)
- Medium threshold: 135.8 (vs fixed 10)

## 📊 Real-World Example Analysis

From the COD Black Ops 6 test:
```
Fixed Thresholds:
- High: 237 segments (99.6%) - almost everything
- Medium: 0 segments (0%)
- Low: 1 segment (0.4%)

Adaptive Thresholds:
- High: 24 segments (10.1%) - only the best action
- Medium: 48 segments (20.2%) - good action
- Low: 166 segments (69.7%) - relative low action
```

## 🎮 Content Type Detection

Uses the same variables to detect content types:

```python
def detect_content_type(metrics):
    if metrics.motion_intensity > 12 and metrics.edge_density > 8:
        return "gaming"  # High motion + high detail
    elif metrics.motion_intensity > 8 and metrics.scene_complexity > 5:
        return "action"  # High motion + moderate complexity
    elif metrics.motion_intensity < 3 and metrics.edge_density < 4:
        return "dialogue"  # Low motion + low detail
    elif 3 <= metrics.motion_intensity <= 8 and metrics.color_variance > 2000:
        return "transition"  # Moderate motion + high color change
```

## 🔍 Why Adaptive Thresholds Matter

**Gaming Content Problem**: 
- Gaming footage is naturally high-action
- Fixed thresholds mark 99%+ as "high action"
- Makes it impossible to distinguish the BEST action moments

**Adaptive Solution**:
- Looks at the video's own distribution
- Only top 10% becomes "high action"
- Gives you the cream of the crop for TikTok clips

**Statistical Selectivity**:
- P90: Top 10% (truly exceptional moments)
- P70: Top 30% (good action moments)
- Everything else: Relatively lower action

This ensures you always get the most statistically relevant, high-action segments regardless of the video's baseline action level.
