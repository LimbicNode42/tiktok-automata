#!/usr/bin/env python3
"""
Comprehensive test for the Action Analyzer.
Tests action intensity detection, categorization, and statistical analysis.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video.analyzers.action_analyzer import VideoActionAnalyzer, ActionMetrics, ContentTypeDetector
from video import FootageManager
from loguru import logger

async def test_action_analyzer_basic():
    """Test basic action analyzer functionality."""
    logger.info("🎯 Testing Action Analyzer - Basic Functionality")
    
    try:
        analyzer = VideoActionAnalyzer()
        
        # Test with sample metrics
        test_metrics = [
            ActionMetrics(motion_intensity=25.0, edge_density=15.0, scene_complexity=8.0, overall_score=28.0, timestamp=10.0),
            ActionMetrics(motion_intensity=15.0, edge_density=8.0, scene_complexity=5.0, overall_score=15.0, timestamp=20.0),
            ActionMetrics(motion_intensity=5.0, edge_density=3.0, scene_complexity=2.0, overall_score=5.0, timestamp=30.0),
            ActionMetrics(motion_intensity=30.0, edge_density=20.0, scene_complexity=10.0, overall_score=35.0, timestamp=40.0),
        ]
        
        # Test categorization
        categorized = analyzer._categorize_metrics(test_metrics)
        
        logger.info(f"📊 Categorization Results:")
        logger.info(f"   🔥 High Action: {len(categorized['high'])} segments")
        logger.info(f"   🔸 Medium Action: {len(categorized['medium'])} segments") 
        logger.info(f"   🔹 Low Action: {len(categorized['low'])} segments")
        
        # Test segment extraction
        segments = analyzer.get_best_action_segments(categorized, segment_duration=30.0, max_segments=2)
        logger.info(f"🎯 Best segments: {len(segments)} extracted")
        for i, (start, end) in enumerate(segments):
            logger.info(f"   Segment {i+1}: {start:.1f}s - {end:.1f}s")
        
        # Test statistical analysis
        stats = analyzer.analyze_action_distribution(categorized)
        logger.info(f"📈 Statistics: {stats['high_action_ratio']:.1%} high action, avg score: {stats['avg_overall_score']:.1f}")
        
        logger.success("✅ Basic action analyzer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic action analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_adaptive_thresholds():
    """Test adaptive threshold calculation."""
    logger.info("🎚️ Testing Adaptive Thresholds")
    
    try:
        analyzer = VideoActionAnalyzer()
        
        # Create test data with different distributions
        
        # Dataset 1: High variance (mix of very low and very high action)
        high_variance_data = []
        for i in range(20):
            score = 5.0 if i < 10 else 30.0  # Clear bimodal distribution
            high_variance_data.append(ActionMetrics(overall_score=score, timestamp=i*2.0))
        
        # Dataset 2: Low variance (all medium action)
        low_variance_data = []
        for i in range(20):
            score = 12.0 + np.random.normal(0, 2)  # Normal distribution around 12
            low_variance_data.append(ActionMetrics(overall_score=score, timestamp=i*2.0))
        
        # Test adaptive thresholds for both datasets
        for name, data in [("High Variance", high_variance_data), ("Low Variance", low_variance_data)]:
            logger.info(f"   Testing {name} Dataset:")
            
            thresholds = analyzer.get_adaptive_thresholds(data)
            logger.info(f"      Adaptive High Threshold: {thresholds['high_threshold']:.1f}")
            logger.info(f"      Adaptive Medium Threshold: {thresholds['medium_threshold']:.1f}")
            
            # Test categorization with adaptive thresholds
            categorized = analyzer.categorize_with_adaptive_thresholds(data)
            total = len(data)
            high_pct = len(categorized['high']) / total * 100
            medium_pct = len(categorized['medium']) / total * 100
            low_pct = len(categorized['low']) / total * 100
            
            logger.info(f"      Distribution: {high_pct:.0f}% high, {medium_pct:.0f}% medium, {low_pct:.0f}% low")
        
        logger.success("✅ Adaptive thresholds test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive thresholds test failed: {e}")
        return False

async def test_content_type_detection():
    """Test content type detection functionality."""
    logger.info("🎮 Testing Content Type Detection")
    
    try:
        # Test different content type scenarios
        test_cases = [
            {
                "name": "Gaming Content",
                "metrics": ActionMetrics(motion_intensity=15.0, edge_density=10.0, scene_complexity=6.0, color_variance=1500),
                "expected": "gaming"
            },
            {
                "name": "Action Scene", 
                "metrics": ActionMetrics(motion_intensity=12.0, edge_density=6.0, scene_complexity=7.0, color_variance=1000),
                "expected": "action"
            },
            {
                "name": "Dialogue Scene",
                "metrics": ActionMetrics(motion_intensity=2.0, edge_density=3.0, scene_complexity=2.0, color_variance=800),
                "expected": "dialogue"
            },
            {
                "name": "Transition Scene",
                "metrics": ActionMetrics(motion_intensity=6.0, edge_density=5.0, scene_complexity=4.0, color_variance=2500),
                "expected": "transition"
            }
        ]
        
        correct_detections = 0
        for test_case in test_cases:
            detected_type = ContentTypeDetector.detect_content_type(test_case["metrics"])
            is_correct = detected_type == test_case["expected"]
            
            status = "✅" if is_correct else "❌"
            logger.info(f"   {status} {test_case['name']}: detected '{detected_type}' (expected '{test_case['expected']}')")
            
            if is_correct:
                correct_detections += 1
        
        accuracy = correct_detections / len(test_cases)
        logger.info(f"📊 Content Type Detection Accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.75:  # 75% accuracy threshold
            logger.success("✅ Content type detection test passed")
            return True
        else:
            logger.warning("⚠️ Content type detection accuracy below threshold")
            return False
            
    except Exception as e:
        logger.error(f"❌ Content type detection test failed: {e}")
        return False

async def test_with_real_video():
    """Test action analyzer with real video footage."""
    logger.info("🎥 Testing Action Analyzer with Real Video")
    
    try:
        fm = FootageManager()
        analyzer = VideoActionAnalyzer()
        
        # Check if we have downloaded videos
        if not fm.metadata.get("videos"):
            logger.warning("⚠️ No videos available for testing")
            logger.info("💡 This test requires downloaded video footage")
            return True  # Don't fail if no test data
        
        # Test with first available video
        video_id = list(fm.metadata["videos"].keys())[0]
        video_info = fm.metadata["videos"][video_id]
        video_path = Path(video_info["file_path"])
        
        if not video_path.exists():
            logger.warning(f"⚠️ Video file not found: {video_path}")
            return True
        
        logger.info(f"🔍 Analyzing real video: {video_id}")
        
        # Run action analysis
        action_data = await analyzer.analyze_video_action(video_path)
        
        # Verify results
        total_segments = len(action_data["high"]) + len(action_data["medium"]) + len(action_data["low"])
        
        if total_segments > 0:
            logger.info(f"📊 Analysis Results:")
            logger.info(f"   Total segments analyzed: {total_segments}")
            logger.info(f"   🔥 High action: {len(action_data['high'])}")
            logger.info(f"   🔸 Medium action: {len(action_data['medium'])}")
            logger.info(f"   🔹 Low action: {len(action_data['low'])}")
            
            # Test statistical analysis
            stats = analyzer.analyze_action_distribution(action_data)
            logger.info(f"   📈 Action ratio: {stats['high_action_ratio']:.1%}")
            logger.info(f"   📈 Avg intensity: {stats['avg_overall_score']:.1f}")
            
            # Test adaptive thresholds
            all_metrics = action_data["high"] + action_data["medium"] + action_data["low"]
            adaptive_categorized = analyzer.categorize_with_adaptive_thresholds(all_metrics)
            
            logger.info(f"   🎚️ Adaptive categorization:")
            logger.info(f"      High: {len(adaptive_categorized['high'])}")
            logger.info(f"      Medium: {len(adaptive_categorized['medium'])}")
            logger.info(f"      Low: {len(adaptive_categorized['low'])}")
            
            logger.success("✅ Real video analysis test passed")
            return True
        else:
            logger.error("❌ No segments analyzed from real video")
            return False
            
    except Exception as e:
        logger.error(f"❌ Real video analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def benchmark_action_analysis():
    """Benchmark action analysis performance."""
    logger.info("⏱️ Benchmarking Action Analysis Performance")
    
    try:
        analyzer = VideoActionAnalyzer()
        
        # Create large test dataset
        large_dataset = []
        for i in range(100):
            metrics = ActionMetrics(
                motion_intensity=np.random.uniform(0, 40),
                edge_density=np.random.uniform(0, 20),
                scene_complexity=np.random.uniform(0, 15),
                overall_score=np.random.uniform(0, 50),
                timestamp=i * 2.0
            )
            large_dataset.append(metrics)
        
        # Benchmark categorization
        import time
        start_time = time.time()
        
        categorized = analyzer._categorize_metrics(large_dataset)
        categorization_time = time.time() - start_time
        
        # Benchmark adaptive thresholds
        start_time = time.time()
        adaptive_categorized = analyzer.categorize_with_adaptive_thresholds(large_dataset)
        adaptive_time = time.time() - start_time
        
        # Benchmark segment extraction
        start_time = time.time()
        segments = analyzer.get_best_action_segments(categorized, segment_duration=30.0, max_segments=10)
        segment_time = time.time() - start_time
        
        logger.info(f"📊 Performance Results (100 segments):")
        logger.info(f"   Basic categorization: {categorization_time*1000:.1f}ms")
        logger.info(f"   Adaptive categorization: {adaptive_time*1000:.1f}ms")
        logger.info(f"   Segment extraction: {segment_time*1000:.1f}ms")
        
        # Performance should be reasonable (< 100ms for each operation)
        if categorization_time < 0.1 and adaptive_time < 0.1 and segment_time < 0.1:
            logger.success("✅ Performance benchmark passed")
            return True
        else:
            logger.warning("⚠️ Performance slower than expected")
            return False
            
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False

async def main():
    """Run all action analyzer tests."""
    logger.info("🧪 Starting Action Analyzer Comprehensive Tests")
    
    tests = [
        ("Basic Functionality", test_action_analyzer_basic),
        ("Adaptive Thresholds", test_adaptive_thresholds),
        ("Content Type Detection", test_content_type_detection),
        ("Real Video Analysis", test_with_real_video),
        ("Performance Benchmark", benchmark_action_analysis),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            if result:
                passed += 1
                logger.success(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"🏁 Action Analyzer Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.success("🎉 All action analyzer tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
