#!/usr/bin/env python3
"""
Quick GPU test and pipeline runner with sequential model loading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_gpu_status():
    """Check GPU availability and memory."""
    print("🔍 Checking GPU Status...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"   Memory: {props.total_memory / 1024**3:.1f}GB total")
                
                # Check current memory usage
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   Current: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False

if __name__ == "__main__":
    print("🚀 TikTok Automata: GPU-Optimized Pipeline")
    print("=" * 50)
    
    # Check GPU first
    has_gpu = check_gpu_status()
    
    if has_gpu:
        print("\n🔥 GPU available - running optimized sequential pipeline")
        
        # Import and run the pipeline
        import asyncio
        from complete_pipeline import CompleteTikTokPipeline
        
        async def run_optimized_pipeline():
            pipeline = CompleteTikTokPipeline()
            return await pipeline.run_batch_processing(max_articles=2)  # Smaller batch for testing
        
        result = asyncio.run(run_optimized_pipeline())
        
        if result.get('success'):
            print("\n🎉 Pipeline completed successfully!")
        else:
            print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
    
    else:
        print("\n⚠️ No GPU available - pipeline may be slow")
        print("Consider using a system with CUDA support for optimal performance")
