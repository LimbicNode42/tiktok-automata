#!/usr/bin/env python3
"""
CUDA Setup Test Script
Tests PyTorch CUDA functionality for RTX 3070 GPU
"""

import sys
import torch
import time

def test_cuda_setup():
    """Test CUDA setup and GPU functionality"""
    print("=" * 60)
    print("CUDA SETUP TEST")
    print("=" * 60)
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available - check installation")
        return False
    
    # CUDA details
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # GPU information
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Performance test
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    # CPU test
    print("Testing CPU performance...")
    start_time = time.time()
    cpu_tensor = torch.randn(1000, 1000)
    cpu_result = torch.mm(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start_time
    print(f"CPU matrix multiplication: {cpu_time:.4f} seconds")
    
    # GPU test
    print("Testing GPU performance...")
    start_time = time.time()
    gpu_tensor = torch.randn(1000, 1000).cuda()
    gpu_result = torch.mm(gpu_tensor, gpu_tensor)
    torch.cuda.synchronize()  # Wait for GPU operation to complete
    gpu_time = time.time() - start_time
    print(f"GPU matrix multiplication: {gpu_time:.4f} seconds")
    
    # Performance comparison
    speedup = cpu_time / gpu_time
    print(f"\nSpeedup: {speedup:.2f}x faster on GPU")
    
    # Memory test
    print("\n" + "=" * 60)
    print("MEMORY TEST")
    print("=" * 60)
    
    # Check available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    cached_memory = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"Total GPU memory: {total_memory:.2f} GB")
    print(f"Allocated memory: {allocated_memory:.2f} GB")
    print(f"Cached memory: {cached_memory:.2f} GB")
    print(f"Available memory: {total_memory - cached_memory:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✅ CUDA SETUP SUCCESSFUL!")
    print("✅ RTX 3070 is ready for ML workloads")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = test_cuda_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during CUDA test: {e}")
        sys.exit(1)
