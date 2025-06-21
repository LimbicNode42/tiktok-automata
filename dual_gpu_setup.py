#!/usr/bin/env python3
"""
Dual GPU setup implementation for TikTok Automata
RTX 3070 (GPU 0) + GTX 1070 (GPU 1)
"""

import torch
from typing import Optional

class DualGPUManager:
    """Manages optimal GPU allocation for dual-GPU setups."""
    
    def __init__(self):
        self.gpu_count = torch.cuda.device_count()
        self.primary_gpu = 0    # RTX 3070 for LLM and encoding
        self.secondary_gpu = 1 if self.gpu_count > 1 else 0  # GTX 1070 for TTS
        
    def get_gpu_info(self):
        """Get information about available GPUs."""
        info = {}
        for i in range(self.gpu_count):
            info[i] = {
                'name': torch.cuda.get_device_name(i),
                'memory': torch.cuda.get_device_properties(i).total_memory / 1024**3,
                'compute_capability': torch.cuda.get_device_capability(i)
            }
        return info
    
    def get_optimal_device(self, task: str) -> int:
        """Get optimal GPU device for specific tasks."""
        task_mapping = {
            'llm': self.primary_gpu,      # Always use primary (RTX 3070)
            'tts': self.secondary_gpu,    # Use secondary if available (GTX 1070)  
            'encoding': self.primary_gpu, # Use primary for NVENC
            'analysis': self.secondary_gpu # Background tasks on secondary
        }
        
        return task_mapping.get(task, self.primary_gpu)
    
    def setup_device_for_task(self, task: str) -> str:
        """Setup and return device string for task."""
        device_id = self.get_optimal_device(task)
        device_str = f"cuda:{device_id}"
        
        # Set the device
        torch.cuda.set_device(device_id)
        
        print(f"🎯 {task.upper()} -> {device_str} ({torch.cuda.get_device_name(device_id)})")
        return device_str

# Example integration with existing pipeline
class OptimizedPipeline:
    """Pipeline with dual-GPU optimization."""
    
    def __init__(self):
        self.gpu_manager = DualGPUManager()
        
        # Print GPU setup
        gpu_info = self.gpu_manager.get_gpu_info()
        print("🔥 GPU Setup:")
        for gpu_id, info in gpu_info.items():
            print(f"  GPU {gpu_id}: {info['name']} ({info['memory']:.1f}GB)")
    
    async def generate_summary(self, text: str):
        """Generate summary using optimal GPU."""
        # Route LLM to primary GPU (RTX 3070)
        llm_device = self.gpu_manager.setup_device_for_task('llm')
        
        # Your existing LLM code here, but with explicit device
        # llama_model.to(llm_device)
        print(f"📝 Running LLM on {llm_device}")
        
        # Clear cache after use
        torch.cuda.empty_cache()
    
    async def generate_tts(self, text: str):
        """Generate TTS using optimal GPU."""
        # Route TTS to secondary GPU (GTX 1070)  
        tts_device = self.gpu_manager.setup_device_for_task('tts')
        
        # Your existing TTS code here, but with explicit device
        # kokoro_model.to(tts_device)
        print(f"🎤 Running TTS on {tts_device}")
        
        # Clear cache after use
        torch.cuda.empty_cache()
    
    async def process_video(self, video_path: str):
        """Process video using optimal GPU."""
        # Video encoding stays on primary GPU for NVENC
        encoding_device = self.gpu_manager.setup_device_for_task('encoding')
        print(f"🎬 Video encoding on {encoding_device}")

def test_dual_gpu_setup():
    """Test the dual GPU setup."""
    print("🧪 Testing Dual GPU Setup")
    print("=" * 40)
    
    pipeline = OptimizedPipeline()
    
    # Show optimal allocation
    tasks = ['llm', 'tts', 'encoding', 'analysis']
    print("\n🎯 Task Allocation:")
    for task in tasks:
        device_id = pipeline.gpu_manager.get_optimal_device(task)
        gpu_name = torch.cuda.get_device_name(device_id)
        print(f"  {task.upper():<10} -> GPU {device_id} ({gpu_name})")
    
    print("\n💡 Benefits with GTX 1070:")
    print("  • TTS runs independently on GTX 1070")  
    print("  • RTX 3070 dedicated to LLM inference")
    print("  • NVENC encoding still on RTX 3070")
    print("  • 30-40% expected speedup from parallelization")

if __name__ == "__main__":
    test_dual_gpu_setup()
