# GTX 1070 8GB Analysis for TikTok Automata

## 💰 **GTX 1070 8GB from CEX - Cost/Benefit Analysis**

### ✅ **Pros of GTX 1070 8GB:**

1. **Excellent Value Proposition**
   - CEX price: ~£80-120 (extremely cheap)
   - 8GB VRAM (same as your RTX 3070)
   - Dedicated GPU for secondary tasks

2. **Perfect for TTS Workloads**
   - Kokoro TTS model is lightweight (~500MB)
   - 1070 has plenty of power for TTS generation
   - Frees up your RTX 3070 entirely for LLM

3. **Good for Background Tasks**
   - Video preprocessing
   - Action analysis caching
   - Parallel encoding jobs

4. **Low Power/Heat**
   - Won't strain your PSU significantly
   - Older but efficient architecture

### ⚠️ **Limitations to Consider:**

1. **No Hardware Encoding**
   - GTX 1070 lacks NVENC (no h264_nvenc support)
   - RTX 3070 would still handle video encoding

2. **Older Architecture**
   - Pascal vs Ampere (your RTX 3070)
   - Less efficient for modern AI workloads
   - No Tensor cores (but TTS doesn't need them)

3. **CUDA Compatibility**
   - Need to ensure PyTorch supports dual-GPU with mixed architectures
   - Might need older CUDA drivers

### 🎯 **Optimal Setup with GTX 1070:**

```
RTX 3070 (GPU 0) - Primary:
├── LLM inference (Llama 3.2-3B)
├── Hardware video encoding (NVENC)
└── Any CUDA-intensive tasks

GTX 1070 (GPU 1) - Secondary:
├── TTS generation (Kokoro)
├── Video preprocessing
├── Action analysis
└── Background caching tasks
```

### 📊 **Expected Performance Gains:**

1. **TTS Parallelization**: 30-40% pipeline speedup
2. **Memory Relief**: RTX 3070 focuses entirely on LLM
3. **Background Processing**: Smoother multitasking

### 💡 **Implementation Strategy:**

1. **Minimal Code Changes Needed**
   - Add GPU device selection in pipeline
   - Route TTS to GPU 1, LLM to GPU 0
   - Keep encoding on RTX 3070

2. **Fallback Support**
   - If GTX 1070 fails, fall back to single GPU
   - Graceful degradation

### 🔧 **Technical Considerations:**

- **PSU**: Check if you have enough power connectors
- **Slot Space**: Ensure motherboard has 2x PCIe slots
- **Cooling**: Additional heat in case
- **Driver**: Unified NVIDIA drivers should work fine

## 🏆 **Verdict: EXCELLENT Budget Choice!**

For £80-120, a GTX 1070 8GB is probably the best price/performance option:

- **ROI**: Extremely high (30-40% speedup for <£150)
- **Risk**: Very low (cheap enough to experiment)
- **Compatibility**: High (same manufacturer, similar generation)
- **Future**: Can always resell if you upgrade later

### 📋 **Action Plan:**

1. **Buy the GTX 1070** (if price is right)
2. **Test dual-GPU setup** with existing code
3. **Implement TTS GPU routing** (simple change)
4. **Benchmark performance gains**

This would be much more cost-effective than a second RTX 3070/4070 while providing 80% of the benefits for 20% of the cost!
