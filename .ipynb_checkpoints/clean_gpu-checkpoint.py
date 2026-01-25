import torch
import gc
import os

# 强制垃圾回收
gc.collect()

# 清空CUDA缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    
    print("GPU memory cleared")
    print(f"Current memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Current memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")