import numpy as np
from runners.engine import QwenEngine

def debug_kv_cache(weights_dir):
    # 1. Initialize engine
    engine = QwenEngine(weights_dir)
    
    # 2. Run one dummy inference pass (using a zero vector or actual embedding)
    dummy_input = np.zeros(1536, dtype=np.float16)
    _ = engine.run_inference(dummy_input)
    
    # 3. Pull the KV Cache back to the CPU
    # Format: [Layers][Max_Seq][2 * K_DIM]
    # K_DIM is 256 (2 heads * 128)
    k_cache_raw = np.frombuffer(engine.buf_k_cache.contents().as_buffer(engine.buf_k_cache.length()), dtype=np.float16)
    v_cache_raw = np.frombuffer(engine.buf_v_cache.contents().as_buffer(engine.buf_v_cache.length()), dtype=np.float16)
    
    # 4. Inspect Layer 0, Position 0
    # The first 256 elements of the buffer should be the K/V vectors for the first token
    l0_p0_k = k_cache_raw[:256]
    l0_p0_v = v_cache_raw[:256]
    
    print("--- KV CACHE DIAGNOSTIC (Layer 0, Pos 0) ---")
    print(f"K-Cache Mean: {np.mean(l0_p0_k):.6f} | Non-zero count: {np.count_nonzero(l0_p0_k)}/256")
    print(f"V-Cache Mean: {np.mean(l0_p0_v):.6f} | Non-zero count: {np.count_nonzero(l0_p0_v)}/256")
    
    if np.all(l0_p0_k == 0):
        print("\n[CRITICAL ERROR]: K-Cache is empty. Your kernel write index is broken.")
    elif np.any(np.isnan(l0_p0_k)):
        print("\n[CRITICAL ERROR]: NaNs detected in Cache. Check your RoPE or Norm kernels.")
    else:
        print("\n[SUCCESS]: GPU is successfully writing stateful data to the cache.")

if __name__ == "__main__":
    import os
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    debug_kv_cache(os.path.join(PROJECT_ROOT, "weights"))
