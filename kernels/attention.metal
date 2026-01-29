#include <metal_stdlib> // error expected if not in xcode
using namespace metal;

#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float 
#define uchar unsigned char
#define thread_position_in_grid 0
#endif

// Attention Score: Score = (Q * K^T) / sqrt(head_dim)
kernel void attention_scores(
    device const half* q_ptr,
    device const half* k_ptr,
    device const half* v_ptr,
    device half* k_cache,
    device half* v_cache,
    device float* scores_out,
    constant uint& layer_idx,
    constant uint& current_pos,
    uint head_idx [[thread_position_in_grid]])
{
    const uint num_kv_heads = 2;
    const uint head_dim = 128;
    const uint max_seq_len = 1024;
    const uint kv_dim = 256;

    uint layer_offset = layer_idx * max_seq_len * kv_dim;
    uint token_offset = current_pos * kv_dim;

    // 1. CACHE WRITE
    // `k_ptr` here is the per-token K projection passed from `engine.py`.
    // RoPE is applied to `buf_k` *before* this kernel runs, so we cache the RoPE-rotated K (Qwen2.5 expects this).
    if (head_idx < num_kv_heads) {
        for (uint d = 0; d < head_dim; d++) {
            uint cache_idx = layer_offset + token_offset + (head_idx * head_dim) + d;
            k_cache[cache_idx] = k_ptr[(head_idx * head_dim) + d];
            v_cache[cache_idx] = v_ptr[(head_idx * head_dim) + d];
        }
    }

    // CRITICAL: Ensure writes are finished before reading back for scores
    threadgroup_barrier(mem_flags::mem_device);

    // 2. DOT PRODUCT LOOP
    device const half* q_head = q_ptr + (head_idx * head_dim);
    uint kv_head_idx = head_idx / 6; 
    float scale = 1.0f / sqrt(128.0f);
    
    for (uint t = 0; t <= current_pos; t++) {
        float sum = 0.0f;
        // Read from the cache we just wrote to
        device half* k_cached = k_cache + layer_offset + (t * kv_dim) + (kv_head_idx * head_dim);
        
        for (uint d = 0; d < head_dim; d++) {
            sum += (float)q_head[d] * (float)k_cached[d];
        }
        scores_out[(head_idx * max_seq_len) + t] = sum * scale;
    }
}
