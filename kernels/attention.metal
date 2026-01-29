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
    device const half* q_ptr [[buffer(0)]],
    device const half* k_ptr [[buffer(1)]],
    device const half* v_ptr [[buffer(2)]],
    device half* k_cache [[buffer(3)]],
    device half* v_cache [[buffer(4)]],
    device float* scores_out [[buffer(5)]],
    constant uint& layer_idx [[buffer(6)]],
    constant uint& current_pos [[buffer(7)]],
    uint head_idx [[thread_position_in_grid]])
{
    const uint num_kv_heads = 2;
    const uint head_dim = 128;
    const uint max_seq_len = 1024;
    const uint kv_heads_per_q = 6; 

    uint layer_offset = layer_idx * max_seq_len * 256;
    uint token_offset = current_pos * 256;
    
    if (head_idx < num_kv_heads) {
        for (uint d = 0; d < head_dim; d++) {
            k_cache[layer_offset + token_offset + (head_idx * head_dim) + d] = k_ptr[(head_idx * head_dim) + d];
            v_cache[layer_offset + token_offset + (head_idx * head_dim) + d] = v_ptr[(head_idx * head_dim) + d];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    device const half* q_head = q_ptr + (head_idx * head_dim);
    uint kv_head_idx = head_idx / kv_heads_per_q;
    float scale = 1.0f / sqrt(128.0f);
    
    for (uint t = 0; t <= current_pos; t++) {
        float sum = 0.0f;
        device half* k_cached = k_cache + layer_offset + (t * 256) + (kv_head_idx * head_dim);
        for (uint d = 0; d < head_dim; d++) sum += (float)q_head[d] * (float)k_cached[d];
        scores_out[(head_idx * max_seq_len) + t] = sum * scale;
    }
}