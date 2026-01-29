#include <metal_stdlib> // error expected if not in xcode
using namespace metal;

#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float 
#define uchar unsigned char
#define thread_position_in_grid 0
#endif

// Because we are using Grouped-Query Attention (GQA),
// the mapping here is the inverse of what we did for the scores.
kernel void attn_weighted_sum(
    device const float* scores [[buffer(0)]],
    device const half* v_cache [[buffer(1)]], 
    device half* output [[buffer(2)]], 
    constant uint& layer_idx [[buffer(3)]],
    constant uint& current_pos [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) 
{
    const uint max_seq_len = 1024;
    uint q_head = tid / 128;
    uint head_dim_idx = tid % 128;
    uint v_head = q_head / 6; 
    uint layer_offset = layer_idx * max_seq_len * 256;

    float weighted_sum = 0.0f;
    for (uint i = 0; i <= current_pos; i++) {
        float attn_weight = scores[q_head * max_seq_len + i];
        weighted_sum += attn_weight * (float)v_cache[layer_offset + (i * 256) + (v_head * 128) + head_dim_idx];
    }
    output[tid] = (half)weighted_sum;
}