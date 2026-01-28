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
    device const float* scores       [[buffer(0)]], // [12 heads * seq_len]
    device const half* v_cache       [[buffer(1)]], // [seq_len * 256]
    device half* output              [[buffer(2)]], // [1536] (Final Attention Out)
    constant uint& current_seq_len   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) // tid = 0 to 1535 (Total Output Dim)
{
    uint q_head = tid / 128; // Which of the 12 heads we are in
    uint head_dim_idx = tid % 128; // Which dimension within that head (0-127)
    // GQA Mapping: 12 Q heads map to 2 V heads
    uint v_head = q_head / 6; 
    float weighted_sum = 0.0f;
    // For every token in the sequence (currently just 1)
    for (uint i = 0; i < current_seq_len; i++) {
        float attn_weight = scores[q_head * current_seq_len + i];
        // Point to the correct Value head for this token
        // Row = i, Head = v_head, Element = head_dim_idx
        device const half* v_ptr = v_cache + (i * 256) + (v_head * 128) + head_dim_idx;
        weighted_sum += attn_weight * (float)(*v_ptr);
    }
    output[tid] = (half)weighted_sum;
}