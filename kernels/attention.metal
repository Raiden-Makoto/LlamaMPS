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
    device const half* q_vec         [[buffer(0)]], // [1536]
    device const half* k_cache       [[buffer(1)]], // [MaxSeq * 256]
    device float* scores             [[buffer(2)]], // [12 heads * MaxSeq]
    device const uint* seq_len       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) // tid = query_head_index
{
    // 1. Identify which K head to use (GQA logic)
    // 12 Q heads / 2 K heads = 6 Q heads per K head
    uint q_head = tid;
    uint k_head = q_head / 6; 
    
    // 2. Head Offsets (each head is 128 dim)
    device const half* q_ptr = q_vec + (q_head * 128);
    
    // 3. Dot Product Loop
    // For now, we only have 1 token in the sequence (current_seq_len = 1)
    // When we do actual inference this will be much mofr advanced
    uint current_seq_len = seq_len[0];
    for (uint i = 0; i < current_seq_len; i++) {
        device const half* k_ptr = k_cache + (i * 256) + (k_head * 128);
        
        float dot = 0.0f;
        for (uint d = 0; d < 128; d++) {
            dot += (float)q_ptr[d] * (float)k_ptr[d];
        }
        
        scores[q_head * current_seq_len + i] = dot * 0.088388f; 
    }
}