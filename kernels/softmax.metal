#include <metal_stdlib> // error expected if not in xcode
using namespace metal;

#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float 
#define uchar unsigned char
#define thread_position_in_grid 0
#endif

// softmax for attention scores
// using online softmax for efficiency
kernel void softmax(
    device float* scores [[buffer(0)]],
    constant uint& current_seq_len [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
){
    uint head_offset = tid * current_seq_len;
    device float* head_scores = scores + head_offset;
    // Find Max for Numerical Stability
    float max_val = -INFINITY;
    for (uint i = 0; i < current_seq_len; i++) {
        if (head_scores[i] > max_val) max_val = head_scores[i];
    }
    // Sum of Exponentials
    float sum = 0.0f;
    for (uint i = 0; i < current_seq_len; i++) {
        head_scores[i] = exp(head_scores[i] - max_val);
        sum += head_scores[i];
    }
    // Normalize
    for (uint i = 0; i < current_seq_len; i++) {
        head_scores[i] /= sum;
    }
}