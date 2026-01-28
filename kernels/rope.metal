#include <metal_stdlib> // error expected if not in xcode
using namespace metal;

#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float 
#define uchar unsigned char
#define thread_position_in_grid 0
#endif

// RoPE for 4-bit quantized weights
kernel void apply_rope_q4(
    device half* vec [[buffer(0)]], // Input vector (1536)
    uint tid [[thread_position_in_grid]]
){
    // Each thread handles ONE PAIR of elements (e.g., 0 and 1, 2 and 3)
    // For a head_dim of 128, we have 64 pairs per head.
    uint idx = tid * 2;
    
    // Position m (currently assuming we are at the first token: m=0)
    // We will pass 'm' as a constant once we start generating multiple tokens.
    float m = 0.0f; 
    
    // Frequency calculation for this specific pair
    // Qwen uses base 10000.0 or 1000000.0 depending on context window
    // We will use the smaller value
    float theta_base = 10000.0f;
    float head_dim = 128.0f;
    float theta = m * pow(theta_base, -((float)(idx % 128) / head_dim));

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float x1 = (float)vec[idx];
    float x2 = (float)vec[idx + 1];

    // Apply the rotation matrix
    vec[idx]     = (half)(x1 * cos_theta - x2 * sin_theta);
    vec[idx + 1] = (half)(x1 * sin_theta + x2 * cos_theta);
}