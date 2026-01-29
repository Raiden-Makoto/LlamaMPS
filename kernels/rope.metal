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
    device half* vec [[buffer(0)]],
    constant uint& current_pos [[buffer(1)]], // Passed from engine.py
    uint tid [[thread_position_in_grid]]
){
    uint idx = tid * 2;
    float m = (float)current_pos; 
    
    // Qwen 2.5 uses base 1000000.0 for 1.5B model to support long context
    float theta_base = 1000000.0f;
    float head_dim = 128.0f;
    
    // Calculate frequency based on the specific dimension pair
    float theta = m * pow(theta_base, -((float)(idx % 128) / head_dim));

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float x1 = (float)vec[idx];
    float x2 = (float)vec[idx + 1];

    // Apply rotation
    vec[idx]     = (half)(x1 * cos_theta - x2 * sin_theta);
    vec[idx + 1] = (half)(x1 * sin_theta + x2 * cos_theta);
}