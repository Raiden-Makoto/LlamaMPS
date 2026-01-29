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
    float m = (float)current_pos; 
    
    // Qwen 2.5 uses base 1000000.0 for 1.5B model to support long context
    float theta_base = 1000000.0f;
    float head_dim = 128.0f;
    
    // Qwen2 uses the "rotate_half" RoPE layout (first half with second half),
    // not adjacent even/odd pairs. For each head, rotate dims:
    //   (x[i], x[i + head_dim/2])  for i in [0, head_dim/2)
    //
    // Dispatch is expected as:
    // - Q: (HIDDEN_DIM/2) threads == (num_heads * head_dim/2)
    // - K: (K_DIM/2) threads       == (num_kv_heads * head_dim/2)
    uint pair = tid % 64;          // i in [0, 63]
    uint head = tid / 64;          // head index
    uint base = head * 128;        // head start (in elements)
    uint idx0 = base + pair;       // first-half index
    uint idx1 = idx0 + 64;         // second-half index

    // Calculate frequency based on the pair index (inv_freq exponent = -2*i/head_dim)
    float theta = m * pow(theta_base, -((float)(pair * 2) / head_dim));

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float x0 = (float)vec[idx0];
    float x1 = (float)vec[idx1];

    // Apply rotation
    vec[idx0] = (half)(x0 * cos_theta - x1 * sin_theta);
    vec[idx1] = (half)(x1 * cos_theta + x0 * sin_theta);
}
