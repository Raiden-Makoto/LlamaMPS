#include <metal_stdlib>
using namespace metal;

// --- IDE Support Header (Doesn't affect GPU) ---
#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float 
#define uchar unsigned char
#define thread_position_in_grid 0
#endif

/*
 * KERNEL: dequant_dot_product
 * Logic: Process 2 weights per thread (1 byte)
 * Math: weight = (packed_val - 8) * scale
 */
kernel void dequant_dot_product(
    device const uchar* packed_weights [[buffer(0)]], // 4-bit packed weights
    device const half* scales          [[buffer(1)]], // FP16 block scales
    device const half* input_vector    [[buffer(2)]], // FP16 activations
    device half* partial_sums          [[buffer(3)]], // Results to sum on CPU
    uint tid [[thread_position_in_grid]]) 
{
    // 1. Fetch the packed byte (contains two 4-bit weights)
    uchar packed = packed_weights[tid];
    
    // 2. Bitwise Surgery
    // Extract lower 4 bits (0x0F) and upper 4 bits (shifted >> 4)
    // We subtract 8 because our packer added 8 to make them unsigned
    half w_low  = (half)((packed & 0x0f) - 8); 
    half w_high = (half)((packed >> 4) - 8);
    
    // 3. Block-Scale Lookup
    // Since each thread handles 2 weights, and block size is 32:
    // Every 16 threads (32 weights) share 1 scale.
    half scale = scales[tid / 16]; 
    
    // 4. Dot Product Math
    // Weight 1 matches input[tid * 2], Weight 2 matches input[tid * 2 + 1]
    half val1 = (w_low * scale) * input_vector[tid * 2];
    half val2 = (w_high * scale) * input_vector[tid * 2 + 1];
    
    // 5. Output
    partial_sums[tid] = val1 + val2;
}