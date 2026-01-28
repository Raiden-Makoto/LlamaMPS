#include <metal_stdlib> // error expected if not in xcode
using namespace metal;

// --- IDE Support Header (Doesn't affect GPU) ---
#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float 
#define uchar unsigned char
#define thread_position_in_grid 0
#endif

// RMS Norm for 4-bit quantized weights
kernel void rms_norm_q4(
    device const half* input  [[buffer(0)]], // Input vector (1536)
    device const half* weight [[buffer(1)]], // Norm weights (1536)
    device half* output       [[buffer(2)]], // Normalized output (1536)
    uint tid [[thread_position_in_grid]]
){
    float val = (float) input[tid];
    float sq_val = val * val;
    float sum_sq = simd_sum(sq_val);
    threadgroup float shared_sums[32];

    if ((tid % 32) == 0) {
        shared_sums[tid / 32] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float final_scale;
    if (tid == 0){
        float total_sum_sq = 0;
        for (int i = 0; i < 48; i++) total_sum_sq += shared_sums[i];
        // RMS Formula: 1 / sqrt(mean(x^2) + epsilon)
        // Qwen 2.5 uses 1e-6 as epsilon
        final_scale = rsqrt(total_sum_sq / 1536.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    output[tid] = (half)(val * final_scale) * weight[tid];
}