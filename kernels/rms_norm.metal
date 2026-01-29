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
    device const half* input  [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output       [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
){
    // Each thread handles 2 elements. Engine should dispatch a single threadgroup
    // of size (HIDDEN_DIM/2) so the reduction is complete.
    uint idx0 = tid * 2;
    uint idx1 = idx0 + 1;

    float v0 = (float)input[idx0];
    float v1 = (float)input[idx1];
    float sum_sq = simd_sum(v0 * v0 + v1 * v1);

    // Compute SIMD-group count dynamically from threadgroup size.
    uint threads = tptg.x;
    uint simd_groups = threads / 32;
    uint lane_id = tid % 32;
    uint simd_id = tid / 32;

    threadgroup float shared_sums[32];
    if (lane_id == 0) {
        shared_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float final_scale;
    if (tid == 0){
        float total_sum_sq = 0.0f;
        for (uint i = 0; i < simd_groups; i++) total_sum_sq += shared_sums[i];
        final_scale = rsqrt(total_sum_sq / 1536.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float w0 = (float)weight[idx0];
    float w1 = (float)weight[idx1];
    output[idx0] = (half)(v0 * final_scale * w0);
    output[idx1] = (half)(v1 * final_scale * w1);
}