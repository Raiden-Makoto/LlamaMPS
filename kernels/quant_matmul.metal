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

// GEMV for 4-bit block-quantized weights. Constants passed via buffer(4): [K, simd_groups, threads].
// K = columns (1536 or 8960). simd_groups = threads/32. threads = threads_per_threadgroup (e.g. 768).
kernel void gemv_q4_0(
    device const uchar* packed_weights [[buffer(0)]], // [Rows * (K/2) bytes]
    device const half* scales          [[buffer(1)]], // [Rows * (K/32) scales]
    device const half* input_vector   [[buffer(2)]], // [K elements]
    device half* output_vector         [[buffer(3)]], // [Rows elements]
    device const uint* constants      [[buffer(4)]], // [K, simd_groups, threads]
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    uint k = constants[0];
    uint simd_groups = constants[1];
    uint threads = constants[2];
    uint bytes_per_row = k / 2;
    uint scales_per_row = k / 32;

    device const uchar* row_weights = packed_weights + (row * bytes_per_row);
    device const half* row_scales = scales + (row * scales_per_row);

    uint stride = (bytes_per_row + threads - 1) / threads;
    float thread_sum = 0.0f;
    for (uint i = 0; i < stride; i++) {
        uint byte_idx = tid + i * threads;
        if (byte_idx >= bytes_per_row) break;

        uchar packed = row_weights[byte_idx];
        float w_low  = (float)((packed & 0x0f) - 8);
        float w_high = (float)((packed >> 4) - 8);

        uint block_idx = byte_idx / 16;
        float scale = (float)row_scales[block_idx];

        float x0 = (float)input_vector[byte_idx * 2];
        float x1 = (float)input_vector[byte_idx * 2 + 1];
        thread_sum += (w_low * scale * x0) + (w_high * scale * x1);
    }

    thread_sum = simd_sum(thread_sum);

    threadgroup float shared_sums[32];
    uint simd_id = tid / 32;
    uint lane_id = tid % 32;

    if (lane_id == 0) {
        shared_sums[simd_id] = thread_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float final_row_sum = 0.0f;
        for (uint i = 0; i < simd_groups; i++) {
            final_row_sum += shared_sums[i];
        }
        output_vector[row] = (half)final_row_sum;
    }
}
