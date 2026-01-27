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

// Optimized GEMV for 4-bit block-quantized weights
kernel void gemv_q4_0(
    device const uchar* packed_weights [[buffer(0)]], // [Rows * 768 bytes]
    device const half* scales          [[buffer(1)]], // [Rows * 48 scales]
    device const half* input_vector    [[buffer(2)]], // [1536 elements]
    device half* output_vector         [[buffer(3)]], // [Rows elements]
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) 
{
    // 1. Point to the start of THIS row
    // Each row has 1536 weights -> 768 bytes
    device const uchar* row_weights = packed_weights + (row * 768);
    // 1536 weights / 32 block size = 48 scales per row
    device const half* row_scales = scales + (row * 48);

    // 2. Each thread calculates partial sum for 2 weights (1 byte)
    // We assume 768 threads per threadgroup
    half thread_sum = 0.0h;
    if (tid < 768) {
        uchar packed = row_weights[tid];
        half w_low  = (half)((packed & 0x0f) - 8);
        half w_high = (half)((packed >> 4) - 8);
        
        half scale = row_scales[tid / 16]; // 16 threads = 32 weights = 1 block
        
        thread_sum = (w_low * scale * input_vector[tid * 2]) + 
                     (w_high * scale * input_vector[tid * 2 + 1]);
    }

    // 3. Sigma Move: SIMD Reduction
    // sum the values across all threads in the SIMD-group (usually 32 threads)
    thread_sum = simd_sum(thread_sum);

    // 4. Threadgroup Reduction (Final Sum for the Row)
    threadgroup half shared_sums[32]; // 768 threads / 32 = 24 SIMD groups
    uint simd_id = tid / 32;
    uint lane_id = tid % 32;

    if (lane_id == 0) {
        shared_sums[simd_id] = thread_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final thread in the group writes the result to global memory
    if (tid == 0) {
        half final_row_sum = 0.0h;
        for (uint i = 0; i < 24; i++) { // 24 = 768 / 32
            final_row_sum += shared_sums[i];
        }
        output_vector[row] = final_row_sum;
    }
}