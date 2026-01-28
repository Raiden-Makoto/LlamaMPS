#include <metal_stdlib> // error expected if not in xcode
using namespace metal;

#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float 
#define uchar unsigned char
#define thread_position_in_grid 0
#endif

// SWIGLU for 4-bit quantized weights
kernel void swiglu(
    device const half* gate_vec [[buffer(0)]],
    device const half* up_vec [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
){
    float x = (float)gate_vec[tid];
    float y = (float) up_vec[tid];
    float silu = x / (1.0f + exp(-x));
    output[tid] = (half)(silu * y);
}