#include <metal_stdlib> // error expected if not in xcode
using namespace metal;

#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float 
#define uchar unsigned char
#define thread_position_in_grid 0
#endif

// Residual Add: x = x + Attention(x)
kernel void residual_add(
    device const half* x [[buffer(0)]],
    device const half* attn_x [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    output[tid] = x[tid] + attn_x[tid];
}