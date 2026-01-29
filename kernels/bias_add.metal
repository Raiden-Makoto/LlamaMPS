#include <metal_stdlib> // error expected if not in xcode
using namespace metal;

#ifndef __METAL_VERSION__
#define kernel
#define device
#define half float
#define thread_position_in_grid 0
#endif

// Elementwise add: out = x + bias
kernel void bias_add(
    device const half* x [[buffer(0)]],
    device const half* bias [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
){
    float a = (float)x[tid];
    float b = (float)bias[tid];
    out[tid] = (half)(a + b);
}

