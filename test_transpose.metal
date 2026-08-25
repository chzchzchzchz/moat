#include <metal_stdlib>
using namespace metal;
kernel void test(device half* out [[buffer(0)]]) {
    simdgroup_matrix<half, 8, 8> m;
    m = transpose(m);
}
