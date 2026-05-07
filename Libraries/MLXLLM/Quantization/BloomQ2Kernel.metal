//
//  BloomQ2Kernel.metal
//  mlx-swift-lm (bloom-q2-fork)
//
//  Q2 mixed-precision dequant + matvec kernel.
//
//  This file contains ONLY the kernel body. MLXFast.metalKernel auto-generates
//  the function signature from inputNames/outputNames passed in the Swift
//  wrapper, plus implicit shape/stride helpers (e.g. x_shape, x_strides).
//
//  Inputs (declared in Swift wrapper):
//    x        : half [B, IN]                — activations (row-major)
//    packed   : uchar [OUT, IN/4]           — packed 2-bit weights, 4 weights/byte LSB-first
//    scales   : half  [OUT, IN/64]          — per-group affine scale
//    zeros    : half  [OUT, IN/64]          — per-group affine zero
//    out_scale: half  [OUT] (or [1] dummy)  — AWQ per-row outScale, or all-ones placeholder
//
//  Output:
//    y        : half [B, OUT]
//
//  Template constants (passed via `template:`):
//    T               : output dtype (half)              — DType
//    HAS_OUT_SCALE   : whether to apply out_scale       — Bool
//    IN_FEATURES     : input feature count              — Int
//    OUT_FEATURES    : output feature count             — Int
//
//  Group size is fixed at 64 (matches tools/quantize/bloom_q2/packing.py).
//  Dispatch: grid = (BLOCK=32, B * OUT, 1), threadGroup = (32, 1, 1).
//  Each threadgroup computes ONE output element; 32 threads share-reduce via simd_sum.

constexpr uint BLOCK = 32;
constexpr uint GS    = 64;        // group size — must match packing.py
constexpr uint BPG   = GS / 4;    // 16 packed bytes per group

uint tid       = thread_position_in_grid.x;        // 0..BLOCK-1 within row
uint flat_idx  = thread_position_in_grid.y;        // 0..(B*OUT-1)
uint batch_idx = flat_idx / OUT_FEATURES;
uint row_idx   = flat_idx % OUT_FEATURES;

uint groups_per_row = IN_FEATURES / GS;

// One thread per stripe of groups: each thread walks group g where (g % BLOCK == tid).
// Per-row pointers for the weight blob.
const uint packed_row_off = row_idx * (IN_FEATURES / 4);
const uint scale_row_off  = row_idx * groups_per_row;

float acc = 0.0f;

for (uint g = tid; g < groups_per_row; g += BLOCK) {
    // Load per-group affine params into a 4-entry LUT: lut[q] = scale*(q-zero), q ∈ {0..3}.
    float scale_f = (float)scales[scale_row_off + g];
    float zero_f  = (float)zeros [scale_row_off + g];
    float lut0 = scale_f * (0.0f - zero_f);
    float lut1 = scale_f * (1.0f - zero_f);
    float lut2 = scale_f * (2.0f - zero_f);
    float lut3 = scale_f * (3.0f - zero_f);

    uint packed_grp_off = packed_row_off + g * BPG;
    uint x_grp_off      = batch_idx * IN_FEATURES + g * GS;

    // 16 packed bytes per group, 4 weights per byte (LSB-first slot order).
    for (uint b = 0; b < BPG; ++b) {
        uchar pb = packed[packed_grp_off + b];
        uint  xi = x_grp_off + b * 4;
        float x0 = (float)x[xi + 0];
        float x1 = (float)x[xi + 1];
        float x2 = (float)x[xi + 2];
        float x3 = (float)x[xi + 3];
        uint q0 = (uint)( pb       & 0x3u);
        uint q1 = (uint)((pb >> 2) & 0x3u);
        uint q2 = (uint)((pb >> 4) & 0x3u);
        uint q3 = (uint)((pb >> 6) & 0x3u);
        float w0 = (q0 == 0) ? lut0 : (q0 == 1 ? lut1 : (q0 == 2 ? lut2 : lut3));
        float w1 = (q1 == 0) ? lut0 : (q1 == 1 ? lut1 : (q1 == 2 ? lut2 : lut3));
        float w2 = (q2 == 0) ? lut0 : (q2 == 1 ? lut1 : (q2 == 2 ? lut2 : lut3));
        float w3 = (q3 == 0) ? lut0 : (q3 == 1 ? lut1 : (q3 == 2 ? lut2 : lut3));
        acc = fma(w0, x0, acc);
        acc = fma(w1, x1, acc);
        acc = fma(w2, x2, acc);
        acc = fma(w3, x3, acc);
    }
}

// SIMD-group reduction: collapses partial sums across the 32 threads.
acc = simd_sum(acc);

if (tid == 0) {
    float final_v = HAS_OUT_SCALE ? (acc * (float)out_scale[row_idx]) : acc;
    y[batch_idx * OUT_FEATURES + row_idx] = (T)final_v;
}
