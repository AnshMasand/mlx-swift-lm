//
//  BloomQ2Kernel.swift
//  mlx-swift-lm (bloom-q2-fork)
//
//  Swift dispatch wrapper for the Bloom Q2 dequant + matvec Metal kernel.
//
//  Mirrors the MLXFast.metalKernel pattern used in `Bitnet.swift` (BitLinear).
//  The Metal source lives alongside this file in `BloomQ2Kernel.metal` — it is
//  embedded verbatim into the `kernelSource` string below at file-scope so that
//  there is no runtime file I/O.
//
//  Phase 3a scope:
//    - Decode (matvec) kernel only. v1: no threadgroup-shared x.
//    - Prefill (B>1) falls back to a per-row matvec loop. v2 (separate phase)
//      will introduce a real GEMM kernel if perf testing demands it.
//
//  See `docs/superpowers/specs/2026-05-07-q2-mixed-metal-dequant-design.md` §6
//  for the full design.

import Foundation
import MLX

// MARK: - Kernel source

/// Kernel body. MLXFast.metalKernel auto-generates the function signature from
/// `inputNames` / `outputNames`. Template constants (T, HAS_OUT_SCALE,
/// IN_FEATURES, OUT_FEATURES) are JIT-specialized via the `template:` arg list.
///
/// Keep this string in sync with `BloomQ2Kernel.metal`.
private let kernelSource: String = """
constexpr uint BLOCK = 32;
constexpr uint GS    = 64;        // group size — must match packing.py
constexpr uint BPG   = GS / 4;    // 16 packed bytes per group

uint tid       = thread_position_in_grid.x;        // 0..BLOCK-1 within row
uint flat_idx  = thread_position_in_grid.y;        // 0..(B*OUT-1)
uint batch_idx = flat_idx / OUT_FEATURES;
uint row_idx   = flat_idx % OUT_FEATURES;

uint groups_per_row = IN_FEATURES / GS;

const uint packed_row_off = row_idx * (IN_FEATURES / 4);
const uint scale_row_off  = row_idx * groups_per_row;

float acc = 0.0f;

for (uint g = tid; g < groups_per_row; g += BLOCK) {
    float scale_f = (float)scales[scale_row_off + g];
    float zero_f  = (float)zeros [scale_row_off + g];
    float lut0 = scale_f * (0.0f - zero_f);
    float lut1 = scale_f * (1.0f - zero_f);
    float lut2 = scale_f * (2.0f - zero_f);
    float lut3 = scale_f * (3.0f - zero_f);

    uint packed_grp_off = packed_row_off + g * BPG;
    uint x_grp_off      = batch_idx * IN_FEATURES + g * GS;

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

acc = simd_sum(acc);

if (tid == 0) {
    float final_v = HAS_OUT_SCALE ? (acc * (float)out_scale[row_idx]) : acc;
    y[batch_idx * OUT_FEATURES + row_idx] = (T)final_v;
}
"""

// MARK: - Singleton kernel manager

/// Registers the kernel exactly once. MLX JIT-compiles per unique
/// (template-arg) tuple internally, so a single `MLXFastKernel` instance is
/// reused across every Q2 layer in the model.
private final class BloomQ2KernelManager: @unchecked Sendable {
    static let shared = BloomQ2KernelManager()

    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "bloom_q2_matvec",
            inputNames: ["x", "packed", "scales", "zeros", "out_scale"],
            outputNames: ["y"],
            source: kernelSource
        )
    }
}

// MARK: - Public dispatch entry point

public enum BloomQ2Kernel {

    /// Run a Q2-dequant matvec/matmul.
    ///
    /// - Parameters:
    ///   - x:           activations, half. Shape `[..., inFeatures]`.
    ///   - packed:      uint8 packed weights, shape `[outFeatures, inFeatures/4]`.
    ///   - scales:      fp16 per-group scales, shape `[outFeatures, inFeatures/64]`.
    ///   - zeros:       fp16 per-group zeros,  shape `[outFeatures, inFeatures/64]`.
    ///   - outScale:    optional fp16 per-row AWQ outScale, shape `[outFeatures]`.
    ///                  Pass `nil` for layers without AWQ outScale (up_proj, gate_proj).
    ///   - outFeatures: number of output rows.
    /// - Returns: half tensor, shape `[..., outFeatures]`.
    public static func matmul(
        x: MLXArray,
        packed: MLXArray,
        scales: MLXArray,
        zeros: MLXArray,
        outScale: MLXArray?,
        outFeatures: Int
    ) -> MLXArray {
        precondition(!x.shape.isEmpty, "x must have at least one dimension")
        let inFeatures = x.shape.last!
        precondition(inFeatures % 64 == 0, "inFeatures (\(inFeatures)) must be a multiple of 64")

        // Flatten leading dims into a single batch axis so the kernel sees [B, IN].
        let originalShape = x.shape
        let leading = originalShape.dropLast()
        let batch = leading.reduce(1, *)
        let xFlat: MLXArray
        if originalShape.count == 1 {
            xFlat = x.reshaped([1, inFeatures])
        } else if originalShape.count == 2 {
            xFlat = x
        } else {
            xFlat = x.reshaped([batch, inFeatures])
        }

        // Out_scale buffer: real one if provided, else a length-1 dummy that the
        // kernel never reads (HAS_OUT_SCALE template constant gates the access).
        let outScaleBuffer: MLXArray
        let hasOutScale: Bool
        if let outScale {
            precondition(outScale.shape == [outFeatures],
                         "outScale shape \(outScale.shape) must equal [\(outFeatures)]")
            outScaleBuffer = outScale
            hasOutScale = true
        } else {
            outScaleBuffer = MLXArray.zeros([1], dtype: .float16)
            hasOutScale = false
        }

        let dtype = xFlat.dtype  // expect .float16

        let outputs = BloomQ2KernelManager.shared.kernel(
            [xFlat, packed, scales, zeros, outScaleBuffer],
            template: [
                ("T", dtype),
                ("HAS_OUT_SCALE", hasOutScale),
                ("IN_FEATURES", inFeatures),
                ("OUT_FEATURES", outFeatures),
            ],
            grid: (32, batch * outFeatures, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[batch, outFeatures]],
            outputDTypes: [dtype]
        )

        let yFlat = outputs[0]

        // Restore original leading dims.
        if originalShape.count == 1 {
            return yFlat.reshaped([outFeatures])
        } else if originalShape.count == 2 {
            return yFlat
        } else {
            return yFlat.reshaped(Array(leading) + [outFeatures])
        }
    }
}
