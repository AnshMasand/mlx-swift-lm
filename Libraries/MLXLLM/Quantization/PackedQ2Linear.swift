//
//  PackedQ2Linear.swift
//  mlx-swift-lm (bloom-q2-fork)
//
//  Q2-packed linear layer used in place of MLX's standard `Linear` for the
//  FFN block in Bloom's SmolLM3-3B. Holds packed Q2 weights (16 bytes per
//  64-weight group + fp16 scale + fp16 zero) plus an optional fp16 outScale
//  vector for AWQ down_proj scaling. Forward pass dispatches the
//  `BloomQ2Kernel` Metal kernel.
//
//  Storage keys (matched to `tools/quantize/bloom_q2/packing.py`):
//    q2_packed: uint8, shape [out_features, in_features / 4]
//    q2_scales: fp16,  shape [out_features, in_features / 64]
//    q2_zeros:  fp16,  shape [out_features, in_features / 64]
//
//  Underscores (vs. dots) are intentional: dotted keys would be parsed by
//  MLX-Swift's `ModuleParameters.unflattened()` as sub-module path
//  components, forcing a nested storage Module. Underscores keep them as
//  flat property names on this Module.
//
//  outScale is loaded post-hoc from `bloom-q2.config.json` via
//  `PackedQ2Loader.injectOutScales(...)` and is NOT a `@ParameterInfo` —
//  it isn't in the safetensors.

import Foundation
import MLX
import MLXNN

public final class PackedQ2Linear: Module, UnaryLayer {
    @ParameterInfo(key: "q2_packed") public var q2Packed: MLXArray
    @ParameterInfo(key: "q2_scales") public var q2Scales: MLXArray
    @ParameterInfo(key: "q2_zeros")  public var q2Zeros:  MLXArray

    /// AWQ outScale for down_proj. Set after weights load by
    /// `PackedQ2Loader.injectOutScales(...)`. `nil` for up_proj / gate_proj
    /// (their AWQ scale is folded into the preceding RMSNorm).
    private var _outScale: MLXArray?

    public let inFeatures: Int
    public let outFeatures: Int

    public init(inFeatures: Int, outFeatures: Int) {
        precondition(
            inFeatures % 64 == 0,
            "PackedQ2Linear: inFeatures (\(inFeatures)) must be a multiple of group size 64"
        )
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures

        let groupsPerRow = inFeatures / 64
        self._q2Packed = ParameterInfo(
            wrappedValue: MLXArray.zeros([outFeatures, inFeatures / 4], dtype: .uint8),
            key: "q2_packed"
        )
        self._q2Scales = ParameterInfo(
            wrappedValue: MLXArray.ones([outFeatures, groupsPerRow], dtype: .float16),
            key: "q2_scales"
        )
        self._q2Zeros = ParameterInfo(
            wrappedValue: MLXArray.zeros([outFeatures, groupsPerRow], dtype: .float16),
            key: "q2_zeros"
        )
        super.init()
    }

    /// Inject AWQ outScale (per-row, fp16). Pass `nil` to clear.
    public func setOutScale(_ outScale: MLXArray?) {
        self._outScale = outScale
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return BloomQ2Kernel.matmul(
            x: x,
            packed: q2Packed,
            scales: q2Scales,
            zeros: q2Zeros,
            outScale: _outScale,
            outFeatures: outFeatures
        )
    }
}
