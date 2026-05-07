// Copyright © 2026 Mitwa.
//
// One-off sanity check for the Bloom Q2 dequant + matvec Metal kernel.
//
// This is INTENTIONALLY a single comparison vs a pure-Swift CPU reference,
// not a property test suite. Per user vibes-test directive: bit-packing bugs
// are otherwise invisible from chat output, so we want exactly enough
// confidence that the kernel computes the right thing once.
//
// Run: swift test --filter BloomQ2KernelSanity

import Foundation
import MLX
import XCTest

@testable import MLXLLM

public class BloomQ2KernelSanityTest: XCTestCase {

    /// Pure-Swift CPU reference. Mirrors `dequant_q2_matrix` in
    /// `tools/quantize/bloom_q2/packing.py` (LSB-first 2-bit slot order, affine
    /// dequant `w = scale * (q - zero)` per group of 64).
    private func referenceMatvec(
        x: [Float],
        packed: [UInt8],            // [out, in/4] row-major
        scales: [Float],            // [out, in/64]
        zeros: [Float],              // [out, in/64]
        outScale: [Float]?,           // nil or [out]
        outFeatures: Int,
        inFeatures: Int
    ) -> [Float] {
        let groupSize = 64
        precondition(inFeatures % groupSize == 0)
        let groupsPerRow = inFeatures / groupSize
        let bytesPerRow = inFeatures / 4

        var y = [Float](repeating: 0, count: outFeatures)
        for r in 0 ..< outFeatures {
            var acc: Float = 0
            for g in 0 ..< groupsPerRow {
                let scale = scales[r * groupsPerRow + g]
                let zero = zeros[r * groupsPerRow + g]
                let lut: [Float] = (0 ..< 4).map { scale * (Float($0) - zero) }
                for b in 0 ..< 16 {
                    let pb = packed[r * bytesPerRow + g * 16 + b]
                    let xi = g * 64 + b * 4
                    for slot in 0 ..< 4 {
                        let q = Int((pb >> (slot * 2)) & 0x3)
                        acc += lut[q] * x[xi + slot]
                    }
                }
            }
            if let outScale {
                acc *= outScale[r]
            }
            y[r] = acc
        }
        return y
    }

    /// Compare BloomQ2Kernel.matmul to the Swift CPU reference on synthetic
    /// random Q2 inputs. Asserts max abs error < 1e-2 (loose; small matrices,
    /// half-precision reduction). Prints max error so we can see how close we
    /// got.
    func testMatchesCPUReference() {
        // Small but non-trivial sizes. in=128 → 2 groups; out=64 → 32×64 batches
        // worth of dispatch coverage. Includes outScale path.
        let inFeatures = 128
        let outFeatures = 64
        let groupSize = 64
        let groupsPerRow = inFeatures / groupSize        // 2
        let bytesPerRow = inFeatures / 4                  // 32

        // Deterministic synthetic data.
        var rng = SplitMix64(seed: 0xB100_0042)

        let xCount = inFeatures
        let xFloats: [Float] = (0 ..< xCount).map { _ in rng.nextFloat(low: -1, high: 1) }

        let packedCount = outFeatures * bytesPerRow
        let packedBytes: [UInt8] = (0 ..< packedCount).map { _ in UInt8(rng.nextU64() & 0xFF) }

        let scaleCount = outFeatures * groupsPerRow
        // Modest scales so half-precision accumulation stays well-behaved.
        let scaleFloats: [Float] = (0 ..< scaleCount).map { _ in rng.nextFloat(low: 0.01, high: 0.1) }
        let zeroFloats: [Float] = (0 ..< scaleCount).map { _ in rng.nextFloat(low: 0.5, high: 2.5) }

        let outScaleFloats: [Float] = (0 ..< outFeatures).map { _ in rng.nextFloat(low: 0.5, high: 1.5) }

        // CPU reference (with outScale path exercised).
        let yRef = referenceMatvec(
            x: xFloats,
            packed: packedBytes,
            scales: scaleFloats,
            zeros: zeroFloats,
            outScale: outScaleFloats,
            outFeatures: outFeatures,
            inFeatures: inFeatures
        )

        // Build MLXArrays. half (fp16) for activations + scales + zeros + outScale,
        // uint8 for packed weights. Mirrors PackedQ2Linear's planned wire format.
        let xHalf = MLXArray(xFloats).asType(.float16).reshaped([1, inFeatures])
        let packedArr = MLXArray(packedBytes).reshaped([outFeatures, bytesPerRow])
        let scalesArr = MLXArray(scaleFloats).asType(.float16).reshaped([outFeatures, groupsPerRow])
        let zerosArr = MLXArray(zeroFloats).asType(.float16).reshaped([outFeatures, groupsPerRow])
        let outScaleArr = MLXArray(outScaleFloats).asType(.float16)

        let yKernel = BloomQ2Kernel.matmul(
            x: xHalf,
            packed: packedArr,
            scales: scalesArr,
            zeros: zerosArr,
            outScale: outScaleArr,
            outFeatures: outFeatures
        )

        // Pull kernel output back to host as fp32 for comparison.
        XCTAssertEqual(yKernel.shape, [1, outFeatures])
        let yKernelFloats = yKernel.asType(.float32).asArray(Float.self)

        // Element-wise compare. Tolerance: 1e-2 absolute. Loose because we
        // accumulate in fp32 inside the kernel but inputs round-trip through
        // half. Synthetic data magnitudes ~|0.01..0.1| × |0..3| × 128 → ~O(1).
        var maxAbsErr: Float = 0
        for i in 0 ..< outFeatures {
            let err = abs(yRef[i] - yKernelFloats[i])
            if err > maxAbsErr { maxAbsErr = err }
        }
        print("[BloomQ2KernelSanity] max abs err = \(maxAbsErr)")
        XCTAssertLessThan(maxAbsErr, 1e-2,
            "Kernel diverged from CPU reference — possible bit-slot order, "
            + "stride, or LUT bug")

        // Also exercise the no-outScale path to make sure HAS_OUT_SCALE=false
        // is reachable and produces sensible output.
        let yNoScaleKernel = BloomQ2Kernel.matmul(
            x: xHalf,
            packed: packedArr,
            scales: scalesArr,
            zeros: zerosArr,
            outScale: nil,
            outFeatures: outFeatures
        )
        let yNoScaleRef = referenceMatvec(
            x: xFloats,
            packed: packedBytes,
            scales: scaleFloats,
            zeros: zeroFloats,
            outScale: nil,
            outFeatures: outFeatures,
            inFeatures: inFeatures
        )
        let yNoScaleFloats = yNoScaleKernel.asType(.float32).asArray(Float.self)
        var maxAbsErrNoScale: Float = 0
        for i in 0 ..< outFeatures {
            let err = abs(yNoScaleRef[i] - yNoScaleFloats[i])
            if err > maxAbsErrNoScale { maxAbsErrNoScale = err }
        }
        print("[BloomQ2KernelSanity] max abs err (no out_scale) = \(maxAbsErrNoScale)")
        XCTAssertLessThan(maxAbsErrNoScale, 1e-2)
    }
}

// MARK: - Tiny seedable RNG (deterministic per seed, no platform-RNG dep)

private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }

    mutating func nextU64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z &>> 31)
    }

    mutating func nextFloat(low: Float, high: Float) -> Float {
        let u = nextU64()
        let unit = Float(u >> 40) / Float(1 << 24)  // 24-bit precision in [0,1)
        return low + (high - low) * unit
    }
}
