//
//  PackedQ2Loader.swift
//  mlx-swift-lm (bloom-q2-fork)
//
//  Companion loader for `PackedQ2Linear`. Reads `bloom-q2.config.json` and
//  injects the AWQ outScale vector into each `down_proj` PackedQ2Linear after
//  the model's safetensors weights have been applied via
//  `Module.update(parameters:)`.
//
//  outScales are NOT stored in the safetensors because they're not really
//  "weights" — they're per-channel multipliers from AWQ that the kernel
//  applies inline. Carrying them out-of-band keeps the safetensors layout
//  symmetric across q2/q4 layers and means stock MLX-Swift loading code can
//  produce a consistent `ModuleParameters` dict without special-casing.
//
//  Usage:
//
//    let weights = try ModuleParameters.unflattened(loadArrays(url: tensorsURL))
//    model.update(parameters: weights)
//    let q2Config = try PackedQ2Loader.loadConfig(from: configURL)
//    PackedQ2Loader.injectOutScales(into: model, config: q2Config)

import Foundation
import MLX
import MLXNN

public struct PackedQ2Config: Decodable, Sendable {
    public let format: String
    public let version: Int
    public let groupSize: Int
    /// path → outScale vector. Path is the dotted module path for the
    /// down_proj layer, e.g. "model.layers.7.mlp.down_proj".
    public let outScales: [String: [Float]]
    public let q2Layers: [String]
    public let q4Layers: [String]

    enum CodingKeys: String, CodingKey {
        case format
        case version
        case groupSize = "group_size"
        case outScales = "out_scales"
        case q2Layers = "q2_layers"
        case q4Layers = "q4_layers"
    }
}

public enum PackedQ2Loader {
    /// Load `bloom-q2.config.json` from a file URL.
    public static func loadConfig(from url: URL) throws -> PackedQ2Config {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PackedQ2Config.self, from: data)
    }

    /// Walk the model's module tree and inject AWQ outScale vectors into the
    /// `PackedQ2Linear` instances at the dotted paths in `config.outScales`.
    /// Call this AFTER `model.update(parameters:)`.
    ///
    /// Layers whose path doesn't resolve to a `PackedQ2Linear` are skipped
    /// silently — this is defensive against config/model schema drift; the
    /// downstream code (model loader) is the one that validates the model
    /// architecture matches the safetensors.
    public static func injectOutScales(into model: Module, config: PackedQ2Config) {
        // Build a path → Module index once. `namedModules()` returns
        // every Module with its dotted path, including intermediate ones.
        // `uniquingKeysWith` collapses any duplicates (the root path is
        // typically empty and emitted once for the model itself; we keep
        // whichever wins — we never look it up).
        let index = Dictionary(
            model.namedModules(),
            uniquingKeysWith: { first, _ in first }
        )

        for (path, vector) in config.outScales {
            guard let layer = index[path] as? PackedQ2Linear else { continue }
            let arr = MLXArray(vector).asType(.float16)
            layer.setOutScale(arr)
        }
    }
}
