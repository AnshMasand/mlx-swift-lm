# Rebasing the bloom-q2-fork on upstream

This fork carries Bloom-specific Q2-quantization changes on top of `ml-explore/mlx-swift-lm`.
The full design and rationale live in the parent repo at
`docs/superpowers/specs/2026-05-07-q2-mixed-metal-dequant-design.md`.

## What we own here

- `Libraries/MLXLLM/Quantization/` — entire directory is ours
  - `PackedQ2Linear.swift` — Module subclass for Q2 FFN layers
  - `PackedQ2Loader.swift` — safetensors loader extension that recognizes our format
  - `BloomQ2Kernel.swift` — Swift wrapper that dispatches the Metal kernel
  - `BloomQ2Kernel.metal` — Metal source for the dequant + matvec kernel
- `Libraries/MLXLLM/Models/SmolLM3.swift` — small patch in the MLP block
  (`Linear` → `PackedQ2Linear` for `up_proj`, `gate_proj`, `down_proj`)

We do **not** modify any other model file or any of MLX-Swift's framework code.

## Recipe to rebase on upstream

```bash
cd Local/mlx-swift-lm

# Add upstream once if missing
git remote -v | grep -q upstream || \
  git remote add upstream https://github.com/ml-explore/mlx-swift-lm.git

git fetch upstream main
git checkout bloom-q2-fork
git rebase upstream/main

# Resolve any conflicts:
#  - Quantization/* → always pick `ours`
#  - Models/SmolLM3.swift → port the 3-line type swap (Linear → PackedQ2Linear) if upstream renamed
git push --force-with-lease origin bloom-q2-fork

# Then in the parent repo:
cd ../..
git add Local/mlx-swift-lm
git commit -m "chore(deps): bump fork pointer to <new SHA>"
```

## When NOT to rebase

- Mid-implementation of Q2 phases. Rebase only at clean checkpoints (after a phase commit lands).
- When upstream has substantial changes to the SmolLM3 model class. Audit first.

## Pinning policy

The submodule is pinned to a specific SHA in the parent repo's `.gitmodules` + index.
Bloom's Xcode project references this submodule path via local SPM. Bumping the
pointer is a deliberate act, not automatic.
