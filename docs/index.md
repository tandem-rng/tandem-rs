# tandem-rs

Rust implementation of Tandem8x32. The crate `tandem-rng` produces the stream of the
[specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) bit for bit, with SIMD
fills on CPUs, `rayon` fills, and a `wgpu` fill on GPUs.

- [API](api.md): the generator, the features, the derived draws and the GPU fill.
- [Design](design.md): the row loop, the features, and how the derived draws work.
- [Tests](tests.md): what each test file checks.
- [Speed](speed.md): CPU, rayon and GPU figures.

## Install

```toml
tandem-rng = { git = "https://github.com/tandem-rng/tandem-rs" }
# optional features: std (default), rayon, serde, simd-intrinsics, wgpu
```

Rust 1.89, edition 2024. The crate is `no_std` without the default `std` feature.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
