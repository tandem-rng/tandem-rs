<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .rs"></p>

# tandem-rs

[![CI](https://github.com/tandem-rng/tandem-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-rs/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tandem--rng.github.io-7fb3ee.svg)](https://tandem-rng.github.io/tandem-rs/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Rust implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator. The crate `tandem-rng` produces the stream the specification
defines, bit for bit, with SIMD fills on CPUs, `rayon` fills, and a `wgpu` fill on GPUs.

Rust 1.89, edition 2024. The crate is `no_std` without the default `std` feature.

```toml
tandem-rng = { git = "https://github.com/tandem-rng/tandem-rs" }
# optional features: std (default), rayon, serde, simd-intrinsics, wgpu
```

```rust
use tandem_rng::Tandem;

let mut rng = Tandem::new(42);                 // 128-bit seed, default K
let mut words = vec![0u32; 1 << 20];
rng.fill_u32(&mut words);
let mut worker = rng.split(7);                 // by index, from the key alone
let z = worker.normal_f64();                   // ziggurat, bit identical to tandem-c with std
```

See [API](docs/api.md) for the `rand` traits, features and GPU fill, and
[design](docs/design.md), [tests](docs/tests.md) and [speed](docs/speed.md) for the rest.

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://tandem-rng.github.io/tandem-rs/) · [Apache 2.0 license](LICENSE)
