<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .rs"></p>

# tandem-rs

Rust implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator. The crate is `tandem-rng`. It produces the stream the
specification defines, bit for bit. It is fast on CPUs and GPUs alike.

## Install

```toml
tandem-rng = "0.1"
# optional features: std (default), rayon, serde, simd-intrinsics, wgpu
```

Rust 1.89, edition 2024. The crate is `no_std` without the default `std` feature.

## Use

```rust
use tandem_rng::Tandem;

let mut rng = Tandem::new(42);                 // 128-bit seed, default K
let x = rng.next_f64();
let mut words = vec![0u32; 1 << 20];
rng.fill_u32(&mut words);
let i = rng.below_u32(10);                     // uniform in 0..10, Lemire
let z = rng.normal_f64();                      // Box-Muller from two f64 draws
let e = rng.exponential_f64();                 // -ln(1 - u) from one f64 draw
let worker = rng.split(7);                     // by index, from the key alone
let kids: Vec<Tandem> = rng.fork(4).collect(); // from the current block, parent moves on
```

```rust
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

let mut rng = Tandem::seed_from_u64(42);       // the same generator as Tandem::new(42)
let z: f64 = rng.sample(StandardNormal);
```

## What it provides

- `Tandem`: a `Copy` generator, 128-bit key, 64-bit bit position, chunk length `K`.
- Every specification type: `bool`, 8 to 128-bit unsigned integers, `f32`, `f64`, binary16
  bit patterns, `char`, complex pairs such as `next_c64`. Scalar draws and `fill_*`.
- `at_u32`, `at_u64`, `at_f32`, `at_f64` for random access without advancing.
- `split`, `fork`, `sub`, `key`, `position`, `chunk_length`.
- `below_u32`, `below_u64`, `fill_below_u32`, `fill_below_u64`. A fill cut anywhere equals the
  whole fill.
- `normal_f64`, `normal_f32`, `normal2_f64`, `normal2_f32`, `fill_normal_f64`,
  `fill_normal_f32`. With `std` they are bit identical to tandem-c.
- `exponential_f64`, `exponential_f32`, `fill_exponential_f64`, `fill_exponential_f32`.
  With `std` they are bit identical to tandem-c and tandem-cuda.
- `rand_core::TryRng` and `SeedableRng`, so `Tandem` drives every `rand` distribution.
- `rayon`: `par_fill_u32`, `par_fill_u64`, `par_fill_f32`, `par_fill_f64`, `par_fill_below_u32`,
  `par_fill_below_u64`, `par_fill_normal_f64`, `par_fill_normal_f32`. Each equals its serial fill.
- `serde`: `Serialize` and `Deserialize` through the transport form.
- `simd-intrinsics`: NEON, SSE2 and run-time AVX2 row steps. It admits `unsafe` in one module.
  Set `TANDEM_NO_AVX2` to turn the AVX2 path off.
- `wgpu`: `gpu::GpuFill` fills `u32` words on any GPU wgpu drives. It advances the generator as
  the CPU fill does, so CPU and GPU draws mix on one stream.
- Parallel use: element `i` of a fill is draw `i`, so any decomposition reproduces a serial run.
  See [Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative).

## Tests

```sh
cargo test
```

- The specification vectors in `tests/vectors_data`, generated from the spec's `vectors.json`.
- Long fills, scalar draws and random access against the stream dumps in `tests/data`.
- Bounded integers, normals and exponentials against the tandem-c cross fixtures, with the
  hashes of `tests/normal_bits.rs` and `tests/exponential_bits.rs`.
- Feature tests: `--features rayon`, `serde`, `simd-intrinsics` and `wgpu`. The `wgpu` test skips
  without an adapter.

## Speed

One thread, `cargo run --release --example bench`, minimum of seven runs of 2^24 elements,
in GiB/s.

| Apple M4 | default | `simd-intrinsics` |
|---|---|---|
| `fill_u32` | 21.8 | 21.4 |
| `fill_u64` | 21.5 | 21.4 |
| `fill_f32` | 17.1 | 18.6 |
| `fill_f64` | 16.9 | 18.5 |
| `fill_exponential_f32` | 6.2 | 6.4 |
| `fill_exponential_f64` | 5.7 | 5.9 |
| `rand_distr::Exp1` `f32`, `StdRng` | 0.88 | 0.88 |
| `rand_distr::Exp1` `f64`, `StdRng` | 1.8 | 1.8 |
| `next_f64` chain, ns per draw | 1.39 | 1.42 |

| AMD EPYC 7702P | default | `simd-intrinsics`, SSE2 | `simd-intrinsics`, AVX2 |
|---|---|---|---|
| `fill_u32` | 5.6 | 5.8 | 11.4 |
| `fill_u64` | 5.7 | 5.7 | 11.8 |
| `fill_f32` | 5.0 | 5.2 | 9.0 |
| `fill_f64` | 3.8 | 4.4 | 7.2 |
| `fill_exponential_f32` | 0.27 | 0.27 | 4.2 |
| `fill_exponential_f64` | 0.23 | 0.33 | 3.0 |
| `rand_distr::Exp1` `f32`, `StdRng` | 0.64 | 0.65 | 0.66 |
| `rand_distr::Exp1` `f64`, `StdRng` | 1.3 | 1.3 | 1.3 |
| `next_f64` chain, ns per draw | 4.96 | 4.94 | 3.96 |

The SSE2 column is the AVX2 build with `TANDEM_NO_AVX2` set. Without AVX2 and FMA the
exponentials and normals fall behind `Exp1`.

With `rayon`, `cargo run --release --features rayon --example bench_par`, 2^25 elements on
14 threads of an Apple M4, in GiB/s of output.

| Apple M4, 14 threads | serial | parallel |
|---|---|---|
| `fill_u32` | 20.6 | 118 |
| `fill_f64` | 15.9 | 109 |
| `fill_below_u32`, n = 1000 | 6.1 | 47 |
| `fill_below_u64`, n = 1000 | 11.4 | 58 |
| `fill_normal_f64` | 3.4 | 31 |
| `fill_normal_f32` | 4.1 | 36 |

GPU fill into device memory, `cargo run --release --features wgpu --example bench_gpu
[log2 words]`, minimum of seven. `TANDEM_GPU_ADAPTER=<index>` picks the adapter.

| | words | one fill per submit | 32 fills per submit |
|---|---|---|---|
| Apple M4 Pro, Metal | 2^26 | 164 GiB/s | 189 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan | 2^26 | 775 GiB/s | 1040 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan | 2^28 | 1100 GiB/s | 1215 GiB/s |

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
