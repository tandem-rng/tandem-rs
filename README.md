<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .rs"></p>

# tandem-rs

Rust implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. The crate is
`tandem-rng`. It produces the stream the specification defines, bit for bit.

- `no_std` and `#![forbid(unsafe_code)]`. The default `std` feature only adds fused
  multiply-adds and the square root from the standard library. Two dependencies: `rand_core`
  for the traits and `wide` for portable vectors, which lower to NEON, SSE/AVX2 or scalar code.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`)
  plus a cache of the current 1024-bit row. It is `Copy`.
- Every type in the specification: `bool`, 8 to 128-bit unsigned integers, `f32`, `f64`,
  binary16 as bit patterns, `char`, complex `f32` and `f64` as `[re, im]` pairs. Random
  access without advancing. Split by index, fork at the current block, sub by purpose.
- Bounded integers (`below_u32`, `below_u64`) and standard normals (`normal_f64`,
  `normal_f32`, and the pairs `normal2_f64`, `normal2_f32`) with fills. They are not in the
  specification. They follow the shared device core in `tandem-cuda`, so every port returns
  the same integers and `f64` normals up to the last bits of `log`, `cos` and `sin`. A bound
  of 0 returns 0 after one draw.
- `fill_below_*` takes draw `i` of the plain fill for element `i` and consumes exactly one
  draw per element, so it parallelises. A rejected draw retries on `sub(purpose).split(i)` of
  the key, as the device core does. That equals the scalar calls whenever nothing is rejected.
- A Box-Muller pair uses two uniform draws. `normal_*` returns its cos half and `normal2_*`
  the `[cos, sin]` pair. `fill_normal_*` fills pairs from draws `2j` and `2j + 1`, so an odd
  length uses the cos half of its last pair and consumes both draws. `f32` normals use `f32`
  draws and run in `f32`, so ports agree on them to a few ulps, not bit for bit.
- Normals are Box-Muller on whole blocks of uniforms with the crate's own logarithm (fdlibm's
  algorithm) and range-free sine and cosine, in plain Rust that the compiler vectorises. The
  logarithm is within one ulp of the system one over 10^8 random draws. The `std` feature
  fuses the multiply-adds on aarch64 and FMA targets, so the last bits of a normal differ
  between targets by an ulp or two. The scalar draws and the fills agree bit for bit on one
  target.
- Implements `rand_core::TryRng` (and so `Rng`) and `SeedableRng`, so it drives every `rand`
  distribution.
- The `wgpu` feature adds the same fill as a compute shader on any GPU wgpu drives.
- The `rayon` feature adds `par_fill_u32`, `par_fill_u64`, `par_fill_f32`, `par_fill_f64`,
  `par_fill_below_u32`, `par_fill_below_u64`, `par_fill_normal_f64` and `par_fill_normal_f32`.
  They split the output at row boundaries, fill the parts on separate threads and move the
  position once, so they equal the serial fills. The feature pulls in `std`.
- The `serde` feature implements `Serialize` and `Deserialize` for `Tandem` through its
  transport form (key, position, `K`). Deserializing rejects an invalid `K`.
- The `simd-intrinsics` feature spells the widening multiply and the float row stores with
  NEON or SSE2 intrinsics. It admits `unsafe` in one private module, so the crate root
  then says `deny(unsafe_code)` instead of `forbid`. The stream is the same.

## Use

```rust
use tandem_rng::Tandem;

let mut rng = Tandem::new(42);                 // 128-bit seed, default K
let x = rng.next_f64();
let mut words = vec![0u32; 1 << 20];
rng.fill_u32(&mut words);
let c = rng.next_c64();                        // [re, im], two f64 draws
let i = rng.below_u32(10);                     // uniform in 0..10, Lemire
let z = rng.normal_f64();                      // Box-Muller from two f64 draws
let worker = rng.split(7);                     // by index, from the key alone
let kids: Vec<Tandem> = rng.fork(4).collect(); // from the current block, parent moves on
let (key, pos, k) = (rng.key(), rng.position(), rng.chunk_length());
```

With `rand`:

```rust
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use tandem_rng::Tandem;

let mut rng = Tandem::seed_from_u64(42);       // the same generator as Tandem::new(42)
let z: f64 = rng.sample(StandardNormal);
```

`SeedableRng::from_seed` reads its 16 bytes as a little-endian 128-bit seed and whitens it as
the specification requires, and `SeedableRng::fork` is the specification's fork of one child.

Parallel use: element `i` of a fill is draw `i`, so ranks, threads or devices that start at the
position of their first element, or draw from `split(task)`, reproduce a serial run for any
decomposition, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

## GPU

```toml
tandem-rng = { version = "0.1", features = ["wgpu"] }
```

```rust
use tandem_rng::{Tandem, gpu::GpuFill};

let gpu = GpuFill::new_default().expect("a wgpu adapter");
let mut rng = Tandem::new(42);
let words: Vec<u32> = gpu.read_u32(&mut rng, 1 << 20);  // the values of rng.fill_u32
let out = gpu.device().create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: GpuFill::buffer_size(&rng, 32, 1 << 24),
    usage: wgpu::BufferUsages::STORAGE,
    mapped_at_creation: false,
});
let span = gpu.fill_words(&mut rng, &out, 1 << 24);     // stays on the GPU
```

`GpuFill::new` takes a device and queue the application owns. Every GPU fill advances the
generator exactly as the CPU fill does, so CPU and GPU draws can be mixed on one stream.
`src/tandem.wgsl` is the shader. WGSL has no 64-bit integers, so it builds the 32x32 to
64-bit product from 16-bit halves, which is exact. The feature pulls in `std`.

## Tests

```sh
cargo test
```

`tests/vectors.rs` checks every vector of the specification. `tests/vectors_data/mod.rs` is
generated from the spec repository's `vectors.json` by `tools/gen_vectors.py`, and CI fails
when it is out of date. `tests/streams.rs` compares long fills, scalar draws and random access
against reference stream dumps in `tests/data`, complex fills included.
`tests/derived.rs` compares bounded integers, bounded fills and normals with the cross-check
values of `tandem-c`, which it generates from the `tandem-cuda` core (`tools/gen_derived.py`
converts them), and the fills with their definitions.
`tests/rand_core.rs` checks the trait implementations against the inherent API.
`tests/parallel.rs` (with `--features rayon`) compares each parallel fill with the serial fill
at offsets and lengths that cut rows and tasks, and checks the final position.
`tests/serde.rs` (with `--features serde`) round-trips a generator through JSON.
`tests/intrinsics.rs` (with `--features simd-intrinsics`) compares every fill with the
stream built from the scalar `block`, at offsets and lengths that cut rows and chunks.
`tests/gpu.rs` (with `--features wgpu`) compares GPU fills with the CPU fills over keys, chunk
lengths, positions and lengths, and with the dumps. It skips without an adapter.

## Speed

One thread, `cargo run --release --example bench` (add `--features simd-intrinsics` for the
second column), minimum of seven runs of 2^24 elements after a warm-up, in GiB/s:

| Apple M4 | default | `simd-intrinsics` |
|---|---|---|
| `fill_u32` | 21.8 | 21.4 |
| `fill_u64` | 21.5 | 21.4 |
| `fill_f32` | 17.1 | 18.6 |
| `fill_f64` | 16.9 | 18.5 |
| `next_f64` chain, ns per draw | 1.39 | 1.42 |

| AMD EPYC 7702P, SSE2 baseline | default | `simd-intrinsics` |
|---|---|---|
| `fill_u32` | 5.2 | 5.7 |
| `fill_u64` | 5.2 | 5.7 |
| `fill_f32` | 4.6 | 4.8 |
| `fill_f64` | 3.5 | 4.4 |
| `next_f64` chain, ns per draw | 4.99 | 4.88 |

With the `rayon` feature, `cargo run --release --features rayon --example bench_par` times the
serial and parallel fills of 2^25 elements on all 14 threads of an Apple M4, minimum of seven
runs, in GiB/s of output:

| Apple M4, 14 threads | serial | parallel |
|---|---|---|
| `fill_u32` | 20.6 | 118 |
| `fill_f64` | 15.9 | 109 |
| `fill_below_u32`, n = 1000 | 6.1 | 47 |
| `fill_below_u64`, n = 1000 | 11.4 | 58 |
| `fill_normal_f64` | 3.4 | 31 |
| `fill_normal_f32` | 4.1 | 36 |

The eight lane states of a row stay in registers as `u32x4` vectors, the row store is a
4x4 transpose by interleaves, and every integer fill writes the same byte stream, so one
routine serves all widths. By default floats convert in a second pass over the row while it
is in L1. The seed rounds run on a local that swaps its halves by value, since `mem::swap`
of the fields made LLVM split the vectors into 64-bit halves.

`simd-intrinsics` maps each 16-byte block to floats before the store: `ucvtf` with 24 or
53 fraction bits on AArch64, the signed convert for `f32` and an exact two-part
exponent trick for `f64` on SSE2, which has no 64-bit convert. It also forms both halves of
the 32x32 to 64-bit products from two widening multiplies (`umull` and `umull2` with two
unzips, or two `pmuludq`). On AArch64 LLVM already uses `umull` for the high half, so
integers do not gain there. On SSE2 the portable multiply costs four `pmuludq` per
product.

The struct is `repr(C)`: with the
default layout the compiler pairs the loads of the position and the cached row index into
one 16-byte load right after the 8-byte store of the position, which defeats store
forwarding and doubles the cost of a scalar draw.

GPU fill into device memory, `cargo run --release --features wgpu --example bench_gpu [log2 words]`,
minimum of seven after a half-second warm-up. `TANDEM_GPU_ADAPTER=<index>` picks the adapter on
hosts with several GPUs.

| | words | one fill per submit | 32 fills per submit |
|---|---|---|---|
| Apple M4 Pro, Metal | 2^26 | 164 GiB/s | 189 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan, GPU idle | 2^26 | 775 GiB/s | 1040 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan, GPU idle | 2^28 | 1100 GiB/s | 1215 GiB/s |

The Apple fill is bound by the GPU's integer throughput, not by memory. On the A100 the fill
with direct 16-byte stores runs near the card's bandwidth once the buffer is large enough to
hide the submit and clock ramp. The A100 host had no system Vulkan loader: a conda-forge
`libvulkan-loader` on `LD_LIBRARY_PATH` with the driver's own ICD was enough.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
