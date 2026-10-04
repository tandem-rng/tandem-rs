# tandem-rs notes

Detail moved out of the README. Sections follow the README headings.

## What it provides

- `no_std` and `#![forbid(unsafe_code)]`. The default `std` feature only adds fused
  multiply-adds and the square root from the standard library. Two dependencies: `rand_core`
  for the traits and `wide` for portable vectors, which lower to NEON, SSE/AVX2 or scalar code.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`)
  plus a cache of the current 1024-bit row. It is `Copy`.
- Bounded integers (`below_u32`, `below_u64`) and standard normals (`normal_f64`,
  `normal_f32`, and the pairs `normal2_f64`, `normal2_f32`) with fills. They are not in the
  specification. They follow the shared device core in `tandem-cuda`, so every port returns
  the same integers and `f64` normals up to the last bits of `log`, `cos` and `sin`. A bound
  of 0 returns 0 after one draw.
- `fill_below_*` takes one draw of the plain fill per element and consumes exactly one draw
  per element, so it parallelises. A rejected draw retries on `sub(purpose).split(g)` of
  the key, `g` being the global draw index, so a fill cut anywhere equals the whole. That
  equals the scalar calls whenever nothing is rejected.
- A Box-Muller pair uses two uniform draws. `normal_*` returns its cos half and `normal2_*`
  the `[cos, sin]` pair. `fill_normal_*` fills pairs from draws `2j` and `2j + 1`, so an odd
  length uses the cos half of its last pair and consumes both draws. `f32` normals use `f32`
  draws and run in `f32`. They match tandem-c bit for bit and the device core to a few ulps.
- Normals are Box-Muller on whole blocks of uniforms in plain Rust that the compiler
  vectorises, with tandem-c's arithmetic: an exponent split and a short atanh series for the
  logarithm, an exact quarter-turn reduction for the sine and cosine. No libm is called. With
  `std` every multiply-add is a fused `mul_add`, so the normals are bit identical to
  tandem-c's on every target, and to each other across scalar draws and fills. The x86_64
  build with `simd-intrinsics` compiles a copy of the loop with `fma` and picks it at run
  time. Without a fused instruction `mul_add` is a correct but slower library call. Without
  `std` the plain form differs from tandem-c in the last bits.
- Exponentials of rate 1, `-ln(1 - u)` from one uniform each, defined in Appendix A of the
  specification. Element `i` of a fill is draw `i`, so a fill cut anywhere equals the whole and
  the scalar draws. `f32` draws run in `f32`. The logarithm is the normals', so with `std` the
  values are bit identical to tandem-c's and `tandem-cuda`'s, and the x86_64 build with
  `simd-intrinsics` picks the `fma` copy of the loop at run time as for the normals. A length
  of 0 leaves the position unchanged.
- The `wgpu` feature adds the same fill as a compute shader on any GPU wgpu drives.
- The `rayon` feature splits the output at row boundaries, fills the parts on separate threads
  and moves the position once, so the parallel fills equal the serial fills. It pulls in `std`.
- The `serde` feature implements `Serialize` and `Deserialize` for `Tandem` through its
  transport form (key, position, `K`). Deserializing rejects an invalid `K`.
- The `simd-intrinsics` feature spells the widening multiply and the float row stores with
  NEON or SSE2 intrinsics. With `std` on x86_64 it also runs the row step, the seeding and
  the transpose on 256-bit AVX2 registers when the CPU has them, chosen at run time, so the
  eight lanes of a group fill one register per state word. Setting `TANDEM_NO_AVX2` turns
  the AVX2 path off. It admits `unsafe` in one private module, so the crate root then says
  `deny(unsafe_code)` instead of `forbid`. The stream is the same.
- `SeedableRng::from_seed` reads its 16 bytes as a little-endian 128-bit seed and whitens it as
  the specification requires, and `SeedableRng::fork` is the specification's fork of one child.
  `let (key, pos, k) = (rng.key(), rng.position(), rng.chunk_length());` reads the transport form.

### GPU

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
generator exactly as the CPU fill does. `src/tandem.wgsl` is the shader. WGSL has no 64-bit
integers, so it builds the 32x32 to 64-bit product from 16-bit halves, which is exact. The
feature pulls in `std`.

## Tests

`tests/vectors.rs` checks every vector of the specification. `tests/vectors_data/mod.rs` is
generated from the spec repository's `vectors.json` by `tools/gen_vectors.py`.
`tests/streams.rs` compares long fills, scalar draws and random access against reference
stream dumps in `tests/data`, complex fills included. `tests/derived.rs` compares bounded
integers, bounded fills and normals with the cross-check values of `tandem-c`, which it
generates from the `tandem-cuda` core (`tools/gen_derived.py` converts them), and the fills
with their definitions. Those values come from the device core and its libm, so the comparison
has a tolerance.

`tests/normal_bits.rs` (with `std`) hashes 1e6 pairs of `f64` and `f32` normals from five
positions and compares with the hash in tandem-c's `tests/test_normal_bits.c`.
`cargo run --release --example dump_normals | shasum -a 256` writes the same bytes as
tandem-c's `tools/dump_normals.c`.

`tests/derived.rs` also compares the exponential fills and scalar draws with tandem-c's
`tests/cross_exponential.h`, bit for bit with `std`, at five positions, and checks fills cut
anywhere, a length of 0, and the Exp(1) moments to the fourth order and a Kolmogorov-Smirnov
statistic on 1e7 `f64` and 1e7 `f32` samples. `tests/exponential_bits.rs` (with `std`) hashes
1e6 `f64` and 1e6 `f32` exponentials from five positions and compares with the hash in
tandem-c's `tests/test_exponential_bits.c`. `cargo run --release --example dump_exponentials
| shasum -a 256` writes the same bytes as tandem-c's `tools/dump_exponentials.c`.

`tests/rand_core.rs` checks the trait implementations against the inherent API.
`tests/parallel.rs` (with `--features rayon`) compares each parallel fill with the serial fill
at offsets and lengths that cut rows and tasks, and checks the final position.
`tests/serde.rs` (with `--features serde`) round-trips a generator through JSON.
`tests/intrinsics.rs` (with `--features simd-intrinsics`) compares every fill with the
stream built from the scalar `block`, at offsets and lengths that cut rows and chunks.
`tests/gpu.rs` (with `--features wgpu`) compares GPU fills with the CPU fills over keys, chunk
lengths, positions and lengths, and with the dumps. It skips without an adapter.

## Speed

The README AVX2 column and the SSE2 column come from one session, except that the exponential
and `Exp1` rows come from a later one. The `Exp1` rows draw one sample per element from
`rand`'s default generator. Without AVX2 and FMA the x86_64 target has no fused instruction, so
each `mul_add` of the exponentials and normals is a library call, which is why those columns
fall behind `Exp1`. The default and SSE2 columns of the EPYC table use no AVX2.

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

The struct is `repr(C)`: with the default layout the compiler pairs the loads of the position
and the cached row index into one 16-byte load right after the 8-byte store of the position,
which defeats store forwarding and doubles the cost of a scalar draw.

GPU fill: the minimum is of seven runs after a half-second warm-up, with the GPU idle on the
A100 host. The Apple fill is bound by the GPU's integer throughput, not by memory. On the A100
the fill with direct 16-byte stores runs near the card's bandwidth once the buffer is large
enough to hide the submit and clock ramp. The A100 host had no system Vulkan loader: a
conda-forge `libvulkan-loader` on `LD_LIBRARY_PATH` with the driver's own ICD was enough.
