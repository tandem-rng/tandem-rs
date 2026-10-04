# API

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

## Details

- `no_std` and `#![forbid(unsafe_code)]`. The default `std` feature only adds fused
  multiply-adds and the square root from the standard library. Two dependencies: `rand_core`
  for the traits and `wide` for portable vectors, which lower to NEON, SSE/AVX2 or scalar code.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`)
  plus a cache of the current 1024-bit row. It is `Copy`.
- Bounded integers (`below_u32`, `below_u64`) and standard normals (`normal_f64`,
  `normal_f32`, and the pairs `normal2_f64`, `normal2_f32`) with fills. They follow Appendix A
  of the specification, which is not normative, and the shared device core in `tandem-cuda`,
  so every port returns the same integers and, with tandem-c's polynomials, the same `f64`
  normals bit for bit. A bound of 0 returns 0 after one draw.
- `fill_below_*` takes one draw of the plain fill per element and consumes exactly one draw
  per element, so it parallelises. A rejected draw retries on `sub(purpose).split(g)` of
  the key, `g` being the global draw index, so a fill cut anywhere equals the whole. That
  equals the scalar calls whenever nothing is rejected.
- A Box-Muller pair uses two uniform draws. `normal_*` returns its cos half and `normal2_*`
  the `[cos, sin]` pair. `fill_normal_*` fills pairs from draws `2j` and `2j + 1`, so an odd
  length uses the cos half of its last pair and consumes both draws. `f32` normals use `f32`
  draws and run in `f32`. They match tandem-c bit for bit. On the device the `f64` normals
  match too, and the `f32` normals, which use `__sincosf`, agree to a few ulps.
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

## GPU

```toml
tandem-rng = { git = "https://github.com/tandem-rng/tandem-rs", features = ["wgpu"] }
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
