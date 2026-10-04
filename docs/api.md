# API

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

## Reference

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
- The `serde` feature implements `Serialize` and `Deserialize` for `Tandem` through its
  transport form (key, position, `K`). Deserializing rejects an invalid `K`.
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

## Parallel use

Element `i` of a fill is draw `i`, so any decomposition reproduces a serial run.
See [Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative).
