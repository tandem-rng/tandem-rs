# API

## Use

```rust
use tandem_rng::{ChoiceTable, Tandem};

let mut rng = Tandem::new(42);                 // 128-bit seed, default K
let x = rng.next_f64();
let mut words = vec![0u32; 1 << 20];
rng.fill_u32(&mut words);
let i = rng.below_u32(10);                     // uniform in 0..10, Lemire
let z = rng.normal_f64();                      // ziggurat from one u64 draw
let e = rng.exponential_f64();                 // -ln(1 - u) from one f64 draw
let table = ChoiceTable::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
let j = rng.choice(&table);                    // index 3 with probability 0.4
let worker = rng.split(7);                     // by index, from the key alone
let kids: Vec<Tandem> = rng.fork(4).collect(); // from the current block, parent moves on
```

```rust
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

let mut rng = Tandem::seed_from_u64(42);       // the same generator as Tandem::new(42)
let z: f64 = rng.sample(StandardNormal);
```

## Reference

- `Tandem`: a `Copy` generator, 128-bit key, 64-bit bit position, chunk length `K`.
- Every specification type: `bool`, 8 to 128-bit unsigned integers, `f32`, `f64`, binary16
  bit patterns, `char`, complex pairs such as `next_c64`. Scalar draws and `fill_*`.
- `at_u32`, `at_u64`, `at_f32`, `at_f64` for random access without advancing.
- `split`, `fork`, `sub`, `key`, `position`, `chunk_length`. `from_key` and `set_position`
  panic on a start position at or above 2^63, the bound of the specification, section 5.
  Draws and fills may still move past 2^63.
- `below_u32`, `below_u64`, `fill_below_u32`, `fill_below_u64`. A fill cut anywhere equals the
  whole fill. An empty fill leaves the position where it is.
- `normal_f64`, `normal_f32`, `normal2_f32`, `fill_normal_f64`, `fill_normal_f32`. `f64`
  normals are the ziggurat of Appendix A, `f32` normals Box-Muller pairs. With `std` they are
  bit identical to tandem-c. An empty `f64` fill aligns the position to 64 bits.
- `exponential_f64`, `exponential_f32`, `fill_exponential_f64`, `fill_exponential_f32`.
  With `std` they are bit identical to tandem-c and tandem-cuda.
- `ChoiceTable::new(&weights)`, `choice`, `fill_choice`: weighted indices by the alias table of
  Appendix C, one `u64` draw each, the same indices as every port. `new` returns
  `InvalidWeights` unless there are 1 to 2^32 − 1 finite, non-negative weights with a
  positive one. `capacity`, `cut` and `alias` read the table. An empty fill aligns the
  position to 64 bits. It needs the `alloc` feature, which `std` implies.
- `rand_core::TryRng` and `SeedableRng`, so `Tandem` drives every `rand` distribution.
- `rand`: the distributions `StandardNormal`, `Exp1`, `Below` and `ChoiceTable` give the Tandem
  stream's normals, exponentials, bounded integers and weighted choices. See [rand distributions](#rand-distributions).
- `rayon`: `par_fill_u32`, `par_fill_u64`, `par_fill_f32`, `par_fill_f64`, `par_fill_below_u32`,
  `par_fill_below_u64`, `par_fill_normal_f64`, `par_fill_normal_f32`, `par_fill_choice`. Each
  equals its serial fill.
- `serde`: `Serialize` and `Deserialize` through the transport form.
- `simd-intrinsics`, on by default: NEON, SSE2 and run-time AVX2 and FMA loops. It admits
  `unsafe` in one module. Set `TANDEM_NO_AVX2` to turn the AVX2 path off.
- `wgpu`: `gpu::GpuFill` fills `u32` words on any GPU wgpu drives. It advances the generator as
  the CPU fill does, so CPU and GPU draws mix on one stream.
- The `serde` feature implements `Serialize` and `Deserialize` for `Tandem` through its
  transport form (key, position, `K`). Deserializing rejects an invalid `K` and a position at
  or above 2^63.
- `SeedableRng::from_seed` reads its 16 bytes as a little-endian 128-bit seed and whitens it as
  the specification requires, and `SeedableRng::fork` is the specification's fork of one child.
  `let (key, pos, k) = (rng.key(), rng.position(), rng.chunk_length());` reads the transport form.

## rand distributions

```toml
tandem-rng = { git = "https://github.com/tandem-rng/tandem-rs", features = ["rand"] }
```

```rust
use rand::{RngExt, distr::Distribution};
use tandem_rng::{Below, Exp1, StandardNormal, Tandem};

let mut rng = Tandem::new(42);
let z: f64 = rng.sample(StandardNormal);       // rng.normal_f64()
let p: [f32; 2] = rng.sample(StandardNormal);  // rng.normal2_f32()
let e: f32 = rng.sample(Exp1);                 // rng.exponential_f32()
let i: u64 = rng.sample(Below(10u64));         // rng.below_u64(10)
let zs: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(1000).collect();
```

| Distribution | Type | Inherent draw |
|---|---|---|
| `StandardNormal` | `f64` | `normal_f64` |
| `StandardNormal` | `f32` | `normal_f32` |
| `StandardNormal` | `[f32; 2]` | `normal2_f32` |
| `Exp1` | `f64`, `f32` | `exponential_f64`, `exponential_f32` |
| `Below(n)` | `u32`, `u64` | `below_u32`, `below_u64` |
| `ChoiceTable` | `u32` | `choice` |

- Each sample equals its inherent draw bit for bit and advances the generator the same way.
  So `sample_iter` equals the fills of the normals and exponentials, with the `[f32; 2]`
  pairs flattened for `fill_normal_f32`. It equals `fill_below_*` up to the first rejected
  draw, because a fill retries on a fallback and the scalar draw on the stream.
- The `f32` normals, the exponentials, `Below` and `ChoiceTable` use only the `u32` and `u64`
  draws, so they are exact on any generator that yields the Tandem stream.
- An `f64` normal that misses the ziggurat's fast path continues on a fallback keyed by the
  generator's key and the draw's index (Appendix A), and `rand` passes a generic generator. So
  `StandardNormal` checks the generator's type: a `Tandem` or a `&mut Tandem` takes the exact
  path. Any other generator, a `dyn Rng` or a wrapper of a `Tandem` among them, gets the same
  ziggurat with its misses continued on its own stream. Those values are standard normals, but
  they leave the Tandem stream at the first miss.
- The type check compares type ids with lifetimes erased (`typeid`) and folds to a constant.
  It is the feature's one `unsafe` block. The feature keeps `no_std` and needs `rand` 0.10.

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
