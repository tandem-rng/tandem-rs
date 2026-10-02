<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .rs"></p>

# tandem-rs

Rust implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. The crate is
`tandem-rng`. It produces the stream the specification defines, bit for bit.

- `no_std`, `#![forbid(unsafe_code)]`. Two dependencies: `rand_core` for the traits and
  `wide` for portable vectors, which lower to NEON, SSE/AVX2 or scalar code.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`)
  plus a cache of the current 1024-bit row. It is `Copy`.
- Every type in the specification: `bool`, 8 to 128-bit unsigned integers, `f32`, `f64`,
  binary16 as bit patterns, `char`. Random access without advancing. Split by index, fork at
  the current block, sub by purpose.
- Implements `rand_core::TryRng` (and so `Rng`) and `SeedableRng`, so it drives every `rand`
  distribution.
- The `wgpu` feature adds the same fill as a compute shader on any GPU wgpu drives.

## Use

```rust
use tandem_rng::Tandem;

let mut rng = Tandem::new(42);                 // 128-bit seed, default K
let x = rng.next_f64();
let mut words = vec![0u32; 1 << 20];
rng.fill_u32(&mut words);
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
against reference stream dumps in `tests/data`.
`tests/rand_core.rs` checks the trait implementations against the inherent API.
`tests/gpu.rs` (with `--features wgpu`) compares GPU fills with the CPU fills over keys, chunk
lengths, positions and lengths, and with the dumps. It skips without an adapter.

## Speed

Apple M4, one thread, `cargo run --release --example bench`, minimum of seven runs of 2^24
elements after a warm-up:

| | GiB/s |
|---|---|
| `fill_u32` | 16.4 |
| `fill_u64` | 16.4 |
| `fill_f32` | 13.8 |
| `fill_f64` | 13.0 |
| `next_f64` chain, ns per draw | 1.42 |

The eight lane states of a row stay in registers as `u32x4` vectors, the row store is a
4x4 transpose by interleaves, and every integer fill writes the same byte stream, so one
routine serves all widths and floats convert in place. The struct is `repr(C)`: with the
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

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
