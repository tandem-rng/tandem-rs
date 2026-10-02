<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .rs"></p>

# tandem-rs

Rust implementation of [Tandem8x32](https://github.com/tandem-rng/spec), a noncryptographic
pseudorandom number generator built to be fast on CPUs and GPUs alike. The crate is
`tandem-rng`. It produces the same stream, bit for bit, as the Julia reference
[TandemRNG.jl](https://github.com/tandem-rng/TandemRNG.jl) and the C reference
[tandem-c](https://github.com/tandem-rng/tandem-c), and runs as fast as the C.

- `no_std`, `#![forbid(unsafe_code)]`. Two dependencies: `rand_core` for the traits and
  `wide` for portable vectors, which lower to NEON, SSE/AVX2 or scalar code.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`)
  plus a cache of the current 1024-bit row. It is `Copy`.
- Every type in the specification: `bool`, 8 to 128-bit unsigned integers, `f32`, `f64`,
  binary16 as bit patterns, `char`. Random access without advancing. Split by index, fork at
  the current block, sub by purpose.
- Implements `rand_core::TryRng` (and so `Rng`) and `SeedableRng`, so it drives every `rand`
  distribution.

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

## Tests

```sh
cargo test
```

`tests/vectors.rs` checks every vector of the specification. `tests/vectors_data/mod.rs` is
generated from the spec repository's `vectors.json` by `tools/gen_vectors.py`, and CI fails
when it is out of date. `tests/streams.rs` compares long fills, scalar draws and random access
against dumps written by TandemRNG.jl (`tests/data`, shared with tandem-c).
`tests/rand_core.rs` checks the trait implementations against the inherent API.

## Speed

Apple M4, one thread, `cargo run --release --example bench`, minimum of seven runs of 2^24
elements after a warm-up, next to tandem-c's `make bench` in the same minute (load 5):

| | GiB/s | tandem-c |
|---|---|---|
| `fill_u32` | 15.9 | 16.0 |
| `fill_u64` | 15.9 | 16.0 |
| `fill_f32` | 12.5 | 12.6 |
| `fill_f64` | 12.5 | 12.5 |
| `next_f64` chain, ns per draw | 1.47 | 1.58 |

The eight lane states of a row stay in registers as `u32x4` vectors, the row store is a
4x4 transpose by interleaves, and every integer fill writes the same byte stream, so one
routine serves all widths and floats convert in place. The struct is `repr(C)`: with the
default layout the compiler pairs the loads of the position and the cached row index into
one 16-byte load right after the 8-byte store of the position, which defeats store
forwarding and doubles the cost of a scalar draw.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
