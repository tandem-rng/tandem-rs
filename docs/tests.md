# Tests

```sh
cargo test
```

## Suite

- The specification vectors in `tests/vectors_data`, generated from the spec's `vectors.json`.
- Long fills, scalar draws and random access against the stream dumps in `tests/data`.
- Bounded integers, normals and exponentials against the tandem-c cross fixtures, with the
  hashes of `tests/normal_bits.rs` and `tests/exponential_bits.rs`.
- Feature tests: `--features rayon`, `serde`, `simd-intrinsics` and `wgpu`. The `wgpu` test skips
  without an adapter.

`tests/vectors.rs` checks every vector of the specification.
`tests/streams.rs` compares long fills, scalar draws and random access against reference
stream dumps in `tests/data`, complex fills included. `tests/derived.rs` compares bounded
integers, bounded fills and normals with the cross-check values of `tandem-c`, and the fills
with their definitions. The `f64` normal rows of `tests/cross_normal.h` start at unaligned
positions and include a wedge accept, a wedge reject and a tail, for fills and scalar draws.
With `std` the normals compare bit for bit. Without it they compare within the tolerance of
Appendix A. It also checks `f64` normal fills cut anywhere, the alignment of an empty fill,
and the mean and variance of 2^20 normals.

`tests/normal_bits.rs` (with `std`) compares FNV-1a hashes with tandem-c's
`tests/test_normal_bits.c`: 1e6 `f64` normals from five positions, 2e5 `f64` normals at two
positions of the spec's Python reference, and 2e6 - 1 `f32` normals from the five positions.

`tests/derived.rs` also compares the exponential fills and scalar draws with tandem-c's
`tests/cross_exponential.h`, bit for bit with `std`, at five positions, and checks fills cut
anywhere, a length of 0, and the Exp(1) moments to the fourth order and a Kolmogorov-Smirnov
statistic on 1e7 `f64` and 1e7 `f32` samples. `tests/exponential_bits.rs` (with `std`) hashes
1e6 `f64` and 1e6 `f32` exponentials from five positions and compares with the hash in
tandem-c's `tests/test_exponential_bits.c`.

`tests/rand_core.rs` checks the trait implementations against the inherent API.
`tests/parallel.rs` (with `--features rayon`) compares each parallel fill with the serial fill
at offsets and lengths that cut rows and tasks, and checks the final position.
`tests/serde.rs` (with `--features serde`) round-trips a generator through JSON.
`tests/intrinsics.rs` (with `--features simd-intrinsics`) compares every fill with the
stream built from the scalar `block`, at offsets and lengths that cut rows and chunks.
`tests/gpu.rs` (with `--features wgpu`) compares GPU fills with the CPU fills over keys, chunk
lengths, positions and lengths, and with the dumps. It skips without an adapter.

## Fixtures

`tests/vectors_data/mod.rs` is generated from the spec repository's `vectors.json` by
`tools/gen_vectors.py`. `tools/gen_derived.py` converts the cross-check values of `tandem-c`.
`tools/gen_zig_tables.py` writes `src/zig_tables.rs` from the spec's
`tables/normal_f64_zig1024.json`, and CI checks all three are current.
`cargo run --release --example dump_normals | shasum -a 256` writes the same bytes as
tandem-c's `tools/dump_normals.c`, SHA-256
`700ec4d2f4d6b82aaa56c6eff18a4e5919585fdbd093988773383d580ea610d1`. `cargo run --release --example dump_exponentials
| shasum -a 256` writes the same bytes as tandem-c's `tools/dump_exponentials.c`.
