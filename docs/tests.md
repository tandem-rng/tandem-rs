# Tests

```sh
cargo test
```

- The specification vectors in `tests/vectors_data`, generated from the spec's `vectors.json`.
- Long fills, scalar draws and random access against the stream dumps in `tests/data`.
- Bounded integers, normals and exponentials against the tandem-c cross fixtures, with the
  hashes of `tests/normal_bits.rs` and `tests/exponential_bits.rs`.
- Feature tests: `--features rayon`, `serde`, `simd-intrinsics` and `wgpu`. The `wgpu` test skips
  without an adapter.

`tests/vectors.rs` checks every vector of the specification. `tests/vectors_data/mod.rs` is
generated from the spec repository's `vectors.json` by `tools/gen_vectors.py`.
`tests/streams.rs` compares long fills, scalar draws and random access against reference
stream dumps in `tests/data`, complex fills included. `tests/derived.rs` compares bounded
integers, bounded fills and normals with the cross-check values of `tandem-c`, which it
generates from the `tandem-cuda` core (`tools/gen_derived.py` converts them), and the fills
with their definitions. With `std` the normals compare bit for bit. Without it they compare
within the tolerance of Appendix A.

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
