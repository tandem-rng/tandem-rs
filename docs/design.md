# Design

## Fills

- `no_std` and `#![forbid(unsafe_code)]` with `--no-default-features`. The default `std`
  feature adds the FMA instruction and the square root from the standard library, and the
  run-time AVX2 check. Two dependencies: `rand_core` for the traits and `wide` for portable
  vectors, which lower to NEON, SSE/AVX2 or scalar code.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`)
  plus a cache of two 1024-bit rows, 432 bytes in all, the layout of tandem-c. It is `Copy`.
- The `simd-intrinsics` feature, on by default, spells the widening multiply and the float row
  stores with NEON or SSE2 intrinsics. With `std` on x86_64 it also runs the row step, the seeding and
  the transpose on 256-bit AVX2 registers when the CPU has them, chosen at run time, so the
  eight lanes of a group fill one register per state word. Setting `TANDEM_NO_AVX2` turns
  the AVX2 path off. It admits `unsafe` in one private module, so the crate root then says
  `deny(unsafe_code)` instead of `forbid`. The stream is the same.
- The `rayon` feature splits the output at row boundaries, fills the parts on separate threads
  and moves the position once, so the parallel fills equal the serial fills. It pulls in `std`.
- The `wgpu` feature adds the same fill as a compute shader on any GPU wgpu drives.

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

The 32- and 64-bit draws inline into the caller. A draw at an aligned position in the readable
row is a load and an add, and the one call is the row refill, after which the draw stores the
position, so a loop keeps the position in a register. The refill makes its row readable and
steps the cache one row ahead into the other row slot. The next refill then finds its row
computed, and no draw reads words just written by vector stores, which store forwarding serves
badly. The slots hold the exposed words in stream order, the order the draws read.

The struct is `repr(C)`: with the default layout the compiler pairs the loads of the position
and the readable row's position into one 16-byte load right after the 8-byte store of the
position, which defeats store forwarding and doubles the cost of a scalar draw.

## Bounded integers

- Bounded integers (`below_u32`, `below_u64`) and standard normals (`normal_f64`,
  `normal_f32`, and the `f32` pair `normal2_f32`) with fills. They follow Appendix A of the
  specification, which is not normative, so every port returns the same integers and the same
  `f64` normals bit for bit. A bound of 0 returns 0 after one draw.
- `fill_below_*` takes one draw of the plain fill per element and consumes exactly one draw
  per element, so it parallelises. A rejected draw retries on `sub(purpose).split(g)` of
  the key, `g` being the global draw index, so a fill cut anywhere equals the whole. That
  equals the scalar calls whenever nothing is rejected.

## Normals

- `f64` normals are the 1024-layer ziggurat of Appendix A, one `u64` draw per element. Element
  `i` of a fill comes from draw `i` of the `u64` fill. 99.57 % of the draws land in an inner
  rectangle and cost a table lookup and a multiply. A draw that misses continues on its own
  fallback generator, `sub(0x4e524d3634).split(g)` of the key at position 0, `g` being the
  global draw index. So a fill cut anywhere equals the whole and the scalar draws, and the fill
  never consumes more than `n` draws. An empty fill aligns the position to 64 bits.
- `src/zig_tables.rs` is generated from the spec's `tables/normal_f64_zig1024.json` by
  `tools/gen_zig_tables.py`, which checks the file's SHA-256. CI regenerates it and fails on
  a difference.
- A fill writes every candidate in one pass over 512 draws and lists the misses. The misses
  queue across passes and their fallbacks are seeded eight at a time on the row lanes, as
  tandem-c does. Each fallback then computes one block per two draws instead of a whole row.
- A Box-Muller pair uses two `f32` draws. `normal_f32` returns its cos half and `normal2_f32`
  the `[cos, sin]` pair. `fill_normal_f32` fills pairs from draws `2j` and `2j + 1`, so an odd
  length uses the cos half of its last pair and consumes both draws. `f32` normals run in
  `f32` on whole blocks of uniforms in plain Rust that the compiler vectorises, with
  tandem-c's arithmetic: an exponent split and a short atanh series for the logarithm, an
  exact quarter-turn reduction for the sine and cosine. No libm is called.
- Every multiply-add of the logarithm and the `f32` normals is fused, rounded once, so both
  kinds of normal and the exponentials are bit identical to tandem-c's on every target and in
  `no_std`, and to each other across scalar draws and fills. The x86_64 build with
  `simd-intrinsics` compiles a copy of these loops with `fma` and picks it at run time, and
  targets built with FMA use `mul_add`. Elsewhere `mul_add` would be a library call that keeps
  the loops scalar, and `no_std` has none, so each multiply-add is an exact emulation: `f32`
  takes the product exactly in `f64` and rounds the sum to odd before the rounding to `f32`,
  and `f64` adds Dekker's exact product by two error-free sums, the last rounded to odd (Boldo
  and Melquiond, IEEE Trans. Computers 57, 2008).

## Weighted choice

- `ChoiceTable` is the alias table of Appendix C, built in exact integers so that every port
  gets the same `capacity`, `cut` and `alias`. Two passes scale the masses: the first bounds
  the sum, and the second brings it just below 2^63 for weights of any magnitude. A weight's
  mass is the ceiling of `w · 2^t`, computed from its 53-bit significand, so a subnormal weight
  keeps a mass of at least 1 where `w · 2^t` in floats would round to 0. The columns pair in
  place, as tandem-c does.
- A draw maps one `u64` to a column by the high word of `r · m`, and compares the low word
  times `capacity` with the column's cut. There is no retry, so element `i` of a fill is draw
  `i` of the `u64` fill and a fill cut anywhere equals the whole. The table and the draw use
  no floating point, so there is no `std` difference.
- The table needs an allocator, hence the `alloc` feature.

## Exponentials

- Exponentials of rate 1, `-ln(1 - u)` from one uniform each, defined in Appendix A of the
  specification. Element `i` of a fill is draw `i`, so a fill cut anywhere equals the whole and
  the scalar draws. `f32` draws run in `f32`. The `f64` logarithm is the normals'. The `f32`
  one is tandem-c's `neg_log_f32`, which carries `(2 - 2m) / (m + 1)` in two floats and adds
  `k ln 2` by an exact two-sum, within 0.571 ulp, so `1 - exp(-x)` maps every draw back to its
  own 2^-24 grid point. With `std` the values are bit identical to tandem-c's and
  `tandem-cuda`'s, and the x86_64 build with
  `simd-intrinsics` picks the `fma` copy of the loop at run time as for the `f32` normals. A length
  of 0 leaves the position unchanged.
