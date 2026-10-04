# Design

## Fills

- `no_std` and `#![forbid(unsafe_code)]`. The default `std` feature only adds fused
  multiply-adds and the square root from the standard library. Two dependencies: `rand_core`
  for the traits and `wide` for portable vectors, which lower to NEON, SSE/AVX2 or scalar code.
- A generator is its transport form (128-bit key, 64-bit bit position, chunk length `K`)
  plus a cache of the current 1024-bit row. It is `Copy`.
- The `simd-intrinsics` feature spells the widening multiply and the float row stores with
  NEON or SSE2 intrinsics. With `std` on x86_64 it also runs the row step, the seeding and
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

The struct is `repr(C)`: with the default layout the compiler pairs the loads of the position
and the cached row index into one 16-byte load right after the 8-byte store of the position,
which defeats store forwarding and doubles the cost of a scalar draw.

## Bounded integers

- Bounded integers (`below_u32`, `below_u64`) and standard normals (`normal_f64`,
  `normal_f32`, and the pairs `normal2_f64`, `normal2_f32`) with fills. They follow Appendix A
  of the specification, which is not normative, and the shared device core in `tandem-cuda`,
  so every port returns the same integers and, with tandem-c's polynomials, the same `f64`
  normals bit for bit. A bound of 0 returns 0 after one draw.
- `fill_below_*` takes one draw of the plain fill per element and consumes exactly one draw
  per element, so it parallelises. A rejected draw retries on `sub(purpose).split(g)` of
  the key, `g` being the global draw index, so a fill cut anywhere equals the whole. That
  equals the scalar calls whenever nothing is rejected.

## Normals

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

## Exponentials

- Exponentials of rate 1, `-ln(1 - u)` from one uniform each, defined in Appendix A of the
  specification. Element `i` of a fill is draw `i`, so a fill cut anywhere equals the whole and
  the scalar draws. `f32` draws run in `f32`. The logarithm is the normals', so with `std` the
  values are bit identical to tandem-c's and `tandem-cuda`'s, and the x86_64 build with
  `simd-intrinsics` picks the `fma` copy of the loop at run time as for the normals. A length
  of 0 leaves the position unchanged.
