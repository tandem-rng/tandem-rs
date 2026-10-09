# Speed

`cargo run --release --example bench`, `bench_distr`, `bench_par`, `bench_gpu` and
`bench_curand` produce the figures.

## CPU

One thread, `cargo run --release --example bench`, minimum of seven runs of 2^24 elements,
in GiB/s of output. A `next_u64` or `next_f64` draw counts 8 bytes. Each Apple M4 Pro table
comes from one session and gives the median of three runs.

The baselines are `rand`'s `SmallRng` (xoshiro256++) and `StdRng` (ChaCha12): `fill` for the
integers, one `random()` per element for the floats, `rand_distr::Exp1` for the exponentials.
A chain sums 2^24 dependent draws. The 32- and 64-bit draws inline into the loop, which keeps
the position in a register. Their one call, the row refill, also steps the cache one row ahead
into the other of its two row slots, so no draw reads a row just stored. Before that, the
`next_f64` chain ran at 6.9 GiB/s.

The default features are `std` and `simd-intrinsics`, so a plain dependency gets the default
column. The portable column is `--no-default-features --features std`. On the M4 the two
builds compile to the same AArch64 instructions as before `simd-intrinsics` became a default,
so the figures stand from their session: the default column was measured as the
`simd-intrinsics` build, the portable column as the former default.

| Apple M4 Pro | default | portable | `SmallRng` | `StdRng` |
|---|---|---|---|---|
| `fill_u32` | 19.2 | 19.3 | 10.5 | 2.7 |
| `fill_u64` | 19.1 | 19.2 | 10.6 | 2.7 |
| `fill_f32` | 16.8 | 16.6 | 5.4 | 2.0 |
| `fill_f64` | 16.7 | 16.7 | 10.7 | 2.3 |
| `fill_exponential_f32` | 6.5 | 6.4 | 3.0 | 0.88 |
| `fill_exponential_f64` | 5.9 | 5.9 | 6.0 | 1.8 |
| `next_u64` chain | 10.1 | 10.3 | 10.7 | 2.6 |
| `next_f64` chain | 9.3 | 9.5 | 10.4 | 2.5 |

| AMD EPYC 7702P | default | default, `TANDEM_NO_AVX2` | portable | `SmallRng` | `StdRng` |
|---|---|---|---|---|---|
| `fill_u32` | 11.1 | 5.8 | 5.6 | 6.3 | 2.9 |
| `fill_u64` | 11.0 | 5.8 | 5.6 | 6.2 | 3.0 |
| `fill_f32` | 8.9 | 5.2 | 5.1 | 3.0 | 1.9 |
| `fill_f64` | 7.0 | 4.3 | 3.8 | 5.9 | 2.2 |
| `fill_exponential_f32` | 3.4 | 0.15 | 0.15 | 1.3 | 0.67 |
| `fill_exponential_f64` | 2.9 | 0.33 | 0.23 | 2.7 | 1.4 |
| `next_u64` chain | 4.2 | 3.1 | 3.0 | 7.3 | 2.4 |
| `next_f64` chain | 3.6 | 2.8 | 2.7 | 5.8 | 2.3 |

A plain dependency now gets the default column on every x86_64 CPU with AVX2 and FMA. Every
EPYC figure comes from one session on one pinned core, rustc 1.99, and is the median of three
runs. That session predates the new defaults and the exact fma emulation. The default column
is the former `simd-intrinsics` build with AVX2, whose code is unchanged. The
`TANDEM_NO_AVX2` column, the SSE2 path of the default build, and the portable column
(`--no-default-features --features std`) await a remeasure: their exponential rows still show
the library `fma` call that the emulation replaces. Before the inline scalar draws, the
`next_f64` chain ran at 1.6 GiB/s. Only the AVX2 path leads `SmallRng` on the integer and
`f64` fills.

The `rand` distributions, `cargo run --release --features rand --example bench_distr`, one
thread, 2^22 elements, in GiB/s of output, each run the minimum of seven. The inherent column
loops over the scalar method, the `Distribution` column over `rng.sample`. Both inline into the
caller, so they cost the same. `StandardNormal.sample_iter` writes 4.98. The last two columns are
the `rand_distr` distribution, or `random_range(0..1000)`, on `SmallRng` and `StdRng`. The `f32`
normals of Tandem are pairs.

| Apple M4 Pro | inherent | `Distribution` | fill | `SmallRng` | `StdRng` |
|---|---|---|---|---|---|
| normal `f64` | 4.92 | 4.93 | 7.25 | 6.82 | 1.85 |
| normal `f32` | 1.78 | 1.77 | 4.51 | 3.46 | 0.92 |
| exponential `f64` | 3.51 | 3.54 | 5.93 | 6.11 | 1.71 |
| `Below(1000u32)` | 5.43 | 5.40 | 5.40 | 1.51 | 2.04 |

With `rayon`, `cargo run --release --features rayon --example bench_par`, 2^25 elements on
14 threads of an Apple M4 Pro, in GiB/s of output. The parallel `SmallRng` fill seeds one
generator per 2^16 elements by chunk index. Its normals are `rand_distr::StandardNormal`.

| Apple M4 Pro, 14 threads | serial | parallel | `SmallRng` serial | `SmallRng` parallel |
|---|---|---|---|---|
| `fill_u32` | 19.2 | 116 | 10.6 | 100 |
| `fill_f64` | 16.5 | 105 | 10.6 | 97 |
| `fill_below_u32`, n = 1000 | 5.4 | 44 | 1.5 | 15 |
| `fill_below_u64`, n = 1000 | 10.8 | 53 | 3.2 | 32 |
| `fill_normal_f64` | 7.3 | 70 | 6.2 | 56 |
| `fill_normal_f32` | 4.5 | 42 | 3.1 | 28 |

Without AVX2 and FMA the x86_64 target has no fused instruction. Each multiply-add of the
exponentials and normals there now rounds once by an exact emulation in the vectorized loop,
with the same bits. The `TANDEM_NO_AVX2` and portable columns of the EPYC table still show
the library `fma` call it replaces, which is why their exponential rows fall behind
`rand_distr::Exp1`.

## GPU

GPU fill into device memory, `cargo run --release --features wgpu --example bench_gpu
[log2 words]`. `TANDEM_GPU_ADAPTER=<index>` picks the adapter.

The baselines are Philox4x32-10 in WGSL with the same `mul_hi` as the Tandem shader, since no
wgpu crate ships a generator, and on the A100 cuRAND's Philox4x32-10 `curandGenerate`,
`cargo run --release --example bench_curand [log2 words]` with cuRAND and a CUDA 12 runtime on
the loader path.

| | words | cuRAND, one per call | cuRAND, 32 per sync | one fill per submit | 32 fills per submit | Philox, one per submit | Philox, 32 per submit |
|---|---|---|---|---|---|---|---|
| Apple M4 Pro, Metal | 2^26 | - | - | 137 GiB/s | 153 GiB/s | 120 GiB/s | 128 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan | 2^26 | 1234 GiB/s | 1262 GiB/s | 774 GiB/s | 1026 GiB/s | 408 GiB/s | 449 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan | 2^28 | 1282 GiB/s | 1290 GiB/s | 1142 GiB/s | 1245 GiB/s | 452 GiB/s | 461 GiB/s |

GPU fill: every row is the median of 21 runs after a 2 s warm-up per row. The A100 rows come
from one session on GPU 1, idle before the run. The "32 per submit" columns submit 32 fills
back to back and wait once. WGSL has no high multiply, and on the A100 the emulated one holds
Philox to about 450 GiB/s. The Apple fill is bound by the GPU's integer throughput, not by
memory.

On the A100 the shader itself is not the gap to cuRAND. Timestamp queries put the 2^28 kernel
at 1350 GiB/s, above cuRAND's whole call, and tandem-cuda's kernel reaches 1389. The rest is
the cost of a submit through wgpu and Vulkan: an empty submit and wait takes 0.05 ms, and back
to back each submit leaves the GPU idle about 0.06 ms, a fifth of a 2^26 fill. `fill_words`
submits once per call, so batching fills in one submit is the only way past that.

Off Metal, `GpuFill` runs `fill_tile`, which stages four rows of every chunk group in 16 KiB of
workgroup memory so 32 invocations store 512 contiguous bytes, as tandem-cuda's tile kernel
does. That lifted the A100 kernel from 1294 to 1344 GiB/s and the 2^28 fill from 1196 to 1245
back to back and from 1099 to 1142 one per submit. On the Apple M4 Pro the tile runs five times
slower than the direct stores, so Metal keeps them. Measured and not taken, all on the A100 at
2^28:

- A native high multiply through `SHADER_INT64` added 0.5 % to the kernel.
- Block offsets computed once per chunk instead of in 64-bit pairs per row changed nothing.
- Eight tile rows in 32 KiB instead of four added 0.5 % and exceed WebGPU's default limit.
- Fill parameters as immediates instead of a fresh uniform per fill gained 2 to 6 % one per
  submit, within the run-to-run spread, and nothing back to back.
- Reusing one uniform and bind group for every fill gained 4 % one per submit, but each fill
  needs its own parameters.

naga's SPIR-V holds no array but the tile, so the row state stays in scalars and vectors the
driver keeps in registers. The A100 host had no system Vulkan loader: a conda-forge
`libvulkan-loader` on `LD_LIBRARY_PATH` with the driver's own ICD was enough.
