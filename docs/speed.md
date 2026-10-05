# Speed

`cargo run --release --example bench`, `bench_distr`, `bench_par` and `bench_gpu` produce the
figures.

## CPU

One thread, `cargo run --release --example bench`, minimum of seven runs of 2^24 elements,
in GiB/s of output. A `next_f64` draw counts 8 bytes. Every Apple M4 Pro figure on this page
comes from one session and is the median of three runs.

| Apple M4 Pro | default | `simd-intrinsics` |
|---|---|---|
| `fill_u32` | 19.0 | 18.9 |
| `fill_u64` | 19.0 | 18.8 |
| `fill_f32` | 15.5 | 16.5 |
| `fill_f64` | 15.5 | 16.5 |
| `fill_exponential_f32` | 6.4 | 6.7 |
| `fill_exponential_f64` | 6.0 | 6.0 |
| `rand_distr::Exp1` `f32`, `StdRng` | 0.86 | 0.86 |
| `rand_distr::Exp1` `f64`, `StdRng` | 1.8 | 1.8 |
| `next_f64` chain | 6.7 | 6.6 |

| AMD EPYC 7702P | default | `simd-intrinsics`, SSE2 | `simd-intrinsics`, AVX2 |
|---|---|---|---|
| `fill_u32` | 5.6 | 5.8 | 11.4 |
| `fill_u64` | 5.7 | 5.7 | 11.8 |
| `fill_f32` | 5.0 | 5.2 | 9.0 |
| `fill_f64` | 3.8 | 4.4 | 7.2 |
| `fill_exponential_f32` | 0.27 | 0.27 | 4.2 |
| `fill_exponential_f64` | 0.23 | 0.33 | 3.0 |
| `rand_distr::Exp1` `f32`, `StdRng` | 0.64 | 0.65 | 0.66 |
| `rand_distr::Exp1` `f64`, `StdRng` | 1.3 | 1.3 | 1.3 |

The SSE2 column is the AVX2 build with `TANDEM_NO_AVX2` set. Without AVX2 and FMA the
exponentials and normals fall behind `Exp1`.

The `rand` distributions, `cargo run --release --features rand --example bench_distr`, one
thread, 2^22 elements, in GiB/s of output, each run the minimum of seven. The inherent column loops over the scalar method, the `Distribution` column over
`rng.sample`. Both inline into the caller, so they cost the same. The last column is the
`rand_distr` distribution, or `random_range(0..1000)`, on `rand`'s default generator, in the
same session. The `f32` normals of Tandem are pairs.

| Apple M4 Pro | inherent | `Distribution` | fill | `rand_distr`, `StdRng` |
|---|---|---|---|---|
| normal `f64` | 4.27 | 4.26 | 7.47 | 1.87 |
| normal `f64`, `sample_iter` | | 4.27 | | |
| normal `f32` | 1.67 | 1.67 | 4.63 | 0.94 |
| exponential `f64` | 3.15 | 3.15 | 6.00 | 1.80 |
| `Below(1000u32)` | 3.99 | 3.99 | 5.42 | 2.08 |

With `rayon`, `cargo run --release --features rayon --example bench_par`, 2^25 elements on
14 threads of an Apple M4 Pro, in GiB/s of output.

| Apple M4 Pro, 14 threads | serial | parallel |
|---|---|---|
| `fill_u32` | 19.0 | 114 |
| `fill_f64` | 15.5 | 115 |
| `fill_below_u32`, n = 1000 | 5.4 | 46 |
| `fill_below_u64`, n = 1000 | 10.9 | 56 |
| `fill_normal_f64` | 7.3 | 72 |
| `fill_normal_f32` | 4.6 | 46 |

The AVX2 column and the SSE2 column come from one session, except that the exponential
and `Exp1` rows come from a later one. The `Exp1` rows draw one sample per element from
`rand`'s default generator. Without AVX2 and FMA the x86_64 target has no fused instruction, so
each `mul_add` of the exponentials and normals is a library call, which is why those columns
fall behind `Exp1`. The default and SSE2 columns of the EPYC table use no AVX2.

## GPU

GPU fill into device memory, `cargo run --release --features wgpu --example bench_gpu
[log2 words]`, minimum of seven. `TANDEM_GPU_ADAPTER=<index>` picks the adapter.

| | words | one fill per submit | 32 fills per submit |
|---|---|---|---|
| Apple M4 Pro, Metal | 2^26 | 165 GiB/s | 186 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan | 2^26 | 775 GiB/s | 1040 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan | 2^28 | 1100 GiB/s | 1215 GiB/s |

GPU fill: the minimum is of seven runs after a half-second warm-up, with the GPU idle on the
A100 host. The Apple fill is bound by the GPU's integer throughput, not by memory. On the A100
the fill with direct 16-byte stores runs near the card's bandwidth once the buffer is large
enough to hide the submit and clock ramp. The A100 host had no system Vulkan loader: a
conda-forge `libvulkan-loader` on `LD_LIBRARY_PATH` with the driver's own ICD was enough.
