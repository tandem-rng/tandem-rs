# Speed

`cargo run --release --example bench`, `bench_distr`, `bench_par`, `bench_gpu` and
`bench_curand` produce the figures.

## CPU

One thread, `cargo run --release --example bench`, minimum of seven runs of 2^24 elements,
in GiB/s of output. A `next_f64` draw counts 8 bytes. Every Apple M4 Pro figure on this page
comes from one session and is the median of three runs.

The baselines are `rand`'s `SmallRng` (xoshiro256++) and `StdRng` (ChaCha12): `fill` for the
integers, one `random()` per element for the floats, `rand_distr::Exp1` for the exponentials.
`SmallRng` leads the scalar chain, whose xoshiro256++ step inlines into the loop.

| Apple M4 Pro | default | `simd-intrinsics` | `SmallRng` | `StdRng` |
|---|---|---|---|---|
| `fill_u32` | 19.3 | 19.2 | 10.7 | 2.7 |
| `fill_u64` | 19.2 | 19.2 | 10.6 | 2.7 |
| `fill_f32` | 16.7 | 16.4 | 5.4 | 2.1 |
| `fill_f64` | 16.5 | 16.5 | 10.7 | 2.3 |
| `fill_exponential_f32` | 6.6 | 6.4 | 3.1 | 0.88 |
| `fill_exponential_f64` | 6.0 | 5.9 | 6.0 | 1.8 |
| `next_f64` chain | 6.9 | 6.6 | 10.4 | 2.5 |

| AMD EPYC 7702P | default | `simd-intrinsics`, SSE2 | `simd-intrinsics`, AVX2 | `SmallRng` | `StdRng` |
|---|---|---|---|---|---|
| `fill_u32` | 5.7 | 5.8 | 11.9 | 6.3 | 3.0 |
| `fill_u64` | 5.6 | 5.8 | 11.6 | 6.3 | 3.0 |
| `fill_f32` | 5.0 | 5.2 | 9.0 | 3.0 | 1.9 |
| `fill_f64` | 3.8 | 4.4 | 7.2 | 5.9 | 2.2 |
| `fill_exponential_f32` | 0.27 | 0.28 | 4.2 | 1.4 | 0.64 |
| `fill_exponential_f64` | 0.23 | 0.33 | 3.0 | 2.7 | 1.4 |
| `next_f64` chain | 1.6 | 1.6 | 2.1 | 5.9 | 2.3 |

Every EPYC figure comes from one session on one pinned core and is the median of three runs.
The SSE2 column is the AVX2 build with `TANDEM_NO_AVX2` set. Only the AVX2 build leads
`SmallRng` on the fills.

The `rand` distributions, `cargo run --release --features rand --example bench_distr`, one
thread, 2^22 elements, in GiB/s of output, each run the minimum of seven. The inherent column
loops over the scalar method, the `Distribution` column over `rng.sample`. Both inline into the
caller, so they cost the same. `StandardNormal.sample_iter` writes 4.38. The last two columns are
the `rand_distr` distribution, or `random_range(0..1000)`, on `SmallRng` and `StdRng`. The `f32`
normals of Tandem are pairs.

| Apple M4 Pro | inherent | `Distribution` | fill | `SmallRng` | `StdRng` |
|---|---|---|---|---|---|
| normal `f64` | 4.38 | 4.35 | 7.38 | 6.89 | 1.85 |
| normal `f32` | 1.68 | 1.65 | 4.59 | 3.45 | 0.92 |
| exponential `f64` | 3.13 | 3.13 | 5.89 | 6.02 | 1.76 |
| `Below(1000u32)` | 3.82 | 4.03 | 5.43 | 1.49 | 2.05 |

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

Without AVX2 and FMA the x86_64 target has no fused instruction, so each `mul_add` of the
exponentials and normals is a library call, which is why the default and SSE2 columns of the
EPYC table fall behind `rand_distr::Exp1`. Those two columns use no AVX2.

## GPU

GPU fill into device memory, `cargo run --release --features wgpu --example bench_gpu
[log2 words]`. `TANDEM_GPU_ADAPTER=<index>` picks the adapter.

The baselines are Philox4x32-10 in WGSL with the same `mul_hi` as the Tandem shader, since no
wgpu crate ships a generator, and on the A100 cuRAND's Philox4x32-10 `curandGenerate`,
`cargo run --release --example bench_curand [log2 words]` with cuRAND and a CUDA 12 runtime on
the loader path.

| | words | one fill per submit | 32 fills per submit | Philox, one per submit | Philox, 32 per submit | cuRAND, one per call | cuRAND, 32 per sync |
|---|---|---|---|---|---|---|---|
| Apple M4 Pro, Metal | 2^26 | 137 GiB/s | 153 GiB/s | 120 GiB/s | 128 GiB/s | - | - |
| NVIDIA A100 40 GB PCIe, Vulkan | 2^26 | 769 GiB/s | 1015 GiB/s | 406 GiB/s | 445 GiB/s | 1220 GiB/s | 1258 GiB/s |
| NVIDIA A100 40 GB PCIe, Vulkan | 2^28 | 1066 GiB/s | 1203 GiB/s | 449 GiB/s | 459 GiB/s | 1271 GiB/s | 1293 GiB/s |

GPU fill: every row is the median of 21 runs after a 2 s warm-up per row. The A100 rows come
from one session on GPU 1, idle before the run. The "32 per submit" columns submit 32 fills
back to back and wait once. cuRAND is a native CUDA kernel and leads the Vulkan shader. For CUDA,
tandem-cuda has the Tandem kernels. WGSL has no high multiply, and on the A100 the emulated one holds Philox to
about 450 GiB/s, while Tandem's two multiplies per step leave it near the card's bandwidth.
The Apple fill is bound by the GPU's integer throughput, not by memory. On the A100
the fill with direct 16-byte stores runs near the card's bandwidth once the buffer is large
enough to hide the submit and clock ramp. The A100 host had no system Vulkan loader: a
conda-forge `libvulkan-loader` on `LD_LIBRARY_PATH` with the driver's own ICD was enough.
