//! Throughput of the `rand` distributions against the inherent scalar draws and fills, and of
//! `rand_distr` on `StdRng`, one thread, in GiB/s of output.
//! Run: cargo run --release --features rand --example bench_distr

use std::hint::black_box;
use std::time::Instant;

use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use tandem_rng::{Below, Exp1, StandardNormal, Tandem};

const N: usize = 1 << 22;

/// GiB/s of `bytes` of output, minimum of seven runs.
fn best(bytes: usize, mut body: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t0 = Instant::now();
        body();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    bytes as f64 / best / (1u64 << 30) as f64
}

fn main() {
    let mut rng = Tandem::new(42);
    let mut std_rng = StdRng::seed_from_u64(42);
    let mut f64s = vec![0f64; N];
    let mut pairs = vec![[0f32; 2]; N / 2];
    let mut f32s = vec![0f32; N];
    let mut u32s = vec![0u32; N];
    let (b64, b32) = (N * 8, N * 4);

    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 0.5 {
        rng.fill_normal_f64(&mut f64s);
    }

    let rows: [(&str, f64, f64, f64, f64); 4] = [
        (
            "normal f64",
            best(b64, || {
                f64s.iter_mut().for_each(|x| *x = rng.normal_f64());
                black_box(&mut f64s);
            }),
            best(b64, || {
                f64s.iter_mut()
                    .for_each(|x| *x = rng.sample(StandardNormal));
                black_box(&mut f64s);
            }),
            best(b64, || rng.fill_normal_f64(black_box(&mut f64s))),
            best(b64, || {
                f64s.iter_mut()
                    .for_each(|x| *x = std_rng.sample(rand_distr::StandardNormal));
                black_box(&mut f64s);
            }),
        ),
        (
            "normal f32",
            best(b32, || {
                pairs.iter_mut().for_each(|x| *x = rng.normal2_f32());
                black_box(&mut pairs);
            }),
            best(b32, || {
                pairs
                    .iter_mut()
                    .for_each(|x| *x = rng.sample(StandardNormal));
                black_box(&mut pairs);
            }),
            best(b32, || rng.fill_normal_f32(black_box(&mut f32s))),
            best(b32, || {
                f32s.iter_mut()
                    .for_each(|x| *x = std_rng.sample(rand_distr::StandardNormal));
                black_box(&mut f32s);
            }),
        ),
        (
            "exponential f64",
            best(b64, || {
                f64s.iter_mut().for_each(|x| *x = rng.exponential_f64());
                black_box(&mut f64s);
            }),
            best(b64, || {
                f64s.iter_mut().for_each(|x| *x = rng.sample(Exp1));
                black_box(&mut f64s);
            }),
            best(b64, || rng.fill_exponential_f64(black_box(&mut f64s))),
            best(b64, || {
                f64s.iter_mut()
                    .for_each(|x| *x = std_rng.sample(rand_distr::Exp1));
                black_box(&mut f64s);
            }),
        ),
        (
            "below u32, n = 1000",
            best(b32, || {
                u32s.iter_mut().for_each(|x| *x = rng.below_u32(1000));
                black_box(&mut u32s);
            }),
            best(b32, || {
                u32s.iter_mut()
                    .for_each(|x| *x = rng.sample(Below(1000u32)));
                black_box(&mut u32s);
            }),
            best(b32, || rng.fill_below_u32(black_box(&mut u32s), 1000)),
            best(b32, || {
                u32s.iter_mut()
                    .for_each(|x| *x = std_rng.random_range(0..1000u32));
                black_box(&mut u32s);
            }),
        ),
    ];
    let iter = best(b64, || {
        for (x, z) in f64s.iter_mut().zip(StandardNormal.sample_iter(&mut rng)) {
            *x = z;
        }
        black_box(&mut f64s);
    });
    println!("GiB/s                inherent  Distribution   fill  rand_distr StdRng");
    for (label, inherent, distr, fill, base) in rows {
        println!("{label:<20} {inherent:8.2} {distr:12.2} {fill:6.2} {base:10.2}");
    }
    println!("normal f64, sample_iter {iter:.2}");
}
