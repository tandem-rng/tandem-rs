//! Cost of the `rand` distributions against the inherent scalar draws and fills, one thread.
//! Run: cargo run --release --features rand --example bench_distr

use std::hint::black_box;
use std::time::Instant;

use rand::RngExt;
use rand::distr::Distribution;
use tandem_rng::{Below, Exp1, StandardNormal, Tandem};

const N: usize = 1 << 22;

/// Nanoseconds per element, minimum of seven runs.
fn best(mut body: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t0 = Instant::now();
        body();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best * 1e9 / N as f64
}

fn main() {
    let mut rng = Tandem::new(42);
    let mut f64s = vec![0f64; N];
    let mut f32s = vec![[0f32; 2]; N / 2];
    let mut flat = vec![0f32; N];
    let mut u32s = vec![0u32; N];

    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 0.5 {
        rng.fill_normal_f64(&mut f64s);
    }

    let rows: [(&str, f64, f64, f64); 4] = [
        (
            "normal f64",
            best(|| {
                f64s.iter_mut().for_each(|x| *x = rng.normal_f64());
                black_box(&mut f64s);
            }),
            best(|| {
                f64s.iter_mut()
                    .for_each(|x| *x = rng.sample(StandardNormal));
                black_box(&mut f64s);
            }),
            best(|| rng.fill_normal_f64(black_box(&mut f64s))),
        ),
        (
            "normal f32 pairs",
            best(|| {
                f32s.iter_mut().for_each(|x| *x = rng.normal2_f32());
                black_box(&mut f32s);
            }),
            best(|| {
                f32s.iter_mut()
                    .for_each(|x| *x = rng.sample(StandardNormal));
                black_box(&mut f32s);
            }),
            best(|| rng.fill_normal_f32(black_box(&mut flat))),
        ),
        (
            "exponential f64",
            best(|| {
                f64s.iter_mut().for_each(|x| *x = rng.exponential_f64());
                black_box(&mut f64s);
            }),
            best(|| {
                f64s.iter_mut().for_each(|x| *x = rng.sample(Exp1));
                black_box(&mut f64s);
            }),
            best(|| rng.fill_exponential_f64(black_box(&mut f64s))),
        ),
        (
            "below u32, n = 1000",
            best(|| {
                u32s.iter_mut().for_each(|x| *x = rng.below_u32(1000));
                black_box(&mut u32s);
            }),
            best(|| {
                u32s.iter_mut()
                    .for_each(|x| *x = rng.sample(Below(1000u32)));
                black_box(&mut u32s);
            }),
            best(|| rng.fill_below_u32(black_box(&mut u32s), 1000)),
        ),
    ];
    let iter = best(|| {
        for (x, z) in f64s.iter_mut().zip(StandardNormal.sample_iter(&mut rng)) {
            *x = z;
        }
        black_box(&mut f64s);
    });
    println!("ns per element      inherent  Distribution  fill");
    for (label, inherent, distr, fill) in rows {
        println!("{label:<20} {inherent:8.2} {distr:12.2} {fill:7.2}");
    }
    println!("normal f64, sample_iter {iter:.2}");
}
