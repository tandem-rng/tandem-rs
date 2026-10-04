//! Throughput of the sequential fills and the scalar chain. Run: cargo run --release --example bench

use std::hint::black_box;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp1};
use tandem_rng::Tandem;

const N: usize = 1 << 24;

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
    let mut u32s = vec![0u32; N];
    let mut u64s = vec![0u64; N];
    let mut f32s = vec![0f32; N];
    let mut f64s = vec![0f64; N];

    // Half a second of work so the clock has ramped up; the vectors are touched by the fills.
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 0.5 {
        rng.fill_u32(&mut u32s);
    }

    let rows = [
        (
            "fill_u32",
            best(N * 4, || rng.fill_u32(black_box(&mut u32s))),
        ),
        (
            "fill_u64",
            best(N * 8, || rng.fill_u64(black_box(&mut u64s))),
        ),
        (
            "fill_f32",
            best(N * 4, || rng.fill_f32(black_box(&mut f32s))),
        ),
        (
            "fill_f64",
            best(N * 8, || rng.fill_f64(black_box(&mut f64s))),
        ),
        (
            "fill_exponential_f32",
            best(N * 4, || rng.fill_exponential_f32(black_box(&mut f32s))),
        ),
        (
            "fill_exponential_f64",
            best(N * 8, || rng.fill_exponential_f64(black_box(&mut f64s))),
        ),
        // rand's default generator and its ziggurat, one sample per element.
        (
            "rand_distr Exp1 f32, StdRng",
            best(N * 4, || {
                f32s.iter_mut().for_each(|x| *x = Exp1.sample(&mut std_rng));
                black_box(&mut f32s);
            }),
        ),
        (
            "rand_distr Exp1 f64, StdRng",
            best(N * 8, || {
                f64s.iter_mut().for_each(|x| *x = Exp1.sample(&mut std_rng));
                black_box(&mut f64s);
            }),
        ),
    ];
    for (label, gibs) in rows {
        println!("{label:<28} {gibs:8.2} GiB/s");
    }
    let mut best_ns = f64::INFINITY;
    for _ in 0..7 {
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..N {
            acc += rng.next_f64();
        }
        black_box(acc);
        best_ns = best_ns.min(t0.elapsed().as_secs_f64() * 1e9 / N as f64);
    }
    println!("{:<28} {best_ns:8.2} ns per draw", "chain next_f64");
}
