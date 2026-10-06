//! Serial against parallel fills on all threads, for Tandem and for `SmallRng` (xoshiro256++).
//! Run: cargo run --release --features rayon --example bench_par

use std::hint::black_box;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use tandem_rng::Tandem;

const N: usize = 1 << 25;
/// Elements per `SmallRng` in the parallel baseline, each seeded by its chunk index.
const CHUNK: usize = 1 << 16;

fn best(bytes: usize, mut body: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t0 = Instant::now();
        body();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    bytes as f64 / best / (1u64 << 30) as f64
}

/// Serial and parallel GiB/s of `fill` run by `SmallRng`.
fn small<T: Send>(out: &mut [T], fill: impl Fn(&mut SmallRng, &mut [T]) + Sync) -> (f64, f64) {
    let bytes = std::mem::size_of_val(out);
    let mut rng = SmallRng::seed_from_u64(42);
    let serial = best(bytes, || fill(&mut rng, black_box(&mut *out)));
    let par = best(bytes, || {
        black_box(&mut *out)
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(i, c)| fill(&mut SmallRng::seed_from_u64(i as u64), c));
    });
    (serial, par)
}

fn main() {
    let mut rng = Tandem::new(42);
    let mut u32s = vec![0u32; N];
    let mut u64s = vec![0u64; N];
    let mut f64s = vec![0f64; N];
    let mut f32s = vec![0f32; N];
    println!("threads: {}", rayon::current_num_threads());
    // Touch every page and wake the thread pool before timing.
    for _ in 0..3 {
        rng.par_fill_u32(&mut u32s);
        rng.par_fill_u64(&mut u64s);
        rng.par_fill_f64(&mut f64s);
        rng.par_fill_f32(&mut f32s);
    }
    let rows = [
        (
            "u32",
            best(N * 4, || rng.fill_u32(black_box(&mut u32s))),
            best(N * 4, || rng.par_fill_u32(black_box(&mut u32s))),
            small(&mut u32s, |r, c| r.fill(c)),
        ),
        (
            "f64",
            best(N * 8, || rng.fill_f64(black_box(&mut f64s))),
            best(N * 8, || rng.par_fill_f64(black_box(&mut f64s))),
            small(&mut f64s, |r, c| c.iter_mut().for_each(|x| *x = r.random())),
        ),
        (
            "below_u32 (n = 1000)",
            best(N * 4, || rng.fill_below_u32(black_box(&mut u32s), 1000)),
            best(N * 4, || rng.par_fill_below_u32(black_box(&mut u32s), 1000)),
            small(&mut u32s, |r, c| {
                c.iter_mut().for_each(|x| *x = r.random_range(0..1000u32))
            }),
        ),
        (
            "below_u64 (n = 1000)",
            best(N * 8, || rng.fill_below_u64(black_box(&mut u64s), 1000)),
            best(N * 8, || rng.par_fill_below_u64(black_box(&mut u64s), 1000)),
            small(&mut u64s, |r, c| {
                c.iter_mut().for_each(|x| *x = r.random_range(0..1000u64))
            }),
        ),
        (
            "normal_f64",
            best(N * 8, || rng.fill_normal_f64(black_box(&mut f64s))),
            best(N * 8, || rng.par_fill_normal_f64(black_box(&mut f64s))),
            small(&mut f64s, |r, c| {
                c.iter_mut()
                    .for_each(|x| *x = r.sample(rand_distr::StandardNormal))
            }),
        ),
        (
            "normal_f32",
            best(N * 4, || rng.fill_normal_f32(black_box(&mut f32s))),
            best(N * 4, || rng.par_fill_normal_f32(black_box(&mut f32s))),
            small(&mut f32s, |r, c| {
                c.iter_mut()
                    .for_each(|x| *x = r.sample(rand_distr::StandardNormal))
            }),
        ),
    ];
    println!(
        "{:<24} {:>10} {:>10} {:>16} {:>16}",
        "", "serial", "parallel", "SmallRng serial", "SmallRng par"
    );
    for (label, serial, par, (s_serial, s_par)) in rows {
        println!("{label:<24} {serial:>10.2} {par:>10.2} {s_serial:>16.2} {s_par:>16.2}  GiB/s");
    }
}
