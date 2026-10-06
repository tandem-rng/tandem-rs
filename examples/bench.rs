//! Throughput of the sequential fills and the scalar chain against `SmallRng` (xoshiro256++)
//! and `StdRng` (ChaCha12), in GiB/s of output. Run: cargo run --release --example bench

use std::hint::black_box;
use std::time::Instant;

use rand::rngs::{SmallRng, StdRng};
use rand::{Rng, RngExt, SeedableRng};
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

struct Bufs {
    u32s: Vec<u32>,
    u64s: Vec<u64>,
    f32s: Vec<f32>,
    f64s: Vec<f64>,
}

/// The rows of a third-party generator: rand's fills and draws, one sample per element.
fn baseline<R: Rng>(rng: &mut R, b: &mut Bufs) -> [f64; 7] {
    [
        best(N * 4, || rng.fill(black_box(&mut b.u32s[..]))),
        best(N * 8, || rng.fill(black_box(&mut b.u64s[..]))),
        best(N * 4, || {
            b.f32s.iter_mut().for_each(|x| *x = rng.random());
            black_box(&mut b.f32s);
        }),
        best(N * 8, || {
            b.f64s.iter_mut().for_each(|x| *x = rng.random());
            black_box(&mut b.f64s);
        }),
        best(N * 4, || {
            b.f32s.iter_mut().for_each(|x| *x = Exp1.sample(rng));
            black_box(&mut b.f32s);
        }),
        best(N * 8, || {
            b.f64s.iter_mut().for_each(|x| *x = Exp1.sample(rng));
            black_box(&mut b.f64s);
        }),
        // A dependent chain of scalar draws, 8 bytes of output each.
        best(N * 8, || {
            let mut acc = 0.0;
            for _ in 0..N {
                acc += rng.random::<f64>();
            }
            black_box(acc);
        }),
    ]
}

fn main() {
    let mut rng = Tandem::new(42);
    let mut b = Bufs {
        u32s: vec![0u32; N],
        u64s: vec![0u64; N],
        f32s: vec![0f32; N],
        f64s: vec![0f64; N],
    };

    // Half a second of work so the clock has ramped up; the vectors are touched by the fills.
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 0.5 {
        rng.fill_u32(&mut b.u32s);
    }

    let tandem = [
        best(N * 4, || rng.fill_u32(black_box(&mut b.u32s))),
        best(N * 8, || rng.fill_u64(black_box(&mut b.u64s))),
        best(N * 4, || rng.fill_f32(black_box(&mut b.f32s))),
        best(N * 8, || rng.fill_f64(black_box(&mut b.f64s))),
        best(N * 4, || rng.fill_exponential_f32(black_box(&mut b.f32s))),
        best(N * 8, || rng.fill_exponential_f64(black_box(&mut b.f64s))),
        best(N * 8, || {
            let mut acc = 0.0;
            for _ in 0..N {
                acc += rng.next_f64();
            }
            black_box(acc);
        }),
    ];
    let small = baseline(&mut SmallRng::seed_from_u64(42), &mut b);
    let std = baseline(&mut StdRng::seed_from_u64(42), &mut b);
    let labels = [
        "fill_u32",
        "fill_u64",
        "fill_f32",
        "fill_f64",
        "fill_exponential_f32",
        "fill_exponential_f64",
        "next_f64 chain",
    ];
    println!("GiB/s                        Tandem  SmallRng    StdRng");
    for i in 0..labels.len() {
        println!(
            "{:<26} {:8.2} {:9.2} {:9.2}",
            labels[i], tandem[i], small[i], std[i]
        );
    }
}
