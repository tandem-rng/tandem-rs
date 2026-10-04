//! Serial against parallel fills on all threads. Run: cargo run --release --features rayon --example bench_par

use std::hint::black_box;
use std::time::Instant;

use tandem_rng::Tandem;

const N: usize = 1 << 25;

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
    let mut u32s = vec![0u32; N];
    let mut u64s = vec![0u64; N];
    let mut f64s = vec![0f64; N];
    println!("threads: {}", rayon::current_num_threads());
    // Touch every page and wake the thread pool before timing.
    for _ in 0..3 {
        rng.par_fill_u32(&mut u32s);
        rng.par_fill_u64(&mut u64s);
        rng.par_fill_f64(&mut f64s);
    }
    let rows = [
        (
            "u32",
            best(N * 4, || rng.fill_u32(black_box(&mut u32s))),
            best(N * 4, || rng.par_fill_u32(black_box(&mut u32s))),
        ),
        (
            "f64",
            best(N * 8, || rng.fill_f64(black_box(&mut f64s))),
            best(N * 8, || rng.par_fill_f64(black_box(&mut f64s))),
        ),
        (
            "below_u32 (n = 1000)",
            best(N * 4, || rng.fill_below_u32(black_box(&mut u32s), 1000)),
            best(N * 4, || rng.par_fill_below_u32(black_box(&mut u32s), 1000)),
        ),
        (
            "below_u64 (n = 1000)",
            best(N * 8, || rng.fill_below_u64(black_box(&mut u64s), 1000)),
            best(N * 8, || rng.par_fill_below_u64(black_box(&mut u64s), 1000)),
        ),
        (
            "normal_f64",
            best(N * 8, || rng.fill_normal_f64(black_box(&mut f64s))),
            best(N * 8, || rng.par_fill_normal_f64(black_box(&mut f64s))),
        ),
    ];
    println!("{:<24} {:>10} {:>10}", "", "serial", "parallel");
    for (label, serial, par) in rows {
        println!("{label:<24} {serial:>10.2} {par:>10.2}  GiB/s");
    }
}
