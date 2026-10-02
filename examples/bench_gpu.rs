//! Throughput of the GPU fill into device memory.
//! Run: cargo run --release --features wgpu --example bench_gpu [log2 words, default 26]

use std::time::Instant;

use tandem_rng::Tandem;
use tandem_rng::gpu::GpuFill;

fn main() {
    let Some(gpu) = GpuFill::new_default() else {
        eprintln!("no wgpu adapter");
        return;
    };
    println!("adapter: {}", gpu.adapter());
    let log2n: u32 = std::env::args()
        .nth(1)
        .map_or(26, |a| a.parse().expect("log2 of the word count"));
    let n = 1usize << log2n;
    let mut rng = Tandem::new(42);
    let out = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench"),
        size: GpuFill::buffer_size(&rng, 32, n),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let wait = || {
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait");
    };

    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < 0.5 {
        gpu.fill_words(&mut rng, &out, n);
        wait();
    }

    for (label, per_submit) in [("one fill per submit", 1), ("32 fills per submit", 32)] {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let t0 = Instant::now();
            for _ in 0..per_submit {
                gpu.fill_words(&mut rng, &out, n);
            }
            wait();
            best = best.min(t0.elapsed().as_secs_f64() / per_submit as f64);
        }
        println!(
            "{label:<24} 2^{log2n} words {:8.1} GiB/s",
            (n * 4) as f64 / best / (1u64 << 30) as f64
        );
    }
}
