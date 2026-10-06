//! Throughput of the GPU fill into device memory, against a plain Philox4x32-10 shader.
//! Run: cargo run --release --features wgpu --example bench_gpu [log2 words, default 26]

use std::time::Instant;

use tandem_rng::Tandem;
use tandem_rng::gpu::GpuFill;

/// The baseline: Philox4x32-10 of Salmon et al. (2011) with the 16-bit-half `mul_hi` of the
/// Tandem shader, since no wgpu crate ships a generator. Invocation t writes blocks t,
/// t + stride, ..., block b from counter (b, 0, 0, 0) under key (42, 0).
const PHILOX: &str = r"@group(0) @binding(0) var<storage, read_write> out: array<vec4<u32>>;
fn mul_hi(a: u32, b: u32) -> u32 {
  let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u;
  let lh = al * bh; let hl = ah * bl;
  let mid = ((al * bl) >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
  return ah * bh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
}
@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  for (var b = id.x; b < arrayLength(&out); b += nw.x * 256u) {
    var c = vec4<u32>(b, 0u, 0u, 0u);
    var k = vec2<u32>(42u, 0u);
    for (var r = 0u; r < 10u; r++) {
      c = vec4<u32>(mul_hi(0xcd9e8d57u, c.z) ^ c.y ^ k.x, 0xcd9e8d57u * c.z,
                    mul_hi(0xd2511f53u, c.x) ^ c.w ^ k.y, 0xd2511f53u * c.x);
      k += vec2<u32>(0x9e3779b9u, 0xbb67ae85u);
    }
    out[b] = c;
  }
}";

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
    let device = gpu.device();
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench"),
        size: GpuFill::buffer_size(&rng, 32, n),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let wait = || {
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait");
    };

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("philox"),
        source: wgpu::ShaderSource::Wgsl(PHILOX.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("philox"),
        layout: None,
        module: &module,
        entry_point: Some("fill"),
        compilation_options: Default::default(),
        cache: None,
    });
    // The Tandem buffer may hold a block more than n words, so bind exactly n words.
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("philox"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &out,
                offset: 0,
                size: wgpu::BufferSize::new(4 * n as u64),
            }),
        }],
    });
    // One submit per fill, as GpuFill does.
    let philox = || {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(((n / 4).div_ceil(256)).min(65535) as u32, 1, 1);
        }
        gpu.queue().submit([encoder.finish()]);
    };

    // Each row warms its own fill for 2 s, so the GPU clock reaches its steady state, then
    // reports the median and the fastest of 21 runs.
    let measure = |fills: usize, fill: &mut dyn FnMut()| {
        let mut run = || {
            let t0 = Instant::now();
            for _ in 0..fills {
                fill();
            }
            wait();
            t0.elapsed().as_secs_f64() / fills as f64
        };
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < 2.0 {
            run();
        }
        let mut t: Vec<f64> = (0..21).map(|_| run()).collect();
        t.sort_by(f64::total_cmp);
        (t[10], t[0])
    };
    let gibs = |seconds: f64| (n * 4) as f64 / seconds / (1u64 << 30) as f64;
    for (label, fills) in [("one fill per submit", 1), ("32 fills back to back", 32)] {
        let ours = measure(fills, &mut || {
            gpu.fill_words(&mut rng, &out, n);
        });
        let theirs = measure(fills, &mut || philox());
        println!(
            "{label:<24} 2^{log2n} words, median (fastest) GiB/s: Tandem {:7.1} ({:7.1}), Philox4x32-10 {:7.1} ({:7.1})",
            gibs(ours.0),
            gibs(ours.1),
            gibs(theirs.0),
            gibs(theirs.1)
        );
    }
}
