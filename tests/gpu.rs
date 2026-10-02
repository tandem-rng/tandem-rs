//! The GPU fill reads the same stream as the CPU fills. Skips without an adapter.
#![cfg(feature = "wgpu")]

use std::fs;

use tandem_rng::Tandem;
use tandem_rng::gpu::GpuFill;

fn gpu() -> Option<GpuFill> {
    let g = GpuFill::new_default();
    if g.is_none() {
        eprintln!("no wgpu adapter: skipping");
    }
    g
}

fn dump(name: &str) -> Vec<u8> {
    fs::read(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).expect(name)
}

#[test]
fn words_equal_cpu_fill() {
    let Some(gpu) = gpu() else { return };
    for key in [[1, 2, 3, 4], [0xdead_beef, 7, 0, 0xffff_ffff]] {
        for k in [1, 8, 32, 128] {
            for pos in [0u64, 32, 96, 1000, 1024, 4096 + 64, 1 << 20] {
                for n in [0usize, 1, 7, 33, 1000, 70_000] {
                    let mut a = Tandem::from_key(key, pos, k);
                    let mut b = a;
                    let got = gpu.read_u32(&mut a, n);
                    let mut want = vec![0u32; n];
                    b.fill_u32(&mut want);
                    assert_eq!(got, want, "key {key:?} K {k} pos {pos} n {n}");
                    assert_eq!(a.position(), b.position(), "position after the fill");
                }
            }
        }
    }
}

#[test]
fn wider_types_equal_cpu_fill() {
    let Some(gpu) = gpu() else { return };
    for pos in [0u64, 8, 40, 100, 1 << 15] {
        let mut a = Tandem::new(99);
        a.set_position(pos);
        let mut b = a;
        let mut want64 = vec![0u64; 5000];
        b.fill_u64(&mut want64);
        assert_eq!(gpu.read_u64(&mut a, 5000), want64);
        let mut want_f32 = vec![0f32; 777];
        b.fill_f32(&mut want_f32);
        assert_eq!(gpu.read_f32(&mut a, 777), want_f32);
        let mut want_f64 = vec![0f64; 777];
        b.fill_f64(&mut want_f64);
        assert_eq!(gpu.read_f64(&mut a, 777), want_f64);
        assert_eq!(a.position(), b.position());
    }
}

#[test]
fn mixed_cpu_and_gpu_draws_share_the_stream() {
    let Some(gpu) = gpu() else { return };
    let mut a = Tandem::new(5);
    let mut b = a;
    let mut want = vec![0u32; 300];
    b.fill_u32(&mut want);
    let mut got = gpu.read_u32(&mut a, 100);
    let mut mid = vec![0u32; 100];
    a.fill_u32(&mut mid);
    got.extend(mid);
    got.extend(gpu.read_u32(&mut a, 100));
    assert_eq!(got, want);
}

#[test]
fn stream_dumps() {
    let Some(gpu) = gpu() else { return };
    let key = [1, 2, 3, 4];
    let want: Vec<u32> = dump("k1234_K32_u32.bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(
        gpu.read_u32(&mut Tandem::from_key(key, 0, 32), want.len()),
        want
    );
    let want: Vec<u32> = dump("k1234_K8_u32.bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(
        gpu.read_u32(&mut Tandem::from_key(key, 0, 8), want.len()),
        want
    );
    let want: Vec<u64> = dump("k1234_K32_u64.bin")
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    assert_eq!(
        gpu.read_u64(&mut Tandem::from_key(key, 0, 32), want.len()),
        want
    );
    let want: Vec<f32> = dump("seed42_K32_f32.bin")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(gpu.read_f32(&mut Tandem::new(42), want.len()), want);
    let want: Vec<f64> = dump("seed42_K32_f64.bin")
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    assert_eq!(gpu.read_f64(&mut Tandem::new(42), want.len()), want);
}

#[test]
fn buffer_size_covers_the_span() {
    let rng = Tandem::from_key([1, 2, 3, 4], 100, 32);
    // Position 100 aligns to 128 for u64. Ten values end at bit 768: blocks 1 to 5.
    assert_eq!(GpuFill::buffer_size(&rng, 64, 10), 5 * 16);
    assert_eq!(GpuFill::buffer_size(&rng, 32, 0), 16);
}
