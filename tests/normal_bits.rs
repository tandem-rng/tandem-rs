//! The normal fills are bit identical to tandem-c on every target: an FNV-1a hash of 1e6 pairs
//! of f64 and f32 normals from five positions equals the value in tandem-c's
//! tests/test_normal_bits.c, and `cargo run --example dump_normals` writes the same bytes as
//! its tools/dump_normals.c. Without `std` there is no fused multiply-add and the last bits
//! differ.
#![cfg(feature = "std")]

use tandem_rng::Tandem;

const EXPECTED_HASH: u64 = 0x9414_e131_5e26_53be;
const PAIRS: usize = 1_000_000;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[test]
fn normal_fills_hash_like_tandem_c() {
    let mut h = 0xcbf2_9ce4_8422_2325;
    let mut d = vec![0f64; 2 * PAIRS - 1];
    let mut f = vec![0f32; 2 * PAIRS - 1];
    for start in [0u64, 1, 77, 12345, 1 << 30] {
        let mut rng = Tandem::new(2026 | 7 << 64);
        rng.set_position(start);
        rng.fill_normal_f64(&mut d);
        for x in &d {
            h = fnv(h, &x.to_le_bytes());
        }
        rng.fill_normal_f32(&mut f);
        for x in &f {
            h = fnv(h, &x.to_le_bytes());
        }
    }
    assert_eq!(h, EXPECTED_HASH, "hash {h:016x}");
}
