//! The exponential fills are bit identical to tandem-c on every target: an FNV-1a hash of 1e6
//! `f64` and 1e6 `f32` exponentials from five positions equals the value in tandem-c's
//! tests/test_exponential_bits.c, and `cargo run --example dump_exponentials` writes the same
//! bytes as its tools/dump_exponentials.c. Without `std` there is no fused multiply-add and
//! the last bits differ.
#![cfg(feature = "std")]

use tandem_rng::Tandem;

const EXPECTED_HASH: u64 = 0x47f8_f982_97d9_4ee2;
const N: usize = 1_000_000;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[test]
fn exponential_fills_hash_like_tandem_c() {
    let mut h = 0xcbf2_9ce4_8422_2325;
    let mut d = vec![0f64; N];
    let mut f = vec![0f32; N];
    for start in [0u64, 1, 77, 12345, 1 << 30] {
        let mut rng = Tandem::new(2026 | 7 << 64);
        rng.set_position(start);
        rng.fill_exponential_f64(&mut d);
        for x in &d {
            h = fnv(h, &x.to_le_bytes());
        }
        rng.fill_exponential_f32(&mut f);
        for x in &f {
            h = fnv(h, &x.to_le_bytes());
        }
    }
    assert_eq!(h, EXPECTED_HASH, "hash {h:016x}");
}
