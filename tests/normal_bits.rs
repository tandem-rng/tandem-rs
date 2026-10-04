//! The normal fills are bit identical to tandem-c on every target. FNV-1a hashes of the fills
//! equal the values in tandem-c's tests/test_normal_bits.c:
//! - `f64`: 1e6 normals from each of five positions, the bytes that
//!   `cargo run --example dump_normals` and tandem-c's tools/dump_normals.c write.
//! - `f64` against a Python implementation of Appendix A: 2e5 normals from key {1, 2, 3, 4},
//!   K = 32, at bits 0 and 2373, with their end positions.
//! - `f32`: 2e6 - 1 normals from each of the five positions.
//!
//! Without `std` there is no fused multiply-add and the last bits differ.
#![cfg(feature = "std")]

use tandem_rng::Tandem;

const STARTS: [u64; 5] = [0, 1, 77, 12345, 1 << 30];
const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn hash_f64(h: u64, x: &[f64]) -> u64 {
    x.iter().fold(h, |h, x| fnv(h, &x.to_le_bytes()))
}

#[test]
fn f64_fills_hash_like_tandem_c() {
    let mut h = FNV_BASIS;
    let mut d = vec![0f64; 1_000_000];
    for start in STARTS {
        let mut rng = Tandem::new(2026 | 7 << 64);
        rng.set_position(start);
        rng.fill_normal_f64(&mut d);
        h = hash_f64(h, &d);
    }
    assert_eq!(h, 0xa61c_fa84_4c85_f7c1, "hash {h:016x}");
}

#[test]
fn f64_fills_hash_like_the_python_reference() {
    let mut d = vec![0f64; 200_000];
    for (start, want, end) in [
        (0, 0x0c40_59ed_409d_578d, 12_800_000),
        (2373, 0x30ce_40c8_6b29_5193, 12_802_432),
    ] {
        let mut rng = Tandem::from_key([1, 2, 3, 4], start, 32);
        rng.fill_normal_f64(&mut d);
        let h = hash_f64(FNV_BASIS, &d);
        assert_eq!(h, want, "hash {h:016x} at {start}");
        assert_eq!(rng.position(), end);
    }
}

#[test]
fn f32_fills_hash_like_tandem_c() {
    let mut h = FNV_BASIS;
    let mut f = vec![0f32; 1_999_999];
    for start in STARTS {
        let mut rng = Tandem::new(2026 | 7 << 64);
        rng.set_position(start);
        rng.fill_normal_f32(&mut f);
        h = f.iter().fold(h, |h, x| fnv(h, &x.to_le_bytes()));
    }
    assert_eq!(h, 0xaa1e_a656_ce73_a4fb, "hash {h:016x}");
}
