//! Every fill under `simd-intrinsics` against the stream built from `block`, which runs the
//! scalar step and not the vector path, at offsets and lengths that cut rows and chunks.

#![cfg(feature = "simd-intrinsics")]

use tandem_rng::{Tandem, block};

const KEY: [u32; 4] = [0x0123_4567, 0x89ab_cdef, 0xdead_beef, 0x0bad_f00d];
const ROWS: u64 = 128;

/// The first `ROWS` rows of the stream as little-endian bytes.
fn stream(k: u32) -> Vec<u8> {
    let k = u64::from(k);
    (0..ROWS)
        .flat_map(|row| (0..8).map(move |l| block(&KEY, 8 * (row / k) + l, (row % k) as u32)))
        .flatten()
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// Element `i` of `w`-bit draws from element `start` on, as raw little-endian bits.
fn raw(bytes: &[u8], w: usize, start: usize, i: usize) -> u128 {
    let at = (start + i) * w / 8;
    let mut b = [0u8; 16];
    b[..w / 8].copy_from_slice(&bytes[at..at + w / 8]);
    u128::from_le_bytes(b)
}

fn f16_bits(raw: u16) -> u16 {
    let k = u32::from(raw >> 5);
    if k == 0 {
        return 0;
    }
    let m = 31 - k.leading_zeros();
    (((m + 4) << 10) | ((k << (10 - m)) & 0x3ff)) as u16
}

fn char_of(raw: u64) -> char {
    let u = ((u128::from(raw) * 1_112_064) >> 64) as u32;
    char::from_u32(if u < 0xd800 { u } else { u + 0x800 }).unwrap()
}

fn check<T: Copy + Default + PartialEq + std::fmt::Debug>(
    name: &str,
    k: u32,
    bytes: &[u8],
    w: usize,
    fill: fn(&mut Tandem, &mut [T]),
    want: impl Fn(u128) -> T,
) {
    for start in [0, 1, 3, 31, 32, 33, 100] {
        for len in [0, 1, 31, 32, 33, 257, 700] {
            let mut rng = Tandem::from_key(KEY, (start * w) as u64, k);
            let mut got = vec![T::default(); len];
            fill(&mut rng, &mut got);
            let want: Vec<T> = (0..len).map(|i| want(raw(bytes, w, start, i))).collect();
            assert_eq!(got, want, "{name}, K = {k}, start {start}, length {len}");
            assert_eq!(rng.position(), ((start + len) * w) as u64);
        }
    }
}

#[test]
fn fills_match_the_scalar_stream() {
    for k in [8, 32] {
        let b = &stream(k);
        check("u8", k, b, 8, Tandem::fill_u8, |r| r as u8);
        check("u16", k, b, 16, Tandem::fill_u16, |r| r as u16);
        check("u32", k, b, 32, Tandem::fill_u32, |r| r as u32);
        check("u64", k, b, 64, Tandem::fill_u64, |r| r as u64);
        check("u128", k, b, 128, Tandem::fill_u128, |r| r);
        check("f32", k, b, 32, Tandem::fill_f32, |r| {
            (r as u32 >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
        });
        check("f64", k, b, 64, Tandem::fill_f64, |r| {
            (r as u64 >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        });
        check("f16", k, b, 16, Tandem::fill_f16_bits, |r| {
            f16_bits(r as u16)
        });
        check("char", k, b, 64, Tandem::fill_char, |r| char_of(r as u64));
    }
}

#[test]
fn bool_fill_matches_the_scalar_stream() {
    for k in [8, 32] {
        let b = stream(k);
        for start in [0, 1, 7, 1023, 1024, 1025] {
            for len in [0, 1, 63, 1024, 2100] {
                let mut rng = Tandem::from_key(KEY, start as u64, k);
                let mut got = vec![false; len];
                rng.fill_bool(&mut got);
                let want: Vec<bool> = (start..start + len)
                    .map(|p| b[p / 8] >> (p % 8) & 1 == 1)
                    .collect();
                assert_eq!(got, want, "bool, K = {k}, start {start}, length {len}");
            }
        }
    }
}
