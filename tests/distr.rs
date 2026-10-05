//! The `rand` distributions equal the inherent draws on the tandem-c fixtures and in the dump
//! hashes, and `sample_iter` equals the fills across their passes.
#![cfg(feature = "rand")]

#[allow(clippy::excessive_precision, dead_code)]
mod derived_data;

use derived_data::{
    BELOW_U32, BELOW_U64, EXPONENTIAL_F32, EXPONENTIAL_F64, NORMAL_F32, NORMAL_F64,
};
use rand::distr::Distribution;
use rand::{Rng, RngExt, SeedableRng};
use tandem_rng::{Below, Exp1, StandardNormal, Tandem};

fn at(pos: u64) -> Tandem {
    let mut rng = Tandem::new(42);
    rng.set_position(pos);
    rng
}

/// The fixtures of `below`, `NORMAL_F32` start after one bit draw.
fn start() -> Tandem {
    let mut rng = Tandem::new(42);
    rng.next_bool();
    rng
}

fn bits64(x: &[f64]) -> Vec<u64> {
    x.iter().map(|x| x.to_bits()).collect()
}

fn bits32(x: &[f32]) -> Vec<u32> {
    x.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn normal_f64_is_the_inherent_draw_on_the_fixtures() {
    for (pos, want, end) in NORMAL_F64 {
        let n = want.len();
        let mut inherent = at(*pos);
        let scalar: Vec<f64> = (0..n).map(|_| inherent.normal_f64()).collect();
        let mut rng = at(*pos);
        let got: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
        assert_eq!(bits64(&got), bits64(&scalar), "sample at {pos}");
        assert_eq!(rng, inherent);
        assert_eq!(rng.position(), *end);
        let mut rng = at(*pos);
        let got: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(n).collect();
        assert_eq!(bits64(&got), bits64(&scalar), "sample_iter at {pos}");
        assert_eq!(rng, inherent);
        if cfg!(feature = "std") {
            assert_eq!(bits64(&got), bits64(want), "fixture at {pos}");
        }
    }
}

#[test]
fn normal_f32_is_the_inherent_draw_on_the_fixtures() {
    let (want, end) = NORMAL_F32;
    let mut inherent = start();
    let pairs: Vec<[f32; 2]> = (0..want.len() / 2)
        .map(|_| inherent.normal2_f32())
        .collect();
    let mut rng = start();
    let got: Vec<[f32; 2]> = StandardNormal
        .sample_iter(&mut rng)
        .take(pairs.len())
        .collect();
    assert_eq!(
        got.iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        pairs
            .iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!((rng, rng.position()), (inherent, end));
    if cfg!(feature = "std") {
        let flat: Vec<f32> = got.into_iter().flatten().collect();
        assert_eq!(bits32(&flat), bits32(want));
    }
    let (mut a, mut b) = (start(), start());
    for _ in 0..want.len() {
        let z: f32 = a.sample(StandardNormal);
        assert_eq!(z.to_bits(), b.normal_f32().to_bits());
    }
    assert_eq!(a, b);
}

#[test]
fn exponentials_are_the_inherent_draws_on_the_fixtures() {
    for (pos, want, end) in EXPONENTIAL_F64 {
        let (mut rng, mut inherent) = (at(*pos), at(*pos));
        let got: Vec<f64> = Exp1.sample_iter(&mut rng).take(want.len()).collect();
        let scalar: Vec<f64> = want.iter().map(|_| inherent.exponential_f64()).collect();
        assert_eq!(bits64(&got), bits64(&scalar), "f64 at {pos}");
        assert_eq!((rng, rng.position()), (inherent, *end));
        if cfg!(feature = "std") {
            assert_eq!(bits64(&got), bits64(want), "f64 fixture at {pos}");
        }
    }
    for (pos, want, end) in EXPONENTIAL_F32 {
        let (mut rng, mut inherent) = (at(*pos), at(*pos));
        let got: Vec<f32> = (0..want.len()).map(|_| rng.sample(Exp1)).collect();
        let scalar: Vec<f32> = want.iter().map(|_| inherent.exponential_f32()).collect();
        assert_eq!(bits32(&got), bits32(&scalar), "f32 at {pos}");
        assert_eq!((rng, rng.position()), (inherent, *end));
        if cfg!(feature = "std") {
            assert_eq!(bits32(&got), bits32(want), "f32 fixture at {pos}");
        }
    }
}

#[test]
fn below_is_the_inherent_draw_on_the_fixtures() {
    for (n, want, end) in BELOW_U32 {
        let mut rng = start();
        let got: Vec<u32> = Below(*n).sample_iter(&mut rng).take(want.len()).collect();
        assert_eq!((&got[..], rng.position()), (*want, *end), "Below({n}u32)");
    }
    for (n, want, end) in BELOW_U64 {
        let mut rng = start();
        let got: Vec<u64> = (0..want.len()).map(|_| rng.sample(Below(*n))).collect();
        assert_eq!((&got[..], rng.position()), (*want, *end), "Below({n}u64)");
    }
}

// The fills run in passes of 512 (f64 normals), 4096 (f32 normals) and 1024 (exponentials).
const LENGTHS: [usize; 9] = [0, 1, 2, 511, 512, 513, 1025, 4097, 9000];

#[test]
fn sample_iter_equals_the_fills_across_passes() {
    for n in LENGTHS {
        let (mut rng, mut fill) = (start(), start());
        let got: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(n).collect();
        let mut want = vec![0.0; n];
        fill.fill_normal_f64(&mut want);
        assert_eq!(bits64(&got), bits64(&want), "f64 normals, {n}");
        // An empty fill aligns the position, an empty iterator draws nothing.
        if n > 0 {
            assert_eq!(rng, fill, "f64 normals, {n}");
        }

        let (mut rng, mut fill) = (start(), start());
        let got: Vec<f32> = StandardNormal
            .sample_iter(&mut rng)
            .take(n.div_ceil(2))
            .flat_map(|p: [f32; 2]| p)
            .take(n)
            .collect();
        let mut want = vec![0.0; n];
        fill.fill_normal_f32(&mut want);
        assert_eq!(bits32(&got), bits32(&want), "f32 normals, {n}");
        assert_eq!(rng, fill, "f32 normals, {n}");

        let (mut rng, mut fill) = (start(), start());
        let got: Vec<f64> = Exp1.sample_iter(&mut rng).take(n).collect();
        let mut want = vec![0.0; n];
        fill.fill_exponential_f64(&mut want);
        assert_eq!(bits64(&got), bits64(&want), "f64 exponentials, {n}");
        assert_eq!(rng, fill);

        let (mut rng, mut fill) = (start(), start());
        let got: Vec<f32> = Exp1.sample_iter(&mut rng).take(n).collect();
        let mut want = vec![0.0; n];
        fill.fill_exponential_f32(&mut want);
        assert_eq!(bits32(&got), bits32(&want), "f32 exponentials, {n}");
        assert_eq!(rng, fill);
    }
}

#[test]
fn a_rng_by_value_in_sample_iter_is_exact() {
    let mut fill = at(77);
    let mut want = vec![0.0; 3000];
    fill.fill_normal_f64(&mut want);
    let got: Vec<f64> = StandardNormal.sample_iter(at(77)).take(3000).collect();
    assert_eq!(bits64(&got), bits64(&want));
}

/// A generator that is not a `Tandem` to the type check.
struct Wrapped(Tandem);

impl rand::TryRng for Wrapped {
    type Error = core::convert::Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.0.next_u32())
    }
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.0.next_u64())
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.0.fill_u8(dst);
        Ok(())
    }
}

#[test]
fn foreign_generators_get_standard_normals() {
    // The fast path is the stream's, so the first draws agree up to the first miss.
    let n = 1 << 20;
    let mut w = Wrapped(start());
    let mut fill = vec![0.0; n];
    start().fill_normal_f64(&mut fill);
    let got: Vec<f64> = StandardNormal.sample_iter(&mut w).take(n).collect();
    let agree = got
        .iter()
        .zip(&fill)
        .take_while(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    assert!(agree > 0 && agree < n, "{agree} leading draws agree");

    let dynamic: &mut dyn Rng = &mut w;
    let z: Vec<f64> = (0..n).map(|_| StandardNormal.sample(dynamic)).collect();
    let mut std_rng = rand::rngs::StdRng::seed_from_u64(1);
    for z in [
        z,
        StandardNormal.sample_iter(&mut std_rng).take(n).collect(),
    ] {
        let mean = z.iter().sum::<f64>() / n as f64;
        let var = z.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!(
            mean.abs() < 5e-3 && (var - 1.0).abs() < 5e-3,
            "{mean} {var}"
        );
    }
}

#[cfg(feature = "std")]
mod hashes {
    //! The dumps of tandem-c's tools/dump_normals.c: 1e6 `f64` and 2e6 - 1 `f32` normals from
    //! each of five positions of the generator seeded `(2026, 7)`.
    use super::*;

    const STARTS: [u64; 5] = [0, 1, 77, 12345, 1 << 30];

    fn fnv(h: u64, bytes: &[u8]) -> u64 {
        bytes
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
    }

    fn dump(start: u64) -> Tandem {
        let mut rng = Tandem::new(2026 | 7 << 64);
        rng.set_position(start);
        rng
    }

    #[test]
    fn f64_normals_hash_like_tandem_c() {
        let h = STARTS.iter().fold(0xcbf2_9ce4_8422_2325, |h, &s| {
            let mut rng = dump(s);
            (0..1_000_000).fold(h, |h, _| {
                let z: f64 = rng.sample(StandardNormal);
                fnv(h, &z.to_le_bytes())
            })
        });
        assert_eq!(h, 0xa61c_fa84_4c85_f7c1, "hash {h:016x}");
    }

    #[test]
    fn f32_normals_hash_like_tandem_c() {
        let h = STARTS.iter().fold(0xcbf2_9ce4_8422_2325, |h, &s| {
            StandardNormal
                .sample_iter(dump(s))
                .flat_map(|p: [f32; 2]| p)
                .take(1_999_999)
                .fold(h, |h, z| fnv(h, &z.to_le_bytes()))
        });
        assert_eq!(h, 0xaa1e_a656_ce73_a4fb, "hash {h:016x}");
    }
}
