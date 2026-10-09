//! The `rand` distributions equal the inherent draws on the spec's conformance cases and in the
//! dump hashes, and `sample_iter` equals the fills across their passes.
#![cfg(feature = "rand")]

mod common;

use common::*;
use rand::distr::Distribution;
use rand::{Rng, RngExt, SeedableRng};
use tandem_rng::{Below, Exp1, StandardNormal, Tandem};

fn at(pos: u64) -> Tandem {
    let mut rng = Tandem::new(42);
    rng.set_position(pos);
    rng
}

/// One bit draw leaves the position unaligned.
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
fn normals_are_the_inherent_draws_on_the_fixtures() {
    for c in cases("normal.json").iter().filter(|c| n(c) > 0) {
        let n = n(c);
        if c["kind"] == "fill_normal_f64" {
            let mut inherent = rng(c);
            let scalar: Vec<f64> = (0..n).map(|_| inherent.normal_f64()).collect();
            let mut r = rng(c);
            let got: Vec<f64> = (0..n).map(|_| r.sample(StandardNormal)).collect();
            assert_eq!(bits64(&got), bits64(&scalar), "sample, {}", id(c));
            assert_eq!(r, inherent);
            let mut r = rng(c);
            let got: Vec<f64> = StandardNormal.sample_iter(&mut r).take(n).collect();
            assert_eq!(bits64(&got), bits64(&scalar), "sample_iter, {}", id(c));
            assert_eq!(r, inherent);
            assert!(
                got.iter().zip(f64s(c)).all(|(g, w)| same_f64(*g, w)),
                "{}",
                id(c)
            );
        } else {
            // The flattened [f32; 2] pairs are the fill, and an f32 sample is the cos half.
            let mut r = rng(c);
            let got: Vec<f32> = StandardNormal
                .sample_iter(&mut r)
                .take(n.div_ceil(2))
                .flat_map(|p: [f32; 2]| p)
                .take(n)
                .collect();
            assert!(
                got.iter().zip(f32s(c)).all(|(g, w)| same_f32(*g, w)),
                "{}",
                id(c)
            );
            let (mut a, mut b) = (rng(c), rng(c));
            for _ in 0..n {
                let z: f32 = a.sample(StandardNormal);
                assert_eq!(z.to_bits(), b.normal_f32().to_bits());
            }
            assert_eq!(a, b);
        }
    }
}

#[test]
fn exponentials_are_the_inherent_draws_on_the_fixtures() {
    for c in cases("exponential.json").iter().filter(|c| n(c) > 0) {
        let (mut r, mut inherent) = (rng(c), rng(c));
        if c["kind"] == "fill_exponential_f64" {
            let got: Vec<f64> = Exp1.sample_iter(&mut r).take(n(c)).collect();
            let scalar: Vec<f64> = (0..n(c)).map(|_| inherent.exponential_f64()).collect();
            assert_eq!(bits64(&got), bits64(&scalar), "{}", id(c));
            assert!(
                got.iter().zip(f64s(c)).all(|(g, w)| same_f64(*g, w)),
                "{}",
                id(c)
            );
        } else {
            let got: Vec<f32> = (0..n(c)).map(|_| r.sample(Exp1)).collect();
            let scalar: Vec<f32> = (0..n(c)).map(|_| inherent.exponential_f32()).collect();
            assert_eq!(bits32(&got), bits32(&scalar), "{}", id(c));
            assert!(
                got.iter().zip(f32s(c)).all(|(g, w)| same_f32(*g, w)),
                "{}",
                id(c)
            );
        }
        assert_eq!(r, inherent);
    }
}

#[test]
fn below_is_the_inherent_draw_on_the_fixtures() {
    for c in cases("below.json") {
        let (mut r, range) = (rng(&c), hex(&c["range"]));
        let got: Vec<u64> = if c["kind"] == "below_u32" {
            Below(range as u32)
                .sample_iter(&mut r)
                .take(n(&c))
                .map(u64::from)
                .collect()
        } else {
            (0..n(&c)).map(|_| r.sample(Below(range))).collect()
        };
        assert_eq!(got, hexes(&c["values"]), "{}", id(&c));
        assert_eq!(r.position(), int(&c, "end"), "{}", id(&c));
    }
}

#[cfg(feature = "alloc")]
#[test]
fn choice_is_the_inherent_draw_on_the_fixtures() {
    for c in cases("choice.json").iter().filter(|c| n(c) > 0) {
        let w: Vec<f64> = hexes(&c["weights"])
            .into_iter()
            .map(f64::from_bits)
            .collect();
        let t = tandem_rng::ChoiceTable::new(&w).unwrap();
        let got: Vec<u64> = t.sample_iter(rng(c)).take(n(c)).map(u64::from).collect();
        assert_eq!(got, hexes(&c["values"]), "{}", id(c));
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

mod hashes {
    //! The `f64` and `f32` normal dumps of hashes.json through `rng.sample` and `sample_iter`.
    use super::*;
    use serde_json::Value;

    fn fnv(h: u64, bytes: &[u8]) -> u64 {
        bytes
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
    }

    fn dump(id: &str) -> Value {
        let all = load("hashes.json")["dumps"].as_array().unwrap().clone();
        all.into_iter().find(|d| d["id"] == id).expect(id)
    }

    /// The generators of the dump at each of its starts.
    fn starts(d: &Value) -> Vec<Tandem> {
        let starts = d["starts"].as_array().unwrap();
        starts
            .iter()
            .map(|s| Tandem::from_key(key(d), s.as_u64().unwrap(), int(d, "K") as u32))
            .collect()
    }

    fn want(d: &Value) -> u64 {
        hex(&d["fnv1a"])
    }

    #[test]
    fn f64_normals_hash_like_tandem_c() {
        let d = dump("tools/dump_normals.c");
        let h = starts(&d)
            .into_iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, mut r| {
                (0..1_000_000).fold(h, |h, _| {
                    let z: f64 = r.sample(StandardNormal);
                    fnv(h, &z.to_le_bytes())
                })
            });
        assert_eq!(h, want(&d), "hash {h:016x}");
    }

    #[test]
    fn f32_normals_hash_like_tandem_c() {
        let d = dump("tests/test_normal_bits.c normal f32");
        let h = starts(&d).into_iter().fold(0xcbf2_9ce4_8422_2325, |h, r| {
            StandardNormal
                .sample_iter(r)
                .flat_map(|p: [f32; 2]| p)
                .take(1_999_999)
                .fold(h, |h, z| fnv(h, &z.to_le_bytes()))
        });
        assert_eq!(h, want(&d), "hash {h:016x}");
    }
}
