//! The spec's conformance cases, tests/conformance/*.json, and every item of its
//! conformance/CHECKLIST.md. Each fill case also checks its end position, the scalar draws, and
//! the fill cut at elements 1, 7, 20, 21 and n - 1, or 2, 8, 20 and the largest even element
//! below n for `f32` normals, and continued on one generator. The checklist is tandem-spec
//! b31af72's.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
#[cfg(feature = "alloc")]
use tandem_rng::ChoiceTable;
use tandem_rng::Tandem;

fn align(p: u64, w: u64) -> u64 {
    p.next_multiple_of(w)
}

/// The end position of a fill case by Appendix A and C: an empty bounded, `f32` normal or
/// exponential fill stays at its start, an empty `f64` normal or choice fill aligns.
fn end_of(c: &Value) -> u64 {
    let (start, n) = (int(c, "start"), n(c) as u64);
    let (w, draws, empty_aligns) = match c["kind"].as_str().unwrap() {
        "fill_below_u32" | "fill_exponential_f32" => (32, n, false),
        "fill_below_u64" | "fill_exponential_f64" => (64, n, false),
        "fill_normal_f32" => (32, n.next_multiple_of(2), false),
        "fill_normal_f64" | "fill_choice" => (64, n, true),
        k => panic!("{k}"),
    };
    if n == 0 && !empty_aligns {
        start
    } else {
        align(start, w) + w * draws
    }
}

/// Run `fill` on the case's generator, check the end position, and check that the fill cut
/// at each of `cuts` and continued equals the whole fill. Returns the whole fill.
fn whole_and_cuts<T: PartialEq + std::fmt::Debug>(
    c: &Value,
    pairs: bool,
    fill: impl Fn(&mut Tandem, usize) -> Vec<T>,
) -> Vec<T> {
    let n = n(c);
    let mut whole = rng(c);
    let got = fill(&mut whole, n);
    assert_eq!(whole.position(), end_of(c), "{} end", id(c));
    if let Some(end) = c["end"].as_u64() {
        assert_eq!(end, end_of(c), "{} fixture end", id(c));
    }
    // A cut inside an `f32` pair would drop a sin half, so those fills cut at the pair
    // boundaries 2, 8, 20 and the largest even element below n.
    let cuts = if pairs {
        [2, 8, 20, n.saturating_sub(1) & !1, 0]
    } else {
        [1, 7, 20, 21, n.saturating_sub(1)]
    };
    for cut in cuts {
        if cut == 0 || cut >= n {
            continue;
        }
        let mut parts = rng(c);
        let mut v = fill(&mut parts, cut);
        v.extend(fill(&mut parts, n - cut));
        assert_eq!(v, got, "{} cut at {cut}", id(c));
        assert_eq!(parts, whole, "{} position after the cut at {cut}", id(c));
    }
    got
}

fn below_fill(c: &Value) -> impl Fn(&mut Tandem, usize) -> Vec<u64> {
    let (wide, range) = (c["kind"] == "fill_below_u64", hex(&c["range"]));
    move |r, n| {
        if wide {
            let mut o = vec![0; n];
            r.fill_below_u64(&mut o, range);
            o
        } else {
            let mut o = vec![0; n];
            r.fill_below_u32(&mut o, range as u32);
            o.into_iter().map(u64::from).collect()
        }
    }
}

fn normal_f64_fill(r: &mut Tandem, n: usize) -> Vec<u64> {
    let mut o = vec![0.0; n];
    r.fill_normal_f64(&mut o);
    o.into_iter().map(f64::to_bits).collect()
}

fn normal_f32_fill(r: &mut Tandem, n: usize) -> Vec<u32> {
    let mut o = vec![0.0; n];
    r.fill_normal_f32(&mut o);
    o.into_iter().map(f32::to_bits).collect()
}

#[test]
fn scalar_bounded_draws() {
    for c in cases("below.json") {
        let (mut r, range) = (rng(&c), hex(&c["range"]));
        let got: Vec<u64> = (0..n(&c))
            .map(|_| match c["kind"].as_str().unwrap() {
                "below_u32" => u64::from(r.below_u32(range as u32)),
                _ => r.below_u64(range),
            })
            .collect();
        assert_eq!(got, hexes(&c["values"]), "{}", id(&c));
        assert_eq!(r.position(), int(&c, "end"), "{} end", id(&c));
    }
}

#[test]
fn bounded_fills() {
    // The cases with `rejected > 0` retry on the fallback of their global draw index.
    let all = cases("fill_below.json");
    assert!(all.iter().any(|c| c["rejected"].as_u64() > Some(0)));
    for c in &all {
        let got = whole_and_cuts(c, false, below_fill(c));
        assert_eq!(got, hexes(&c["values"]), "{}", id(c));
    }
}

#[test]
fn normal_fills() {
    for c in cases("normal.json") {
        if c["kind"] == "fill_normal_f64" {
            let got = whole_and_cuts(&c, false, normal_f64_fill);
            let want = f64s(&c);
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(g, w)| same_f64(f64::from_bits(*g), *w)),
                "{}",
                id(&c)
            );
            // n scalar draws equal the fill and end at the same position.
            let mut r = rng(&c);
            let scalar: Vec<u64> = (0..n(&c)).map(|_| r.normal_f64().to_bits()).collect();
            if n(&c) > 0 {
                assert_eq!((scalar, r.position()), (got, end_of(&c)), "{}", id(&c));
            }
        } else {
            let got = whole_and_cuts(&c, true, normal_f32_fill);
            let want = f32s(&c);
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(g, w)| same_f32(f32::from_bits(*g), *w)),
                "{}",
                id(&c)
            );
            // The fill is the flattened Box-Muller pairs of the scalar draws.
            let mut r = rng(&c);
            let pairs: Vec<u32> = (0..n(&c).div_ceil(2))
                .flat_map(|_| r.normal2_f32())
                .map(f32::to_bits)
                .collect();
            assert_eq!(pairs[..n(&c)], got, "{} pairs", id(&c));
        }
    }
}

#[test]
fn exponential_fills() {
    for c in cases("exponential.json") {
        let mut r = rng(&c);
        if c["kind"] == "fill_exponential_f64" {
            let got = whole_and_cuts(&c, false, |r, n| {
                let mut o = vec![0.0; n];
                r.fill_exponential_f64(&mut o);
                o
            });
            assert!(
                got.iter().zip(f64s(&c)).all(|(g, w)| same_f64(*g, w)),
                "{}",
                id(&c)
            );
            let scalar: Vec<f64> = (0..n(&c)).map(|_| r.exponential_f64()).collect();
            assert_eq!(scalar, got, "{} scalar", id(&c));
        } else {
            let got = whole_and_cuts(&c, false, |r, n| {
                let mut o = vec![0.0; n];
                r.fill_exponential_f32(&mut o);
                o
            });
            assert!(
                got.iter().zip(f32s(&c)).all(|(g, w)| same_f32(*g, w)),
                "{}",
                id(&c)
            );
            let scalar: Vec<f32> = (0..n(&c)).map(|_| r.exponential_f32()).collect();
            assert_eq!(scalar, got, "{} scalar", id(&c));
        }
        assert_eq!(r.position(), end_of(&c), "{} scalar end", id(&c));
    }
}

#[cfg(feature = "alloc")]
fn weights(c: &Value) -> Vec<f64> {
    hexes(&c["weights"])
        .into_iter()
        .map(f64::from_bits)
        .collect()
}

#[cfg(feature = "alloc")]
#[test]
fn choice_fills() {
    let all = cases("choice.json");
    assert_eq!(all.iter().filter(|c| c.get("cut").is_some()).count(), 5);
    for c in &all {
        let t = ChoiceTable::new(&weights(c)).expect("valid weights");
        assert_eq!(t.capacity(), hex(&c["capacity"]), "{} capacity", id(c));
        if c.get("cut").is_some() {
            assert_eq!(t.cut(), hexes(&c["cut"]), "{} cut", id(c));
            let alias: Vec<u64> = t.alias().iter().map(|&a| u64::from(a)).collect();
            assert_eq!(alias, hexes(&c["alias"]), "{} alias", id(c));
        }
        let got = whole_and_cuts(c, false, |r, n| {
            let mut o = vec![0; n];
            r.fill_choice(&mut o, &t);
            o.into_iter().map(u64::from).collect()
        });
        assert_eq!(got, hexes(&c["values"]), "{}", id(c));
        // A scalar draw is element i of the fill and consumes 64 bits.
        let mut r = rng(c);
        let scalar: Vec<u64> = (0..n(c)).map(|_| u64::from(r.choice(&t))).collect();
        assert_eq!(scalar, got, "{} scalar", id(c));
        if n(c) > 0 {
            assert_eq!(r.position(), end_of(c), "{} scalar end", id(c));
        }
    }
}

#[cfg(feature = "alloc")]
#[test]
fn choice_of_one_weight_and_rejected_weights() {
    let t = ChoiceTable::new(&[0.25]).unwrap();
    let mut r = Tandem::new(9);
    assert_eq!(r.choice(&t), 0);
    assert_eq!(r.position(), 64);
    for w in [
        &[][..],
        &[1.0, -1.0],
        &[1.0, f64::NAN],
        &[1.0, f64::INFINITY],
        &[0.0, -0.0],
    ] {
        assert!(ChoiceTable::new(w).is_err(), "{w:?}");
    }
}

#[test]
fn fallback_is_keyed_by_the_global_draw_index() {
    // A start one draw later shifts the output by one element, rejected elements and
    // ziggurat misses included, because the fallback of element i is keyed by
    // g = align(start, w) / w + i.
    let below = cases("fill_below.json");
    for (at, from0) in [
        ("CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]"),
        ("CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]"),
    ] {
        let (a, b) = (case(&below, at), case(&below, from0));
        assert_eq!(int(a, "start"), 1);
        let shifted = &below_fill(b)(&mut rng(b), n(a) + 1)[1..];
        assert_eq!(below_fill(a)(&mut rng(a), n(a)), shifted, "{at}");
    }
    let normal = cases("normal.json");
    let (a, b) = (
        case(&normal, "CROSS_NORMAL[1]"),
        case(&normal, "CROSS_NORMAL[0]"),
    );
    let shifted = &normal_f64_fill(&mut rng(b), n(a) + 1)[1..];
    assert_eq!(normal_f64_fill(&mut rng(a), n(a)), shifted);
    #[cfg(feature = "alloc")]
    {
        let choice = cases("choice.json");
        let (a, b) = (
            case(&choice, "CROSS_CHOICE[1]"),
            case(&choice, "CROSS_CHOICE[0]"),
        );
        assert_eq!((int(a, "start"), weights(a)), (1, weights(b)));
        let t = ChoiceTable::new(&weights(a)).unwrap();
        let fill = |c: &Value, n| {
            let mut o = vec![0; n];
            rng(c).fill_choice(&mut o, &t);
            o
        };
        assert_eq!(fill(a, n(a)), fill(b, n(a) + 1)[1..]);
    }
}

#[test]
fn width_is_named_by_the_interface() {
    // The bounded draws of this crate name their width, so range 1000 with a u64 result is
    // CROSS_BELOW64[3] and with a u32 result CROSS_BELOW32[3].
    let below = cases("fill_below.json");
    for name in ["CROSS_BELOW32[3]", "CROSS_BELOW64[3]"] {
        let c = case(&below, name);
        assert_eq!(hex(&c["range"]), 1000);
        assert_eq!(
            below_fill(c)(&mut rng(c), n(c)),
            hexes(&c["values"]),
            "{name}"
        );
    }
    // Range 0 returns 0 after one draw of the width.
    let mut r = Tandem::new(3);
    assert_eq!((r.below_u32(0), r.position()), (0, 32));
    assert_eq!((r.below_u64(0), r.position()), (0, 128));
}

#[test]
fn float32_normal_pairs() {
    let normal = cases("normal.json");
    let pairs = case(&normal, "CROSS_NORMALF");
    assert_eq!((int(pairs, "start"), int(pairs, "end")), (1, 4128));
    let n32 = |name| normal_f32_fill(&mut rng(case(&normal, name)), 33);
    assert_eq!(
        normal_f32_fill(&mut rng(pairs), 128)[..33],
        n32("CROSS_NORMAL32[1]")
    );
    // A start one pair later shifts the output by one pair.
    let c2 = case(&normal, "CROSS_NORMAL32[2]");
    assert_eq!(normal_f32_fill(&mut rng(c2), 33), {
        let c0 = case(&normal, "CROSS_NORMAL32[0]");
        normal_f32_fill(&mut rng(c0), 35)[2..].to_vec()
    });
    // An odd fill ends after both draws of its last pair.
    let c0 = case(&normal, "CROSS_NORMAL32[0]");
    let mut r = rng(c0);
    normal_f32_fill(&mut r, 33);
    assert_eq!(r.position(), 1088);
    // A scalar f32 normal is the cos half of two draws.
    let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
    assert_eq!(a.normal_f32().to_bits(), b.normal2_f32()[0].to_bits());
    assert_eq!((a.position(), a), (64, b));
}

fn sha256_hex(h: Sha256) -> String {
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn stream_hashes() {
    // The streams cross 128-bit blocks, rows and chunks, and K = 8 crosses a chunk every 8 rows.
    for s in load("hashes.json")["streams"].as_array().unwrap() {
        let (mut r, n) = (rng(s), n(s));
        let bytes: Vec<u8> = match s["type"].as_str().unwrap() {
            "UInt8" => {
                let mut o = vec![0u8; n];
                r.fill_u8(&mut o);
                o
            }
            "Bool" => {
                let mut o = vec![false; n];
                r.fill_bool(&mut o);
                o.into_iter().map(u8::from).collect()
            }
            "UInt32" => le(n, |o: &mut [u32]| r.fill_u32(o), u32::to_le_bytes),
            "UInt64" => le(n, |o: &mut [u64]| r.fill_u64(o), u64::to_le_bytes),
            "UInt128" => le(n, |o: &mut [u128]| r.fill_u128(o), u128::to_le_bytes),
            "Float32" => le(n, |o: &mut [f32]| r.fill_f32(o), f32::to_le_bytes),
            "Float64" => le(n, |o: &mut [f64]| r.fill_f64(o), f64::to_le_bytes),
            "Float16" => le(n, |o: &mut [u16]| r.fill_f16_bits(o), u16::to_le_bytes),
            "Char" => le(
                n,
                |o: &mut [char]| r.fill_char(o),
                |c| u32::from(c).to_le_bytes(),
            ),
            "ComplexF32" => le(
                n,
                |o: &mut [[f32; 2]]| r.fill_c32(o),
                |[a, b]| [a.to_le_bytes(), b.to_le_bytes()].concat(),
            ),
            "ComplexF64" => le(
                n,
                |o: &mut [[f64; 2]]| r.fill_c64(o),
                |[a, b]| [a.to_le_bytes(), b.to_le_bytes()].concat(),
            ),
            t => panic!("{t}"),
        };
        assert_eq!(bytes.len() as u64, int(s, "bytes"));
        let mut h = Sha256::new();
        h.update(&bytes);
        assert_eq!(
            sha256_hex(h),
            s["sha256"].as_str().unwrap(),
            "{}",
            s["file"]
        );
    }
}

/// `n` elements of `fill` as little-endian bytes.
fn le<T: Default + Clone, B: AsRef<[u8]>>(
    n: usize,
    mut fill: impl FnMut(&mut [T]),
    bytes: impl Fn(T) -> B,
) -> Vec<u8> {
    let mut o = vec![T::default(); n];
    fill(&mut o);
    o.into_iter()
        .flat_map(|x| bytes(x).as_ref().to_vec())
        .collect()
}

/// The long dumps of tandem-c, bit identical in every build. The `f32` normal hash holds
/// because the crate copies the C polynomials.
#[test]
fn dump_hashes() {
    fn fnv(h: u64, bytes: &[u8]) -> u64 {
        bytes
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
    }
    for d in load("hashes.json")["dumps"].as_array().unwrap() {
        let (mut f, mut sha, mut bytes) = (0xcbf2_9ce4_8422_2325, Sha256::new(), 0);
        let mut r = Tandem::from_key(key(d), 0, int(d, "K") as u32);
        for start in d["starts"].as_array().unwrap() {
            r = Tandem::from_key(key(d), start.as_u64().unwrap(), int(d, "K") as u32);
            for draw in d["draws"].as_array().unwrap() {
                let n = n(draw);
                let out: Vec<u8> = match draw["kind"].as_str().unwrap() {
                    "fill_normal_f64" => {
                        le(n, |o: &mut [f64]| r.fill_normal_f64(o), f64::to_le_bytes)
                    }
                    "fill_normal_f32" => {
                        le(n, |o: &mut [f32]| r.fill_normal_f32(o), f32::to_le_bytes)
                    }
                    "fill_exponential_f64" => le(
                        n,
                        |o: &mut [f64]| r.fill_exponential_f64(o),
                        f64::to_le_bytes,
                    ),
                    "fill_exponential_f32" => le(
                        n,
                        |o: &mut [f32]| r.fill_exponential_f32(o),
                        f32::to_le_bytes,
                    ),
                    k => panic!("{k}"),
                };
                f = fnv(f, &out);
                sha.update(&out);
                bytes += out.len() as u64;
            }
        }
        let name = d["id"].as_str().unwrap();
        assert_eq!(bytes, int(d, "bytes"), "{name}");
        assert_eq!(format!("{f:016x}"), d["fnv1a"].as_str().unwrap(), "{name}");
        if let Some(want) = d["sha256"].as_str() {
            assert_eq!(sha256_hex(sha), want, "{name}");
        }
        if let Some(end) = d["end"].as_u64() {
            assert_eq!(r.position(), end, "{name}");
        }
    }
}

#[test]
fn complex_draws_span_blocks() {
    // The real part ends block 0 at bit 128 and the imaginary part comes from block 1.
    let mut r = Tandem::new(42);
    r.set_position(64);
    let (re, im) = (r.at_f64(0), r.at_f64(1));
    assert_eq!(r.next_c64(), [re, im]);
    r.set_position(96);
    let (re, im) = (r.at_f32(0), r.at_f32(1));
    assert_eq!(r.next_c32(), [re, im]);
}

#[test]
fn start_positions_lie_below_2_63() {
    // The checklist's fill that reaches 2^64 cannot be expressed here: starts lie below 2^63,
    // and from there a slice would need 2^57 or more elements to reach 2^64 bits. A start
    // of 2^63 or more panics instead, in from_key, set_position and Deserialize.
    let last = (1u64 << 63) - 1;
    let key = [1, 2, 3, 4];
    let mut r = Tandem::from_key(key, last, 32);
    for bad in [1u64 << 63, u64::MAX] {
        assert!(
            catch_unwind(|| Tandem::from_key(key, bad, 32)).is_err(),
            "{bad}"
        );
        let before = r;
        assert!(catch_unwind(AssertUnwindSafe(|| r.set_position(bad))).is_err());
        assert_eq!(
            r, before,
            "a rejected position leaves the generator unchanged"
        );
    }
    // A u64 draw at 2^63 - 1 aligns to 2^63.
    let want = r.at_u64(0);
    assert_eq!(r.next_u64(), want);
    assert_eq!(r.position(), (1 << 63) + 64);
}
