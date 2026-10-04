//! Bounded integers and normals agree with the shared device core, and fills agree with
//! scalar draws. The fixed values come from tools/gen_derived.cpp.

// The reference values carry 17 digits as the device core prints them.
#[allow(clippy::excessive_precision)]
mod derived_data;

use derived_data::{BELOW_U32, BELOW_U64, FILL_BELOW_U32, FILL_BELOW_U64, NORMAL_F32, NORMAL_F64};
use tandem_rng::Tandem;

/// The fixtures start after one bit draw, which leaves the position unaligned.
fn start() -> Tandem {
    let mut rng = Tandem::new(42);
    rng.next_bool();
    rng
}

#[test]
fn below_matches_the_device_core() {
    for (n, want, end) in BELOW_U32 {
        let mut rng = start();
        let got: Vec<u32> = want.iter().map(|_| rng.below_u32(*n)).collect();
        assert_eq!((&got[..], rng.position()), (*want, *end), "below_u32({n})");
    }
    for (n, want, end) in BELOW_U64 {
        let mut rng = start();
        let got: Vec<u64> = want.iter().map(|_| rng.below_u64(*n)).collect();
        assert_eq!((&got[..], rng.position()), (*want, *end), "below_u64({n})");
    }
}

#[test]
fn bound_zero_returns_zero_after_one_draw() {
    let (mut a, mut b) = (Tandem::new(3), Tandem::new(3));
    assert_eq!(a.below_u32(0), 0);
    b.next_u32();
    assert_eq!(a, b);
    assert_eq!(a.below_u64(0), 0);
    b.next_u64();
    assert_eq!(a, b);
}

#[test]
fn normals_match_the_device_core() {
    // libm and the C library differ in the last bits of log, cos and sin.
    let (want, end) = NORMAL_F64;
    let mut rng = start();
    let got: Vec<f64> = (0..want.len() / 2)
        .flat_map(|_| rng.normal2_f64())
        .collect();
    for (i, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "f64 {i}: {got} against {want}"
        );
    }
    assert_eq!(rng.position(), end);
    let mut rng = start();
    let mut got = vec![0.0; want.len()];
    rng.fill_normal_f64(&mut got);
    assert!(
        got.iter()
            .zip(want)
            .all(|(g, w)| (g - w).abs() <= 1e-12 * w.abs().max(1.0))
    );
    assert_eq!(rng.position(), end);

    let (want, end) = NORMAL_F32;
    let tol = |w: &f32| 8.0 * f32::EPSILON * w.abs() + 1e-6;
    let mut rng = start();
    let got: Vec<f32> = (0..want.len() / 2)
        .flat_map(|_| rng.normal2_f32())
        .collect();
    for (i, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= tol(want),
            "f32 {i}: {got} against {want}"
        );
    }
    assert_eq!(rng.position(), end);
    let mut rng = start();
    let mut got = vec![0.0; want.len()];
    rng.fill_normal_f32(&mut got);
    assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tol(w)));
    assert_eq!(rng.position(), end);
}

#[test]
fn fill_below_matches_the_device_core() {
    for (at, n, want, end) in FILL_BELOW_U32 {
        let mut rng = Tandem::new(42);
        rng.set_position(*at);
        let mut got = vec![0; want.len()];
        rng.fill_below_u32(&mut got, *n);
        assert_eq!(
            (&got[..], rng.position()),
            (*want, *end),
            "fill_below_u32({n}) at {at}"
        );
    }
    for (at, n, want, end) in FILL_BELOW_U64 {
        let mut rng = Tandem::new(42);
        rng.set_position(*at);
        let mut got = vec![0; want.len()];
        rng.fill_below_u64(&mut got, *n);
        assert_eq!(
            (&got[..], rng.position()),
            (*want, *end),
            "fill_below_u64({n}) at {at}"
        );
    }
}

#[test]
fn normal_f32_is_box_muller_of_two_f32_draws() {
    // The oracle is the exact formula in f64 on the same two draws. The f32 rounding of
    // the argument, the log and the root adds up to a few ulps against it.
    let (mut a, mut b) = (start(), start());
    for i in 0..1000 {
        let got = a.normal_f32();
        let (u, v) = (f64::from(b.next_f32()), f64::from(b.next_f32()));
        let want = (-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * v).cos();
        let tol = 16.0 * f64::from(f32::EPSILON) * want.abs().max(1.0);
        assert!(
            (f64::from(got) - want).abs() <= tol,
            "element {i}: {got} against {want}"
        );
    }
    assert_eq!(a, b);
}

/// The bounded fill from its definition: element `i` takes draw `i` of the raw fill, and a
/// rejected draw retries on `sub(purpose).split(g)` of the key at position 0, `g` being the
/// global draw index.
fn below_by_definition(rng: &Tandem, raw: &[u64], n: u64, wide: bool, first: u64) -> Vec<u64> {
    let (bits, purpose) = if wide {
        (64, 0x0042_4c57_3634)
    } else {
        (32, 0x0042_4c57_3332)
    };
    let sub = Tandem::from_key(rng.key(), 0, rng.chunk_length()).sub(purpose);
    let draw = |r: &mut Tandem| {
        if wide {
            r.next_u64()
        } else {
            u64::from(r.next_u32())
        }
    };
    let reject = (1u128 << bits) % u128::from(n.max(1));
    raw.iter()
        .enumerate()
        .map(|(i, &u)| {
            let mut m = u128::from(u) * u128::from(n);
            let low = |m: u128| m & ((1u128 << bits) - 1);
            if low(m) < reject {
                let mut r = sub.split(first + i as u64);
                m = u128::from(draw(&mut r)) * u128::from(n);
                while low(m) < reject {
                    m = u128::from(draw(&mut r)) * u128::from(n);
                }
            }
            (m >> bits) as u64
        })
        .collect()
}

#[test]
fn bounded_fills_follow_the_definition() {
    // Bounds near 2^32 and 2^64 reject a quarter of the draws, so the retry runs.
    for len in [0, 1, 31, 32, 33, 1000, 5000] {
        for n in [0, 1, 3, 1000, 0xc000_0000u32] {
            let mut rng = start();
            let mut raw = start();
            let mut got = vec![0; len];
            rng.fill_below_u32(&mut got, n);
            let mut words = vec![0; len];
            raw.fill_u32(&mut words);
            let words: Vec<u64> = words.iter().map(|&w| u64::from(w)).collect();
            let want = below_by_definition(&rng, &words, u64::from(n), false, 1);
            assert!(
                got.iter().map(|&x| u64::from(x)).eq(want),
                "u32 n={n} len={len}"
            );
            assert_eq!(rng, raw, "u32 consumes exactly len draws");
        }
        for n in [0, 1, 3, 1_000_000_000_000, 0xc000_0000_0000_0000u64] {
            let mut rng = start();
            let mut raw = start();
            let mut got = vec![0; len];
            rng.fill_below_u64(&mut got, n);
            let mut words = vec![0; len];
            raw.fill_u64(&mut words);
            assert_eq!(
                got,
                below_by_definition(&rng, &words, n, true, 1),
                "u64 n={n} len={len}"
            );
            assert_eq!(rng, raw, "u64 consumes exactly len draws");
        }
    }
}

#[test]
fn bounded_fill_without_rejection_is_the_scalar_draws() {
    // A power of two never rejects.
    let (mut a, mut b) = (start(), start());
    let mut got = vec![0; 777];
    a.fill_below_u32(&mut got, 1 << 20);
    assert!(got.iter().all(|&x| x == b.below_u32(1 << 20)));
    assert_eq!(a, b);
}

#[test]
fn normal_fills_are_flattened_pairs() {
    // Lengths cross the block of the normal fill and a row, and include odd ones, which
    // use the cos half of the last pair and still consume both of its draws.
    for len in [0, 1, 2, 127, 128, 129, 255, 257, 300, 1001] {
        let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
        let mut got = vec![0.0; len];
        a.fill_normal_f64(&mut got);
        let want: Vec<f64> = (0..len.div_ceil(2)).flat_map(|_| b.normal2_f64()).collect();
        assert_eq!(got, want[..len], "normal_f64 at {len}");
        assert_eq!(a, b, "normal_f64 position at {len}");

        let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
        let mut got = vec![0.0; len];
        a.fill_normal_f32(&mut got);
        let want: Vec<f32> = (0..len.div_ceil(2)).flat_map(|_| b.normal2_f32()).collect();
        assert_eq!(got, want[..len], "normal_f32 at {len}");
        assert_eq!(a, b, "normal_f32 position at {len}");
    }
}

#[test]
fn scalar_normal_is_the_cos_half() {
    let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
    assert_eq!(a.normal_f64(), b.normal2_f64()[0]);
    assert_eq!(a.normal_f32(), b.normal2_f32()[0]);
    assert_eq!(a, b);
}

#[test]
fn normals_have_unit_moments() {
    // Mean 0 and variance 1 to within 5 standard errors of 2^20 draws.
    let mut z = vec![0.0; 1 << 20];
    let mut y = vec![0.0f32; 1 << 20];
    Tandem::new(1).fill_normal_f64(&mut z);
    Tandem::new(1).fill_normal_f32(&mut y);
    let y: Vec<f64> = y.iter().map(|&x| f64::from(x)).collect();
    let n = z.len() as f64;
    for (name, z) in [("f64", z), ("f32", y)] {
        let mean = z.iter().sum::<f64>() / n;
        let var = z.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 5.0 / n.sqrt(), "{name} mean {mean}");
        assert!(
            (var - 1.0).abs() < 5.0 * (2.0 / n).sqrt(),
            "{name} variance {var}"
        );
    }
}

#[test]
fn normal_pairs_are_box_muller_to_an_ulp() {
    // The oracle is the formula through the system sin and cos on the same draws.
    let (mut a, mut b) = (Tandem::new(5), Tandem::new(5));
    for i in 0..200_000 {
        let z = a.normal2_f64();
        let (u, v) = (b.next_f64(), b.next_f64());
        let r = (-2.0 * (1.0 - u).ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * v).sin_cos();
        // The oracle's rounded angle 2 pi v is off by up to 2^-53 * 2 pi v, about 7e-16.
        let tol = 1.5e-15 * r.max(1.0);
        assert!(
            (z[0] - r * c).abs() <= tol && (z[1] - r * s).abs() <= tol,
            "pair {i}: {z:?} against {:?}",
            [r * c, r * s]
        );
    }
}

#[test]
fn bounded_fills_cut_anywhere_equal_the_whole() {
    // The fallback is keyed by the global draw index, so a fill split at any element and
    // continued gives the whole fill, rejected elements included. The bounds reject a quarter
    // of the draws, and the start is unaligned.
    let (n32, n64) = (0xc000_0000u32, 0xc000_0000_0000_0000u64);
    for cut in [0, 1, 7, 100, 777, 1000] {
        let mut whole = start();
        let mut a = vec![0u32; 1000];
        whole.fill_below_u32(&mut a, n32);
        let mut parts = start();
        let mut b = vec![0u32; 1000];
        let (head, tail) = b.split_at_mut(cut);
        parts.fill_below_u32(head, n32);
        parts.fill_below_u32(tail, n32);
        assert_eq!(a, b, "u32 cut at {cut}");
        assert_eq!(whole, parts);

        let mut whole = start();
        let mut a = vec![0u64; 1000];
        whole.fill_below_u64(&mut a, n64);
        let mut parts = start();
        let mut b = vec![0u64; 1000];
        let (head, tail) = b.split_at_mut(cut);
        parts.fill_below_u64(head, n64);
        parts.fill_below_u64(tail, n64);
        assert_eq!(a, b, "u64 cut at {cut}");
        assert_eq!(whole, parts);
    }
}
