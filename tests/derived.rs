//! Bounded integers and normals agree with the shared device core, and fills agree with
//! scalar draws. The fixed values come from tools/gen_derived.cpp.

// The reference values carry 17 digits as the device core prints them.
#[allow(clippy::excessive_precision)]
mod derived_data;

use derived_data::{BELOW_U32, BELOW_U64, NORMAL_F64};
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
fn normal_matches_the_device_core() {
    // libm and the C library differ in the last bits of log and cos.
    let (want, end) = NORMAL_F64;
    let mut rng = start();
    for (i, want) in want.iter().enumerate() {
        let got = rng.normal_f64();
        assert!(
            (got - want).abs() <= 1e-13 * want.abs().max(1.0),
            "element {i}: {got} against {want}"
        );
    }
    assert_eq!(rng.position(), end);
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

#[test]
fn fills_are_scalar_draws() {
    // Lengths cross the block of the normal fill and a row.
    for len in [0, 1, 127, 128, 129, 300, 1000] {
        let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
        let mut got = vec![0; len];
        a.fill_below_u32(&mut got, 0xc000_0000);
        assert!(got.iter().all(|&x| x == b.below_u32(0xc000_0000)));
        assert_eq!(a, b, "below_u32 position at {len}");

        let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
        let mut got = vec![0; len];
        a.fill_below_u64(&mut got, 1000);
        assert!(got.iter().all(|&x| x == b.below_u64(1000)));
        assert_eq!(a, b, "below_u64 position at {len}");

        let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
        let mut got = vec![0.0; len];
        a.fill_normal_f64(&mut got);
        assert!(got.iter().all(|&x| x == b.normal_f64()));
        assert_eq!(a, b, "normal_f64 position at {len}");

        let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
        let mut got = vec![0.0; len];
        a.fill_normal_f32(&mut got);
        assert!(got.iter().all(|&x| x == b.normal_f32()));
        assert_eq!(a, b, "normal_f32 position at {len}");
    }
}

#[test]
fn normals_have_unit_moments() {
    // Mean 0 and variance 1 to within 5 standard errors of 2^20 draws.
    let mut z = vec![0.0; 1 << 20];
    Tandem::new(1).fill_normal_f64(&mut z);
    let n = z.len() as f64;
    let mean = z.iter().sum::<f64>() / n;
    let var = z.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    assert!(mean.abs() < 5.0 / n.sqrt(), "mean {mean}");
    assert!((var - 1.0).abs() < 5.0 * (2.0 / n).sqrt(), "variance {var}");
}
