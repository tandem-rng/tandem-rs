//! Bounded integers and normals agree with the shared device core, and fills agree with
//! scalar draws. The fixed values come from tools/gen_derived.cpp.

mod derived_data;

use derived_data::{BELOW_U32, BELOW_U64, NORMAL};
use tandem_rng::Tandem;

#[test]
fn below_matches_the_device_core() {
    for (n, want, end) in BELOW_U32 {
        let mut rng = Tandem::new(42);
        let got = want.map(|_| rng.below_u32(*n));
        assert_eq!((&got, rng.position()), (want, *end), "below_u32({n})");
    }
    for (n, want, end) in BELOW_U64 {
        let mut rng = Tandem::new(42);
        let got = want.map(|_| rng.below_u64(*n));
        assert_eq!((&got, rng.position()), (want, *end), "below_u64({n})");
    }
}

#[test]
fn normal_matches_the_device_core() {
    // libm and the C library may differ in the last bits of log and cos.
    let (want, end) = NORMAL;
    let mut rng = Tandem::new(42);
    for (i, bits) in want.iter().enumerate() {
        let (got, want) = (rng.normal_f64(), f64::from_bits(*bits));
        assert!(
            (got - want).abs() <= 1e-14 * want.abs().max(1.0),
            "element {i}: {got} against {want}"
        );
    }
    assert_eq!(rng.position(), end);
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
